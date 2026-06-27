from collections import defaultdict
import datetime as dt
import json
import logging
from pathlib import Path
import re
import tempfile
from typing import Literal, TypedDict
import zipfile

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    format_document,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
import streamlit as st

from constants import (
    CONTEXT_WINDOW,
    EMBEDDINGS_MODEL_NAME,
    EXPECTED_SN_VERSION,
    FETCH_ALL_MAX,
    MODEL_NAME,
    MONTH_MAP,
    PERSIST_DIR,
    SEARCH_K,
    SINGLE_PASS_BUDGET,
    TAG_ALIASES,
    TOKEN_WARN_RATIO,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("python_multipart").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# --- Parser ---
class StandardNotesData(TypedDict):
    uuid: str
    title: str
    text: str
    date: dt.date
    tags: set[str]


class StandardNotesTag(TypedDict):
    title: str
    references: list[str]


def parse_standard_notes(
    backup_zip_path: Path, notes_json: str = "Standard Notes Backup and Import File.txt"
) -> list[StandardNotesData]:
    tag_data = []
    sn_data = None
    tags_path = Path("Items/Tag")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        with zipfile.ZipFile(backup_zip_path) as zf:
            zf.extractall(tmp_dir)
        with open(tmp_dir / notes_json) as f:
            sn_data = json.load(f)

        if (v := sn_data.get("version")) != EXPECTED_SN_VERSION:
            logger.warning(
                "Standard Notes backup version changed. Expected %r, found %r.",
                EXPECTED_SN_VERSION,
                v,
            )

        tag_file_paths = (tmp_dir / tags_path).glob("*.txt")
        for tag_file_path in tag_file_paths:
            with open(tag_file_path) as f:
                tag_file_data = json.load(f)
                tag_data.append(
                    StandardNotesTag(
                        title=tag_file_data["title"],
                        references=[r["uuid"] for r in tag_file_data["references"]],
                    )
                )

    id_tag_map = defaultdict(set)
    for item in tag_data:
        title = item["title"]
        for ref in item["references"]:
            id_tag_map[ref].add(title)

    iso_date_fmt = "yyyy-mm-dd"
    parsed_data = []

    for item in sn_data["items"]:
        if not item.get("deleted") and item.get("content_type") == "Note":
            content = item["content"]
            uuid = item["uuid"]
            tags = id_tag_map[uuid]
            text = content.get("text", "")
            title = content.get("title", "")

            try:
                date = dt.date.fromisoformat(title[: len(iso_date_fmt)])
                title = title[len(iso_date_fmt) :]
                parsed_data.append(
                    StandardNotesData(
                        uuid=uuid, title=title, text=text, date=date, tags=tags
                    )
                )
            except Exception:
                continue

    parsed_data.sort(key=lambda x: x["date"])
    return parsed_data


# --- Query understanding & retrieval ---
class DiarySearchQuery(BaseModel):
    """Extract search parameters from the user's question about their diary."""

    query: str = Field(
        description="The semantic search text to find relevant diary entries"
    )
    year: int | None = Field(default=None, description="Filter by year (e.g. 2024)")
    month: int | None = Field(default=None, description="Filter by month (1-12)")
    day: int | None = Field(default=None, description="Filter by day of month (1-31)")
    tags: list[str] = Field(
        default_factory=list,
        description="All diary tags this question is about (may be several); "
        "empty if none apply",
    )
    recent: int | None = Field(
        default=None,
        description=(
            "Number of most-recent entries to fetch. Set ONLY when the user asks for "
            "a specific count of the LATEST items (e.g. 'last 10 climbing sessions' "
            "-> 10, 'my last run' -> 1). Leave null otherwise."
        ),
    )
    breadth: Literal["specific", "all"] = Field(
        default="specific",
        description=(
            "'all' when the user wants an overview, summary, recap, trend, or "
            "progression across the WHOLE filtered scope (e.g. 'summarize my "
            "bouldering progression'). 'specific' for point lookups (the default)."
        ),
    )


def _build_where(
    year: int | None, month: int | None, day: int | None, tags: list[str]
) -> dict | None:
    """Build a Chroma metadata `where` filter from extracted query parameters.

    Tags are OR-ed (an entry matches if its tag list contains ANY of them);
    `$and`/`$or` need >=2 children, so single conditions are emitted bare."""
    conditions: list[dict] = []
    if year is not None:
        conditions.append({"year": {"$eq": year}})
    if month is not None:
        conditions.append({"month": {"$eq": month}})
    if day is not None:
        conditions.append({"day": {"$eq": day}})

    # `tags` is stored as a list; $contains tests list membership.
    tag_conds = [{"tags": {"$contains": t}} for t in tags]
    if len(tag_conds) == 1:
        conditions.append(tag_conds[0])
    elif len(tag_conds) >= 2:
        conditions.append({"$or": tag_conds})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class DiaryQueryRouter:
    """Turns a natural-language question into structured query parameters (mode +
    filters), then retrieves diary entries either by semantic search (point lookups)
    or by a full metadata fetch (aggregate summaries)."""

    def __init__(
        self,
        vectorstore: Chroma,
        llm: ChatOllama,
        available_tags: list[str] | None = None,
        k: int = SEARCH_K,
    ) -> None:
        self.vectorstore = vectorstore
        self.llm = llm
        self.available_tags = available_tags or []
        self.k = k

    def _fallback_extract_query(self, query: str, today: dt.date) -> DiarySearchQuery:
        """Best-effort extraction when structured output is unavailable."""
        q = query.lower()

        year: int | None = None
        month: int | None = None
        day: int | None = None

        # --- Relative date expressions ---
        if "today last year" in q or ("today" in q and "last year" in q):
            year = today.year - 1
            month = today.month
            day = today.day
        else:
            if "last year" in q:
                year = today.year - 1
            elif "this year" in q:
                year = today.year

            if "this month" in q:
                month = today.month
            elif "last month" in q:
                last_month = today.month - 1
                month = 12 if last_month == 0 else last_month
                if year is None and today.month == 1:
                    year = today.year - 1

            if re.search(r"\btoday\b", q) and year is None:
                month = today.month
                day = today.day

        # --- ISO date: 2025-05-18 ---
        iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", q)
        if iso_match:
            year = int(iso_match.group(1))
            month = int(iso_match.group(2))
            day = int(iso_match.group(3))

        # --- Month name + year: "may 2025", "in january 2024" ---
        month_names_pattern = "|".join(MONTH_MAP.keys())
        month_year_match = re.search(rf"\b({month_names_pattern})\s+(\d{{4}})\b", q)
        if month_year_match:
            month = MONTH_MAP[month_year_match.group(1)]
            year = int(month_year_match.group(2))

        # --- Year + month name: "2025 may" (less common but possible) ---
        if month is None:
            year_month_match = re.search(rf"\b(\d{{4}})\s+({month_names_pattern})\b", q)
            if year_month_match:
                year = int(year_month_match.group(1))
                month = MONTH_MAP[year_month_match.group(2)]

        # --- Standalone month name without year: "in march", "during june" ---
        if month is None:
            standalone_month = re.search(rf"\b({month_names_pattern})\b", q)
            if standalone_month:
                month = MONTH_MAP[standalone_month.group(1)]

        # --- "Month day, year" or "Month day year": "May 18, 2025" ---
        if day is None:
            mdy_match = re.search(
                rf"\b({month_names_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,\s]+(\d{{4}})\b",
                q,
            )
            if mdy_match:
                month = MONTH_MAP[mdy_match.group(1)]
                day = int(mdy_match.group(2))
                year = int(mdy_match.group(3))

        # --- "day(th) of Month (year)": "18th of may 2025" ---
        if day is None:
            dom_match = re.search(
                rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_names_pattern})(?:\s+(\d{{4}}))?\b",
                q,
            )
            if dom_match:
                day = int(dom_match.group(1))
                month = MONTH_MAP[dom_match.group(2)]
                if dom_match.group(3):
                    year = int(dom_match.group(3))

        # --- Standalone 4-digit year if nothing else matched it ---
        if year is None:
            year_match = re.search(r"\b(20\d{2})\b", q)
            if year_match:
                year = int(year_match.group(1))

        # --- Tag matching: match query against each tag's name + multilingual
        # aliases; a query word like "skiing" can select several tags (lyže, skialp).
        matched_tags: list[str] = []
        for tag in self.available_tags:
            aliases = [tag.lower(), *(a.lower() for a in TAG_ALIASES.get(tag, []))]
            if any(alias in q for alias in aliases):
                matched_tags.append(tag)

        # --- Coarse intent heuristics ---
        breadth: Literal["specific", "all"] = "specific"
        if re.search(r"\b(summ|overview|recap|progress|trend|over time|evolv)", q):
            breadth = "all"

        recent: int | None = None
        if m := re.search(r"\b(?:last|latest|recent|past)\s+(\d{1,3})\b", q):
            recent = int(m.group(1))
        elif re.search(r"\bmy\s+last\b", q):
            recent = 1

        return DiarySearchQuery(
            query=query,
            year=year,
            month=month,
            day=day,
            tags=matched_tags,
            recent=recent,
            breadth=breadth,
        )

    def extract(self, query: str, chat_history: list[BaseMessage]) -> DiarySearchQuery:
        """Extract mode + filters, resolving follow-ups against the chat history."""
        today = dt.date.today()
        structured_llm = self.llm.with_structured_output(
            DiarySearchQuery, method="function_calling"
        )

        tags_hint = ""
        if self.available_tags:
            lines = [
                f"- {tag}: {', '.join(TAG_ALIASES[tag])}"
                if TAG_ALIASES.get(tag)
                else f"- {tag}"
                for tag in self.available_tags
            ]
            tags_hint = (
                "\n\nThe diary uses these fixed tags (with example synonyms in "
                "Czech/English). Map the user's topic — in ANY language — to ALL "
                "matching tags. A broad term may match SEVERAL (e.g. 'skiing' -> "
                "lyže and skialp). Use only exact tag values from this list; leave "
                "tags empty if none clearly apply.\n" + "\n".join(lines)
            )

        system_msg = (
            f"Today's date is {today.isoformat()} "
            f"(year={today.year}, month={today.month}, day={today.day}). "
            "Extract search parameters from the user's question about their personal "
            "diary. The question may be a follow-up to the conversation above; resolve "
            "any references so that 'query' is a self-contained search phrase. "
            "Resolve relative dates (e.g. 'last year' means "
            f"year={today.year - 1}, 'this month' means month={today.month}, and "
            f"'today last year' means year={today.year - 1}, month={today.month}, "
            f"day={today.day}). Always provide a semantic search query. Only set "
            "date/tag filters when the user explicitly or implicitly refers to a time "
            "period or tag. Set 'recent' to N only when the user asks for a specific "
            "number of the latest items ('last 10 climbing sessions' -> 10, 'my last "
            "run' -> 1). Set 'breadth' to 'all' for overviews/summaries/progressions "
            f"over the whole scope, else 'specific'.{tags_hint}"
        )

        extraction_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = extraction_prompt | structured_llm
        try:
            parsed: DiarySearchQuery = chain.invoke(
                {"input": query, "chat_history": chat_history}
            )
        except Exception as exc:
            logger.warning("Structured extraction failed, using fallback: %s", exc)
            parsed = self._fallback_extract_query(query, today)

        if not parsed.query or not parsed.query.strip():
            parsed.query = query

        # Keep only real tags (drop hallucinated values), de-duplicated in order.
        allowed = set(self.available_tags)
        parsed.tags = [t for t in dict.fromkeys(parsed.tags) if t in allowed]

        logger.info(
            "Routed query=%r year=%s month=%s day=%s tags=%s recent=%s breadth=%s",
            parsed.query,
            parsed.year,
            parsed.month,
            parsed.day,
            parsed.tags,
            parsed.recent,
            parsed.breadth,
        )
        return parsed

    @staticmethod
    def _docs_from_get(fetched: dict) -> list[Document]:
        """Turn a Chroma .get() result into Documents, sorted chronologically."""
        docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(
                fetched["documents"], fetched["metadatas"], strict=True
            )
        ]
        docs.sort(key=lambda d: d.metadata.get("date_str", ""))
        return docs

    def retrieve(self, parsed: DiarySearchQuery) -> list[Document]:
        """Pick a retrieval strategy from the actual DB cardinality (not a brittle
        upfront mode): most-recent-N, fetch-all, or similarity top-K."""
        where = _build_where(parsed.year, parsed.month, parsed.day, parsed.tags)
        logger.debug("Chroma where filter: %s", where)

        # Explicit "most recent N": ask the DB for matching dates, take the N latest.
        if parsed.recent:
            meta = self.vectorstore.get(where=where, include=["metadatas"])
            pairs = sorted(
                zip(meta["ids"], meta["metadatas"], strict=True),
                key=lambda p: p[1].get("date_str", ""),
                reverse=True,
            )
            top_ids = [i for i, _ in pairs[: parsed.recent]]
            if not top_ids:
                return []
            fetched = self.vectorstore.get(
                ids=top_ids, include=["documents", "metadatas"]
            )
            docs = self._docs_from_get(fetched)  # re-sorted chronologically
            logger.info(
                "retrieve: most-recent-%d -> %d entries", parsed.recent, len(docs)
            )
            return docs

        # No filter at all: can't enumerate the whole diary -> semantic search only.
        if where is None:
            results = self.vectorstore.similarity_search(parsed.query, k=self.k)
            logger.info(
                "retrieve: no filter, similarity k=%d -> %d", self.k, len(results)
            )
            return results

        # Filtered: the real match count decides breadth.
        count = len(self.vectorstore.get(where=where, include=[])["ids"])
        if count == 0:
            logger.info("retrieve: 0 matches for filter")
            return []
        if count <= FETCH_ALL_MAX or parsed.breadth == "all":
            fetched = self.vectorstore.get(
                where=where, include=["documents", "metadatas"]
            )
            docs = self._docs_from_get(fetched)
            logger.info(
                "retrieve: fetch-all %d entries (count=%d, breadth=%s)",
                len(docs),
                count,
                parsed.breadth,
            )
            return docs
        results = self.vectorstore.similarity_search(
            parsed.query, k=self.k, filter=where
        )
        logger.info(
            "retrieve: count=%d > %d & breadth=specific -> similarity k=%d -> %d",
            count,
            FETCH_ALL_MAX,
            self.k,
            len(results),
        )
        return results


# --- Answer generation ---
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "Date: {date_str}\nTags: {tags}\nContent: {page_content}"
)


def _prepare_for_prompt(docs: list[Document]) -> list[Document]:
    """Normalise metadata so DOCUMENT_PROMPT can always render (untagged entries
    have no 'tags' key, and tags are stored as a list)."""
    prepared = []
    for d in docs:
        tags = d.metadata.get("tags")
        meta = {
            **d.metadata,
            "date_str": d.metadata.get("date_str", "?"),
            "tags": ", ".join(tags) if tags else "—",
        }
        prepared.append(Document(page_content=d.page_content, metadata=meta))
    return prepared


def estimate_tokens(text: str) -> int:
    """Rough offline token estimate (~4 chars/token). `get_num_tokens` would download
    a GPT-2 tokenizer over the network, breaking the app's offline guarantee."""
    return max(1, len(text) // 4)


def _format_context(docs: list[Document]) -> str:
    """Stuff entries into the prompt context, one block per entry."""
    return "\n\n".join(
        format_document(d, DOCUMENT_PROMPT) for d in _prepare_for_prompt(docs)
    )


def stream_with_metrics(llm: ChatOllama, messages: list[BaseMessage], sink: dict):
    """Stream an LLM call token-by-token (for st.write_stream) while accumulating the
    chunks. After the generator is exhausted, sink['message'] holds the final merged
    chunk, whose response_metadata/usage_metadata carry the token counts + durations."""
    full = None
    for chunk in llm.stream(messages):
        full = chunk if full is None else full + chunk
        if chunk.content:
            yield chunk.content
    sink["message"] = full


def _usage_of(msg) -> dict:
    """Extract {prompt, gen, eval_ns} token counts from an AIMessage(Chunk), zeros if
    absent. (eval_ns is generation time in nanoseconds.)"""
    md = getattr(msg, "response_metadata", None) or {}
    return {
        "prompt": md.get("prompt_eval_count") or 0,
        "gen": md.get("eval_count") or 0,
        "eval_ns": md.get("eval_duration") or 0,
    }


def format_metrics(usage: dict) -> str | None:
    """Human-readable perf line from accumulated {prompt, gen, eval_ns}, or None."""
    if not usage or not usage.get("gen"):
        return None
    line = f"{usage['prompt']} prompt + {usage['gen']} gen tokens"
    if usage.get("eval_ns"):
        line = f"⚡ {usage['gen'] / (usage['eval_ns'] / 1e9):.0f} tok/s · " + line
    return line


def _scope_phrase(parsed: DiarySearchQuery) -> str:
    """Human-readable description of what was actually retrieved (topic + date), used
    to anchor the answer so terse follow-ups ('and in 2026?') aren't pulled toward an
    earlier turn. Built from the filters that were applied, so it is ground truth."""
    topic = parsed.query.strip() if parsed.query else ""
    if parsed.year and parsed.month and parsed.day:
        when = f"on {parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d}"
    elif parsed.year and parsed.month:
        when = f"in {parsed.year:04d}-{parsed.month:02d}"
    elif parsed.year:
        when = f"in {parsed.year}"
    elif parsed.month:
        when = f"in month {parsed.month}"
    else:
        when = ""
    return " ".join(p for p in (topic, when) if p)


def _batch_by_chars(docs: list[Document], budget: int) -> list[list[Document]]:
    """Group consecutive (chronological) docs into batches under a char budget."""
    batches: list[list[Document]] = []
    current: list[Document] = []
    size = 0
    for d in docs:
        dlen = len(d.page_content)
        if current and size + dlen > budget:
            batches.append(current)
            current, size = [], 0
        current.append(d)
        size += dlen
    if current:
        batches.append(current)
    return batches


def plan_generation(
    docs: list[Document],
    user_query: str,
    chat_history: list[BaseMessage],
    today: dt.date,
    scope: str,
    gen_llm: ChatOllama,
) -> tuple[list[BaseMessage] | None, dict, str | None]:
    """Plan size-adaptive generation over the retrieved (chronological) entries.

    Returns (messages_to_stream, premap_usage, canned_answer):
    - empty docs   -> (None, {}, "couldn't find…")
    - fits budget  -> (messages, {}, None)            single streamed pass (+ history)
    - too large    -> (reduce_messages, premap_usage, None)  map steps already run

    `premap_usage` carries the token usage of the non-streamed map calls so the final
    metrics cover the whole job. `scope` anchors the answer on the actually-retrieved
    topic/period so prior turns can't pull it off-target.
    """
    if not docs:
        return (
            None,
            {},
            "I couldn't find any diary entries matching that. "
            "Try a different tag or time period.",
        )

    total_chars = sum(len(d.page_content) for d in docs)

    if total_chars <= SINGLE_PASS_BUDGET:
        logger.info("Generating from %d entries in a single pass", len(docs))
        anchor = (
            f"The user's current request is about: {scope}.\n"
            "The conversation history may mention other periods or topics — answer the "
            "CURRENT request only. The diary excerpts below have ALREADY been filtered "
            "to match it; treat them as the complete set for this question.\n"
            if scope
            else "Answer the user's question using ONLY the diary excerpts below.\n"
        )
        system_prompt = (
            "/no_think\n"
            "You are a compassionate personal assistant helping the user review their "
            "diary.\n"
            f"Today's date is {today.isoformat()}.\n"
            f"{anchor}"
            "The entries are in CHRONOLOGICAL order. Answer the question directly; if "
            "the user asked for an overview, summary, recap, or progression, give a "
            "concise chronological summary instead. Either way, cite the specific "
            "date(s) you use.\n"
            "If there are no excerpts below, say honestly that you have no entries for "
            "that.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        messages = prompt.format_messages(
            context=_format_context(docs), chat_history=chat_history, input=user_query
        )
        return messages, {}, None

    # Too much text for one context window: summarize chronological batches (map),
    # then stream a single narrative that combines the dated batch-summaries (reduce).
    dated_docs = [
        Document(
            page_content=f"[{d.metadata.get('date_str', '?')}] {d.page_content}",
            metadata=d.metadata,
        )
        for d in docs
    ]
    batches = _batch_by_chars(dated_docs, SINGLE_PASS_BUDGET)
    logger.info(
        "Generating from %d entries (%d chars) via map-reduce in %d batches",
        len(docs),
        total_chars,
        len(batches),
    )

    partials: list[str] = []
    premap = {"prompt": 0, "gen": 0, "eval_ns": 0}
    for batch in batches:
        joined = "\n\n".join(d.page_content for d in batch)
        map_input = (
            "/no_think\n"
            "Summarize the key points of these chronologically-ordered diary entries "
            "as a few concise bullet points, keeping the [YYYY-MM-DD] date next to "
            f"each point.\n\n{joined}\n\nDated summary:"
        )
        resp = gen_llm.invoke(map_input)
        partials.append(resp.content)
        for k, v in _usage_of(resp).items():
            premap[k] += v

    combined = "\n\n".join(partials)
    focus_line = f"Focus: {scope}.\n" if scope else ""
    reduce_input = (
        "/no_think\n"
        f"Today's date is {today.isoformat()}.\n"
        f"{focus_line}"
        "Below are dated summaries of diary entries, in chronological order. Combine "
        "them into ONE coherent narrative that follows the progression OVER TIME, "
        "citing specific dates. Use ONLY this information.\n\n"
        f"User's question: {user_query}\n\n{combined}\n\nProgression summary:"
    )
    return [HumanMessage(reduce_input)], premap, None


# --- Streamlit UI ---
st.set_page_config(page_title="Local Diary Chat", page_icon="📔", layout="wide")
st.title("📔 Local Diary Assistant (Private & Offline)")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: ingestion
with st.sidebar:
    st.header("1. Ingest Data")
    uploaded_file = st.file_uploader("Upload Standard Notes ZIP backup", type=["zip"])

    if uploaded_file and st.button("Process & Index Diary"):
        with st.spinner("Processing backup and embedding text..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = Path(tmp_file.name)

            logger.info("Parsing Standard Notes backup from %s", tmp_file_path)
            parsed_notes = parse_standard_notes(tmp_file_path)
            logger.info("Parsed %d notes from backup", len(parsed_notes))

            # One Document per diary entry (entries are tiny — no chunking needed).
            documents = []
            for note in parsed_notes:
                if not note["text"].strip():
                    continue
                metadata = {
                    "uuid": note["uuid"],
                    "year": note["date"].year,
                    "month": note["date"].month,
                    "day": note["date"].day,
                    "date_str": note["date"].isoformat(),
                    "title": note["title"].strip(),
                }
                # Chroma forbids empty lists; omit the key for untagged entries.
                if note["tags"]:
                    metadata["tags"] = sorted(note["tags"])
                documents.append(Document(page_content=note["text"], metadata=metadata))

            embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL_NAME)

            logger.info(
                "Embedding %d entries into ChromaDB at %s", len(documents), PERSIST_DIR
            )
            vectorstore = Chroma(
                persist_directory=PERSIST_DIR, embedding_function=embeddings
            )
            vectorstore.reset_collection()  # replace any prior index, don't append
            vectorstore.add_documents(documents)
            st.session_state.vectorstore = vectorstore

            # Collect all unique tags for query assistance
            all_tags: set[str] = set()
            for note in parsed_notes:
                all_tags.update(note["tags"])
            st.session_state.available_tags = sorted(all_tags - {""})
            logger.info("Available tags: %s", st.session_state.available_tags)

            logger.info("Indexing complete")
            st.success(
                f"Indexed {len(documents)} entries from {len(parsed_notes)} notes!"
            )

# Sidebar: conversation controls + context-window gauge
with st.sidebar:
    st.header("2. Conversation")
    if st.button("🆕 New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    convo = "".join(m["content"] for m in st.session_state.messages)
    est = estimate_tokens(convo)
    ratio = min(est / CONTEXT_WINDOW, 1.0)
    st.progress(ratio, text=f"~{est:,} / {CONTEXT_WINDOW:,} tokens (est.)")
    last_prompt = next(
        (
            m["usage"]["prompt"]
            for m in reversed(st.session_state.messages)
            if m.get("usage", {}).get("prompt")
        ),
        None,
    )
    if last_prompt:
        st.caption(f"Last prompt: {last_prompt:,} tokens (exact)")
    if ratio >= TOKEN_WARN_RATIO:
        st.warning(
            "Conversation is getting long — older history may be dropped. "
            "Start a New chat for an unrelated question."
        )

# Load an existing DB if one was built in a previous session
if st.session_state.vectorstore is None:
    if Path(PERSIST_DIR).exists():
        st.session_state.vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OllamaEmbeddings(model=EMBEDDINGS_MODEL_NAME),
        )
        if "available_tags" not in st.session_state:
            stored = st.session_state.vectorstore.get(include=["metadatas"])
            all_tags = set()
            for meta in stored["metadatas"]:
                all_tags.update(meta.get("tags", []))
            st.session_state.available_tags = sorted(all_tags)
            logger.info("Recovered tags from DB: %s", st.session_state.available_tags)
        st.info("Loaded existing diary database from disk.")
    else:
        st.warning(
            "Please upload and process your Standard Notes backup in the "
            "sidebar to begin."
        )

# Chat interface
if st.session_state.vectorstore is not None:
    st.header("2. Chat with your Diary")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            cap = format_metrics(message.get("usage"))
            if cap:
                st.caption(cap)

    if user_query := st.chat_input(
        "Ask something (e.g., 'Summarize my running progression' "
        "or 'What did I do in May 2024?')"
    ):
        with st.chat_message("user"):
            st.markdown(user_query)

        # Conversation so far (before this turn) — lets follow-ups resolve references.
        chat_history: list[BaseMessage] = [
            HumanMessage(m["content"])
            if m["role"] == "user"
            else AIMessage(m["content"])
            for m in st.session_state.messages
        ]
        st.session_state.messages.append({"role": "user", "content": user_query})
        logger.info("User query: %s", user_query)

        today = dt.date.today()
        gen_llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0.3,
            verbose=True,
            num_ctx=16384,
            num_predict=2048,
            reasoning=False,
        )
        query_llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0,
            num_predict=256,
            num_ctx=8192,  # else history silently truncates during follow-up resolution
            reasoning=False,
        )
        router = DiaryQueryRouter(
            vectorstore=st.session_state.vectorstore,
            llm=query_llm,
            available_tags=st.session_state.get("available_tags", []),
        )

        with st.chat_message("assistant"):
            # Routing + retrieval (+ map-reduce) happen behind a step-by-step status;
            # the answer itself streams below it.
            with st.status("Understanding your question…") as status:
                parsed = router.extract(user_query, chat_history)
                scope = _scope_phrase(parsed)
                status.update(label="Searching your diary…")
                docs = router.retrieve(parsed)
                status.update(label=f"Reviewing {len(docs)} entries…")
                messages, premap, canned = plan_generation(
                    docs, user_query, chat_history, today, scope, gen_llm
                )
                status.update(label="Writing answer…", state="complete")

            # A whole-diary overview can't be enumerated into one prompt; flag the
            # fallback to similarity search so the user can narrow the scope.
            if (
                parsed.breadth == "all"
                and not parsed.recent
                and not (parsed.tags or parsed.year or parsed.month or parsed.day)
            ):
                st.caption(
                    "Tip: add a tag or time period for a full overview — showing the "
                    "most relevant entries instead."
                )

            logger.info(
                "Answered using %d entries (recent=%s breadth=%s)",
                len(docs),
                parsed.recent,
                parsed.breadth,
            )

            usage = dict(premap) if premap else {}
            if canned is not None:
                answer = canned
                st.markdown(answer)
            else:
                sink: dict = {}
                # Streamed display may briefly show raw <think> on reasoning models, but
                # /no_think keeps qwen quiet; we strip it from the stored text below.
                answer = st.write_stream(stream_with_metrics(gen_llm, messages, sink))
                for k, v in _usage_of(sink.get("message")).items():
                    usage[k] = usage.get(k, 0) + v

            # Strip Qwen3 <think>...</think> blocks from the stored/replayed answer
            if "<think>" in answer:
                answer = re.sub(
                    r"<think>.*?</think>", "", answer, flags=re.DOTALL
                ).strip()

            cap = format_metrics(usage)
            if cap:
                st.caption(cap)
                logger.info("Generation: %s", cap)

            with st.expander("View Retrieved Sources"):
                for doc in docs:
                    tags = doc.metadata.get("tags") or []
                    st.caption(
                        f"**Date:** {doc.metadata.get('date_str')} | "
                        f"**Tags:** {', '.join(tags) if tags else '—'}"
                    )
                    st.text(doc.page_content[:200] + "...")
                    st.write("---")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "usage": usage}
        )
