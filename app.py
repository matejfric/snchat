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
    MODEL_NAME,
    PERSIST_DIR,
    SEARCH_K,
    SUMMARY_CHAR_BUDGET,
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

    mode: Literal["search", "summarize"] = Field(
        default="search",
        description=(
            "Use 'summarize' for aggregate questions that ask for an overview, "
            "recap, trend, or progression across MANY entries (often scoped to a "
            "tag, e.g. 'summarize my bouldering progression', 'how did my running "
            "develop?'). Use 'search' for specific point-in-time lookups (e.g. "
            "'what was I working on in May 2024?')."
        ),
    )
    query: str = Field(
        description="The semantic search text to find relevant diary entries"
    )
    year: int | None = Field(default=None, description="Filter by year (e.g. 2024)")
    month: int | None = Field(default=None, description="Filter by month (1-12)")
    day: int | None = Field(default=None, description="Filter by day of month (1-31)")
    tags: str | None = Field(default=None, description="Filter by tag keyword")


def _build_where(
    year: int | None, month: int | None, day: int | None, tags: str | None
) -> dict | None:
    """Build a Chroma metadata `where` filter from extracted query parameters."""
    conditions: list[dict] = []
    if year is not None:
        conditions.append({"year": {"$eq": year}})
    if month is not None:
        conditions.append({"month": {"$eq": month}})
    if day is not None:
        conditions.append({"day": {"$eq": day}})
    if tags is not None:
        # `tags` is stored as a list; $contains tests list membership.
        conditions.append({"tags": {"$contains": tags}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class DiaryQueryRouter:
    """Turns a natural-language question into structured query parameters (mode +
    filters), then retrieves diary entries either by semantic search (point lookups)
    or by a full metadata fetch (aggregate summaries)."""

    _MONTH_MAP: dict[str, int] = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }

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
        month_names_pattern = "|".join(self._MONTH_MAP.keys())
        month_year_match = re.search(rf"\b({month_names_pattern})\s+(\d{{4}})\b", q)
        if month_year_match:
            month = self._MONTH_MAP[month_year_match.group(1)]
            year = int(month_year_match.group(2))

        # --- Year + month name: "2025 may" (less common but possible) ---
        if month is None:
            year_month_match = re.search(rf"\b(\d{{4}})\s+({month_names_pattern})\b", q)
            if year_month_match:
                year = int(year_month_match.group(1))
                month = self._MONTH_MAP[year_month_match.group(2)]

        # --- Standalone month name without year: "in march", "during june" ---
        if month is None:
            standalone_month = re.search(rf"\b({month_names_pattern})\b", q)
            if standalone_month:
                month = self._MONTH_MAP[standalone_month.group(1)]

        # --- "Month day, year" or "Month day year": "May 18, 2025" ---
        if day is None:
            mdy_match = re.search(
                rf"\b({month_names_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,\s]+(\d{{4}})\b",
                q,
            )
            if mdy_match:
                month = self._MONTH_MAP[mdy_match.group(1)]
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
                month = self._MONTH_MAP[dom_match.group(2)]
                if dom_match.group(3):
                    year = int(dom_match.group(3))

        # --- Standalone 4-digit year if nothing else matched it ---
        if year is None:
            year_match = re.search(r"\b(20\d{2})\b", q)
            if year_match:
                year = int(year_match.group(1))

        # --- Tag matching: fuzzy match against available tags ---
        matched_tag: str | None = None
        if self.available_tags:
            q_words = set(q.split())
            for tag in self.available_tags:
                tag_lower = tag.lower()
                # Exact substring match in query
                if tag_lower in q:
                    matched_tag = tag
                    break
                # Match if all words of the tag appear in the query
                tag_words = set(tag_lower.split())
                if tag_words and tag_words.issubset(q_words):
                    matched_tag = tag
                    break

        # --- Coarse intent heuristic ---
        mode: Literal["search", "summarize"] = "search"
        if re.search(r"\b(summ|overview|recap|progress|trend|over time|evolv)", q):
            mode = "summarize"

        return DiarySearchQuery(
            mode=mode, query=query, year=year, month=month, day=day, tags=matched_tag
        )

    def extract(self, query: str, chat_history: list[BaseMessage]) -> DiarySearchQuery:
        """Extract mode + filters, resolving follow-ups against the chat history."""
        today = dt.date.today()
        structured_llm = self.llm.with_structured_output(
            DiarySearchQuery, method="function_calling"
        )

        tags_hint = ""
        if self.available_tags:
            tags_hint = (
                f" Available tags in the diary: {', '.join(self.available_tags)}."
                f" Only use one of these exact tag values for the 'tags' filter."
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
            f"period or tag.{tags_hint}"
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

        logger.info(
            "Routed mode=%s query=%r year=%s month=%s day=%s tags=%s",
            parsed.mode,
            parsed.query,
            parsed.year,
            parsed.month,
            parsed.day,
            parsed.tags,
        )
        return parsed

    def search(self, parsed: DiarySearchQuery) -> list[Document]:
        """Semantic similarity search restricted by the metadata filter."""
        where_filter = _build_where(parsed.year, parsed.month, parsed.day, parsed.tags)
        logger.debug("Chroma where filter (search): %s", where_filter)

        kwargs: dict[str, object] = {"k": self.k}
        if where_filter:
            kwargs["filter"] = where_filter
        results = self.vectorstore.similarity_search(parsed.query, **kwargs)
        logger.info("Retrieved %d documents (search)", len(results))
        return results

    def fetch_all(self, parsed: DiarySearchQuery) -> list[Document]:
        """Fetch ALL entries matching the metadata filter (no similarity ranking),
        sorted chronologically, for aggregate summaries."""
        where_filter = _build_where(parsed.year, parsed.month, parsed.day, parsed.tags)
        if where_filter is None:
            logger.warning(
                "Summarize query has no tag/date filter; fetching all entries"
            )
        logger.debug("Chroma where filter (summarize): %s", where_filter)

        fetched = self.vectorstore.get(
            where=where_filter, include=["documents", "metadatas"]
        )
        docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(
                fetched["documents"], fetched["metadatas"], strict=True
            )
        ]
        docs.sort(key=lambda d: d.metadata.get("date_str", ""))
        logger.info(
            "Fetched %d entries for summary (filter=%s)", len(docs), where_filter
        )
        return docs


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


def build_answer_messages(
    docs: list[Document],
    user_query: str,
    chat_history: list[BaseMessage],
    today: dt.date,
) -> list[BaseMessage]:
    """Build the chat messages for a point-lookup answer (system + history + human),
    with the retrieved entries stuffed into the context."""
    system_prompt = (
        "/no_think\n"
        "You are a compassionate personal assistant helping the user "
        "review their diary.\n"
        f"Today's date is {today.isoformat()}.\n"
        "Answer the user's question accurately using ONLY the provided "
        "diary excerpts below.\n"
        "Always reference the specific date(s) of the entries you are citing.\n"
        "If you cannot find the answer in the contexts, say honestly that you can't "
        "recall details about that.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt.format_messages(
        context=_format_context(docs), chat_history=chat_history, input=user_query
    )


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


def summarize_plan(
    docs: list[Document], user_query: str, today: dt.date
) -> tuple[list[BaseMessage] | None, dict, str | None, ChatOllama | None]:
    """Plan an adaptive summary of chronologically-ordered entries.

    Returns (messages_to_stream, premap_usage, canned_answer, llm):
    - empty docs  -> (None, {}, "couldn't find…", None)
    - single pass -> (messages, {}, None, llm)            entries fit one window
    - map-reduce  -> (reduce_messages, premap_usage, None, llm)  map steps already run

    The handler streams `messages` with `llm`; `premap_usage` carries the token usage
    of the non-streamed map calls so the final metrics cover the whole job.
    """
    if not docs:
        return (
            None,
            {},
            "I couldn't find any diary entries matching that. "
            "Try a different tag or time period.",
            None,
        )

    summarize_llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.3,
        num_ctx=16384,
        num_predict=2048,
        reasoning=False,
    )
    total_chars = sum(len(d.page_content) for d in docs)

    if total_chars <= SUMMARY_CHAR_BUDGET:
        logger.info("Summarizing %d entries in a single pass", len(docs))
        system_prompt = (
            "/no_think\n"
            "You are a compassionate personal assistant reviewing the user's diary.\n"
            f"Today's date is {today.isoformat()}.\n"
            "The diary entries below are in CHRONOLOGICAL order. Write a concise, "
            "coherent summary that follows how things developed OVER TIME, citing the "
            "specific dates of notable entries. Use ONLY the provided entries.\n\n"
            "Entries (chronological):\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )
        messages = prompt.format_messages(
            context=_format_context(docs), input=user_query
        )
        return messages, {}, None, summarize_llm

    # Too much text for one context window: summarize chronological batches (map),
    # then stream a single narrative that combines the dated batch-summaries (reduce).
    dated_docs = [
        Document(
            page_content=f"[{d.metadata.get('date_str', '?')}] {d.page_content}",
            metadata=d.metadata,
        )
        for d in docs
    ]
    batches = _batch_by_chars(dated_docs, SUMMARY_CHAR_BUDGET)
    logger.info(
        "Summarizing %d entries (%d chars) via map-reduce in %d batches",
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
        resp = summarize_llm.invoke(map_input)
        partials.append(resp.content)
        for k, v in _usage_of(resp).items():
            premap[k] += v

    combined = "\n\n".join(partials)
    reduce_input = (
        "/no_think\n"
        f"Today's date is {today.isoformat()}.\n"
        "Below are dated summaries of diary entries, in chronological order. Combine "
        "them into ONE coherent narrative that follows the progression OVER TIME, "
        "citing specific dates. Use ONLY this information.\n\n"
        f"User's question: {user_query}\n\n{combined}\n\nProgression summary:"
    )
    return [HumanMessage(reduce_input)], premap, None, summarize_llm


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
        answer_llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0.3,
            verbose=True,
            num_ctx=8192,
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
                if parsed.mode == "summarize":
                    status.update(label="Gathering matching entries…")
                    docs = router.fetch_all(parsed)
                    status.update(label=f"Summarizing {len(docs)} entries…")
                    messages, premap, canned, gen_llm = summarize_plan(
                        docs, user_query, today
                    )
                else:
                    status.update(label="Searching your diary…")
                    docs = router.search(parsed)
                    messages = build_answer_messages(
                        docs, user_query, chat_history, today
                    )
                    premap, canned, gen_llm = {}, None, answer_llm
                status.update(label="Writing answer…", state="complete")

            logger.info("Answered using %d entries (mode=%s)", len(docs), parsed.mode)

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
