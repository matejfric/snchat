import datetime as dt
import logging
from pathlib import Path
import re
import tempfile

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
import streamlit as st

from constants import (
    CONTEXT_WINDOW,
    EMBEDDINGS_MODEL_NAME,
    MODEL_NAME,
    PERSIST_DIR,
    SINGLE_PASS_BUDGET,
    TOKEN_WARN_RATIO,
)
from diary_query_router import DiaryQueryRouter
from diary_search_query import DiarySearchQuery
from parser import parse_standard_notes

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


# --- Answer generation ---
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "Date: {date_str}\nTags: {tags}\nContent: {page_content}"
)


def _tags_str(metadata: dict) -> str:
    """Render an entry's tags list as prompt text ('—' when untagged). Shared by both
    generation paths so the LLM sees tags consistently (single-pass AND map-reduce)."""
    tags = metadata.get("tags")
    return ", ".join(tags) if tags else "—"


def _prepare_for_prompt(docs: list[Document]) -> list[Document]:
    """Normalise metadata so DOCUMENT_PROMPT can always render (untagged entries
    have no 'tags' key, and tags are stored as a list)."""
    prepared = []
    for d in docs:
        meta = {
            **d.metadata,
            "date_str": d.metadata.get("date_str", "?"),
            "tags": _tags_str(d.metadata),
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
            page_content=(
                f"[{d.metadata.get('date_str', '?')}] "
                f"(tags: {_tags_str(d.metadata)}) {d.page_content}"
            ),
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
            "as a few concise bullet points, keeping the [YYYY-MM-DD] date (and any "
            f"tags) next to each point.\n\n{joined}\n\nDated summary:"
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
