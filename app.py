import contextlib
import datetime as dt
import logging
from pathlib import Path
import tempfile

import chromadb
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
import streamlit as st
from streamlit.runtime.scriptrunner_utils.exceptions import (
    RerunException,
    StopException,
)

from constants import (
    CONTEXT_WINDOW,
    EMBEDDINGS_MODEL_NAME,
    GEN_NUM_CTX,
    GEN_NUM_PREDICT,
    MODEL_NAME,
    PERSIST_DIR,
    TAG_EMOJI,
    TOKEN_WARN_RATIO,
)
from diary_query_router import DiaryQueryRouter
from diary_search_query import DiarySearchQuery
from generation import (
    _scope_phrase,
    _usage_of,
    estimate_tokens,
    format_metrics,
    plan_generation,
    stream_with_metrics,
)
from parser import documents_from_notes, parse_standard_notes

logging.basicConfig(
    level=logging.WARNING,  # third-party stays quiet without a per-library blocklist
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
# Full diagnostics for our own modules only (e.g. the routed Chroma `where` filter).
for _mod in ("__main__", "parser", "diary_query_router", "generation"):
    logging.getLogger(_mod).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# langchain-chroma's default collection name — existing DBs were built with it and
# the startup load path below opens it implicitly via Chroma(persist_directory=…).
MAIN_COLLECTION = "langchain"
TMP_COLLECTION = "diary_ingest_tmp"  # staging collection, renamed over MAIN on success

# Past this many retrieved entries the sources expander gets a fixed-height
# scrolling container instead of growing unboundedly.
SOURCES_SCROLL_AFTER = 5


def _route_caption(parsed: DiarySearchQuery, n_docs: int) -> str:
    """What was actually searched (resolved scope, tags, keywords, recent-N, entry
    count), so a misroute (wrong year, dropped tag, spurious keywords) is visible
    instead of silently producing a fluent answer grounded in the wrong entries.
    Built from the applied filters, not the LLM's free-text query — tags/keywords
    already name the topic, so the query is shown only when neither is set."""
    route_bits = []
    if parsed.tags:
        route_bits.append(" ".join(TAG_EMOJI.get(t, t) for t in parsed.tags))
    if parsed.keywords:
        route_bits.append("keywords: " + ", ".join(parsed.keywords))
    if not route_bits and parsed.query.strip():
        route_bits.append(parsed.query.strip())
    if parsed.date_from or parsed.date_to:
        route_bits.append(f"{parsed.date_from or '…'} → {parsed.date_to or '…'}")
    elif parsed.year:
        route_bits.append(
            "-".join(f"{v:02d}" for v in (parsed.year, parsed.month, parsed.day) if v)
        )
    elif parsed.month:
        route_bits.append(f"month {parsed.month}")
    if parsed.recent:
        route_bits.append(f"latest {parsed.recent}")
    entries_text = "entries" if n_docs != 1 else "entry"
    return f"🔎 Searched: {' · '.join(route_bits)} · {n_docs} {entries_text}"


def _overview_tip(parsed: DiarySearchQuery) -> str | None:
    """A whole-diary overview can't be enumerated into one prompt (in a reasonable
    amount of time - documented limitation); flag the fallback to similarity search
    so the user can narrow the scope. Only for a TRULY unscoped overview — keyword
    (lexical) and filtered/range overviews DID enumerate every match, so the tip
    would be wrong there."""
    if (
        parsed.breadth == "all"
        and not parsed.recent
        and not parsed.keywords
        and not (
            parsed.tags
            or parsed.year
            or parsed.month
            or parsed.day
            or parsed.date_from
            or parsed.date_to
        )
    ):
        return (
            "Tip: add a tag or time period for a full overview — showing the "
            "most relevant entries instead."
        )
    return None


def _render_message_extras(message: dict) -> None:
    """Captions + sources stored on an assistant message, re-rendered on every
    history replay (the live generation run ends in an immediate rerun, so anything
    not persisted on the message would vanish)."""
    if message.get("stopped"):
        st.caption("⏹ Stopped by user")
    cap = format_metrics(message.get("usage"))
    if cap:
        st.caption(cap)
    if message.get("route"):
        st.caption(message["route"])
    if message.get("tip"):
        st.caption(message["tip"])
    sources = message.get("sources")
    if sources:
        with st.expander("View Retrieved Sources"):
            box = (
                st.container(height=400)
                if len(sources) > SOURCES_SCROLL_AFTER
                else st.container()
            )
            with box:
                for src in sources:
                    tags = src["tags"]
                    st.caption(
                        f"**Date:** {src['date']} | "
                        f"**Tags:** {', '.join(tags) if tags else '—'}"
                    )
                    st.text(src["text"] + "...")
                    st.write("---")


def _capture_stream(gen):
    """Mirror streamed chunks into session state so a Stop click — which interrupts
    the script mid-stream — can preserve the partial answer on the next run."""
    for chunk in gen:
        st.session_state.partial += chunk
        yield chunk


# --- Streamlit UI ---
st.set_page_config(page_title="Local Diary Chat", page_icon="📔", layout="wide")
st.title("📔 Diary Chat")
st.caption("Private & offline — your diary never leaves this machine.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "generating" not in st.session_state:
    st.session_state.generating = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "partial" not in st.session_state:
    st.session_state.partial = ""

# Sidebar: ingestion (collapsed once a diary DB exists; disabled while generating
# so a stray click can't interrupt and restart a running answer)
with st.sidebar:
    has_db = st.session_state.vectorstore is not None or Path(PERSIST_DIR).exists()
    with st.expander("📥 Ingest data", expanded=not has_db):
        uploaded_file = st.file_uploader(
            "Upload Standard Notes ZIP backup",
            type=["zip"],
            disabled=st.session_state.generating,
        )

        if uploaded_file and st.button(
            "Process & Index Diary", disabled=st.session_state.generating
        ):
            with st.spinner("Processing backup and embedding text..."):
                tmp_file_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".zip"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = Path(tmp_file.name)

                    logger.info("Parsing Standard Notes backup from %s", tmp_file_path)
                    parsed_notes, skipped = parse_standard_notes(tmp_file_path)
                    logger.info(
                        "Parsed %d notes from backup (%d skipped)",
                        len(parsed_notes),
                        skipped,
                    )

                    documents = documents_from_notes(parsed_notes)
                    skipped_note = (
                        f" {skipped} note(s) were skipped — missing yyyy-mm-dd title "
                        "prefix or unreadable (encrypted?) content."
                        if skipped
                        else ""
                    )

                    if not documents:
                        st.error(
                            "No diary entries found in the backup. Make sure it is a "
                            "**decrypted** Standard Notes export and that note titles "
                            "start with an ISO date (yyyy-mm-dd). The existing index "
                            f"(if any) was left untouched.{skipped_note}"
                        )
                    else:
                        embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL_NAME)

                        logger.info(
                            "Embedding %d entries into ChromaDB at %s",
                            len(documents),
                            PERSIST_DIR,
                        )
                        # Embed into a TEMP collection so a failed or interrupted run
                        # can't destroy the existing index; swap names only on success.
                        client = chromadb.PersistentClient(path=PERSIST_DIR)
                        with contextlib.suppress(Exception):
                            # crashed-run residue
                            client.delete_collection(TMP_COLLECTION)
                        tmp_store = Chroma(
                            client=client,
                            collection_name=TMP_COLLECTION,
                            embedding_function=embeddings,
                        )
                        tmp_store.add_documents(documents)

                        with contextlib.suppress(Exception):
                            # first ingest: none
                            client.delete_collection(MAIN_COLLECTION)
                        tmp_store._collection.modify(name=MAIN_COLLECTION)
                        st.session_state.vectorstore = Chroma(
                            client=client,
                            collection_name=MAIN_COLLECTION,
                            embedding_function=embeddings,
                        )

                        # Collect all unique tags for query assistance
                        all_tags: set[str] = set()
                        for note in parsed_notes:
                            all_tags.update(note["tags"])
                        st.session_state.available_tags = sorted(all_tags - {""})
                        logger.info(
                            "Available tags: %s", st.session_state.available_tags
                        )

                        logger.info("Indexing complete")
                        st.success(
                            f"Indexed {len(documents)} entries "
                            f"from {len(parsed_notes)} notes!{skipped_note}"
                        )
                except Exception as exc:
                    logger.exception("Ingest failed")
                    st.error(
                        "Ingest failed — the existing index was left untouched. "
                        f"({exc}) "
                        "Check that the file is a decrypted Standard Notes ZIP backup "
                        "and that Ollama is running with the embedding model pulled."
                    )
                finally:
                    # Don't leave a plaintext copy of the diary in the temp dir.
                    if tmp_file_path is not None:
                        tmp_file_path.unlink(missing_ok=True)

# Sidebar: conversation controls + context-window gauge
with st.sidebar:
    st.header("Conversation")
    if st.button(
        "🆕 New chat",
        use_container_width=True,
        disabled=st.session_state.generating,
    ):
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
            if any("date_int" not in m for m in stored["metadatas"]):
                st.warning(
                    "This index predates date-range support — re-upload your "
                    "backup to enable questions like “what did I do last week?”."
                )
    else:
        st.warning(
            "Please upload and process your Standard Notes backup in the "
            "sidebar to begin."
        )

# Chat interface
if st.session_state.vectorstore is not None:
    if not st.session_state.messages:
        n_entries = st.session_state.vectorstore._collection.count()
        n_tags = len(st.session_state.get("available_tags", []))
        st.caption(f"{n_entries:,} diary entries indexed · {n_tags} tags")
        st.markdown(
            "Ask about your diary, e.g.:\n"
            "- *What did I do last week?*\n"
            "- *Summarize my running progression this year*\n"
            "- *When did I last go skiing?*"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_message_extras(message)

    user_query = st.chat_input(
        "Ask something (e.g., 'Summarize my running progression' "
        "or 'What did I do in May 2024?')",
        key="chat_input",
        disabled=st.session_state.generating,
    )
    if user_query:
        # Two-phase turn: stash the question and rerun, so the next run renders a
        # DISABLED input + a Stop button before doing any LLM work. A new question
        # can no longer interrupt a running generation — only Stop can.
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.pending_query = user_query
        st.session_state.generating = True
        st.rerun()

    if st.session_state.generating:
        # The answer streams into this slot; the Stop button is created after it,
        # so it sits BELOW the streaming text, just above the chat input.
        answer_slot = st.container()
        if st.button("⏹ Stop generating"):
            # A click interrupts the streaming run; THIS run finalizes the partial
            # answer captured by _capture_stream into history.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": st.session_state.partial
                    or "*Stopped before an answer was generated.*",
                    "stopped": True,
                }
            )
            st.session_state.generating = False
            st.session_state.pending_query = None
            st.rerun()

        user_query = st.session_state.pending_query
        logger.info("User query: %s", user_query)
        # Conversation before this turn ([:-1] excludes the pending question) —
        # lets follow-ups resolve references.
        chat_history: list[BaseMessage] = [
            HumanMessage(m["content"])
            if m["role"] == "user"
            else AIMessage(m["content"])
            for m in st.session_state.messages[:-1]
        ]

        today = dt.date.today()
        gen_llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0.3,
            verbose=True,
            num_ctx=GEN_NUM_CTX,
            num_predict=GEN_NUM_PREDICT,
            reasoning=False,
        )
        query_llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0,
            num_predict=256,
            # else history silently truncates during follow-up resolution
            num_ctx=CONTEXT_WINDOW,
            reasoning=False,
        )
        router = DiaryQueryRouter(
            vectorstore=st.session_state.vectorstore,
            llm=query_llm,
            available_tags=st.session_state.get("available_tags", []),
        )

        st.session_state.partial = ""  # reset here so a restarted run can't double up
        with answer_slot, st.chat_message("assistant"):
            try:
                # Routing + retrieval (+ map-reduce) happen behind a step-by-step
                # status; the answer itself streams below it.
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

                route = _route_caption(parsed, len(docs))
                st.caption(route)
                tip = _overview_tip(parsed)
                if tip:
                    st.caption(tip)

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
                    answer = st.write_stream(
                        _capture_stream(stream_with_metrics(gen_llm, messages, sink))
                    )
                    for k, v in _usage_of(sink.get("message")).items():
                        usage[k] = usage.get(k, 0) + v

                cap = format_metrics(usage)
                if cap:
                    logger.info("Generation: %s", cap)
            except (RerunException, StopException):
                raise  # script control (Stop click / rerun) — not an error
            except Exception as exc:
                logger.exception("Answer generation failed")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"⚠️ Something went wrong while answering. ({exc}) "
                            "Check that Ollama is running with the models pulled, "
                            "then try again."
                        ),
                    }
                )
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "usage": usage,
                        "route": route,
                        "tip": tip,
                        "sources": [
                            {
                                "date": d.metadata.get("date_str"),
                                "tags": d.metadata.get("tags") or [],
                                "text": d.page_content[:200],
                            }
                            for d in docs
                        ],
                    }
                )
        st.session_state.generating = False
        st.session_state.pending_query = None
        st.rerun()
