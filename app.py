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

# langchain-chroma's default collection name — existing DBs were built with it and
# the startup load path below opens it implicitly via Chroma(persist_directory=…).
MAIN_COLLECTION = "langchain"
TMP_COLLECTION = "diary_ingest_tmp"  # staging collection, renamed over MAIN on success


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
                        client.delete_collection(TMP_COLLECTION)  # crashed-run residue
                    tmp_store = Chroma(
                        client=client,
                        collection_name=TMP_COLLECTION,
                        embedding_function=embeddings,
                    )
                    tmp_store.add_documents(documents)

                    with contextlib.suppress(Exception):
                        client.delete_collection(MAIN_COLLECTION)  # first ingest: none
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
                    logger.info("Available tags: %s", st.session_state.available_tags)

                    logger.info("Indexing complete")
                    st.success(
                        f"Indexed {len(documents)} entries "
                        f"from {len(parsed_notes)} notes!{skipped_note}"
                    )
            except Exception as exc:
                logger.exception("Ingest failed")
                st.error(
                    f"Ingest failed — the existing index was left untouched. ({exc}) "
                    "Check that the file is a decrypted Standard Notes ZIP backup "
                    "and that Ollama is running with the embedding model pulled."
                )
            finally:
                # Don't leave a plaintext copy of the diary in the temp dir.
                if tmp_file_path is not None:
                    tmp_file_path.unlink(missing_ok=True)

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
            if any("date_int" not in m for m in stored["metadatas"]):
                st.warning(
                    "This index predates date-range support — re-upload your "
                    "backup to enable questions like “what did I do last week?”."
                )
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

            # Show what was actually searched, so a misroute (wrong year, dropped
            # tag, spurious keywords) is visible instead of silently producing a
            # fluent answer grounded in the wrong entries. Built from the applied
            # filters, not the LLM's free-text query — tags/keywords already name
            # the topic, so the query is shown only when neither is set.
            route_bits = []
            if parsed.tags:
                route_bits.append(" ".join(TAG_EMOJI.get(t, t) for t in parsed.tags))
            if parsed.keywords:
                route_bits.append("keywords: " + ", ".join(parsed.keywords))
            if not route_bits and parsed.query.strip():
                route_bits.append(parsed.query.strip())
            if parsed.date_from or parsed.date_to:
                route_bits.append(
                    f"{parsed.date_from or '…'} → {parsed.date_to or '…'}"
                )
            elif parsed.year:
                route_bits.append(
                    "-".join(
                        f"{v:02d}" for v in (parsed.year, parsed.month, parsed.day) if v
                    )
                )
            elif parsed.month:
                route_bits.append(f"month {parsed.month}")
            if parsed.recent:
                route_bits.append(f"latest {parsed.recent}")
            _entries_text = "entries" if len(docs) != 1 else "entry"
            st.caption(
                f"🔎 Searched: {' · '.join(route_bits)} · {len(docs)} {_entries_text}"
            )

            # A whole-diary overview can't be enumerated into one prompt (in a
            # reasonable amount of time - documented limitation); flag the
            # fallback to similarity search so the user can narrow the scope. Only
            # for a TRULY unscoped overview — keyword (lexical) and filtered/range
            # overviews DID enumerate every match, so the tip would be wrong there.
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
                answer = st.write_stream(stream_with_metrics(gen_llm, messages, sink))
                for k, v in _usage_of(sink.get("message")).items():
                    usage[k] = usage.get(k, 0) + v

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
