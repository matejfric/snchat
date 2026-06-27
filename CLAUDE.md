# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SNChat is a fully local, offline RAG chatbot over a personal diary exported from [Standard Notes](https://standardnotes.com/). A Streamlit UI ingests a Standard Notes ZIP backup, embeds the notes into a local ChromaDB vector store, and answers natural-language questions about the diary using a local Ollama LLM. No data leaves the machine.

## Prerequisites

The app requires [Ollama](https://ollama.com/) running locally with two models pulled — without them it cannot embed or answer:

```bash
ollama pull qwen3.5:9b   # LLM for query parsing + answers  (MODEL_NAME in app.py)
ollama pull bge-m3       # embeddings                        (EMBEDDINGS_MODEL_NAME in app.py)
```

## Commands

Dependencies are managed with `uv` (Python 3.12+; see `uv.lock`).

```bash
uv sync                          # install all dependencies (incl. dev)
uv run streamlit run app.py      # run the app (diary chat UI)
uv run ruff check .              # lint
uv run ruff check --fix .        # lint + autofix
uv run ruff format .             # format
```

There is no test suite — Ruff is the only automated check. The VS Code workspace runs Ruff fix + import-organize on save.

## Architecture

**`app.py` is the entire application.** Its only local imports are configuration constants and the multilingual tag glossary (`TAG_ALIASES`) from `constants.py`. The end-to-end flow:

1. **Ingestion** (sidebar): `parse_standard_notes()` unzips the backup, reads notes from `Standard Notes Backup and Import File.txt` and tags from `Items/Tag/*.txt`, then maps tags onto notes by UUID. Each note becomes **one** LangChain `Document` (no chunking — entries are tiny, so one entry = one retrievable unit) with metadata `year`/`month`/`day`/`date_str`/`title`/`uuid`, plus a `tags` **list** that is omitted entirely for untagged entries (Chroma forbids empty lists). Before indexing, the collection is `reset_collection()`-ed so re-ingest **replaces** rather than appends, then embedded via Ollama `bge-m3` and persisted to `./diary_vector_db` (ChromaDB).
2. **Routing + retrieval**: `DiaryQueryRouter.extract()` asks the query LLM for structured output (`DiarySearchQuery`) to pick a **mode** (`search` vs `summarize`) plus a semantic query and date/tag filters, resolving relative dates and follow-up references (the chat history is passed in via a `MessagesPlaceholder`). The notes/tags are Czech but questions may be English, so the extraction prompt is given `constants.TAG_ALIASES` (Czech+English synonyms for the fixed tags) and may select **several** tags — e.g. "skiing" → `lyže` + `skialp`; returned tags are clamped to the actually-present tags. `_build_where()` turns the filters into a Chroma `where` clause (date via `$eq`; tags via `{"tags": {"$contains": ...}}` list-membership, OR-ed when there are several — `$and`/`$or` need ≥2 children, so single conditions are emitted bare). If function-calling fails, `_fallback_extract_query()` recovers via regex + the same aliases.
   - **search** → `similarity_search(query, k=SEARCH_K, filter=where)` for point lookups.
   - **summarize** → `vectorstore.get(where=...)` fetches **all** matching entries (no similarity ranking, so it isn't capped by `k`), sorted chronologically — this is what makes "summarize my `lezení` progression" work across ~50 entries.
3. **Generation**: generation helpers (`build_answer_messages` for search, `summarize_plan` for summaries) return a `messages` list rather than running a chain, so the handler can **stream** the answer token-by-token via `stream_with_metrics` + `st.write_stream` (the old `create_stuff_documents_chain` was dropped because its `StrOutputParser` discards token metadata). Summaries are **adaptive**: a single streamed pass when total entry text fits `SUMMARY_CHAR_BUDGET` (`num_ctx=16384`), otherwise a manual map-reduce — the map steps run non-streamed and their token usage is folded into the metrics, then the reduce step streams. Exact token counts + tokens/sec come from Ollama `response_metadata` (`prompt_eval_count`/`eval_count`/`eval_duration`) on the accumulated chunk; Qwen3 `<think>` blocks are stripped from the stored answer. The answer/summary prompts are anchored on a resolved-scope phrase (`_scope_phrase()`, built from the applied filters, e.g. "skiing in 2026") and told the excerpts are already filtered — so terse follow-ups like "and in 2026?" don't get pulled toward an earlier turn's year/topic via the chat history.

On startup the app reloads an existing `./diary_vector_db` from disk if present; otherwise it prompts the user to upload a backup.

### UI / observability

- A **status** container (`st.status`) shows the routing → retrieval → generation steps; the answer then streams below it.
- The sidebar has a **New chat** button (clears `st.session_state.messages`) and a **context-window gauge**: `estimate_tokens()` gives an *offline* char-based estimate of the conversation size vs `CONTEXT_WINDOW` (the answer LLM's `num_ctx`), warns past `TOKEN_WARN_RATIO`, and shows the exact last-turn prompt size from stored metrics. Avoid `get_num_tokens` — it downloads a GPT-2 tokenizer over the network and breaks offline use.
- Per-answer `usage` (`{prompt, gen, eval_ns}`) is stored on each assistant message so the tokens/sec caption re-renders on history replay.

### Key constraints

- **Data contract:** each note's title must begin with an ISO date (`yyyy-mm-dd`), optionally followed by title text. Notes whose title doesn't parse as a date are silently skipped — all date sorting/filtering derives from this prefix.
- **Embedding-model consistency:** the model that queries the vector store must match the one that built it (`bge-m3`, 1024-dim). Querying a DB built with a different model yields garbage. `EXPECTED_SN_VERSION` (`constants.py`) warns if the backup format changes.
- **`k` vs context window:** `create_stuff_documents_chain` concatenates *all* passed docs with no truncation, and `ChatOllama` lets the Ollama server silently truncate past `num_ctx`. Keep `SEARCH_K` small; route large aggregations through `summarize` mode, not a bigger `k`.
- `diary_vector_db/` and `*.zip` backups are gitignored (the diary is private); a built DB may already exist locally. Changing the chunking or `tags` metadata format requires re-indexing (re-upload the backup).

## Coding Guideliness

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.