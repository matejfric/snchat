# Error modes & fixes

A running catalog of bugs / failure modes discovered in SNChat and how each is currently
addressed. All code lives in `app.py` unless noted; config in `constants.py`.

> Many of these are **query-side** fixes (no re-index). The ones marked **[re-index]**
> change ingestion and require re-uploading the backup to rebuild `diary_vector_db/`.

- [1. Ingestion \& vector store](#1-ingestion--vector-store)
  - [1.1. Tag filtering silently broken — no usable `tags` metadata **\[re-index\]**](#11-tag-filtering-silently-broken--no-usable-tags-metadata-re-index)
  - [1.2. Re-ingest duplicated/conflicted instead of replacing **\[re-index\]**](#12-re-ingest-duplicatedconflicted-instead-of-replacing-re-index)
  - [1.3. Tiny entries fragmented by chunking **\[re-index\]**](#13-tiny-entries-fragmented-by-chunking-re-index)
- [2. Retrieval \& routing](#2-retrieval--routing)
  - [2.1. Multilingual tag resolution failure (English → fixed Czech tags)](#21-multilingual-tag-resolution-failure-english--fixed-czech-tags)
  - [2.2. Chroma `$and`/`$or` require ≥2 children](#22-chroma-andor-require-2-children)
  - [2.3. Brittle binary `mode` under-retrieved](#23-brittle-binary-mode-under-retrieved)
  - [2.4. "Last N sessions" processed the whole tag](#24-last-n-sessions-processed-the-whole-tag)
  - [2.5. Whole-diary overview isn't enumerable](#25-whole-diary-overview-isnt-enumerable)
- [3. Generation \& conversation](#3-generation--conversation)
  - [3.1. Cross-turn answer confusion (follow-ups drift to the wrong scope)](#31-cross-turn-answer-confusion-follow-ups-drift-to-the-wrong-scope)
  - [3.2. Oversized context silently truncated](#32-oversized-context-silently-truncated)
  - [3.3. Extraction LLM truncated chat history](#33-extraction-llm-truncated-chat-history)
- [4. Offline / library constraints](#4-offline--library-constraints)
  - [4.1. `get_num_tokens` breaks the offline guarantee](#41-get_num_tokens-breaks-the-offline-guarantee)
  - [4.2. Streaming dropped token metrics](#42-streaming-dropped-token-metrics)
  - [4.3. Conversation can outgrow the context window unnoticed](#43-conversation-can-outgrow-the-context-window-unnoticed)
- [5. Data contract (violations skip data silently)](#5-data-contract-violations-skip-data-silently)

## 1. Ingestion & vector store

### 1.1. Tag filtering silently broken — no usable `tags` metadata **[re-index]**

- **Symptom:** tag-scoped questions (e.g. anything about `lezení`) matched nothing.
- **Cause:** tags were written inconsistently (a Python list in one path, a comma-joined
  string in another) and the persisted DB ended up with no usable `tags` field; Chroma
  also rejects **empty lists** as metadata.
- **Fix:** store `tags` as a **list of strings**, and **omit the key entirely** for
  untagged entries. Filter with `{"tags": {"$contains": <tag>}}` (valid list-membership
  test). See ingestion in the sidebar block + `_build_where()`.

### 1.2. Re-ingest duplicated/conflicted instead of replacing **[re-index]**

- **Symptom:** re-processing a backup grew the collection / mixed old and new metadata.
- **Cause:** `Chroma.from_documents(persist_directory=…)` **appends** to an existing
  collection.
- **Fix:** open the store, call `reset_collection()`, then `add_documents()` — re-ingest
  now replaces.

### 1.3. Tiny entries fragmented by chunking **[re-index]**

- **Symptom:** a single day's entry split across chunks; retrieval returned fragments.
- **Cause:** `RecursiveCharacterTextSplitter` (1000/200) applied to already-tiny notes.
- **Fix:** **one `Document` per entry, no chunking** (entries are < ½ A4, well under
  `bge-m3`'s input limit). One entry = one retrievable unit.

## 2. Retrieval & routing

### 2.1. Multilingual tag resolution failure (English → fixed Czech tags)

- **Symptom:** "summarize my skiing" selected **no** tags; the model couldn't map
  English "skiing" to Czech `lyže`/`skialp` (and `skialp` is not a common word for text embeddings).
- **Cause:** the prompt listed bare tag names with no meaning, and the schema allowed
  only **one** tag — but "skiing" spans two.
- **Fix:** `constants.TAG_ALIASES` (Czech+English synonyms; a term can appear under
  several tags) is fed to the extraction LLM **and** the regex fallback; `tags` is a
  **list**; returned tags are clamped to actually-present tags. `_build_where()` OR-s
  multiple tags.

### 2.2. Chroma `$and`/`$or` require ≥2 children

- **Symptom:** a filter with a single condition wrapped in `$and`/`$or` errors.
- **Fix:** `_build_where()` emits single conditions **bare** and only wraps in
  `$and`/`$or` when there are ≥2.

### 2.3. Brittle binary `mode` under-retrieved

- **Symptom:** questions that should pull **all** tagged entries in a period only got
  top-`K`=10, because the LLM mis-routed to "search".
- **Cause:** retrieval breadth hinged on a fragile upfront `mode: search|summarize`.
- **Fix:** `mode` removed. `retrieve()` decides from the **actual DB match count**:
  `count ≤ FETCH_ALL_MAX` (or `breadth=="all"`) → **fetch all**; else → similarity
  top-K. Small/medium filtered scopes are now always returned in full.

### 2.4. "Last N sessions" processed the whole tag

- **Symptom:** "summarize my last 10 climbing sessions" tried to load every `lezení`
  entry; the model doesn't know the dates.
- **Cause:** no recency-limited retrieval; the model can't pick dates it can't see.
- **Fix:** a `recent: int` field drives a **dates-first** path — `get(include=["metadatas"])`
  → sort by `date_str` desc → take the N newest ids → fetch only those. Note: Chroma
  `get(ids=…)` does **not** preserve order, so results are re-sorted in Python.

### 2.5. Whole-diary overview isn't enumerable

- **Symptom:** "summarize my whole diary" (no tag/date) can't be stuffed into one prompt.
- **Fix (deliberate limitation):** with no filter, `retrieve()` falls back to similarity
  top-K and the UI shows a one-line caption suggesting the user add a tag or period.

## 3. Generation & conversation

### 3.1. Cross-turn answer confusion (follow-ups drift to the wrong scope)

- **Symptom:** after "…ski in 2025?", asking "and in 2026?" retrieved the correct 2026
  entries but the answer framed them as if 2025 ("I cannot recall…").
- **Cause:** **not** a qwen capacity limit (prompt was 6.3k < 8192 ctx). The answer step
  got the prior 2025 Q&A (chat history) + a terse "and in 2026?" + 2026 context, and the
  router's resolved scope was never told to the answer LLM.
- **Fix:** `_scope_phrase()` builds a ground-truth scope from the applied filters (e.g.
  "skiing in 2026") and the prompt states the excerpts are **already filtered** to the
  current request; chat history is kept but explicitly demoted.

### 3.2. Oversized context silently truncated

- **Symptom:** large retrieved sets degraded answers with no error.
- **Cause:** the stuff prompt concatenates **all** docs; `ChatOllama` lets the Ollama
  server **silently truncate** past `num_ctx`.
- **Fix:** keep `SEARCH_K` small; route breadth via count (#7); `plan_generation()` is
  **size-adaptive** — single pass under `SINGLE_PASS_BUDGET`, else map-reduce over
  char-budgeted batches. Generation LLM runs at `num_ctx=16384`.

### 3.3. Extraction LLM truncated chat history

- **Symptom:** follow-up resolution degraded in longer conversations.
- **Cause:** `query_llm` had no `num_ctx`, so it used Ollama's small default and silently
  dropped history.
- **Fix:** set `query_llm` `num_ctx=8192`.

## 4. Offline / library constraints

### 4.1. `get_num_tokens` breaks the offline guarantee

- **Symptom:** counting tokens tried to download a GPT-2 tokenizer over the network.
- **Fix:** `estimate_tokens()` uses an offline ~4-chars/token heuristic for the sidebar
  context gauge; **exact** counts come only from Ollama `response_metadata` after a call.

### 4.2. Streaming dropped token metrics

- **Symptom:** couldn't show tokens/sec when streaming the answer.
- **Cause:** `create_stuff_documents_chain` ends in `StrOutputParser`, which discards the
  `AIMessageChunk` metadata.
- **Fix:** stream the LLM directly via `stream_with_metrics()`, accumulating chunks so the
  final merged chunk carries `prompt_eval_count`/`eval_count`/`eval_duration`.

### 4.3. Conversation can outgrow the context window unnoticed

- **Symptom:** long chats silently lose older history.
- **Fix:** sidebar **context gauge** (vs `CONTEXT_WINDOW`, the query-LLM/history budget)
  warns past `TOKEN_WARN_RATIO`, plus a **New chat** button to reset.

## 5. Data contract (violations skip data silently)

- Each note's **title must begin with an ISO date** (`yyyy-mm-dd`); notes whose title
  doesn't parse as a date are **silently skipped**. All date sorting/filtering derives
  from this prefix.
