# Error modes & fixes

A running catalog of bugs / failure modes discovered in SNChat and how each is currently
addressed. Code locations: backup parsing in `parser.py`, routing/retrieval in
`diary_query_router.py`, generation planning in `generation.py`, UI wiring in `app.py`;
config in `constants.py`.

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
  - [2.6. Scattered entity mentions exceeded top-K](#26-scattered-entity-mentions-exceeded-top-k)
  - [2.7. Structured extraction returned None (no tool call)](#27-structured-extraction-returned-none-no-tool-call)
  - [2.8. Multi-day periods were inexpressible ("last week", "this winter") **\[re-index\]**](#28-multi-day-periods-were-inexpressible-last-week-this-winter-re-index)
- [3. Generation \& conversation](#3-generation--conversation)
  - [3.1. Cross-turn answer confusion (follow-ups drift to the wrong scope)](#31-cross-turn-answer-confusion-follow-ups-drift-to-the-wrong-scope)
  - [3.2. Oversized context silently truncated](#32-oversized-context-silently-truncated)
  - [3.3. Extraction LLM truncated chat history](#33-extraction-llm-truncated-chat-history)
  - [3.4. Map-reduce dropped tags from the context](#34-map-reduce-dropped-tags-from-the-context)
- [4. Offline / library constraints](#4-offline--library-constraints)
  - [4.1. `get_num_tokens` breaks the offline guarantee](#41-get_num_tokens-breaks-the-offline-guarantee)
  - [4.2. Streaming dropped token metrics](#42-streaming-dropped-token-metrics)
  - [4.3. Conversation can outgrow the context window unnoticed](#43-conversation-can-outgrow-the-context-window-unnoticed)
- [5. Data contract (violations skip data silently)](#5-data-contract-violations-skip-data-silently)
- [6. Example prompts (regression reference)](#6-example-prompts-regression-reference)

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

### 2.6. Scattered entity mentions exceeded top-K

- **Symptom:** "what were my impressions of Game of Thrones?" returned an incomplete
  answer — a named entity (a TV series, book, film, person) is mentioned across many
  days, **more than `SEARCH_K`**, so similarity top-K only saw a slice; embeddings also
  smear over proper nouns/acronyms, so even the top-K were not the most on-topic.
- **Cause:** retrieval was purely **semantic**; there was no keyword/lexical path. Exact
  substring matching wouldn't fix it either — Czech is heavily inflected (`Hra o trůny`
  → `ve Hře o trůny`) and people abbreviate (GoT ↔ Game of Thrones).
- **Fix:** the extraction LLM expands a named entity into a `keywords` list of surface
  forms (abbreviation + full name + Czech/English variants). When `keywords` are set,
  `retrieve()` takes a **lexical branch** (`_fuzzy_retrieve()`): fetch the full
  (optionally date/tag-filtered) set and keep **every** entry matching any candidate via
  `_keyword_hit()` — a **whole-word** match for short acronyms ("GoT"; `WRatio`
  under-scores them and a bare substring test false-matches "forgot"), else rapidfuzz
  `partial_ratio ≥ FUZZY_MATCH_THRESHOLD` (best-matching window: a verbatim mention
  scores 100 regardless of entry length, and declension still scores high). The full
  match set then flows into the existing fetch-all → map-reduce generation; `recent`
  trims to the N latest. Covered by `tests/test_keyword_hit.py`.
- **Gotcha — don't use `WRatio` here:** it was the first choice and silently broke on
  real entries. `WRatio` scales partial matches down to 0.6 once the entry is >~8x
  longer than the query, so a **verbatim** "Campfire Cooking" in a normal diary
  paragraph scored ~60 and fell below the threshold → 0 results. Synthetic tests with
  short entries (length ratio <8) didn't catch it; the long-entry regression test now
  does. `_fuzzy_retrieve()` also logs `scanned`/`matched`/`best_score` so a 0-result is
  diagnosable (a high `best_score` below the threshold points straight at this class of
  bug).

### 2.7. Structured extraction returned None (no tool call)

- **Symptom:** "summarize my whole diary" crashed with `AttributeError: 'NoneType'
  object has no attribute 'query'` in `extract()`.
- **Cause:** `with_structured_output(method="function_calling")` returns **None** — not
  an exception — when the model emits no tool call (likely for a bare overview with
  nothing to extract). The `try/except` only caught exceptions, so the `None` reached
  `parsed.query`.
- **Fix:** treat a `None` result like a failure — fall back to
  `_fallback_extract_query()` (which routes a bare overview to `breadth="all"`, no
  filter). Covered by `tests/test_extract_fallback.py`.

### 2.8. Multi-day periods were inexpressible ("last week", "this winter") **[re-index]**

- **Symptom:** range questions either lost their date filter entirely (top-K
  similarity over the whole diary) or were squeezed into one wrong month — "this
  winter" → December only, silently dropping January/February of the next year.
- **Cause:** `DiarySearchQuery` offered only exact-equality `year`/`month`/`day`;
  Chroma's `$gte`/`$lte` are numeric-only, and entries had no numeric date key to
  compare on.
- **Fix:** entries carry `date_int` (`yyyymmdd`, built in
  `parser.documents_from_notes()`); the schema gained inclusive ISO
  `date_from`/`date_to` (either may stand alone for "since …"/"until …"), which
  `_build_where()` turns into `$gte`/`$lte` on `date_int`. The extraction prompt
  states today's **weekday** so "last week" resolves to the previous Mon–Sun, and
  spells out that winter-style ranges cross the year boundary; `extract()` drops
  invalid dates and swaps a reversed range. Old indexes lack `date_int` (a range
  filter matches nothing), so the app warns on startup until the backup is
  re-uploaded. Covered by range tests in `tests/test_mock_retrieval.py`.

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

### 3.4. Map-reduce dropped tags from the context

- **Symptom:** large scopes that fell to **map-reduce** generation summarized entries
  without their tags, while the single-pass path included them — the LLM saw different
  information depending on the (size-driven) path.
- **Cause:** single-pass renders each entry via `DOCUMENT_PROMPT` (`Date:` / `Tags:` /
  `Content:`), but the map-reduce path built its own per-entry string with only a
  `[YYYY-MM-DD]` date prefix and **no tags**.
- **Fix:** both paths now render tags through a shared `_tags_str()` helper (`—` when
  untagged), so they can't drift apart again. The map-reduce prefix is
  `[YYYY-MM-DD] (tags: …) <content>`, and the map prompt keeps tags next to each point
  so they survive into the reduce step.

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

## 6. Example prompts (regression reference)

Prompts the app should handle, grouped by the behavior they exercise, with the expected
routing/retrieval. Use as a manual checklist after touching the router, `retrieve()`, or
generation; each group cross-references the failure mode above.

> **Partially automated against the mock diary** (`tests/mock_diary.py`, see
> [mock_diary.md](mock_diary.md)): parse-level ground truth and keyword matching in
> `tests/test_mock_diary.py`, and the routing/retrieval behavior below against a real
> in-memory Chroma in `tests/test_mock_retrieval.py`. Answer-*quality* checks (the LLM
> steps) remain manual — build the ZIP with `uv run python tests/mock_diary.py`, ingest
> it in the sidebar, and use the answer key in mock_diary.md. When testing against your
> real backup instead, substitute entities/topics that actually occur in it. Tags shown
> are the fixed Czech tags from `constants.TAG_ALIASES`.

### 6.1. Multilingual tag mapping (→ 2.1)

- "summarize my skiing" → tags `lyže` **+** `skialp`, `breadth=all`.
- "how is my climbing going?" → tag `lezení`, `breadth=all`.
- "kolik jsem toho naběhal?" → tag `běh` (Czech question, Czech tag).

### 6.2. Broad scope returns ALL matching entries (→ 2.3, 2.5)

- "what did I do skiing in January?" → tags `lyže`+`skialp` **and** `month=1`; fetches
  **every** match, not top-K.
- "summarize my whole diary" → no filter → similarity top-K **plus** the UI caption
  suggesting a tag/period (deliberate limitation, not a bug).

### 6.3. Recent-N, dates-first (→ 2.4)

- "summarize my last 10 climbing sessions" → `recent=10`, tag `lezení`, newest 10 by date.
- "my last run" → `recent=1`, tag `běh`.

### 6.4. Dates, relative dates & point lookups (→ 3.1, §5)

- "what did I do on 2025-05-18?" → exact `year`/`month`/`day`.
- "what was I up to today last year?" → `year`=today−1, plus today's `month`/`day`.
- **Follow-up scope:** ask "how was skiing in 2025?", then "and in 2026?" → the second
  answer must be framed as **2026** (not drift back to 2025); anchored by `_scope_phrase`.

### 6.5. Keyword / named-entity lexical lookup (→ 2.6)

- "what were my impressions of Game of Thrones?" → `keywords` ≈ `[GoT, Game of Thrones,
  Hra o trůny]`; fuzzy-fetches **all** mentions across dates, including Czech-declined.
- "did I enjoy watching GoT?" → abbreviation expands to full forms; the short token "got"
  matches as a whole word (and must **not** match inside "forgot").
- "what did I think of <a book/film/person in the diary>?" → keyword expansion finds
  scattered mentions a top-K similarity search would miss.
- **Negative control:** "when did I feel anxious?" → thematic, no named entity →
  `keywords` stays **empty** → must route to the semantic/other path, not the lexical one.

### 6.6. Date ranges (→ 2.8)

- "what did I do last week?" → `date_from`/`date_to` = previous Mon–Sun.
- "how was my skiing this winter?" → tags `lyže`+`skialp` **and** a Dec-1 → end-of-Feb
  range **across the year boundary**.
- "what happened between March and May 2025?" → `2025-03-01` → `2025-05-31`.
- "what have I written since June?" → open-ended: `date_from` only.
- **Negative control:** "what did I do in May 2025?" → single month → `year`+`month`,
  **no** range.

### 6.7. Conversation window (→ 4.3)

- A long multi-turn chat → sidebar context gauge warns past `TOKEN_WARN_RATIO`; **New
  chat** resets `st.session_state.messages`.
