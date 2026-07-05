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
  - [2.9. Regex fallback guessed wrong filters](#29-regex-fallback-guessed-wrong-filters)
  - [2.10. LLM echoed a tag alias; the clamp silently dropped the filter](#210-llm-echoed-a-tag-alias-the-clamp-silently-dropped-the-filter)
  - [2.11. Short acronym candidates matched common words ("GoT" ⊨ "got")](#211-short-acronym-candidates-matched-common-words-got--got)
- [3. Generation \& conversation](#3-generation--conversation)
  - [3.1. Cross-turn answer confusion (follow-ups drift to the wrong scope)](#31-cross-turn-answer-confusion-follow-ups-drift-to-the-wrong-scope)
  - [3.2. Oversized context silently truncated](#32-oversized-context-silently-truncated)
  - [3.3. Extraction LLM truncated chat history](#33-extraction-llm-truncated-chat-history)
  - [3.4. Map-reduce dropped tags from the context](#34-map-reduce-dropped-tags-from-the-context)
  - [3.5. Map step discarded the details a specific question needed](#35-map-step-discarded-the-details-a-specific-question-needed)
- [4. Offline / library constraints](#4-offline--library-constraints)
  - [4.1. `get_num_tokens` breaks the offline guarantee](#41-get_num_tokens-breaks-the-offline-guarantee)
  - [4.2. Streaming dropped token metrics](#42-streaming-dropped-token-metrics)
  - [4.3. Conversation can outgrow the context window unnoticed](#43-conversation-can-outgrow-the-context-window-unnoticed)
- [5. Data contract (violations skip data — with a visible count)](#5-data-contract-violations-skip-data--with-a-visible-count)
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
  several tags) is fed to the extraction LLM; `tags` is a
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
  The caption checks **every** scope field (tags, exact dates, ranges, keywords,
  recent) — keyword and filtered overviews *did* enumerate all matches, so showing
  it there would be wrong.

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
  under-scores them and a bare substring test false-matches "forgot"; case-exact
  when the form carries case, see §2.11), else rapidfuzz
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

### 2.9. Regex fallback guessed wrong filters

- **Symptom:** with structured extraction unavailable (§2.7), innocuous phrasings
  produced confidently wrong scopes: modal "may" became `month=5` ("what may have
  caused my knee pain?"), alias substrings over-matched ("run" ⊂ "brunch" → tag
  `běh`), "last 2 weeks"
  became `recent=2` *entries*, and bare "today"/"last month" matched that month/day
  in **every** year (the year was never set).
- **Cause:** ~100 lines of date/tag/count regexes tried to replicate the extraction
  LLM's job. A wrongly guessed filter silently narrows retrieval to the wrong
  entries — a fluent answer built on the wrong subset, strictly worse than a broad
  one. The fallback also never filled `keywords`, so it could not match the
  structured path anyway.
- **Fix (deliberate simplification):** `_fallback_extract_query()` guesses **no
  filters** — it keeps the raw question as the semantic query and only detects
  overview intent (`breadth="all"` for summarize/overview/progression phrasings, so
  §2.7's bare-overview case still routes right). Degraded mode is now predictable:
  unfiltered semantic top-K, visible as such in the 🔎 route caption. `MONTH_MAP`
  left with it. Covered by `tests/test_extract_fallback.py`.

### 2.10. LLM echoed a tag alias; the clamp silently dropped the filter

- **Symptom:** "my last swim" routed to `recent=1` with **no** tag filter — the
  newest entry of *any* kind, answered fluently as if it were a swim.
- **Cause:** the extraction prompt lists tag names next to their aliases, and a
  small local model sometimes returns the alias ("swimming") or re-cases the tag
  ("Lyže"); the clamp was exact-match, so the value was dropped and the scope
  silently widened.
- **Fix:** returned tags are normalized before clamping — casefolded and resolved
  through a reverse alias→tag map built from `TAG_ALIASES` (an echoed "skiing"
  fans out to `lyže`+`skialp`, matching the alias table's intent); only unknown
  values are dropped. Covered by `tests/test_extract_fallback.py`.

### 2.11. Short acronym candidates matched common words ("GoT" ⊨ "got")

- **Symptom:** entity lookups whose expansion contains a short form — "GoT", "Duna"
  — also returned every entry containing the ordinary word: English snippets with
  "i got up early", a sand "duna". Each false hit flows into generation as a
  supposed mention of the entity.
- **Cause:** the lexical branch lowercased both candidate and entry text before the
  whole-word match (§2.6), erasing exactly the signal — casing — that separates an
  acronym/proper noun from a common word.
- **Fix:** candidates keep their original casing; a short form that **carries case**
  ("GoT", "Duna") must match that exact casing, while an all-lowercase short form
  stays case-insensitive (the escape hatch for diaries that write acronyms
  lowercase — alternate casings like "GOT" come from the LLM's variant expansion).
  Long/multi-word forms are casefolded as before. Covered by
  `tests/test_keyword_hit.py`.

### 2.12. One junk field voided the whole extraction (int `date_from` → silent fallback)

- **Symptom:** "what was I up to today last year?" routed with **no** filters at
  all — unfiltered semantic top-K — although the model had correctly extracted
  `year=2025`. Surfaced by the answer-key replay (docs/diagnostics.md); the trace
  showed the tool call `{"date_from": 2025, "date_to": 2025, "year": 2025, …}`.
- **Cause:** the date-range fields added in §2.8 are `str | None`, and gemma4
  sometimes fills them with bare integers. One invalid field fails pydantic
  validation of the WHOLE tool call, `with_structured_output` raises, and
  extract() drops to the no-filter fallback (§2.9) — every *valid* filter in the
  same call is silently lost. A regression risk of any schema growth.
- **Fix:** a lenient `mode="before"` validator on the two date-range fields
  degrades a non-string value to `None` so the rest of the call survives; the
  extraction prompt additionally pins date_from/date_to to yyyy-mm-dd strings.
  Deliberately scoped to the observed failure — junk in other fields still
  falls back predictably (§2.9), and the answer-key replay
  (docs/diagnostics.md) will surface it if it ever happens. Covered by
  `tests/test_extract_fallback.py` with the verbatim traced payload.

### 2.13. Verbatim typed dates under-parsed ("on 2025-05-18" → `year=2025` only)

- **Symptom:** "what did I do on 2025-05-18?" routed as year-2025 similarity
  top-K — the Pálava entry itself wasn't retrieved, filler was.
- **Cause:** the model under-parses a literal ISO date into just the year
  (attention diluted by the §2.8 range instructions). Yet a `yyyy-mm-dd` typed
  by the user needs no interpretation at all — trusting the LLM here is trusting
  it to copy.
- **Fix:** deterministic backfill in extract() post-processing: exactly one
  valid ISO date found verbatim in the question sets year+month+day — but ONLY
  when extraction produced no day-precision filter, so a correctly extracted
  "the week after 2025-05-18" range is never overwritten. Applies to the
  fallback path too (a literal date is copied, not guessed — §2.9's rule is
  about guessing). A single-day range (`date_from == date_to`, how the model
  answers "today last year") is likewise canonicalized to year/month/day.
  Covered by `tests/test_extract_fallback.py`.

### 2.14. Cross-lingual entity expansion hallucinated ("The Witcher" → "Věštík")

- **Symptom:** "what did I think of the Witcher books?" retrieved 1 of 4
  mentions — only the entry with the English name. The trace showed keywords
  `["The Witcher", "Witcher", "Věštík", "Sapiekowski"]`: the Czech title
  "Zaklínač" is a knowledge gap for a local 12B, and no prompt fixes missing
  knowledge; every declined Czech mention was silently lost.
- **Documented limitation (no fix):** keyword-branch recall depends entirely on
  the LLM's own variant expansion. A curated entity glossary (a `TAG_ALIASES`
  counterpart) was considered and rejected — hand-listing every book/series/
  person defeats the point of using an LLM, and reduced recall is accepted
  instead. The miss is at least visible: the 🔎 route caption shows the exact
  keywords searched, so a hallucinated variant can be spotted and the question
  re-asked with the native name ("what did I think of Zaklínač?"). The replay
  answer key asserts only the reliably-matched English-form entry.

### 2.15. Relative single-day date under-filled to year-only ("today last year" → year)

- **Symptom:** "what was I up to today last year?" (today 2026-07-05) routed as
  `year=2025` with no month/day — 93 matches > `FETCH_ALL_MAX` → year-wide
  similarity top-K, so the actual July-5 entry was never retrieved. The model
  *had* resolved the day correctly in its prose `query` ("What was I doing on
  July 5th, 2025?") but mirrored only the year into the structured fields.
- **Cause:** a third variant of this same query after §2.12 (int range) and
  §2.13 (single-day range). The omnibus extraction juggles the semantic query
  plus eight fields, the tag glossary and the range rules; date attention is
  diluted (§2.13) and the day/month drop. The prompt already spells out "'today
  last year' means year=… month=… day=…" *and* "fill year AND month AND day
  together" — prompting is maxed and the model still under-fills. It can't be
  patched deterministically the way §2.12/§2.13 were: `year=2025` alone is *also*
  the correct output for "in 2025", so the intent lives only in the question, and
  a phrase matcher on the question is out — the app is **multilingual** (a
  literal-phrase matcher only works for hand-listed languages, the same objection
  as the curated glossary in §2.14).
- **Fix:** a **focused second extraction call** (`_resolve_date()` →
  `ResolvedDate`) that does ONLY date resolution — nothing to dilute its
  attention — returning a precision-honest ISO string (`yyyy` / `yyyy-mm` /
  `yyyy-mm-dd`); `_iso_precision()` backfills whichever of year/month/day it
  carries. It is **gated on the structured output, not the question text** (year
  set, nothing finer), so it stays language-agnostic and fires rarely: it runs
  *after* the deterministic backfills (a typed literal date, §2.13, needs no
  call), a genuine "in 2025" trips it but the specialist just re-confirms the
  year (no-op), and no-date/thematic questions never set a year so never fire. A
  failed / unparseable / null specialist answer degrades to the year-only scope —
  predictable, never a crash or a wrong narrowing. This is a reliability bet on
  the same 12B, so it's validated through the answer-key replay
  (docs/diagnostics.md) on the `today-last-year` case; if the focused call proves
  unreliable there, the fallback is to accept it as a documented limitation like
  §2.14. Covered by `tests/test_extract_fallback.py`.
- **Gotcha — the specialist's `date` field must be REQUIRED:** a first cut made it
  optional (`str | None = None`); gemma4 then emitted **no tool call at all**
  (§2.7) and `with_structured_output` returned None for *every* question — the
  replay showed `year=2025 month=None day=None` unchanged, with no specialist
  effect. A required `date` (with a `"none"` sentinel for the no-date case) forces
  the call, exactly as the omnibus schema's required `query` field always does.

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

### 3.5. Map step discarded the details a specific question needed

- **Symptom:** in map-reduce mode, a point question ("what was the name of the hut
  on the January ski traverse?") got a vague answer — the detail existed in the
  entries but never reached the final prompt.
- **Cause:** map prompts asked for generic "key points"; the user's question first
  appeared at the **reduce** step, which only sees the generic bullets. Map-reduce
  is reachable for specific lookups too — `count ≤ FETCH_ALL_MAX` forces fetch-all
  regardless of `breadth`, and long entries can exceed `SINGLE_PASS_BUDGET`.
- **Fix:** every map/condense call carries the user's question and is told to
  prioritize details relevant to it. Covered by `tests/test_generation.py`.

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

## 5. Data contract (violations skip data — with a visible count)

- Each note's **title must begin with an ISO date** (`yyyy-mm-dd`); notes whose title
  doesn't parse as a date (or whose content is unreadable/encrypted) are **skipped
  and counted** — the count is logged and appended to the ingest success/error
  message, so a typo'd title (`2025-1-4`) is no longer invisible. Deleted items and
  non-Note items are expected non-data and aren't counted. All date sorting/filtering
  derives from this prefix.

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
