# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SNChat is a fully local, offline RAG chatbot over a personal diary exported from [Standard Notes](https://standardnotes.com/). A Streamlit UI ingests a Standard Notes ZIP backup, embeds the notes into a local ChromaDB vector store, and answers natural-language questions about the diary using a local Ollama LLM. No data leaves the machine.

## Prerequisites

The app requires [Ollama](https://ollama.com/) running locally with two models pulled — without them it cannot embed or answer:

```bash
ollama pull gemma4:12b   # LLM for query parsing + answers  (MODEL_NAME in constants.py)
ollama pull bge-m3       # embeddings                        (EMBEDDINGS_MODEL_NAME in constants.py)
```

## Commands

Dependencies are managed with `uv` (Python 3.12+; see `uv.lock`).

```bash
uv sync                          # install all dependencies (incl. dev)
uv run streamlit run app.py      # run the app (diary chat UI)
uv run ruff check .              # lint
uv run ruff check --fix .        # lint + autofix
uv run ruff format .             # format
uv run pytest                    # run tests (tests/)
```

Automated checks are Ruff (lint/format) and a pytest suite in `tests/` (keyword matcher, extraction fallback, generation planning, and mock-diary parse/retrieval ground truth). `uv run python tests/mock_diary.py` builds a deterministic synthetic Standard Notes backup ZIP for manual end-to-end testing — see `docs/mock_diary.md` for its answer key. The VS Code workspace runs Ruff fix + import-organize on save.

## Architecture

**Module map:** `app.py` is the Streamlit UI and wiring; backup parsing lives in `parser.py` (`parse_standard_notes()`), query routing + retrieval in `diary_query_router.py` (with the `DiarySearchQuery` schema in `diary_search_query.py`), generation planning + token/metrics helpers in `generation.py`, and configuration — including the multilingual tag glossary `TAG_ALIASES` — in `constants.py`. The end-to-end flow:

1. **Ingestion** (sidebar): `parse_standard_notes()` unzips the backup, reads notes from `Standard Notes Backup and Import File.txt` and tags from `Items/Tag/*.txt`, then maps tags onto notes by UUID. Each note becomes **one** LangChain `Document` (no chunking — entries are tiny, so one entry = one retrievable unit) with metadata `year`/`month`/`day`/`date_str`/`date_int` (numeric `yyyymmdd` for range filters)/`title`/`uuid`, plus a `tags` **list** that is omitted entirely for untagged entries (Chroma forbids empty lists). The note→`Document` mapping lives in `parser.py` (`documents_from_notes()`), shared with the retrieval tests so the metadata contract can't drift. Entries are embedded (Ollama `bge-m3`) into a **temporary Chroma collection** and renamed over the live one only on success — so re-ingest **replaces** rather than appends, and a failed or interrupted ingest can't destroy the existing index at `./diary_vector_db`. The whole ingest is wrapped in an error boundary (`st.error` instead of a raw traceback), a backup that parses to zero entries aborts with a hint (encrypted exports parse to zero — the backup must be **decrypted**), and the uploaded temp ZIP is always deleted afterwards.
2. **Routing + retrieval**: `DiaryQueryRouter.extract()` asks the query LLM for structured output (`DiarySearchQuery`) — a semantic query, date/tag filters (exact `year`/`month`/`day` OR an inclusive `date_from`/`date_to` range for multi-day periods like "last week"/"this winter", possibly open-ended), a `recent` count, a `keywords` list (expanded surface forms of a named entity for lexical lookup), and a coarse `breadth` (`specific`/`all`) — resolving relative dates (the prompt includes today's weekday so "last week" resolves to the previous Mon–Sun) and follow-up references (chat history via a `MessagesPlaceholder`). Extraction post-processing drops invalid range dates and swaps a reversed range. The notes/tags are Czech but questions may be English, so the extraction prompt is given `constants.TAG_ALIASES` (Czech+English synonyms for the fixed tags) and may select **several** tags — e.g. "skiing" → `lyže` + `skialp`; returned tags are normalized (casefold + reverse alias→tag lookup, so an echoed "skiing" or "Lyže" still filters) and clamped to the actually-present tags. For a named entity that is *not* a tag (a TV series, book, person), it instead fills `keywords` with abbreviation/translation variants (GoT → Game of Thrones / Hra o trůny) for the lexical branch below. `_build_where()` turns the filters into a Chroma `where` clause (exact dates via `$eq`; ranges via `$gte`/`$lte` on the numeric `date_int`; tags via `{"tags": {"$contains": ...}}` list-membership, OR-ed when there are several — `$and`/`$or` need ≥2 children, so single conditions are emitted bare). If function-calling fails (or returns `None`), `_fallback_extract_query()` deliberately guesses **no** filters — the raw question goes to unfiltered semantic top-K with only the overview intent (`breadth`) detected, so degraded mode is predictable rather than confidently mis-scoped (error_modes §2.9).
   - `DiaryQueryRouter.retrieve()` picks the strategy from the **actual DB match count** (not a brittle upfront mode): if `keywords` are set (a named entity whose mentions scatter beyond top-K), **fuzzy-match** the expanded surface forms over the full (optionally date/tag-filtered) text and return *every* hit — `_keyword_hit()` uses a whole-word match for short acronyms — case-exact when the form carries case, so "GoT" doesn't match the common word "got"; `WRatio` under-scores short forms and a substring test would false-match in "forgot" and rapidfuzz `partial_ratio ≥ FUZZY_MATCH_THRESHOLD` for longer/multi-word forms (best-matching window, so a verbatim mention scores 100 regardless of entry length and Czech declension still matches — `WRatio` is wrong here, it scales partial matches down to ~60 in long entries), then `recent` trims to the N latest; elif `recent=N`, do a dates-first metadata fetch → sort by `date_str` desc → take the N newest ids → fetch those (so "last 10 climbing sessions" never processes the whole tag); elif there's **no** filter → `similarity_search(k=SEARCH_K)` (can't enumerate the whole diary); else count matches and **fetch ALL** when `count ≤ FETCH_ALL_MAX` or `breadth=="all"` (this is what makes "what did I do skiing in January" return *every* matching entry, not just top-K), otherwise fall back to `similarity_search(filter=where)` for a specific lookup inside a large scope.
3. **Generation**: `plan_generation()` (`generation.py`) is **size-adaptive** and returns a `messages` list built directly (no prompt template — braces in user text can't break formatting), so the handler can **stream** the answer token-by-token via `stream_with_metrics` + `st.write_stream`. When the **full prompt** — stuffed context + chat history + question — fits `SINGLE_PASS_BUDGET` it's one streamed pass; otherwise a manual map-reduce: the map steps run non-streamed with their token usage folded into the metrics (each map/condense call carries the user's question so specific details survive the bullet compression), the joined batch-summaries are **re-condensed in further rounds if they still exceed the budget**, then the reduce step streams. A single generation LLM at `num_ctx=GEN_NUM_CTX` (16384, with `GEN_NUM_PREDICT` reserved for the answer) is used for both. The prompt answers a point lookup OR, if the user asked for an overview/progression, summarizes — citing dates either way. It's anchored on a resolved-scope phrase (`_scope_phrase()`, e.g. "skiing in 2026") and told the excerpts are already filtered, so terse follow-ups like "and in 2026?" don't drift toward an earlier turn. Exact token counts + tokens/sec come from Ollama `response_metadata` (`prompt_eval_count`/`eval_count`/`eval_duration`); any `<think>` blocks are stripped from the stored answer (defensive — gemma doesn't emit them).

On startup the app reloads an existing `./diary_vector_db` from disk if present; otherwise it prompts the user to upload a backup.

### UI / observability

- A **status** container (`st.status`) shows the routing → retrieval → generation steps; the answer then streams below it. A **🔎 route caption** under the status shows what was actually searched (resolved scope, tags, keywords, recent-N, entry count), so a misroute is visible instead of silently producing a fluent answer from the wrong entries.
- The sidebar has a **New chat** button (clears `st.session_state.messages`) and a **context-window gauge**: `estimate_tokens()` gives an *offline* char-based estimate (`CHARS_PER_TOKEN≈3`, Czech-calibrated) of the conversation size vs `CONTEXT_WINDOW` (the query LLM's `num_ctx`, i.e. the chat-history budget — decoupled from the larger generation LLM), warns past `TOKEN_WARN_RATIO`, and shows the exact last-turn prompt size from stored metrics. Avoid `get_num_tokens` — it downloads a GPT-2 tokenizer over the network and breaks offline use.
- Per-answer `usage` (`{prompt, gen, eval_ns}`) is stored on each assistant message so the tokens/sec caption re-renders on history replay.

### Key constraints

- **Data contract:** each note's title must begin with an ISO date (`yyyy-mm-dd`), optionally followed by title text. Notes whose title doesn't parse as a date are silently skipped — all date sorting/filtering derives from this prefix.
- **Embedding-model consistency:** the model that queries the vector store must match the one that built it (`bge-m3`, 1024-dim). Querying a DB built with a different model yields garbage. `EXPECTED_SN_VERSION` (`constants.py`) warns if the backup format changes.
- **Breadth is count-driven, not `k`:** `ChatOllama` lets the Ollama server silently truncate past `num_ctx`, so don't lean on a big `k`. `SEARCH_K` stays small for similarity lookups; broad scopes are handled by `retrieve()` fetching ALL matching entries (`count ≤ FETCH_ALL_MAX` or `breadth=="all"`) and `plan_generation()` map-reducing when the full prompt exceeds `SINGLE_PASS_BUDGET` (derived as `(GEN_NUM_CTX − GEN_NUM_PREDICT) × CHARS_PER_TOKEN`).
- `diary_vector_db/` and `*.zip` backups are gitignored (the diary is private); a built DB may already exist locally. Changing the chunking or entry metadata format (`tags`, `date_int`, …) requires re-indexing (re-upload the backup) — the app warns on startup when the on-disk index predates `date_int`.

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