# Diagnostics: tracing & answer-key replay

Dev-only tooling for inspecting what the pipeline *actually did* on each
question — the extracted `DiarySearchQuery`, the Chroma filter, the retrieval
strategy and the exact entries it returned, the generation plan, and every LLM
prompt/response — and for discovering new error modes (candidates for
[error_modes.md](error_modes.md)). Everything runs on localhost; the app's
offline guarantee is untouched, and with `SNCHAT_TRACE` unset the tracing code
is a no-op.

## Trace UI (Phoenix)

```bash
uv run phoenix serve                          # trace UI at http://localhost:6006
SNCHAT_TRACE=1 uv run streamlit run app.py    # second terminal
```

Chat as usual; each question appears in Phoenix seconds later as one
`diary_turn` trace. Implementation lives in `tracing.py` (`setup()` +
`turn()`), wired into `app.py`:

- **LLM spans** — every LangChain call (extraction, map/condense, streamed
  reduce) is auto-traced via OpenInference with full prompts, outputs, token
  counts, and latency.
- **Decision events** — the log lines `diary_query_router`/`generation` already
  emit (routed filters, `where` clause, chosen retrieval strategy, plan) are
  attached to the turn span as events, so the modules need no instrumentation
  code of their own.
- **Turn attributes** — `snchat.extraction` (the parsed query as JSON),
  `snchat.scope`, `snchat.retrieval.count`/`.dates`, `snchat.plan`
  (`single_pass`/`map_reduce`/`canned`), `snchat.usage`, and
  `input.value`/`output.value`. Turns of one Streamlit session share a
  `session.id`, so multi-turn conversations group in Phoenix's Sessions view.

Findings can be labeled in the Phoenix UI (annotations) to build an error-mode
taxonomy over time. Traces persist in `~/.phoenix/` (SQLite).

## Answer-key replay

```bash
SNCHAT_TRACE=1 uv run python tests/replay_answer_key.py [--generate] [--case ID]
```

Replays the [mock_diary.md](mock_diary.md) answer key through the **live**
pipeline: real `extract()` (Ollama query LLM) + real `retrieve()` (bge-m3
embeddings over the mock diary in an ephemeral Chroma — `./diary_vector_db` is
never touched), then checks routing fields and retrieved entry dates against
the fixture's ground-truth constants. One Phoenix trace per case, tagged
`snchat.replay.case` / `snchat.replay.failed`, so a failed check jumps straight
to the trace that explains it.

This is a *diagnostic*, not a unit test: extraction is model behavior, so a
failure is a finding to inspect, not necessarily a code bug — and it is exactly
how routing regressions surface after changing the extraction prompt,
`TAG_ALIASES`, or the model. `--generate` additionally runs (un-judged) answer
generation for complete traces; answer grading (LLM-as-judge) is future work.
