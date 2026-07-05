# Diagnostics: trace capture & answer-key replay

Dev-only tooling for inspecting what the pipeline *actually did* on each
question — the extracted `DiarySearchQuery`, the retrieval strategy and the
exact entries it returned, the generation plan, and the answer — and for
discovering new error modes (candidates for [error_modes.md](error_modes.md)).
Everything runs locally and writes to a file; with `SNCHAT_TRACE` unset the
tracing code is a no-op, so the app's offline guarantee is untouched.

## Trace capture

```bash
SNCHAT_TRACE=1 uv run streamlit run app.py    # appends diagnostics/traces.jsonl
```

Chat as usual; each question→answer turn is appended as **one JSON line** to
`diagnostics/traces.jsonl` (gitignored — it holds raw diary questions and
answers). Implementation is `tracing.py` (`setup()` + the `turn()` context
manager), wired into `app.py` and `tests/replay_answer_key.py`; the JSONL is
stdlib-only, no server, no dependencies.

Each record holds:

- **`id`, `ts`, `session.id`, `input.value`** — a stable turn id (for the
  future judge to attach verdicts), timestamp, conversation id, the question.
- **structured routing** — `snchat.extraction` (the full `DiarySearchQuery`
  as a nested object), `snchat.scope`, `snchat.retrieval.count`/`.dates`,
  `snchat.plan` (`single_pass`/`map_reduce`/`canned`), `snchat.usage`,
  `output.value` (the answer).
- **`events`** — the routing narration the modules already emit via `logging`
  (which retrieval branch fired + counts, the Chroma `where`-clause), captured
  automatically for the turn.

Inspect the file directly (it's just JSON lines):

```bash
# routing summary per turn
cat diagnostics/traces.jsonl | jq -c '{q: ."input.value", tags: ."snchat.extraction".tags, n: ."snchat.retrieval.count", plan: ."snchat.plan"}'
rm diagnostics/traces.jsonl        # clear between sessions (append-only, no rotation)
```

The records are the shared contract for two consumers — the HTML viewer below
and a future headless LLM-as-judge — both reading the file via
`tracing.read_turns()`.

## HTML viewer

```bash
uv run python diagnostics_report.py            # -> diagnostics/report.html
```

Generates one **self-contained** dark-mode HTML file (no external assets —
opens offline via `file://`; also gitignored, as it inlines diary content).
Sidebar lists turns grouped by session; clicking one shows its routing
pipeline (Question → Extract → Retrieve → Plan → Answer as pure-CSS nodes), the
routing narration, the retrieved entries (each collapsible, inside a bounded
scroll box so a 90-entry result doesn't bury the answer), and the answer. A
few lines of JS switch turns; everything else is HTML/CSS. `render(turns)` is
pure (no I/O), covered by `tests/test_diagnostics_report.py`.

## Answer-key replay

```bash
SNCHAT_TRACE=1 uv run python tests/replay_answer_key.py [--generate] [--case ID]
```

Replays the [mock_diary.md](mock_diary.md) answer key through the **live**
pipeline: real `extract()` (Ollama query LLM) + real `retrieve()` (bge-m3
embeddings over the mock diary in an ephemeral Chroma — `./diary_vector_db` is
never touched), then checks routing fields and retrieved entry dates against
the fixture's ground-truth constants. With `SNCHAT_TRACE=1`, each case's full
turn is appended to `diagnostics/traces.jsonl` for inspection.

This is a *diagnostic*, not a unit test: extraction is model behavior, so a
failure is a finding to inspect, not necessarily a code bug — and it is exactly
how routing regressions surface after changing the extraction prompt,
`TAG_ALIASES`, or the model. `--generate` additionally runs (un-judged) answer
generation for complete records; answer grading (LLM-as-judge) is future work.
