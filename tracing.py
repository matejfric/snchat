"""Dev-only diagnostic tracing to a local JSONL file (docs/diagnostics.md).

Disabled unless SNCHAT_TRACE=1; then each question→answer turn is appended as
one JSON line to diagnostics/traces.jsonl — the question, the structured
routing data set via `.set()` (extracted DiarySearchQuery, retrieval count +
dates, plan, usage, answer), and the routing narration the modules already
emit through `logging` (which retrieval branch fired, the Chroma where-clause).
No network, no server, no dependency beyond the stdlib, so the app's offline
guarantee is untouched. The same file feeds a future viewer and a headless
LLM-as-judge; `read_turns()` is the shared reader.

Usage:

    SNCHAT_TRACE=1 uv run streamlit run app.py   # writes diagnostics/traces.jsonl
"""

from contextlib import contextmanager
import contextvars
import datetime as dt
import json
import logging
import os
from pathlib import Path
import uuid

ENABLED = os.environ.get("SNCHAT_TRACE") == "1"

# Anchored to this file's directory (the repo root), NOT the CWD, so the app
# (writer) and any reader agree on one location regardless of where each is
# launched from (tests patch this global directly).
TRACE_PATH = Path(__file__).parent / "diagnostics" / "traces.jsonl"

# Loggers whose records are captured as `events` on the active turn — the exact
# routing narration (chosen strategy + counts, the where-clause) that the
# structured fields don't spell out.
_BRIDGED_LOGGERS = ("parser", "diary_query_router", "generation")

# Per-thread handle on the turn currently being recorded, so the log handler
# knows where to append. contextvars are per-thread, which is exactly right for
# Streamlit's per-session ScriptRunner threads.
_current: contextvars.ContextVar = contextvars.ContextVar("snchat_turn", default=None)
_installed = False


class _LogCapture(logging.Handler):
    """Append a bridged log record's message to the active turn's event list."""

    def emit(self, record: logging.LogRecord) -> None:
        turn = _current.get()
        if turn is not None:
            turn.rec.setdefault("events", []).append(
                f"{record.name}: {record.getMessage()}"
            )


def setup() -> None:
    """Install the log-capture handler once. No-op unless SNCHAT_TRACE=1;
    idempotent across Streamlit reruns (module globals persist in-process)."""
    global _installed
    if not ENABLED or _installed:
        return
    handler = _LogCapture(level=logging.DEBUG)
    for name in _BRIDGED_LOGGERS:
        logging.getLogger(name).addHandler(handler)
    _installed = True


def read_turns(path: Path | None = None) -> list[dict]:
    """Read all turn records, skipping malformed/partial lines — a concurrent
    read during an append, or a crash mid-write, can leave a truncated final
    line. Shared by the future viewer and the headless judge."""
    path = path or TRACE_PATH
    if not path.exists():
        return []
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return turns


class _Turn:
    def __init__(self, rec: dict) -> None:
        self.rec = rec

    def set(self, key: str, value) -> None:
        """Record a field on the turn (stored natively — dicts/lists stay
        structured, unlike the old span attributes). Empties are skipped, so an
        absent filter doesn't clutter the record."""
        if value is None or value == "" or value == []:
            return
        self.rec[key] = value

    def set_retrieval(self, docs) -> None:
        """Record the retrieved entries' count + full text (date/tags/text).
        Duck-typed over LangChain Documents so tracing stays import-pure; the one
        place the trace's retrieval-doc shape is defined (app + replay share it)."""
        self.set("snchat.retrieval.count", len(docs))
        self.set(
            "snchat.retrieval.docs",
            [
                {
                    "date": d.metadata.get("date_str", "?"),
                    "tags": d.metadata.get("tags") or [],
                    "text": d.page_content,
                }
                for d in docs
            ],
        )


class _NoopTurn:
    def set(self, key: str, value) -> None:
        pass

    def set_retrieval(self, docs) -> None:
        pass


_NOOP = _NoopTurn()


@contextmanager
def turn(question: str, session_id: str | None = None):
    """Record one question→answer turn as a JSONL line, written on exit even if
    the turn is interrupted (Stop click) or raises — so partial turns aren't
    lost. Yields a recorder with `.set()`; a no-op when tracing is disabled."""
    if not ENABLED:
        yield _NOOP
        return

    rec: dict = {
        "id": uuid.uuid4().hex,
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "session.id": session_id or "default",
        "input.value": question,
    }
    token = _current.set(_Turn(rec))
    try:
        yield _current.get()
    finally:
        _current.reset(token)
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # One atomic append per record, so a concurrent reader never observes a
        # half-written line.
        with TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
