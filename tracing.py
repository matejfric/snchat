"""Dev-only diagnostic tracing to a local Phoenix server (docs/diagnostics.md).

Disabled unless SNCHAT_TRACE=1, and then everything stays on localhost — the
app's offline guarantee is untouched. When enabled:

- every LangChain LLM call (extraction, map/condense, streamed answer) is
  auto-traced via OpenInference, nested under one `diary_turn` span per question;
- the routing decisions the modules already narrate through `logging`
  (retrieval strategy, Chroma `where` filter, generation plan) are attached to
  the current span as events, so no core module needs instrumentation code;
- `turn()` records the structured decision data (extracted DiarySearchQuery,
  retrieved dates, plan mode, token usage) as span attributes.

Usage:

    uv run phoenix serve                        # trace UI at http://localhost:6006
    SNCHAT_TRACE=1 uv run streamlit run app.py  # (second terminal)
"""

from contextlib import contextmanager
import json
import logging
import os

ENABLED = os.environ.get("SNCHAT_TRACE") == "1"

# Loggers whose records become events on the current span (see module docstring).
_BRIDGED_LOGGERS = ("parser", "diary_query_router", "generation")

_tracer = None  # set once by setup(); module cache keeps it across Streamlit reruns


class _SpanEventHandler(logging.Handler):
    """Mirror a log record onto the currently active span as an event."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span.is_recording():
                span.add_event(
                    record.getMessage(),
                    {"log.logger": record.name, "log.level": record.levelname},
                )
        except Exception:
            self.handleError(record)


def setup() -> None:
    """Register the Phoenix OTel exporter and the LangChain auto-instrumentor.

    No-op unless SNCHAT_TRACE=1; idempotent across Streamlit reruns. Imports are
    deferred so the (dev-only) tracing packages are never touched in normal runs.
    """
    global _tracer
    if not ENABLED or _tracer is not None:
        return

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    # Exports to http://localhost:6006 unless PHOENIX_COLLECTOR_ENDPOINT overrides;
    # unbatched, so spans appear in the UI as soon as each step finishes.
    provider = register(project_name="snchat", set_global_tracer_provider=True)
    LangChainInstrumentor().instrument(tracer_provider=provider)

    handler = _SpanEventHandler(level=logging.DEBUG)
    for name in _BRIDGED_LOGGERS:
        logging.getLogger(name).addHandler(handler)

    _tracer = provider.get_tracer("snchat")


class _Turn:
    """Attribute recorder for the per-question root span."""

    def __init__(self, span) -> None:
        self._span = span

    def set(self, key: str, value) -> None:
        """Record a span attribute; dicts/objects are JSON-ified, empties skipped."""
        if value is None or value == "" or value == []:
            return
        if not isinstance(value, str | bool | int | float | list | tuple):
            value = json.dumps(value, ensure_ascii=False, default=str)
        self._span.set_attribute(key, value)


class _NoopTurn:
    def set(self, key: str, value) -> None:
        pass


_NOOP = _NoopTurn()


@contextmanager
def turn(question: str, session_id: str | None = None):
    """Root span for one question→answer turn; LLM sub-spans nest under it.

    `session_id` groups turns of one conversation in Phoenix's Sessions view.
    Yields a no-op recorder when tracing is disabled.
    """
    if _tracer is None:
        yield _NOOP
        return

    attributes = {
        "openinference.span.kind": "CHAIN",
        "input.value": question,
    }
    if session_id:
        attributes["session.id"] = session_id
    with _tracer.start_as_current_span("diary_turn", attributes=attributes) as span:
        yield _Turn(span)
