"""Capture-layer integrity for tracing.py (docs/diagnostics.md).

The JSONL is the contract shared by the future viewer and the headless judge,
so the load-bearing bits get a check: a turn is flushed even when it raises
(Stop-click / crash mid-stream must not lose the record), read_turns skips a
truncated final line, and disabled tracing writes nothing.
"""

import json

import pytest

import tracing


def _enable(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "ENABLED", True)
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "traces.jsonl")


class _Doc:
    """Minimal stand-in for a LangChain Document (set_retrieval is duck-typed)."""

    def __init__(self, date, tags, text):
        self.metadata = {"date_str": date, "tags": tags}
        self.page_content = text


def test_turn_writes_a_readable_record(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    with tracing.turn("what did I do skiing?", session_id="s1") as t:
        t.set("snchat.extraction", {"tags": ["lyže"], "month": 1})
        t.set_retrieval([_Doc("2025-01-04", ["lyže"], "Lyžovačka")])
        t.set("empty", [])  # skipped

    (rec,) = tracing.read_turns(tmp_path / "traces.jsonl")
    assert rec["input.value"] == "what did I do skiing?"
    assert rec["session.id"] == "s1"
    assert rec["snchat.extraction"] == {"tags": ["lyže"], "month": 1}  # structured
    assert rec["snchat.retrieval.count"] == 1
    assert rec["snchat.retrieval.docs"][0] == {
        "date": "2025-01-04",
        "tags": ["lyže"],
        "text": "Lyžovačka",
    }
    assert "empty" not in rec
    assert rec["id"] and rec["ts"]  # idempotent-judge key + timestamp present


def test_record_flushed_even_when_turn_raises(tmp_path, monkeypatch):
    # A Stop-click (StopException) or crash mid-stream must still leave a record.
    _enable(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError), tracing.turn("interrupted", session_id="s1") as t:
        t.set("output.value", "partial answer so far")
        raise RuntimeError("boom mid-stream")

    (rec,) = tracing.read_turns(tmp_path / "traces.jsonl")
    assert rec["output.value"] == "partial answer so far"


def test_read_turns_skips_a_truncated_last_line(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps({"id": "1", "input.value": "ok"}) + "\n{ truncated half-lin",
        encoding="utf-8",
    )
    turns = tracing.read_turns(path)
    assert len(turns) == 1 and turns[0]["input.value"] == "ok"


def test_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "ENABLED", False)
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "traces.jsonl")
    with tracing.turn("q") as t:
        t.set("output.value", "x")
    assert not (tmp_path / "traces.jsonl").exists()
