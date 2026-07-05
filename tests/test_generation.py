"""Tests for generation planning (generation.py).

plan_generation() must (a) answer canned on empty retrieval, (b) run a single
streamed pass when the FULL prompt — stuffed context + chat history + question —
fits SINGLE_PASS_BUDGET, (c) otherwise map-reduce over char-budgeted batches,
folding the map calls' token usage into `premap`, and (d) keep condensing the
reduce input until it fits the budget too (Ollama silently truncates past
num_ctx, so every oversized prompt is a silent-wrong-answer bug). Messages are
built directly (no prompt template), so braces in user text must never crash
prompt construction. The LLM is faked; everything runs offline.
"""

import datetime as dt

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)

from constants import SINGLE_PASS_BUDGET
from diary_search_query import DiarySearchQuery
from generation import (
    _batch_by_chars,
    _scope_phrase,
    _usage_of,
    estimate_tokens,
    format_metrics,
    plan_generation,
    stream_with_metrics,
)

TODAY = dt.date(2026, 7, 4)


class FakeGenLLM:
    """Duck-typed ChatOllama: .invoke returns a canned dated summary with usage."""

    def __init__(self, reply: str = "- [2025-01-11] (tags: běh) krátké shrnutí"):
        self.reply = reply
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(prompt)
        return AIMessage(
            content=self.reply,
            response_metadata={
                "prompt_eval_count": 10,
                "eval_count": 5,
                "eval_duration": 2_000_000_000,
            },
        )


def _doc(text: str, date_str: str = "2025-01-11", tags: list[str] | None = None):
    metadata = {"date_str": date_str}
    if tags:
        metadata["tags"] = tags
    return Document(page_content=text, metadata=metadata)


# --- plan_generation: strategy selection ---


def test_empty_docs_returns_canned_answer() -> None:
    messages, premap, canned = plan_generation([], "q", [], TODAY, "", FakeGenLLM())
    assert messages is None
    assert premap == {}
    assert "couldn't find" in canned


def test_small_scope_is_a_single_pass_with_history() -> None:
    llm = FakeGenLLM()
    history = [HumanMessage("how was skiing in 2025?"), AIMessage("great")]
    docs = [_doc("lyžování na Lipně", tags=["lyže"])]
    messages, premap, canned = plan_generation(
        docs, "and in 2026?", history, TODAY, "skiing in 2026", llm
    )
    assert canned is None
    assert premap == {}
    assert llm.calls == []  # no map steps ran
    assert isinstance(messages[0], SystemMessage)
    assert "Date: 2025-01-11" in messages[0].content
    assert "Tags: lyže" in messages[0].content
    assert "skiing in 2026" in messages[0].content  # scope anchor
    assert messages[1:3] == history  # history spliced between system and question
    assert messages[-1] == HumanMessage("and in 2026?")


def test_untagged_entry_renders_dash() -> None:
    messages, _, _ = plan_generation([_doc("x")], "q", [], TODAY, "", FakeGenLLM())
    assert "Tags: —" in messages[0].content


def test_braces_in_query_scope_and_entries_do_not_crash() -> None:
    # Regression: scope used to be f-string-embedded into a ChatPromptTemplate,
    # so a question or scope containing {braces} crashed with KeyError.
    docs = [_doc("code snippet: {'k': 1}")]
    messages, _, _ = plan_generation(
        docs, "what about {something}?", [], TODAY, "notes on {json}", FakeGenLLM()
    )
    assert "{something}" in messages[-1].content
    assert "notes on {json}" in messages[0].content
    assert "{'k': 1}" in messages[0].content


def test_long_history_forces_map_reduce_even_for_small_docs() -> None:
    # Docs alone fit the budget; docs + history do not. The decision must count
    # the FULL prompt, not just the entry text (error_modes §3.2).
    llm = FakeGenLLM()
    docs = [_doc("z" * (SINGLE_PASS_BUDGET // 2))]
    history = [HumanMessage("h" * SINGLE_PASS_BUDGET)]
    messages, premap, canned = plan_generation(docs, "q", history, TODAY, "", llm)
    assert canned is None
    assert len(llm.calls) >= 1  # map ran
    (reduce_msg,) = messages  # reduce prompt only — history is dropped here
    assert isinstance(reduce_msg, HumanMessage)
    assert premap["prompt"] == 10 * len(llm.calls)


# --- plan_generation: map-reduce mechanics ---


def test_map_reduce_batches_and_accumulates_usage() -> None:
    llm = FakeGenLLM()
    entry = "dnes jsem si byl zaběhat kolem přehrady " * 80  # ~3.2k chars
    n = (3 * SINGLE_PASS_BUDGET) // len(entry) + 1  # ~3 budgets worth of text
    docs = [_doc(entry, f"2025-01-{i % 28 + 1:02d}", ["běh"]) for i in range(n)]
    messages, premap, canned = plan_generation(
        docs, "how did my running progress?", [], TODAY, "running", llm
    )
    assert canned is None
    n_calls = len(llm.calls)
    assert n_calls >= 3  # at least one map call per budget's worth of text
    assert premap == {
        "prompt": 10 * n_calls,
        "gen": 5 * n_calls,
        "eval_ns": 2_000_000_000 * n_calls,
    }
    (reduce_msg,) = messages
    assert "how did my running progress?" in reduce_msg.content
    assert llm.reply in reduce_msg.content  # partials made it into the reduce
    # each map batch carried the dated (tags: …) prefix into the prompt
    assert "[2025-01-01] (tags: běh)" in llm.calls[0]


def test_oversized_partials_are_condensed_until_reduce_fits() -> None:
    # Each map call returns a huge partial, so the joined partials exceed the
    # budget; plan_generation must keep condensing instead of overflowing num_ctx.
    llm = FakeGenLLM(reply="- [2025-01-11] " + "dlouhé shrnutí " * 800)  # ~12k chars
    n = (8 * SINGLE_PASS_BUDGET) // 4000  # ~8 budgets worth of entries
    docs = [_doc("x" * 4000, f"2025-{i % 12 + 1:02d}-01") for i in range(n)]
    messages, premap, _ = plan_generation(docs, "q", [], TODAY, "", llm)
    (reduce_msg,) = messages
    assert len(reduce_msg.content) <= SINGLE_PASS_BUDGET
    assert premap["prompt"] == 10 * len(llm.calls)  # condense calls counted too


def test_batch_by_chars_respects_budget_and_order() -> None:
    texts = ["a" * 10, "b" * 10, "c" * 10, "d" * 10, "e" * 10]
    assert _batch_by_chars(texts, 25) == [texts[0:2], texts[2:4], texts[4:]]


def test_batch_by_chars_oversized_item_gets_own_batch() -> None:
    big = "x" * 40
    assert _batch_by_chars(["small", big, "small2"], 25) == [
        ["small"],
        [big],
        ["small2"],
    ]


# --- scope phrase + token estimate ---


def test_scope_phrase_variants() -> None:
    q = DiarySearchQuery
    assert _scope_phrase(q(query="ski", year=2026, month=1, day=5)) == (
        "ski on 2026-01-05"
    )
    assert _scope_phrase(q(query="ski", year=2026, month=1)) == "ski in 2026-01"
    assert _scope_phrase(q(query="ski", year=2026)) == "ski in 2026"
    assert _scope_phrase(q(query="", month=5)) == "in month 5"
    assert _scope_phrase(q(query="běh")) == "běh"
    assert _scope_phrase(
        q(query="ski", date_from="2025-12-01", date_to="2026-02-28")
    ) == "ski from 2025-12-01 to 2026-02-28"
    assert _scope_phrase(q(query="", date_from="2026-03-01")) == "since 2026-03-01"
    assert _scope_phrase(q(query="x", date_to="2024-01-01")) == "x until 2024-01-01"


def test_estimate_tokens_uses_czech_calibrated_divisor() -> None:
    assert estimate_tokens("a" * 30) == 10  # ~3 chars/token
    assert estimate_tokens("") == 1


# --- streaming + metrics helpers ---


class FakeStreamLLM:
    def __init__(self, chunks):
        self.chunks = chunks

    def stream(self, messages):
        yield from self.chunks


def test_stream_with_metrics_yields_content_and_merges_usage() -> None:
    chunks = [
        AIMessageChunk(content="Ahoj "),
        AIMessageChunk(content="světe"),
        AIMessageChunk(  # final Ollama chunk: no content, carries the counters
            content="",
            response_metadata={
                "prompt_eval_count": 7,
                "eval_count": 3,
                "eval_duration": 1_000_000_000,
            },
        ),
    ]
    sink: dict = {}
    out = list(stream_with_metrics(FakeStreamLLM(chunks), [], sink))
    assert out == ["Ahoj ", "světe"]  # empty final chunk isn't yielded
    assert sink["message"].content == "Ahoj světe"
    assert _usage_of(sink["message"]) == {"prompt": 7, "gen": 3, "eval_ns": 10**9}


def test_usage_of_missing_message_is_zeros() -> None:
    assert _usage_of(None) == {"prompt": 0, "gen": 0, "eval_ns": 0}


def test_format_metrics() -> None:
    assert format_metrics({}) is None
    assert format_metrics({"prompt": 5, "gen": 0, "eval_ns": 0}) is None
    line = format_metrics({"prompt": 100, "gen": 50, "eval_ns": 2_000_000_000})
    assert "25 tok/s" in line
    assert "100 prompt + 50 gen tokens" in line
