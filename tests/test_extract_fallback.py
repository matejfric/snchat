"""Tests for DiaryQueryRouter.extract() falling back robustly.

`with_structured_output(method="function_calling")` returns **None** (not an exception)
when the model emits no tool call, e.g. a bare "summarize my whole diary". extract()
must treat it like a failure and use the fallback instead of dereferencing None.

The fallback deliberately guesses NO filters (error_modes §2.9): a regex-guessed
date/tag silently narrows retrieval to the wrong entries — a fluent answer from the
wrong subset — so degraded mode is an unfiltered semantic top-K that only keeps the
overview intent (breadth). The LLM is mocked so these run offline (no Ollama, no
vector store).
"""

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from diary_query_router import DiaryQueryRouter
from diary_search_query import DiarySearchQuery


def _router_returning(
    value,
    available_tags: list[str] | None = None,
    tag_aliases: dict[str, list[str]] | None = None,
):
    """A router whose structured-output chain yields `value` regardless of input."""
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: value)
    return DiaryQueryRouter(
        vectorstore=None,
        llm=llm,
        available_tags=available_tags or [],
        tag_aliases=tag_aliases,
    )


def test_extract_falls_back_when_structured_output_returns_none() -> None:
    # Regression: structured output returning None used to crash with
    # "AttributeError: 'NoneType' object has no attribute 'query'".
    router = _router_returning(None)
    parsed = router.extract("summarize my whole diary", [])
    assert parsed is not None
    assert parsed.query.strip()  # a usable semantic query is always present
    assert parsed.breadth == "all"  # fallback routes a bare overview to fetch-all


def test_extract_falls_back_on_exception() -> None:
    def _boom(_):
        raise RuntimeError("boom")

    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(_boom)
    router = DiaryQueryRouter(vectorstore=None, llm=llm, available_tags=["běh"])
    parsed = router.extract("summarize my running in march", [])
    assert parsed is not None
    assert parsed.query == "summarize my running in march"
    assert parsed.breadth == "all"  # overview intent survives


def test_fallback_never_guesses_filters() -> None:
    # Date-, tag- and count-looking phrasings must NOT become filters: the old
    # regex extractor turned "may" into month=5, "run"⊂"brunch" into tag běh and
    # "last 2 weeks" into recent=2 (error_modes §2.9).
    router = _router_returning(None, available_tags=["běh", "lyže"])
    for q in (
        "what did I do on 2025-05-18?",
        "what may have caused my knee pain?",
        "did I have brunch with Anna?",
        "what did I do in the last 2 weeks?",
    ):
        parsed = router.extract(q, [])
        assert parsed.query == q
        assert (parsed.year, parsed.month, parsed.day) == (None, None, None)
        assert (parsed.date_from, parsed.date_to) == (None, None)
        assert parsed.tags == []
        assert parsed.keywords == []
        assert parsed.recent is None
        assert parsed.breadth == "specific"


def test_extract_normalizes_aliased_and_cased_tags() -> None:
    # The prompt lists tag names next to their aliases, so a small model may echo
    # an alias or re-case the tag instead of returning the exact value. An
    # exact-match clamp dropped those silently — the query then ran with NO tag
    # filter at all (error_modes §2.10). Uses a synthetic alias table so the test
    # doesn't depend on the user-editable TAG_ALIASES config.
    router = _router_returning(
        DiarySearchQuery(
            query="x", tags=["shared-alias", "Gamma", "hallucinated", "alpha"]
        ),
        available_tags=["alpha", "beta", "gamma"],
        tag_aliases={
            "alpha": ["shared-alias"],
            "beta": ["shared-alias", "solo-alias"],
            # gamma deliberately absent: tags without aliases must still normalize
        },
    )
    parsed = router.extract("anything", [])
    # "shared-alias" fans out to alpha+beta, "Gamma" re-cases, junk is dropped,
    # and the echoed exact "alpha" dedupes with the fan-out.
    assert parsed.tags == ["alpha", "beta", "gamma"]


def test_extract_swaps_a_reversed_date_range() -> None:
    router = _router_returning(
        DiarySearchQuery(query="x", date_from="2026-05-31", date_to="2026-03-01")
    )
    parsed = router.extract("between March and May 2026", [])
    assert (parsed.date_from, parsed.date_to) == ("2026-03-01", "2026-05-31")


def test_extract_drops_invalid_range_dates() -> None:
    # A small local model can emit garbage — invalid dates must not reach Chroma.
    router = _router_returning(
        DiarySearchQuery(query="x", date_from="not-a-date", date_to="2026-13-99")
    )
    parsed = router.extract("q", [])
    assert parsed.date_from is None
    assert parsed.date_to is None
