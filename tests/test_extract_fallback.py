"""Tests for DiaryQueryRouter.extract() falling back robustly.

`with_structured_output(method="function_calling")` returns **None** (not an exception)
when the model emits no tool call, e.g. a bare "summarize my whole diary". extract()
must treat it like a failure and use the regex fallback instead of dereferencing None.
The LLM is mocked so these run offline (no Ollama, no vector store).
"""

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from app import DiaryQueryRouter


def _router_returning(value) -> DiaryQueryRouter:
    """A router whose structured-output chain yields `value` regardless of input."""
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: value)
    return DiaryQueryRouter(vectorstore=None, llm=llm, available_tags=[])


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
    router = DiaryQueryRouter(vectorstore=None, llm=llm, available_tags=[])
    parsed = router.extract("what did I do in march?", [])
    assert parsed is not None
    assert parsed.month == 3  # fallback's regex still extracts the month
