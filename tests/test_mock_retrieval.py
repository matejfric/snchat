"""DiaryQueryRouter.retrieve() routing, asserted against the mock diary.

Ingests the fixture into a real in-memory Chroma (deterministic fake embeddings —
no Ollama needed) and checks each retrieval strategy picks the right entries:
fetch-all by count, most-recent-N, keyword/fuzzy entity lookup, similarity top-K
fallbacks, and the Chroma `where` semantics ($contains on the tags list, bare
single conditions). Covers error_modes §1.1, §2.2-2.4, §2.6 at the retrieval layer;
answer-level checks stay manual (docs/mock_diary.md).
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
import pytest

from app import DiaryQueryRouter, DiarySearchQuery
from constants import SEARCH_K
from parser import parse_standard_notes
from tests.mock_diary import (
    CLIMBING_DATES,
    ENTRIES,
    GOT_DATES,
    GOT_KEYWORDS,
    JANUARY_SKIING_DATES,
    LAST_RUN_DATE,
    POINT_LOOKUP_DATE,
    TAGS,
    WITCHER_DATES,
    WITCHER_KEYWORDS,
    build_zip,
)


def _documents(parsed_notes) -> list[Document]:
    """Mirrors the sidebar ingestion mapping in app.py (one Document per entry,
    tags key omitted when untagged)."""
    docs = []
    for note in parsed_notes:
        if not note["text"].strip():
            continue
        metadata = {
            "uuid": note["uuid"],
            "year": note["date"].year,
            "month": note["date"].month,
            "day": note["date"].day,
            "date_str": note["date"].isoformat(),
            "title": note["title"].strip(),
        }
        if note["tags"]:
            metadata["tags"] = sorted(note["tags"])
        docs.append(Document(page_content=note["text"], metadata=metadata))
    return docs


@pytest.fixture(scope="module")
def router(tmp_path_factory):
    zip_path = tmp_path_factory.mktemp("mockdb") / "mock_diary.zip"
    build_zip(zip_path)
    vectorstore = Chroma.from_documents(
        _documents(parse_standard_notes(zip_path)),
        embedding=DeterministicFakeEmbedding(size=64),
        collection_name="snchat_mock_retrieval",
    )
    return DiaryQueryRouter(vectorstore=vectorstore, llm=None, available_tags=TAGS)


def _dates(docs: list[Document]) -> list[str]:
    return [d.metadata["date_str"] for d in docs]


def test_january_skiing_fetches_every_match(router) -> None:
    # Broad tag+month scope -> count <= FETCH_ALL_MAX -> ALL matches, not top-K.
    q = DiarySearchQuery(query="skiing", month=1, tags=["lyže", "skialp"])
    assert _dates(router.retrieve(q)) == sorted(JANUARY_SKIING_DATES)


def test_recent_n_takes_the_newest_by_date(router) -> None:
    q = DiarySearchQuery(query="climbing sessions", tags=["lezení"], recent=10)
    assert _dates(router.retrieve(q)) == list(CLIMBING_DATES[-10:])


def test_recent_one_is_the_last_run(router) -> None:
    q = DiarySearchQuery(query="my last run", tags=["běh"], recent=1)
    assert _dates(router.retrieve(q)) == [LAST_RUN_DATE]


def test_point_lookup_by_exact_date(router) -> None:
    q = DiarySearchQuery(query="what did I do", year=2025, month=5, day=18)
    docs = router.retrieve(q)
    assert _dates(docs) == [POINT_LOOKUP_DATE]
    assert docs[0].metadata["title"] == "Pálava"


def test_large_scope_specific_falls_back_to_top_k(router) -> None:
    # Year 2025 exceeds FETCH_ALL_MAX; a specific lookup must stay at top-K.
    q = DiarySearchQuery(query="a specific thing", year=2025)
    assert len(router.retrieve(q)) == SEARCH_K


def test_large_scope_all_fetches_everything(router) -> None:
    n_2025 = sum(e.date.startswith("2025-") for e in ENTRIES)
    q = DiarySearchQuery(query="summarize my year", year=2025, breadth="all")
    assert len(router.retrieve(q)) == n_2025


def test_no_filter_uses_similarity_top_k(router) -> None:
    q = DiarySearchQuery(query="summarize my whole diary", breadth="all")
    assert len(router.retrieve(q)) == SEARCH_K


def test_empty_scope_returns_nothing(router) -> None:
    q = DiarySearchQuery(query="skiing", year=2024)
    assert router.retrieve(q) == []


@pytest.mark.parametrize(
    ("keywords", "expected_dates"),
    [(GOT_KEYWORDS, GOT_DATES), (WITCHER_KEYWORDS, WITCHER_DATES)],
    ids=["game-of-thrones", "witcher"],
)
def test_fuzzy_entity_lookup_finds_all_scattered_mentions(
    router, keywords, expected_dates
) -> None:
    # More mentions than SEARCH_K, so only the lexical branch can find them all.
    q = DiarySearchQuery(query="impressions", keywords=list(keywords))
    assert _dates(router.retrieve(q)) == sorted(expected_dates)


def test_fuzzy_lookup_with_recent_trims_to_latest(router) -> None:
    q = DiarySearchQuery(query="latest", keywords=list(GOT_KEYWORDS), recent=3)
    assert _dates(router.retrieve(q)) == sorted(GOT_DATES)[-3:]
