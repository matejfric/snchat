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

from constants import SEARCH_K
from diary_query_router import DiaryQueryRouter
from diary_search_query import DiarySearchQuery
from parser import documents_from_notes, parse_standard_notes
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


@pytest.fixture(scope="module")
def router(tmp_path_factory):
    # Ingests through the REAL parser + document mapping (parser.py), so the
    # metadata contract retrieval relies on cannot drift from the app's.
    zip_path = tmp_path_factory.mktemp("mockdb") / "mock_diary.zip"
    build_zip(zip_path)
    vectorstore = Chroma.from_documents(
        documents_from_notes(parse_standard_notes(zip_path)),
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


def test_fuzzy_lookup_respects_date_filter(router) -> None:
    # The lexical branch scans the date/tag-filtered subset — if the `where` were
    # dropped on the way to _fuzzy_retrieve, mentions from other years would leak in.
    assert any(d.startswith("2025-") for d in WITCHER_DATES)  # span crosses years,
    assert any(d.startswith("2026-") for d in WITCHER_DATES)  # else this is vacuous
    q = DiarySearchQuery(query="witcher", keywords=list(WITCHER_KEYWORDS), year=2026)
    expected = sorted(d for d in WITCHER_DATES if d.startswith("2026-"))
    assert _dates(router.retrieve(q)) == expected


def _entry_dates_between(lo: str, hi: str) -> list[str]:
    """Fixture ground truth: all entry dates in the inclusive ISO range."""
    return sorted(e.date for e in ENTRIES if lo <= e.date <= hi)


def test_date_range_fetches_exactly_the_window(router) -> None:
    q = DiarySearchQuery(
        query="what happened", date_from="2025-05-10", date_to="2025-05-20"
    )
    expected = _entry_dates_between("2025-05-10", "2025-05-20")
    assert expected  # the window must actually cover fixture entries
    assert _dates(router.retrieve(q)) == expected


def test_date_range_spans_the_year_boundary(router) -> None:
    # 'this winter'-style ranges are the reason ranges exist: a single
    # year/month filter cannot express December..January.
    q = DiarySearchQuery(
        query="winter", date_from="2025-12-15", date_to="2026-01-15", breadth="all"
    )
    dates = _dates(router.retrieve(q))
    assert dates == _entry_dates_between("2025-12-15", "2026-01-15")
    assert any(d.startswith("2025-12") for d in dates)
    assert any(d.startswith("2026-01") for d in dates)


def test_open_ended_range_since(router) -> None:
    q = DiarySearchQuery(query="recent months", date_from="2026-01-01", breadth="all")
    assert _dates(router.retrieve(q)) == _entry_dates_between("2026-01-01", "9999")


def test_fuzzy_entity_lookup_respects_date_range(router) -> None:
    # The range filter must reach the lexical branch's vectorstore.get(where=…).
    q = DiarySearchQuery(
        query="witcher",
        keywords=list(WITCHER_KEYWORDS),
        date_from="2025-01-01",
        date_to="2025-12-31",
    )
    expected = sorted(d for d in WITCHER_DATES if d <= "2025-12-31")
    assert expected and expected != sorted(WITCHER_DATES)  # range must actually trim
    assert _dates(router.retrieve(q)) == expected
