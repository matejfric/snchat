"""The mock diary fixture stays parseable and its ground truth stays true.

Builds the synthetic Standard Notes ZIP (tests/mock_diary.py), runs it through the
real `parse_standard_notes()`, and pins the fixture's ground-truth constants — so a
change to either the parser, the ingestion contract, or the fixture content that
would invalidate the docs/mock_diary.md answer key fails here first.
"""

from collections import Counter

import pytest

from constants import FETCH_ALL_MAX, SINGLE_PASS_BUDGET
from diary_query_router import _keyword_hit
from parser import parse_standard_notes
from tests.mock_diary import (
    EMPTY_TEXT_DATE,
    ENTRIES,
    GOT_DATES,
    GOT_KEYWORDS,
    JANUARY_SKIING_DATES,
    KEYWORD_NEGATIVE_DATES,
    POINT_LOOKUP_DATE,
    TAGS,
    WITCHER_DATES,
    WITCHER_KEYWORDS,
    build_zip,
)


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    zip_path = tmp_path_factory.mktemp("mock") / "mock_diary.zip"
    build_zip(zip_path)
    return parse_standard_notes(zip_path)


def test_parse_counts_and_uniqueness(parsed) -> None:
    # Everything in ENTRIES plus the one valid-date-but-empty-text edge note.
    assert len(parsed) == len(ENTRIES) + 1
    assert len({n["date"] for n in parsed}) == len(parsed)  # one entry per day
    indexable = [n for n in parsed if n["text"].strip()]
    assert len(indexable) == len(ENTRIES)
    assert any(
        n["date"].isoformat() == EMPTY_TEXT_DATE and not n["text"] for n in parsed
    )


def test_parse_is_chronological(parsed) -> None:
    dates = [n["date"] for n in parsed]
    assert dates == sorted(dates)


def test_contract_violations_are_skipped(parsed) -> None:
    # No-date title, invalid date, deleted item, non-Note item: none may survive.
    titles = {n["title"].strip() for n in parsed}
    assert not {"Nápady na dárky", "Chyba", "Smazáno", "osobní"} & titles


def test_point_lookup_entry(parsed) -> None:
    entry = next(n for n in parsed if n["date"].isoformat() == POINT_LOOKUP_DATE)
    assert entry["title"].strip() == "Pálava"
    assert entry["tags"] == {"turistika"}
    assert "Pálavu" in entry["text"]


def test_tags_map_onto_notes(parsed) -> None:
    expected = Counter(t for e in ENTRIES for t in e.tags)
    actual = Counter(t for n in parsed for t in n["tags"])
    assert actual == expected
    assert {t for n in parsed for t in n["tags"]} == set(TAGS)


def test_scope_sizes_exercise_the_breadth_paths(parsed) -> None:
    """Year 2025 must overflow both breadth limits, so 'summarize my 2025' takes
    fetch-all + map-reduce while a specific 2025 lookup falls back to top-K."""
    year_2025 = [n for n in parsed if n["date"].year == 2025 and n["text"].strip()]
    assert len(year_2025) > FETCH_ALL_MAX
    assert sum(len(n["text"]) for n in year_2025) > SINGLE_PASS_BUDGET
    # The January-skiing scope stays small enough to be fetched in full by count.
    assert 0 < len(JANUARY_SKIING_DATES) <= FETCH_ALL_MAX


@pytest.mark.parametrize(
    ("keywords", "expected_dates"),
    [(GOT_KEYWORDS, GOT_DATES), (WITCHER_KEYWORDS, WITCHER_DATES)],
    ids=["game-of-thrones", "witcher"],
)
def test_keyword_ground_truth_over_every_entry(parsed, keywords, expected_dates):
    """Sweep the WHOLE diary with the expanded surface forms, mirroring
    `_fuzzy_retrieve`: exactly the ground-truth entries match — declined Czech
    forms and the long buried mention included, filler and the look-alike
    negatives ('gotická', 'forgot') excluded."""
    candidates = [k.lower() for k in keywords]
    matched = {
        n["date"].isoformat()
        for n in parsed
        if any(_keyword_hit(c, n["text"].lower()) for c in candidates)
    }
    assert matched == set(expected_dates)
    assert not matched & set(KEYWORD_NEGATIVE_DATES)
