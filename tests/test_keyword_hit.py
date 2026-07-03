"""Tests for the keyword/entity matcher behind DiaryQueryRouter's lexical branch.

`_keyword_hit` decides whether an LLM-expanded surface form occurs in a diary entry.
Short single tokens (acronyms like "GoT") match as WHOLE WORDS; longer / multi-word
forms use rapidfuzz `partial_ratio` so a verbatim mention matches regardless of entry
length and Czech declension still scores high. Inputs are expected lowercased (the
router lowercases both sides before calling), so the tests mirror that contract.
"""

import pytest

from diary_query_router import _keyword_hit

# The surface forms the router would expand the query "GoT" into.
CANDIDATES = ["got", "game of thrones", "hra o trůny"]


def _matches(text: str) -> bool:
    """Does ANY expanded candidate hit this entry? Mirrors `_fuzzy_retrieve`."""
    haystack = text.lower()
    return any(_keyword_hit(c.lower(), haystack) for c in CANDIDATES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # --- should match ---
        ("Dnes večer jsem koukal na Game of Thrones, super díl.", True),  # exact EN
        ("Pustil jsem si další epizodu GoT, konečně se to rozjíždí.", True),  # acronym
        ("Četl jsem, že ve Hře o trůny je spoustu postav.", True),  # cz declension
        ("Bavíme se o seriálu Hra o trůny s kamarády.", True),  # cz base form
        # --- should NOT match ---
        ("Ráno běh 10 km, odpoledne nákupy a vaření večeře.", False),  # unrelated
        ("Sledování fantasy seriálů mě baví.", False),  # theme only
        ("I almost forgot my friend's birthday!", False),  # 'got' inside 'forgot'
        ("Let's go!", False),  # 'go' is not 'got'
    ],
)
def test_entry_matches_expanded_query(text: str, expected: bool) -> None:
    assert _matches(text) is expected


def test_short_acronym_matches_whole_word() -> None:
    assert _keyword_hit("got", "pustil jsem si epizodu got, super díl")


def test_short_acronym_not_matched_as_substring() -> None:
    # The whole-word boundary is what keeps "got" out of "forgot" — the reason the
    # short-token path uses \b...\b instead of a plain substring test or WRatio.
    assert not _keyword_hit("got", "skoro jsem zapomněl forgot zavolat domů")


def test_multiword_matches_verbatim_in_long_entry() -> None:
    # Regression: WRatio scales partial matches down to 0.6 once the entry is >~8x
    # longer than the query, so a VERBATIM multi-word mention in a normal diary
    # paragraph scored ~60 and was dropped. partial_ratio finds the window -> 100.
    entry = (
        "ráno jsem zaspal, pak práce na projektu celý den. večer jsem si pustil "
        "nový díl campfire cooking in another world, pohodové anime o vaření u "
        "táboráku, snědl jsem u toho večeři a šel brzo spát."
    )
    assert _keyword_hit("campfire cooking", entry)
