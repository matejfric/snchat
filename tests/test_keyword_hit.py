"""Tests for the keyword/entity matcher behind DiaryQueryRouter's lexical branch.

`_keyword_hit` decides whether an LLM-expanded surface form occurs in a diary entry.
Short single tokens (acronyms/proper nouns like "GoT") match as WHOLE WORDS and,
when the candidate carries case, case-EXACTLY — lowercasing both sides made "GoT"
match the common English word "got" and "Duna" match a sand "duna" (error_modes
§2.11). An all-lowercase candidate stays case-insensitive. Longer / multi-word
forms use rapidfuzz `partial_ratio` over casefolded text, so a verbatim mention
matches regardless of entry length or casing and Czech declension still scores
high. Candidates and text are passed AS WRITTEN; the matcher folds internally.
"""

import pytest

from diary_query_router import _keyword_hit

# The surface forms the router would expand the query "GoT" into (original case).
CANDIDATES = ["GoT", "Game of Thrones", "Hra o trůny"]


def _matches(text: str) -> bool:
    """Does ANY expanded candidate hit this entry? Mirrors `_fuzzy_retrieve`."""
    return any(_keyword_hit(c, text) for c in CANDIDATES)


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
        ("I got up early and went for a run.", False),  # common word, wrong case
    ],
)
def test_entry_matches_expanded_query(text: str, expected: bool) -> None:
    assert _matches(text) is expected


def test_cased_acronym_does_not_match_common_lowercase_word() -> None:
    # THE case fix: the entity "GoT"/"Duna" must not hit ordinary words in English
    # snippets or Czech text ("i got up early", a sand "duna").
    assert not _keyword_hit("GoT", "i got up early and went for a run")
    assert not _keyword_hit("Duna", "na pláži byla obrovská duna z bílého písku")


def test_cased_acronym_matches_its_own_casing() -> None:
    assert _keyword_hit("GoT", "večer jsem dokoukal GoT, výborný finále")
    assert _keyword_hit("Duna", "Duna od Franka Herberta je skvělá kniha")


def test_lowercase_short_candidate_stays_case_insensitive() -> None:
    # Escape hatch: a diary that writes acronyms lowercase is still searchable by
    # passing (or LLM-expanding) the lowercase form.
    assert _keyword_hit("got", "pustil jsem si epizodu got, super díl")
    assert _keyword_hit("got", "Pustil jsem si další díl GOT večer.")


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


def test_multiword_is_case_insensitive() -> None:
    # Long forms fold case internally — "Game of Thrones" finds a lowercase mention.
    assert _keyword_hit("Game of Thrones", "koukal jsem na game of thrones u večeře")
