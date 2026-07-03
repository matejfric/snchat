# Query understanding & retrieval
from typing import Literal

from pydantic import BaseModel, Field


class DiarySearchQuery(BaseModel):
    """Extract search parameters from the user's question about their diary."""

    query: str = Field(
        description="The semantic search text to find relevant diary entries"
    )
    year: int | None = Field(default=None, description="Filter by year (e.g. 2024)")
    month: int | None = Field(default=None, description="Filter by month (1-12)")
    day: int | None = Field(default=None, description="Filter by day of month (1-31)")
    tags: list[str] = Field(
        default_factory=list,
        description="All diary tags this question is about (may be several); "
        "empty if none apply",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Distinctive named entities to look up by keyword — a TV series, film, "
            "book, person, or place the question is about (NOT a theme or mood). "
            "Include the user's term AND its likely written variants: expand "
            "abbreviations and add Czech + English forms (e.g. 'GoT' -> ['GoT', "
            "'Game of Thrones', 'Hra o trůny']). Leave empty for thematic or mood "
            "questions with no specific named entity."
        ),
    )
    recent: int | None = Field(
        default=None,
        description=(
            "Number of most-recent entries to fetch. Set ONLY when the user asks for "
            "a specific count of the LATEST items (e.g. 'last 10 climbing sessions' "
            "-> 10, 'my last run' -> 1). Leave null otherwise."
        ),
    )
    breadth: Literal["specific", "all"] = Field(
        default="specific",
        description=(
            "'all' when the user wants an overview, summary, recap, trend, or "
            "progression across the WHOLE filtered scope (e.g. 'summarize my "
            "bouldering progression'). 'specific' for point lookups (the default)."
        ),
    )
