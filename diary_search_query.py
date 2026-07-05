# Query understanding & retrieval
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DiarySearchQuery(BaseModel):
    """Extract search parameters from the user's question about their diary.

    The lenient date validator degrades a junk range value from the extraction
    LLM (e.g. `date_from: 2025` emitted as a bare integer) to None. Without it
    one bad optional field fails validation of the WHOLE tool call and every
    valid filter in it is silently lost to the no-filter fallback
    (error_modes §2.12)."""

    query: str = Field(
        description="The semantic search text to find relevant diary entries"
    )
    year: int | None = Field(default=None, description="Filter by year (e.g. 2024)")
    month: int | None = Field(default=None, description="Filter by month (1-12)")
    day: int | None = Field(default=None, description="Filter by day of month (1-31)")
    date_from: str | None = Field(
        default=None,
        description=(
            "Start of a date RANGE, inclusive, ISO yyyy-mm-dd. Use together with "
            "date_to for MULTI-DAY periods ('last week', 'this winter', 'between "
            "March and May'); set alone for open-ended 'since …'. Leave null and "
            "use year/month/day for a single day, month, or year."
        ),
    )
    date_to: str | None = Field(
        default=None,
        description=(
            "End of the date range, inclusive, ISO yyyy-mm-dd; set alone for "
            "open-ended 'until …'."
        ),
    )
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

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def _lenient_date(cls, v):
        # Bare years/numbers are junk here; invalid ISO *strings* are dropped
        # later by the router's range normalization.
        return v if isinstance(v, str) else None


class ResolvedDate(BaseModel):
    """Output of a focused second extraction call that does ONLY date resolution.

    The omnibus extraction sometimes resolves a specific day in its prose `query`
    but mirrors only the year into the fields ("today last year" -> year=2025,
    month/day dropped — error_modes §2.15). When that under-fill is detected, the
    router asks for just the date; a single field can't be half-filled the way
    three independent year/month/day fields can.

    `date` is REQUIRED (with a "none" sentinel), not optional: an all-optional
    single-field schema makes the local model emit no tool call at all (§2.7),
    so `with_structured_output` returned None every time."""

    date: str = Field(
        description=(
            "The ONE calendar date or period the question refers to, as ISO and "
            "PRECISION-HONEST: yyyy-mm-dd for an exact day, yyyy-mm for a whole "
            "month, yyyy for a whole year. Do NOT add finer precision than the "
            "question implies (a whole-year question stays yyyy). The literal "
            '"none" if the question names no specific date.'
        ),
    )
