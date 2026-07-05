import datetime as dt
import logging
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from rapidfuzz import fuzz

from constants import (
    FETCH_ALL_MAX,
    FUZZY_MATCH_THRESHOLD,
    SEARCH_K,
    TAG_ALIASES,
)
from diary_search_query import DiarySearchQuery, ResolvedDate

logger = logging.getLogger(__name__)


def _date_int(iso: str | None) -> int | None:
    """Validated yyyy-mm-dd -> numeric yyyymmdd (the range-filter key); else None."""
    try:
        return int(dt.date.fromisoformat(iso).strftime("%Y%m%d"))
    except (TypeError, ValueError):
        return None


def _iso_precision(s: str) -> tuple[int, int | None, int | None] | None:
    """Parse a precision-honest ISO string from the date specialist into
    (year, month, day), with None for components the string omits: '2025' ->
    (2025, None, None), '2025-07' -> (2025, 7, None), '2025-07-05' -> (2025, 7, 5).
    None if malformed or not a real date. Format parsing, not language matching —
    safe for a multilingual app (see the multilingual-routing note)."""
    parts = s.strip().split("-")
    if not 1 <= len(parts) <= 3:
        return None
    try:
        nums = [int(p) for p in parts]
        y, m, d = (nums + [1, 1])[:3]  # fill absent parts with 1 only to validate
        dt.date(y, m, d)
    except ValueError:
        return None
    return (
        nums[0],
        nums[1] if len(nums) > 1 else None,
        nums[2] if len(nums) > 2 else None,
    )


def _build_where(
    year: int | None,
    month: int | None,
    day: int | None,
    tags: list[str],
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict | None:
    """Build a Chroma metadata `where` filter from extracted query parameters.

    Tags are OR-ed (an entry matches if its tag list contains ANY of them);
    `$and`/`$or` need >=2 children, so single conditions are emitted bare."""
    conditions: list[dict] = []
    if year is not None:
        conditions.append({"year": {"$eq": year}})
    if month is not None:
        conditions.append({"month": {"$eq": month}})
    if day is not None:
        conditions.append({"day": {"$eq": day}})

    # Multi-day ranges compare on the numeric yyyymmdd key (`$gte`/`$lte` are
    # numeric-only in Chroma); one operator per condition, joined by the $and below.
    if (f := _date_int(date_from)) is not None:
        conditions.append({"date_int": {"$gte": f}})
    if (t := _date_int(date_to)) is not None:
        conditions.append({"date_int": {"$lte": t}})

    # `tags` is stored as a list; $contains tests list membership.
    tag_conds = [{"tags": {"$contains": t}} for t in tags]
    if len(tag_conds) == 1:
        conditions.append(tag_conds[0])
    elif len(tag_conds) >= 2:
        conditions.append({"$or": tag_conds})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _keyword_score(candidate: str, text: str) -> float:
    """Match score (0-100) of a keyword candidate against entry text.

    Short single tokens (acronyms/proper nouns like 'GoT', 'Duna') are a binary
    whole-word match — case-EXACT when the candidate carries case, so 'GoT' doesn't
    match the common word 'got' ('i got up early') and 'Duna' doesn't match a sand
    'duna'; an all-lowercase candidate matches case-insensitively (the escape hatch
    for diaries that write acronyms lowercase — alternate casings like 'GOT' must
    come from the LLM's variant expansion). A substring test would false-match
    inside 'forgot', and fuzzy scorers mis-rank 3-char strings (error_modes §2.11).
    Longer / multi-word forms use rapidfuzz `partial_ratio` over casefolded text,
    which scores the best-matching window — so a verbatim mention scores 100
    regardless of entry length and Czech declension still scores high (e.g.
    'hra o trůny' -> 'Hře o trůny'). NOT `WRatio`: it scales partial matches down
    to 0.6 once the entry is >~8x longer than the query, so a verbatim hit in a
    normal diary paragraph collapsed to ~60 and was dropped below the threshold."""
    if len(candidate) <= 4 and " " not in candidate:
        flags = 0 if any(ch.isupper() for ch in candidate) else re.IGNORECASE
        return 100.0 if re.search(rf"\b{re.escape(candidate)}\b", text, flags) else 0.0
    return fuzz.partial_ratio(candidate.casefold(), text.casefold())


def _keyword_hit(candidate: str, haystack: str) -> bool:
    """Whether a keyword candidate's match score clears the threshold."""
    return _keyword_score(candidate, haystack) >= FUZZY_MATCH_THRESHOLD


class DiaryQueryRouter:
    """Turns a natural-language question into structured query parameters (mode +
    filters), then retrieves diary entries either by semantic search (point lookups)
    or by a full metadata fetch (aggregate summaries)."""

    def __init__(
        self,
        vectorstore: Chroma,
        llm: ChatOllama,
        available_tags: list[str] | None = None,
        k: int = SEARCH_K,
        tag_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.vectorstore = vectorstore
        self.llm = llm
        self.available_tags = available_tags or []
        self.k = k
        # The multilingual glossary is injectable (like available_tags) so tests
        # don't depend on the user-editable TAG_ALIASES config; the app uses the
        # config default.
        self.tag_aliases = TAG_ALIASES if tag_aliases is None else tag_aliases
        # Reverse alias→tags map for normalizing extraction output: the prompt
        # shows tag names NEXT TO their aliases, and a small local model may sometimes
        # echo the alias ("skiing") or re-case the tag ("Lyže"). An alias listed
        # under several tags fans out to all of them, like the alias table intends.
        self._alias_to_tags: dict[str, list[str]] = {}
        for tag in self.available_tags:
            for form in (tag, *self.tag_aliases.get(tag, [])):
                self._alias_to_tags.setdefault(form.casefold(), []).append(tag)

    def _fallback_extract_query(self, query: str) -> DiarySearchQuery:
        """Degraded-but-predictable extraction for when structured output fails: no
        filters (unfiltered semantic top-K over the raw question), keeping only the
        overview intent so a bare "summarize my whole diary" still routes to
        breadth="all". A previous regex extractor that guessed dates/tags/counts
        here produced confidently mis-scoped answers instead (error_modes §2.9)."""
        is_overview = re.search(
            r"\b(summ|overview|recap|progress|trend|over time|evolv)", query.lower()
        )
        return DiarySearchQuery(
            query=query, breadth="all" if is_overview else "specific"
        )

    def _resolve_date(
        self, query: str, chat_history: list[BaseMessage], today: dt.date
    ) -> tuple[int, int | None, int | None] | None:
        """Focused fallback for a year-only under-fill (error_modes §2.15): ask the
        LLM for ONLY the date the question refers to, stripped of the omnibus call's
        competing fields/instructions that dilute its date attention (§2.13).
        Multilingual by construction — the LLM resolves the phrase in any language;
        the router gates the call on the structured output, never the question text.
        Returns (year, month, day) at the model's precision, or None on
        no-date/failure/unparseable output (caller then keeps the year-only scope)."""
        structured_llm = self.llm.with_structured_output(
            ResolvedDate, method="function_calling"
        )
        system_msg = (
            f"Today is {today.isoformat()}, a {today.strftime('%A')}. The user asks "
            "about their personal diary, in any language. Give the ONE calendar date "
            "or period the question refers to, resolving relative expressions against "
            f"today — e.g. 'today last year' is {today.year - 1}-{today:%m-%d}, 'this "
            f"month' is {today:%Y-%m}. Use ISO precision that MATCHES the question: "
            "yyyy-mm-dd for a day, yyyy-mm for a whole month, yyyy for a whole year — "
            'add no finer precision than it implies. Use "none" if the question names '
            "no specific date."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        try:
            result = (prompt | structured_llm).invoke(
                {"input": query, "chat_history": chat_history}
            )
        except Exception as exc:
            logger.warning("Date specialist call failed: %s", exc)
            return None
        # None: model emitted no tool call (§2.7). "none": the schema's no-date
        # sentinel. Both mean "no date to backfill".
        if result is None or result.date.strip().lower() == "none":
            return None
        parsed = _iso_precision(result.date)
        if parsed is None:
            logger.info("Date specialist returned unparseable date %r", result.date)
        return parsed

    def extract(self, query: str, chat_history: list[BaseMessage]) -> DiarySearchQuery:
        """Extract mode + filters, resolving follow-ups against the chat history."""
        today = dt.date.today()
        structured_llm = self.llm.with_structured_output(
            DiarySearchQuery, method="function_calling"
        )

        tags_hint = ""
        if self.available_tags:
            lines = [
                f"- {tag}: {', '.join(self.tag_aliases[tag])}"
                if self.tag_aliases.get(tag)
                else f"- {tag}"
                for tag in self.available_tags
            ]
            tags_hint = (
                "\n\nThe diary uses these fixed tags (with example synonyms in "
                "Czech/English). Map the user's topic — in ANY language — to ALL "
                "matching tags. A broad term may match SEVERAL (e.g. 'skiing' -> "
                "lyže and skialp). Use only exact tag values from this list; leave "
                "tags empty if none clearly apply.\n" + "\n".join(lines)
            )

        system_msg = (
            f"Today's date is {today.isoformat()}, a {today.strftime('%A')} "
            f"(year={today.year}, month={today.month}, day={today.day}). "
            "Extract search parameters from the user's question about their personal "
            "diary. The question may be a follow-up to the conversation above; resolve "
            "any references so that 'query' is a self-contained search phrase. "
            "Resolve relative dates (e.g. 'last year' means "
            f"year={today.year - 1}, 'this month' means month={today.month}, and "
            f"'today last year' means year={today.year - 1}, month={today.month}, "
            f"day={today.day}). For a MULTI-DAY period, set date_from and date_to "
            "(inclusive, yyyy-mm-dd) instead of year/month/day: 'last week' means the "
            "previous Monday through Sunday, 'this winter' spans December 1 through "
            "the end of February ACROSS the year boundary, 'between March and May "
            "2026' means 2026-03-01 to 2026-05-31; 'since March' sets only date_from. "
            "date_from/date_to must be yyyy-mm-dd STRINGS — never a bare year or "
            "number. Use year/month/day (never a range) for a single day, month, or "
            "year; when the question names ONE exact day ('on 2025-05-18', 'today "
            "last year'), fill year AND month AND day together. "
            "Always provide a semantic search query. Only set "
            "date/tag filters when the user explicitly or implicitly refers to a time "
            "period or tag. Set 'recent' to N only when the user asks for a specific "
            "number of the latest items ('last 10 climbing sessions' -> 10, 'my last "
            "run' -> 1). Set 'breadth' to 'all' for overviews/summaries/progressions "
            "over the whole scope, else 'specific'. Set 'keywords' ONLY when the "
            "question is about a specific named entity (a TV series, film, book, "
            "person, place) whose mentions are scattered through the diary — list the "
            "user's term plus likely written variants (expand abbreviations, add the "
            f"Czech and English forms); leave empty otherwise.{tags_hint}"
        )

        extraction_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = extraction_prompt | structured_llm
        parsed: DiarySearchQuery | None = None
        try:
            parsed = chain.invoke({"input": query, "chat_history": chat_history})
        except Exception as exc:
            logger.warning("Structured extraction failed, using fallback: %s", exc)

        # `with_structured_output` can return None (the model emitted no tool call, e.g.
        # for a bare "summarize my whole diary") WITHOUT raising; fall back then too.
        if parsed is None:
            logger.info("Extraction produced no tool call; using no-filter fallback")
            parsed = self._fallback_extract_query(query)

        if not parsed.query or not parsed.query.strip():
            logger.debug("Empty semantic query; backfilled from the raw question")
            parsed.query = query

        # Normalize returned tags — exact value, re-cased tag, or an echoed alias
        # ("skiing" fans out to lyže+skialp) — then drop hallucinated leftovers,
        # de-duplicated in order. An exact-match clamp may silently lose the filter
        # when the model echoed an alias (error_modes §2.10).
        raw_tags = list(parsed.tags)
        resolved = [
            tag
            for returned in parsed.tags
            for tag in self._alias_to_tags.get(returned.strip().casefold(), [])
        ]
        parsed.tags = list(dict.fromkeys(resolved))
        if parsed.tags != raw_tags:
            logger.info("Normalized tags %s -> %s", raw_tags, parsed.tags)

        # Normalize the date range: drop invalid dates, swap a reversed range.
        if parsed.date_from and _date_int(parsed.date_from) is None:
            logger.info("Dropped invalid date_from=%r", parsed.date_from)
            parsed.date_from = None
        if parsed.date_to and _date_int(parsed.date_to) is None:
            logger.info("Dropped invalid date_to=%r", parsed.date_to)
            parsed.date_to = None
        if parsed.date_from and parsed.date_to and parsed.date_from > parsed.date_to:
            logger.info(
                "Swapped reversed range %s..%s", parsed.date_from, parsed.date_to
            )
            parsed.date_from, parsed.date_to = parsed.date_to, parsed.date_from

        # Canonicalize a single-day "range" (the model answers e.g. "today last
        # year" with date_from == date_to) to year/month/day, so the scope
        # phrase, route caption and filters all see the point lookup it is.
        if parsed.date_from and parsed.date_from == parsed.date_to:
            logger.info("Collapsed single-day range %s to y/m/d", parsed.date_from)
            d = dt.date.fromisoformat(parsed.date_from)
            parsed.year, parsed.month, parsed.day = d.year, d.month, d.day
            parsed.date_from = parsed.date_to = None

        # One ISO date typed verbatim in the question is authoritative — the
        # model sometimes returns just its year (error_modes §2.13). Only when
        # extraction produced no day-precision filter, so a correct range
        # ("the week after 2025-05-18") is never overwritten; applies to the
        # fallback path too (a literal date is copied, not guessed).
        if not (parsed.day or parsed.date_from or parsed.date_to):
            typed = {
                m
                for m in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
                if _date_int(m) is not None
            }
            if len(typed) == 1:
                d = dt.date.fromisoformat(typed.pop())
                parsed.year, parsed.month, parsed.day = d.year, d.month, d.day
                logger.info("Backfilled y/m/d from verbatim date %s", d.isoformat())

        # Year-only under-fill: the model resolved a specific day/month in its prose
        # `query` but mirrored only the year into the fields (error_modes §2.15).
        # A focused second call recovers the precision. Gated on the STRUCTURED
        # output (year set, nothing finer) so it stays language-agnostic and fires
        # rarely — a true "in 2025" also trips it but the specialist just re-confirms
        # the year (no-op). Runs after the deterministic backfills above so a typed
        # literal date needs no call.
        if parsed.year and not (
            parsed.month or parsed.day or parsed.date_from or parsed.date_to
        ):
            resolved = self._resolve_date(query, chat_history, today)
            if resolved and resolved != (parsed.year, parsed.month, parsed.day):
                logger.info(
                    "Date specialist refined year-only %s -> %s",
                    parsed.year,
                    resolved,
                )
                parsed.year, parsed.month, parsed.day = resolved

        logger.info(
            "Routed query=%r year=%s month=%s day=%s range=%s..%s tags=%s "
            "keywords=%s recent=%s breadth=%s",
            parsed.query,
            parsed.year,
            parsed.month,
            parsed.day,
            parsed.date_from,
            parsed.date_to,
            parsed.tags,
            parsed.keywords,
            parsed.recent,
            parsed.breadth,
        )
        return parsed

    @staticmethod
    def _docs_from_get(fetched: dict) -> list[Document]:
        """Turn a Chroma .get() result into Documents, sorted chronologically."""
        docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(
                fetched["documents"], fetched["metadatas"], strict=True
            )
        ]
        docs.sort(key=lambda d: d.metadata.get("date_str", ""))
        return docs

    def _fuzzy_retrieve(
        self, parsed: DiarySearchQuery, where: dict | None
    ) -> list[Document]:
        """Find entries mentioning a named entity by matching the LLM-expanded surface
        forms against the stored text (see `_keyword_score`). Returns EVERY matching
        entry — the point is to catch all scattered mentions that semantic top-K would
        miss; `recent` then trims to the N latest if the user asked for a count."""
        fetched = self.vectorstore.get(where=where, include=["documents", "metadatas"])
        # Original casing kept: short case-carrying forms match case-exactly.
        candidates = [k.strip() for k in parsed.keywords if k.strip()]
        docs: list[Document] = []
        best = 0.0  # highest score seen, so a 0-match result is explainable in the log
        for content, meta in zip(
            fetched["documents"], fetched["metadatas"], strict=True
        ):
            score = max((_keyword_score(c, content) for c in candidates), default=0.0)
            best = max(best, score)
            if score >= FUZZY_MATCH_THRESHOLD:
                docs.append(Document(page_content=content, metadata=meta))
        docs.sort(key=lambda d: d.metadata.get("date_str", ""))
        matched = len(docs)
        if parsed.recent:
            docs = docs[-parsed.recent :]
        logger.info(
            "retrieve: fuzzy keywords=%s scanned=%d matched=%d best_score=%.0f "
            "(threshold=%d, recent=%s)",
            parsed.keywords,
            len(fetched["documents"]),
            matched,
            best,
            FUZZY_MATCH_THRESHOLD,
            parsed.recent,
        )
        return docs

    def retrieve(self, parsed: DiarySearchQuery) -> list[Document]:
        """Pick a retrieval strategy from the actual DB cardinality (not a brittle
        upfront mode): keyword/entity fuzzy match, most-recent-N, fetch-all, or
        similarity top-K."""
        where = _build_where(
            parsed.year,
            parsed.month,
            parsed.day,
            parsed.tags,
            parsed.date_from,
            parsed.date_to,
        )
        logger.debug("Chroma where filter: %s", where)

        # Keyword/entity lookup: mentions of a named entity (a series, book, place) are
        # scattered across entries that semantic top-K misses, so fuzzy-match the
        # expanded surface forms over the full (optionally date/tag-filtered) text.
        if parsed.keywords:
            return self._fuzzy_retrieve(parsed, where)

        # Explicit "most recent N": ask the DB for matching dates, take the N latest.
        if parsed.recent:
            meta = self.vectorstore.get(where=where, include=["metadatas"])
            pairs = sorted(
                zip(meta["ids"], meta["metadatas"], strict=True),
                key=lambda p: p[1].get("date_str", ""),
                reverse=True,
            )
            top_ids = [i for i, _ in pairs[: parsed.recent]]
            if not top_ids:
                return []
            fetched = self.vectorstore.get(
                ids=top_ids, include=["documents", "metadatas"]
            )
            docs = self._docs_from_get(fetched)  # re-sorted chronologically
            logger.info(
                "retrieve: most-recent-%d -> %d entries", parsed.recent, len(docs)
            )
            return docs

        # No filter at all: can't enumerate the whole diary -> semantic search only.
        if where is None:
            results = self.vectorstore.similarity_search(parsed.query, k=self.k)
            logger.info(
                "retrieve: no filter, similarity k=%d -> %d", self.k, len(results)
            )
            return results

        # Filtered: the real match count decides breadth.
        count = len(self.vectorstore.get(where=where, include=[])["ids"])
        if count == 0:
            logger.info("retrieve: 0 matches for filter")
            return []
        if count <= FETCH_ALL_MAX or parsed.breadth == "all":
            fetched = self.vectorstore.get(
                where=where, include=["documents", "metadatas"]
            )
            docs = self._docs_from_get(fetched)
            logger.info(
                "retrieve: fetch-all %d entries (count=%d, breadth=%s)",
                len(docs),
                count,
                parsed.breadth,
            )
            return docs
        results = self.vectorstore.similarity_search(
            parsed.query, k=self.k, filter=where
        )
        logger.info(
            "retrieve: count=%d > %d & breadth=specific -> similarity k=%d -> %d",
            count,
            FETCH_ALL_MAX,
            self.k,
            len(results),
        )
        return results
