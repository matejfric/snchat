"""Replay the docs/mock_diary.md answer key through the live routing pipeline.

Runs each answer-key prompt through the REAL `DiaryQueryRouter.extract()` (live
Ollama query LLM) + `retrieve()` (real bge-m3 embeddings over the mock diary in
an ephemeral Chroma) and checks the routing and the retrieved entry dates
against the fixture's ground truth — the two layers that are deterministic
enough to assert. Answer wording stays un-judged (no LLM judge yet); pass
--generate to also run generation so traces/eyeballing cover the full turn.

This is a diagnostic, not a unit test: extraction is model behavior, so a
failure is a FINDING to inspect (set SNCHAT_TRACE=1 to append each case's full
turn to diagnostics/traces.jsonl), not necessarily a code bug. Requires Ollama
running with gemma4:12b + bge-m3 pulled; the real ./diary_vector_db is never
touched.

    SNCHAT_TRACE=1 uv run python tests/replay_answer_key.py [--generate] [--case ID]

Exits 1 if any check failed.
"""

# ruff: noqa: E402  (sys.path bootstrap so the script runs directly or via -m)

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from dataclasses import dataclass, field
import datetime as dt
import logging
import tempfile

import chromadb
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from constants import (
    CONTEXT_WINDOW,
    EMBEDDINGS_MODEL_NAME,
    GEN_NUM_CTX,
    GEN_NUM_PREDICT,
    MODEL_NAME,
    SEARCH_K,
)
from diary_query_router import DiaryQueryRouter
from generation import _scope_phrase, plan_generation, stream_with_metrics
from parser import documents_from_notes, parse_standard_notes
from tests import mock_diary as md
import tracing

TODAY = dt.date.today()
SKIING_DATES = tuple(e.date for e in md.SKIING)
SKIING_2026_DATES = tuple(d for d in SKIING_DATES if d.startswith("2026"))
Y2025_DATES = tuple(e.date for e in md.ENTRIES if e.date.startswith("2025"))
NO_FILTERS = {
    "year": None,
    "month": None,
    "day": None,
    "date_from": None,
    "date_to": None,
    "tags": set(),
    "keywords": False,
    "recent": None,
}


@dataclass(frozen=True)
class Case:
    """One answer-key prompt with its checkable expectations.

    `route` maps DiarySearchQuery field -> expected value; `tags` compares as a
    set, `keywords` only as presence/absence (the exact expansions are model
    behavior). `dates` asserts the exact set of retrieved entry dates;
    `dates_include` is the soft variant for semantic top-K (real-embedding
    ranking isn't ground truth, but the known-relevant entries should surface).
    """

    id: str
    question: str
    history: tuple[tuple[str, str], ...] = ()  # prior (user, assistant) turns
    route: dict = field(default_factory=dict)
    dates: tuple[str, ...] | None = None
    dates_include: tuple[str, ...] = ()
    max_docs: int | None = None


CASES = [
    Case(
        "skiing-overview",
        "summarize my skiing",
        route={"tags": {"lyže", "skialp"}, "breadth": "all", "keywords": False},
        dates=SKIING_DATES,
    ),
    Case(
        "skiing-january",
        "what did I do skiing in January?",
        route={"tags": {"lyže", "skialp"}, "month": 1, "year": None},
        dates=md.JANUARY_SKIING_DATES,
    ),
    Case(
        "skiing-followup",
        "and in 2026?",
        history=(
            (
                "how was skiing in 2025?",
                "In 2025 you skied five times (2025-01-04 … 2025-03-01): your old "
                "skis were worn out, you broke a pole at Špindlerův Mlýn, toured "
                "Sněžka and Chopok, and closed the season on spring firn.",
            ),
        ),
        route={"tags": {"lyže", "skialp"}, "year": 2026},
        dates=SKIING_2026_DATES,
    ),
    Case(
        "climbing-recent-10",
        "summarize my last 10 climbing sessions",
        route={"tags": {"lezení"}, "recent": 10},
        dates=md.CLIMBING_DATES[-10:],
    ),
    Case(
        "climbing-progression",
        "how is my climbing going?",
        route={"tags": {"lezení"}, "breadth": "all", "keywords": False},
        dates=md.CLIMBING_DATES,
    ),
    Case(
        "last-run",
        "my last run",
        route={"tags": {"běh"}, "recent": 1},
        dates=(md.LAST_RUN_DATE,),
    ),
    Case(
        "running-czech",
        "kolik jsem toho naběhal?",
        route={"tags": {"běh"}, "keywords": False},
        dates=md.RUNNING_DATES,
    ),
    Case(
        "point-lookup",
        "what did I do on 2025-05-18?",
        route={"year": 2025, "month": 5, "day": 18},
        dates=(md.POINT_LOOKUP_DATE,),
    ),
    Case(
        "got-impressions",
        "what were my impressions of Game of Thrones?",
        route={"keywords": True},
        dates=md.GOT_DATES,
    ),
    Case(
        "witcher-books",
        "what did I think of the Witcher books?",
        # Recall here is best-effort by design (error_modes §2.14): finding the
        # declined Czech mentions depends on the LLM expanding "Witcher" to
        # "Zaklínač", which a local 12B does unreliably. Only the routing and
        # the reliably-matched English-form entry are asserted.
        route={"keywords": True},
        dates_include=("2026-01-15",),
    ),
    Case(
        "anxiety-thematic",
        "when did I feel anxious?",
        route={"keywords": False, "tags": set()},
        dates_include=md.ANXIETY_DATES,
        max_docs=SEARCH_K,
    ),
    Case(
        "whole-diary",
        "summarize my whole diary",
        route={**NO_FILTERS, "breadth": "all"},
        max_docs=SEARCH_K,
    ),
    Case(
        "year-2025",
        "summarize my year 2025",
        route={"year": 2025, "breadth": "all"},
        dates=Y2025_DATES,
    ),
    Case(
        "today-last-year",
        "what was I up to today last year?",
        # Only the year is guaranteed for a RELATIVE single-day phrase: gemma4
        # resolves the date but flip-flops on which fields carry it (full
        # y/m/d, a single-day range, or year-only). The §2.12 validator and
        # the single-day-range collapse make year the reliable floor; only a
        # date typed verbatim guarantees day precision (§2.13). No dates
        # expectation either: filler-grid coverage depends on the run date.
        route={"year": TODAY.year - 1},
    ),
]


def build_vectorstore() -> tuple[Chroma, list[str]]:
    """Mock ZIP -> real ingestion path -> ephemeral Chroma with real embeddings."""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = md.build_zip(Path(tmp) / "mock_diary.zip")
        notes, _skipped = parse_standard_notes(zip_path)
    docs = documents_from_notes(notes)
    store = Chroma(
        client=chromadb.EphemeralClient(),
        collection_name="replay",
        embedding_function=OllamaEmbeddings(model=EMBEDDINGS_MODEL_NAME),
    )
    store.add_documents(docs)
    tags = sorted({t for n in notes for t in n["tags"]})
    return store, tags


def evaluate(case: Case, parsed, dates: list[str]) -> list[tuple[str, bool, str]]:
    """Check routing fields + retrieved dates; returns (name, ok, detail) rows."""
    checks = []
    for fld, expected in case.route.items():
        actual = getattr(parsed, fld)
        if fld == "tags":
            ok = set(actual) == expected
        elif fld == "keywords":
            ok = bool(actual) == expected
        else:
            ok = actual == expected
        checks.append((f"route.{fld}", ok, f"expected {expected!r}, got {actual!r}"))

    if case.dates is not None:
        missing = sorted(set(case.dates) - set(dates))
        extra = sorted(set(dates) - set(case.dates))
        detail = "; ".join(
            f"{label}: {', '.join(vals[:5])}{'…' if len(vals) > 5 else ''}"
            for label, vals in (("missing", missing), ("extra", extra))
            if vals
        )
        checks.append(("dates", not missing and not extra, detail or "exact match"))
    for d in case.dates_include:
        checks.append(("dates.include", d in dates, f"{d} not retrieved"))
    if case.max_docs is not None:
        checks.append(
            (
                "max_docs",
                len(dates) <= case.max_docs,
                f"{len(dates)} docs > {case.max_docs}",
            )
        )
    return checks


def run_case(
    case: Case, router: DiaryQueryRouter, gen_llm: ChatOllama | None
) -> list[tuple[str, bool, str]]:
    history: list[BaseMessage] = []
    for user, assistant in case.history:
        history += [HumanMessage(user), AIMessage(assistant)]

    with tracing.turn(case.question, session_id=f"replay:{case.id}") as trace:
        parsed = router.extract(case.question, history)
        trace.set("snchat.extraction", parsed.model_dump(exclude_none=True))
        docs = router.retrieve(parsed)
        dates = [d.metadata.get("date_str", "?") for d in docs]  # for evaluate()
        trace.set_retrieval(docs)

        checks = evaluate(case, parsed, dates)
        trace.set("snchat.replay.case", case.id)
        trace.set("snchat.replay.failed", [n for n, ok, _ in checks if not ok])

        if gen_llm is not None:
            messages, _premap, canned = plan_generation(
                docs, case.question, history, TODAY, _scope_phrase(parsed), gen_llm
            )
            answer = canned or "".join(stream_with_metrics(gen_llm, messages, {}))
            trace.set("output.value", answer)
            print(f"    ↳ answer: {answer[:120].replace(chr(10), ' ')}…")
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--generate",
        action="store_true",
        help="also run (un-judged) answer generation for complete traces",
    )
    ap.add_argument("--case", help="run only the case with this id")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    for mod in ("parser", "diary_query_router", "generation"):
        logging.getLogger(mod).setLevel(logging.DEBUG)  # feeds the trace log-bridge
    tracing.setup()

    cases = [c for c in CASES if args.case in (None, c.id)]
    if not cases:
        print(f"no case named {args.case!r}; ids: {', '.join(c.id for c in CASES)}")
        return 2

    print(f"Embedding mock diary ({EMBEDDINGS_MODEL_NAME}, ephemeral Chroma)…")
    store, tags = build_vectorstore()
    router = DiaryQueryRouter(
        vectorstore=store,
        llm=ChatOllama(
            model=MODEL_NAME,
            temperature=0,
            num_predict=256,
            num_ctx=CONTEXT_WINDOW,
            reasoning=False,
        ),
        available_tags=tags,
    )
    gen_llm = (
        ChatOllama(
            model=MODEL_NAME,
            temperature=0.3,
            num_ctx=GEN_NUM_CTX,
            num_predict=GEN_NUM_PREDICT,
            reasoning=False,
        )
        if args.generate
        else None
    )

    failed_checks = 0
    for case in cases:
        checks = run_case(case, router, gen_llm)
        bad = [(n, d) for n, ok, d in checks if not ok]
        failed_checks += len(bad)
        mark = "✓" if not bad else "✗"
        print(f"{mark} {case.id:20s} {case.question!r} — {len(checks)} checks")
        for name, detail in bad:
            print(f"    ✗ [{name}] {detail}")

    print(
        f"\n{len(cases)} cases, {failed_checks} failed check(s)"
        + (f" — traces in {tracing.TRACE_PATH}" if tracing.ENABLED else "")
    )
    return 1 if failed_checks else 0


if __name__ == "__main__":
    sys.exit(main())
