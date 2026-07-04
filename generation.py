"""Size-adaptive answer generation over the retrieved diary entries.

Pure planning/streaming helpers with no Streamlit dependency, extracted from
app.py so tests can import them without triggering the app's import-time UI
side effects.
"""

import datetime as dt
import logging

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from constants import CHARS_PER_TOKEN, SINGLE_PASS_BUDGET
from diary_search_query import DiarySearchQuery

logger = logging.getLogger(__name__)


def _tags_str(metadata: dict) -> str:
    """Render an entry's tags list as prompt text ('—' when untagged). Shared by both
    generation paths so the LLM sees tags consistently (single-pass AND map-reduce)."""
    tags = metadata.get("tags")
    return ", ".join(tags) if tags else "—"


def estimate_tokens(text: str) -> int:
    """Rough offline token estimate (~CHARS_PER_TOKEN chars/token, calibrated for the
    Czech corpus). `get_num_tokens` would download a GPT-2 tokenizer over the network,
    breaking the app's offline guarantee."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _format_entry(d: Document) -> str:
    """One prompt block per entry ('—' when untagged)."""
    return (
        f"Date: {d.metadata.get('date_str', '?')}\n"
        f"Tags: {_tags_str(d.metadata)}\n"
        f"Content: {d.page_content}"
    )


def _format_context(docs: list[Document]) -> str:
    """Stuff entries into the prompt context, one block per entry."""
    return "\n\n".join(_format_entry(d) for d in docs)


def stream_with_metrics(llm: ChatOllama, messages: list[BaseMessage], sink: dict):
    """Stream an LLM call token-by-token (for st.write_stream) while accumulating the
    chunks. After the generator is exhausted, sink['message'] holds the final merged
    chunk, whose response_metadata/usage_metadata carry the token counts + durations."""
    full = None
    for chunk in llm.stream(messages):
        full = chunk if full is None else full + chunk
        if chunk.content:
            yield chunk.content
    sink["message"] = full


def _usage_of(msg) -> dict:
    """Extract {prompt, gen, eval_ns} token counts from an AIMessage(Chunk), zeros if
    absent. (eval_ns is generation time in nanoseconds.)"""
    md = getattr(msg, "response_metadata", None) or {}
    return {
        "prompt": md.get("prompt_eval_count") or 0,
        "gen": md.get("eval_count") or 0,
        "eval_ns": md.get("eval_duration") or 0,
    }


def format_metrics(usage: dict) -> str | None:
    """Human-readable perf line from accumulated {prompt, gen, eval_ns}, or None."""
    if not usage or not usage.get("gen"):
        return None
    line = f"{usage['prompt']} prompt + {usage['gen']} gen tokens"
    if usage.get("eval_ns"):
        line = f"⚡ {usage['gen'] / (usage['eval_ns'] / 1e9):.0f} tok/s · " + line
    return line


def _scope_phrase(parsed: DiarySearchQuery) -> str:
    """Human-readable description of what was actually retrieved (topic + date), used
    to anchor the answer so terse follow-ups ('and in 2026?') aren't pulled toward an
    earlier turn. Built from the filters that were applied, so it is ground truth."""
    topic = parsed.query.strip() if parsed.query else ""
    if parsed.year and parsed.month and parsed.day:
        when = f"on {parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d}"
    elif parsed.year and parsed.month:
        when = f"in {parsed.year:04d}-{parsed.month:02d}"
    elif parsed.year:
        when = f"in {parsed.year}"
    elif parsed.month:
        when = f"in month {parsed.month}"
    else:
        when = ""
    return " ".join(p for p in (topic, when) if p)


def _batch_by_chars(texts: list[str], budget: int) -> list[list[str]]:
    """Group consecutive (chronological) texts into batches under a char budget."""
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for t in texts:
        if current and size + len(t) > budget:
            batches.append(current)
            current, size = [], 0
        current.append(t)
        size += len(t)
    if current:
        batches.append(current)
    return batches


def _summarize(gen_llm: ChatOllama, joined: str, usage: dict) -> str:
    """One non-streamed map/condense call; its token usage is folded into `usage`."""
    resp = gen_llm.invoke(
        "/no_think\n"
        "Summarize the key points of these chronologically-ordered diary entries "
        "as a few concise bullet points, keeping the [YYYY-MM-DD] date (and any "
        f"tags) next to each point.\n\n{joined}\n\nDated summary:"
    )
    for k, v in _usage_of(resp).items():
        usage[k] += v
    return resp.content


def _reduce_input(user_query: str, combined: str, today: dt.date, scope: str) -> str:
    focus_line = f"Focus: {scope}.\n" if scope else ""
    return (
        "/no_think\n"
        f"Today's date is {today.isoformat()}.\n"
        f"{focus_line}"
        "Below are dated summaries of diary entries, in chronological order. Combine "
        "them into ONE coherent narrative that follows the progression OVER TIME, "
        "citing specific dates. Use ONLY this information.\n\n"
        f"User's question: {user_query}\n\n{combined}\n\nProgression summary:"
    )


def plan_generation(
    docs: list[Document],
    user_query: str,
    chat_history: list[BaseMessage],
    today: dt.date,
    scope: str,
    gen_llm: ChatOllama,
) -> tuple[list[BaseMessage] | None, dict, str | None]:
    """Plan size-adaptive generation over the retrieved (chronological) entries.

    Returns (messages_to_stream, premap_usage, canned_answer):
    - empty docs   -> (None, {}, "couldn't find…")
    - fits budget  -> (messages, {}, None)            single streamed pass (+ history)
    - too large    -> (reduce_messages, premap_usage, None)  map steps already run

    The single-pass check measures the FULL prompt — system text (with the stuffed
    context), chat history, and the question — against SINGLE_PASS_BUDGET, because
    Ollama silently truncates anything past num_ctx (error_modes §3.2). Messages are
    built directly rather than through a prompt template, so braces in the user's
    question or in entries can't break formatting. `premap_usage` carries the token
    usage of the non-streamed map calls so the final metrics cover the whole job.
    `scope` anchors the answer on the actually-retrieved topic/period so prior turns
    can't pull it off-target.
    """
    if not docs:
        return (
            None,
            {},
            "I couldn't find any diary entries matching that. "
            "Try a different tag or time period.",
        )

    anchor = (
        f"The user's current request is about: {scope}.\n"
        "The conversation history may mention other periods or topics — answer the "
        "CURRENT request only. The diary excerpts below have ALREADY been filtered "
        "to match it; treat them as the complete set for this question.\n"
        if scope
        else "Answer the user's question using ONLY the diary excerpts below.\n"
    )
    system_text = (
        "/no_think\n"
        "You are a compassionate personal assistant helping the user review their "
        "diary.\n"
        f"Today's date is {today.isoformat()}.\n"
        f"{anchor}"
        "The entries are in CHRONOLOGICAL order. Answer the question directly; if "
        "the user asked for an overview, summary, recap, or progression, give a "
        "concise chronological summary instead. Either way, cite the specific "
        "date(s) you use.\n"
        "If there are no excerpts below, say honestly that you have no entries for "
        "that.\n\n"
        f"Context:\n{_format_context(docs)}"
    )
    messages: list[BaseMessage] = [
        SystemMessage(system_text),
        *chat_history,
        HumanMessage(user_query),
    ]
    prompt_chars = sum(len(m.content) for m in messages)

    if prompt_chars <= SINGLE_PASS_BUDGET:
        logger.info(
            "Generating from %d entries in a single pass (%d prompt chars)",
            len(docs),
            prompt_chars,
        )
        return messages, {}, None

    # Too much text for one context window: summarize chronological batches (map),
    # then stream a single narrative that combines the dated batch-summaries (reduce).
    dated_texts = [
        f"[{d.metadata.get('date_str', '?')}] "
        f"(tags: {_tags_str(d.metadata)}) {d.page_content}"
        for d in docs
    ]
    batches = _batch_by_chars(dated_texts, SINGLE_PASS_BUDGET)
    logger.info(
        "Generating from %d entries (%d prompt chars) via map-reduce in %d batches",
        len(docs),
        prompt_chars,
        len(batches),
    )

    premap = {"prompt": 0, "gen": 0, "eval_ns": 0}
    partials = [_summarize(gen_llm, "\n\n".join(b), premap) for b in batches]

    # The joined partials can outgrow the window themselves (each map call may emit
    # up to GEN_NUM_PREDICT tokens): condense in rounds until the reduce prompt fits.
    # Each round shrinks the partial count by ~budget/partial-size, so this
    # terminates quickly; a single partial always fits by construction.
    reduce_in = _reduce_input(user_query, "\n\n".join(partials), today, scope)
    while len(reduce_in) > SINGLE_PASS_BUDGET and len(partials) > 1:
        logger.info(
            "Reduce input too large (%d chars > %d) — condensing %d partials",
            len(reduce_in),
            SINGLE_PASS_BUDGET,
            len(partials),
        )
        partials = [
            _summarize(gen_llm, "\n\n".join(group), premap)
            for group in _batch_by_chars(partials, SINGLE_PASS_BUDGET)
        ]
        reduce_in = _reduce_input(user_query, "\n\n".join(partials), today, scope)
    return [HumanMessage(reduce_in)], premap, None
