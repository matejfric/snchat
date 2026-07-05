"""Generate a self-contained dark-mode HTML viewer from diagnostics/traces.jsonl.

Reads `tracing.read_turns()`, groups turns by session, and renders each as a
routing pipeline (Question → Extract → Retrieve → Plan → Answer) plus the
routing narration, the retrieved entries (collapsible, in a bounded scroll
box), and the answer. Pure-CSS diagram, a few lines of JS for sidebar nav; no
external assets, so it opens offline via file://. See docs/diagnostics.md.

    uv run python diagnostics_report.py [out.html]   # default diagnostics/report.html
"""

# ruff: noqa: E501  (the HTML/CSS template is one big string; wrapping it hurts more than it helps)

from pathlib import Path
import sys

from jinja2 import Environment, select_autoescape
from markdown_it import MarkdownIt

import tracing

# Answers are markdown. html=False escapes any raw HTML in the (untrusted) answer
# so the rendered result is safe to mark |safe; image is disabled so a stray
# ![](http://…) can't auto-fetch and break the offline/privacy guarantee (links
# don't auto-load, so they stay).
_MD = MarkdownIt("commonmark", {"html": False}).disable("image")


def _extract_label(ex: dict) -> str:
    """Compact one-line summary of the DiarySearchQuery for the Extract node."""
    bits = []
    if ex.get("tags"):
        bits.append("tags=" + "+".join(ex["tags"]))
    if ex.get("keywords"):
        bits.append("kw=" + "+".join(ex["keywords"]))
    if ex.get("date_from") or ex.get("date_to"):
        bits.append(f"{ex.get('date_from', '…')}→{ex.get('date_to', '…')}")
    elif ex.get("year"):
        bits.append(
            "-".join(
                str(v) for v in (ex.get("year"), ex.get("month"), ex.get("day")) if v
            )
        )
    elif ex.get("month"):
        bits.append(f"month={ex['month']}")
    if ex.get("recent"):
        bits.append(f"recent={ex['recent']}")
    if ex.get("breadth") == "all":
        bits.append("breadth=all")
    return " · ".join(bits) or "—"


def _view(turn: dict) -> dict:
    """Turn record → template view model (compact node labels + full detail)."""
    ex = turn.get("snchat.extraction", {})
    # Retrieval strategy from the routing narration (falls back to the count).
    strat = next(
        (
            e.split("retrieve:", 1)[1].strip()
            for e in turn.get("events", [])
            if "retrieve:" in e
        ),
        f"{turn.get('snchat.retrieval.count', 0)} entries",
    )
    docs = turn.get("snchat.retrieval.docs") or []  # empty for a zero-result turn
    return {
        "id": turn.get("id", "")[:8] or turn.get("ts", "?"),
        "ts": turn.get("ts", ""),
        "q": turn.get("input.value", "(no question)"),
        "extract": _extract_label(ex),
        "strat": strat,
        "count": turn.get("snchat.retrieval.count", len(docs)),
        "plan": turn.get("snchat.plan", "—"),
        # Narration: drop the "logger: " prefix; each event becomes one clean row.
        "events": [e.split(": ", 1)[-1] for e in turn.get("events", [])],
        "docs": docs,
        "answer": turn.get("output.value", ""),
        "answer_html": _MD.render(turn["output.value"])
        if turn.get("output.value")
        else "",
        "usage": turn.get("snchat.usage", {}),
        "failed": turn.get("snchat.replay.failed", []),
    }


def _group(turns: list[dict]) -> list[dict]:
    """Group view models by session.id, preserving first-seen (chronological) order."""
    sessions: dict[str, list[dict]] = {}
    for t in turns:
        sessions.setdefault(t.get("session.id", "default"), []).append(_view(t))
    return [{"id": sid, "turns": views} for sid, views in sessions.items()]


_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SNChat diagnostics</title><style>
*{box-sizing:border-box}
body{margin:0;font:12px/1.55 system-ui,-apple-system,sans-serif;background:#161616;color:#d8d8d8;display:flex}
#side{width:290px;min-width:290px;height:100vh;overflow:auto;background:#1b1b1b;border-right:1px solid #333;padding:12px}
.brand{font-weight:600;color:#eee}
.hint{color:#666;font-size:11px;margin:2px 0 10px}
#side h3{color:#7a7a7a;font-size:11px;text-transform:uppercase;letter-spacing:.6px;margin:16px 0 4px}
#side a{display:block;color:#bbb;text-decoration:none;padding:7px 9px;border-radius:6px;font-size:13px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#side a:hover{background:#2a2a2a;color:#fff}
#side a.active{background:#2a3a55;color:#fff}
#main{flex:1;height:100vh;overflow:auto;padding:22px 30px}
.empty{color:#666;margin-top:40px}
.turn{display:none}.turn.active{display:block}
h2.q{font-size:17px;margin:0 0 2px;color:#fff}
.meta{color:#6d6d6d;font-size:12px;margin-bottom:4px}
.meta .bad{color:#e0736d}
.pipe{display:flex;align-items:stretch;margin:16px 0 22px;flex-wrap:wrap;gap:22px}
.node{background:#242424;border:1px solid #3a3a3a;border-radius:9px;padding:9px 12px;min-width:110px;max-width:230px;position:relative}
.node+.node::before{content:"\\2192";position:absolute;left:-18px;top:50%;transform:translateY(-50%);color:#6aa3ff}
.node .k{color:#6aa3ff;font-size:10px;text-transform:uppercase;letter-spacing:.6px}
.node .v{margin-top:3px;font-size:11px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.sec{margin:16px 0}
.sec>h4{margin:0 0 7px;color:#8fddb0;font-size:11px;text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.sec .count{color:#6d6d6d;font-weight:400;text-transform:none}
.ev{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;color:#b8d8c4;
  border-left:2px solid #3a5a44;background:#191d1a;padding:5px 10px;margin:3px 0;border-radius:0 5px 5px 0;
  white-space:pre-wrap;word-break:break-word}
.doclist{max-height:32vh;overflow:auto;border:1px solid #333;border-radius:8px;background:#1a1a1a}
details{border-bottom:1px solid #262626;padding:7px 12px}
details:last-child{border-bottom:0}
summary{cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
summary .d{color:#6aa3ff}summary .t{color:#7a7a7a}
details>.body{margin-top:8px;color:#cdcdcd;white-space:pre-wrap;word-break:break-word}
.answer{max-height:42vh;overflow:auto;background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:4px 15px;word-break:break-word}
.answer p{margin:.5em 0}.answer ul,.answer ol{margin:.5em 0;padding-left:1.35em}.answer li{margin:.15em 0}
.answer strong{color:#fff}.answer a{color:#6aa3ff}
.answer code{background:#2a2a2a;padding:1px 5px;border-radius:4px;color:#e6db74}
.answer h1,.answer h2,.answer h3{font-size:1.05em;margin:.7em 0 .3em;color:#fff}
</style></head><body>
<nav id="side">
  <div class="brand">📔 SNChat diagnostics</div>
  <div class="hint">{{ total }} turn(s)</div>
  {% for s in sessions %}
  <h3>{{ s.id }}</h3>
  {% for t in s.turns %}
  <a href="#" data-t="{{ t.id }}">{{ t.q }}</a>
  {% endfor %}
  {% endfor %}
</nav>
<main id="main">
  {% if not sessions %}<div class="empty">No traces yet. Run the app or replay with SNCHAT_TRACE=1.</div>{% endif %}
  {% for s in sessions %}{% for t in s.turns %}
  <section class="turn" id="turn-{{ t.id }}">
    <h2 class="q">{{ t.q }}</h2>
    <div class="meta">{{ t.ts }} · {{ s.id }}{% if t.usage.get('gen') %} · {{ t.usage.prompt }}+{{ t.usage.gen }} tok{% endif %}{% if t.failed %} · <span class="bad">failed: {{ t.failed|join(', ') }}</span>{% endif %}</div>
    <div class="pipe">
      <div class="node"><div class="k">Question</div><div class="v" title="{{ t.q }}">{{ t.q }}</div></div>
      <div class="node"><div class="k">Extract</div><div class="v" title="{{ t.extract }}">{{ t.extract }}</div></div>
      <div class="node"><div class="k">Retrieve</div><div class="v" title="{{ t.strat }}">{{ t.strat }}</div></div>
      <div class="node"><div class="k">Plan</div><div class="v">{{ t.plan }}</div></div>
      <div class="node"><div class="k">Answer</div><div class="v">{{ 'yes' if t.answer else '—' }}</div></div>
    </div>
    {% if t.events %}<div class="sec"><h4>Routing narration</h4>{% for e in t.events %}<div class="ev">{{ e }}</div>{% endfor %}</div>{% endif %}
    <div class="sec"><h4>Retrieved <span class="count">{{ t.count }} entries</span></h4>
      <div class="doclist">
      {% for d in t.docs %}
      <details><summary><span class="d">{{ d.date }}</span> <span class="t">{{ d.tags|join(', ') or '' }}</span> — {{ (d.text or '')[:90] }}…</summary><div class="body">{{ d.text }}</div></details>
      {% else %}
      <div style="padding:10px 12px;color:#666">no entries</div>
      {% endfor %}
      </div>
    </div>
    {% if t.answer %}<div class="sec"><h4>Answer</h4><div class="answer">{{ t.answer_html|safe }}</div></div>{% endif %}
  </section>
  {% endfor %}{% endfor %}
</main>
<script>
const links=[...document.querySelectorAll('#side a')],turns=[...document.querySelectorAll('.turn')];
function show(id){turns.forEach(t=>t.classList.toggle('active',t.id==='turn-'+id));
  links.forEach(a=>a.classList.toggle('active',a.dataset.t===id));}
links.forEach(a=>a.onclick=e=>{e.preventDefault();show(a.dataset.t);});
if(links.length)show(links[0].dataset.t);
</script>
</body></html>"""

_ENV = Environment(autoescape=select_autoescape(["html"]))


def render(turns: list[dict]) -> str:
    sessions = _group(turns)
    return _ENV.from_string(_TEMPLATE).render(sessions=sessions, total=len(turns))


def main() -> None:
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else tracing.TRACE_PATH.parent / "report.html"
    )
    turns = tracing.read_turns()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(turns), encoding="utf-8")
    print(f"wrote {out} ({len(turns)} turns)")


if __name__ == "__main__":
    main()
