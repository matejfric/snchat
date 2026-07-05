"""diagnostics_report.render() — the HTML viewer builds and escapes untrusted text.

The diary is Czech and full text is user data, so the load-bearing check is that
render() emits the turn's content AND that HTML-significant characters are escaped
(Jinja2 autoescape), not injected raw into the page.
"""

import diagnostics_report as dr

SAMPLE = [
    {
        "id": "abc12345",
        "ts": "2026-07-05T17:00:00",
        "session.id": "demo",
        "input.value": "what did I do skiing in January?",
        "snchat.extraction": {"tags": ["lyže"], "month": 1, "breadth": "all"},
        "snchat.retrieval.count": 2,
        "events": ["diary_query_router: retrieve: fetch-all 2 entries (count=2)"],
        "snchat.retrieval.docs": [
            # <b> and & exercise autoescape:
            {"date": "2025-01-04", "tags": ["lyže"], "text": "Lyže <b>x</b> & pád"},
            {"date": "2025-01-11", "tags": ["skialp"], "text": "Skialp na Sněžku"},
        ],
        "output.value": (
            "You skied twice:\n\n"
            "- **2025-01-04** ledová\n"
            "- 2025-01-11 Sněžka\n\n"
            "<script>alert(1)</script> ![x](http://evil/x.png)"
        ),
    }
]


def test_render_includes_content_and_nav():
    html = dr.render(SAMPLE)
    assert "what did I do skiing in January?" in html
    assert "Skialp na Sněžku" in html  # Czech text passes through
    assert "tags=lyže" in html and "breadth=all" in html  # compact extract label
    assert "retrieve: fetch-all 2 entries" in html  # narration, logger prefix stripped
    assert "diary_query_router:" not in html
    assert "querySelectorAll" in html  # JS nav present, single self-contained file
    # Self-contained = nothing AUTO-loads externally (img/script/link src). A link
    # href in answer content is fine — it only navigates on click, never on open.
    assert "src=" not in html and "<link" not in html


def test_untrusted_text_is_escaped():
    html = dr.render(SAMPLE)
    # The <b> and & in the diary text must be escaped, not live markup.
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert "<b>x</b>" not in html


def test_answer_rendered_as_markdown_safely():
    html = dr.render(SAMPLE)
    assert "<li>" in html and "<strong>2025-01-04</strong>" in html  # markdown rendered
    assert "&lt;script&gt;" in html and "<script>alert(1)" not in html  # HTML escaped
    assert "<img" not in html  # image disabled — no offline auto-fetch


def test_empty_traces_render_without_crashing():
    html = dr.render([])
    assert "No traces yet" in html


def test_zero_result_turn_renders_no_entries():
    turn = [{"id": "x", "input.value": "q", "snchat.retrieval.count": 0}]
    html = dr.render(turn)
    assert "no entries" in html
