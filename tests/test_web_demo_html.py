"""The static web demo (docs/index.html) SURFACES the autonomous-loop decision trace.

Iteration 4 persisted the per-agent trace into docs/demo_cache.json under the 'loop' key, but the
GitHub Pages page (the primary offline demo/video surface) only rendered the fixed flow. These tests
pin the contract that index.html now renders the loop trace step by step AND that every tool name /
terminal reason the shipped cache can emit is actually handled by the page's renderer - so a later
cache regen that introduces a new tool or terminal reason fails here instead of rendering a raw slug.
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "docs", "index.html")
CACHE = os.path.join(ROOT, "docs", "demo_cache.json")


def _html() -> str:
    return open(HTML, encoding="utf-8").read()


# --- the page has a loop-mode renderer wired to the cached trace ---
def test_index_html_renders_loop_trace():
    html = _html()
    # a mode toggle distinct from the fixed flow, and a dispatcher that plays the loop trace
    assert 'id="modetog"' in html
    assert 'data-mode="loop"' in html
    assert "function playLoop(" in html
    assert "data.loop" in html                       # reads the persisted per-agent traces
    # every step field the cache carries is read by the renderer
    for field in ("s.tool", "s.tool_input", "s.observation", "s.decision", "s.reason", "s.index"):
        assert field in html, f"renderer does not read {field}"
    # the terminal reason is surfaced when the loop stops
    assert "terminal_reason" in html


# --- every tool name in the shipped cache is handled by the page (no raw-slug fallthrough) ---
def test_html_handles_every_cached_tool_and_terminal_reason():
    data = json.load(open(CACHE, encoding="utf-8"))
    html = _html()

    # parse the JS lookup tables the renderer uses
    tool_keys = set(re.findall(r"(\w+):\[", html.split("const TOOLMETA=", 1)[1].split("};", 1)[0]))
    term_keys = set(re.findall(r'"([^"]+)":', html.split("const TERMLABEL=", 1)[1].split("};", 1)[0]))

    cache_tools, cache_terms = set(), set()
    for agent in data["loop"]:
        cache_terms.add(agent["terminal_reason"])
        for step in agent["steps"]:
            cache_tools.add(step["tool"])

    missing_tools = cache_tools - tool_keys
    missing_terms = cache_terms - term_keys
    assert not missing_tools, f"index.html TOOLMETA missing tools: {missing_tools}"
    assert not missing_terms, f"index.html TERMLABEL missing terminal reasons: {missing_terms}"


# --- the loop view stays a client of the shared cache: no re-ported trace data baked into the HTML ---
def test_loop_view_reads_cache_not_inlined_trace():
    html = _html()
    # the loop board is built from CACHE.loop, not a hardcoded copy of the steps
    assert "CACHE.loop" in html
    # sanity: the cache genuinely carries a non-trivial trace for the page to render
    data = json.load(open(CACHE, encoding="utf-8"))
    assert all(len(a["steps"]) >= 4 for a in data["loop"])
