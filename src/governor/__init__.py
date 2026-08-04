"""The Governor — an oversight & eval layer for autonomous recruiting agents.

Public surface used by the CLI, the Streamlit app, and the web-cache generator.
The deterministic core (governor scoring, eval labels) lives in `governor`,
`eval_set`, and `evaluate`; nothing here adds an LLM.
"""
from __future__ import annotations

__all__ = ["govern", "evaluate", "build_cases", "search_by_query"]

__version__ = "0.1.0"


def __getattr__(name):  # lazy re-exports keep import cost (and optional deps) low
    if name == "govern":
        from .governor import govern
        return govern
    if name == "evaluate":
        from .evaluate import evaluate
        return evaluate
    if name == "build_cases":
        from .eval_set import build_cases
        return build_cases
    if name == "search_by_query":
        from .discover import search_by_query
        return search_by_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
