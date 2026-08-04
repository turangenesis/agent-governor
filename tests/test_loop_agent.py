"""Loop-engine tests: with a MOCK LLM emitting scripted tool-calls, the bounded tool-use loop
invokes tools, collects proposals, records a well-formed decision trace, and always terminates
within its guardrails. No network, no key, no real model."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from governor.loop_agent import SourcingLoop, run_loop_agent
from governor.models import ProposedAction, TERMINAL_REASONS
from governor.models import Candidate

TERRITORY = {
    "key": "clean-py", "flavor": "clean",
    "goal": "Python backend engineers", "note": "",
    "query": "language:python type:user", "broaden": "language:python type:user followers:>50",
}


def _cand(name: str) -> Candidate:
    return Candidate(name, "Stripe", "Staff Engineer", "senior",
                     f"{name.split()[0].lower()}@example.com", "Senior Backend Engineer", 0.85)

# A deterministic, offline search tool: four stable leads regardless of query.
LEADS = [_cand(n) for n in ("Ann Lee", "Bo Ng", "Cy Roy", "Di Fox")]


def _search_fn(query, limit=4, **kw):
    return LEADS[:limit]


def _turn(calls, tokens=100):
    """Build one scripted assistant turn: `calls` = list of (tool_name, args) tuples."""
    tool_calls = [
        {"name": name, "args": args, "id": f"call-{i}", "type": "tool_call"}
        for i, (name, args) in enumerate(calls)
    ]
    return AIMessage(content="", tool_calls=tool_calls,
                     usage_metadata={"input_tokens": tokens, "output_tokens": tokens,
                                     "total_tokens": tokens * 2})


class ScriptedLLM:
    """Mock tool-calling LLM: replays a fixed list of scripted turns, one per invoke()."""

    def __init__(self, turns):
        self._turns = list(turns)
        self._i = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        turn = self._turns[min(self._i, len(self._turns) - 1)]
        self._i += 1
        return turn


def _propose_flow():
    """A full search -> per-candidate draft/critique/propose script for 4 candidates."""
    turns = [_turn([("search_github", {"query": "language:python type:user"})])]
    for c in LEADS:
        turns.append(_turn([("draft_email", {"candidate_name": c.name})]))
        turns.append(_turn([("critique_draft", {"candidate_name": c.name})]))
        turns.append(_turn([("propose", {"candidate_name": c.name})]))
    # extra turns that should never be reached once max_candidates is hit
    turns.append(_turn([("search_github", {"query": "more"})]))
    return turns


# --- Loop mechanics: invoke tools, collect proposals, STOP at max_candidates ---
def test_loop_collects_and_stops_at_max_candidates():
    llm = ScriptedLLM(_propose_flow())
    res = run_loop_agent(TERRITORY, llm, use_llm=False, search_fn=_search_fn)

    assert len(res.proposals) == 4
    assert res.terminal_reason == "enough-candidates"
    tools_used = {s.tool for s in res.steps}
    assert {"search_github", "draft_email", "critique_draft", "propose"} <= tools_used
    # It stopped as soon as it had enough — the trailing extra search never ran.
    assert res.steps[-1].tool == "propose"


# --- Proposals: every emitted item is a valid ProposedAction ---
def test_every_proposal_is_valid():
    res = run_loop_agent(TERRITORY, ScriptedLLM(_propose_flow()), use_llm=False, search_fn=_search_fn)
    for pa in res.proposals:
        assert isinstance(pa, ProposedAction)
        assert isinstance(pa.candidate, Candidate)
        assert pa.subject and pa.body
        assert 0.0 <= pa.draft_confidence <= 1.0


# --- Guardrail: force-terminate at max_iterations, still returns valid proposals ---
def test_guardrail_max_iterations():
    # Propose one, then loop forever on refine_search (never enough candidates).
    turns = [
        _turn([("search_github", {"query": "q"})]),
        _turn([("draft_email", {"candidate_name": "Ann Lee"})]),
        _turn([("propose", {"candidate_name": "Ann Lee"})]),
        _turn([("refine_search", {"query": "broader"})]),  # replayed forever
    ]
    loop = SourcingLoop(TERRITORY, ScriptedLLM(turns), use_llm=False,
                        search_fn=_search_fn, max_iterations=6)
    res = loop.run()

    assert res.terminal_reason == "max-iterations"
    assert res.iterations <= 6
    assert 1 <= len(res.proposals) < 4
    assert all(isinstance(p, ProposedAction) for p in res.proposals)


# --- Guardrail: force-terminate at the token budget, still returns valid proposals ---
def test_guardrail_token_budget():
    turns = [
        _turn([("search_github", {"query": "q"})], tokens=100),
        _turn([("propose", {"candidate_name": "Ann Lee"})], tokens=100),
        _turn([("refine_search", {"query": "expensive"})], tokens=50_000),  # blows the budget
    ]
    loop = SourcingLoop(TERRITORY, ScriptedLLM(turns), use_llm=False,
                        search_fn=_search_fn, token_budget=20_000)
    res = loop.run()

    assert res.terminal_reason == "budget"
    assert res.tokens_used >= 20_000
    assert len(res.proposals) == 1
    assert all(isinstance(p, ProposedAction) for p in res.proposals)


# --- Decision trace is well-formed: ordered, typed, ends in an explicit terminal reason ---
def test_decision_trace_well_formed():
    res = run_loop_agent(TERRITORY, ScriptedLLM(_propose_flow()), use_llm=False, search_fn=_search_fn)

    assert res.steps, "trace must not be empty"
    for i, step in enumerate(res.steps):
        assert step.index == i                     # ordered
        assert isinstance(step.tool, str) and step.tool
        assert isinstance(step.tool_input, dict)   # the input it passed
        assert isinstance(step.observation, str) and step.observation
        assert step.decision in ("continue", "stop")
        assert isinstance(step.reason, str) and step.reason

    # Exactly the last step stops, and it carries an explicit terminal reason.
    assert res.steps[-1].decision == "stop"
    assert res.steps[-1].reason in TERMINAL_REASONS
    assert res.terminal_reason == res.steps[-1].reason
    assert all(s.decision == "continue" for s in res.steps[:-1])


# --- Trace round-trips through to_dict for cache persistence ---
def test_trace_serializes():
    res = run_loop_agent(TERRITORY, ScriptedLLM(_propose_flow()), use_llm=False, search_fn=_search_fn)
    d = res.to_dict()
    assert d["terminal_reason"] == "enough-candidates"
    assert len(d["steps"]) == len(res.steps)
    assert all({"index", "tool", "tool_input", "observation", "decision", "reason"} <= set(s)
               for s in d["steps"])
    assert len(d["proposals"]) == 4
