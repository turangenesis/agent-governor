"""Loop-mode wiring in the Orchestrator + the `governor run` CLI.

Covers the acceptance path: source="agents" runs each territory as an autonomous LLM tool-use
loop that emits a decision trace, whose proposals are gated by the UNCHANGED Governor with zero
dangerous auto-sends; with no key the agents fall back to the deterministic flow and still run
offline. A stateless policy mock stands in for a real tool-calling LLM (safe to share across the
5 concurrent agent threads, exactly like the live model).
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from governor.eval_set import BRIEF
from governor.governor import PUSHY_TERMS, SENSITIVE_TERMS, _is_competitor, govern
from governor.loop_agent import run_loop_agent
from governor.models import Candidate, Decision, ProposedAction, TERMINAL_REASONS
from governor.runner import Orchestrator, run_headless
from governor.territories import TERRITORIES


# --- offline search: honest risk keyed off the territory query (no network) ---
def _offline_search(query, limit=4, **kw):
    q = (query or "").lower()
    if "anthropic" in q or "openai" in q:            # competitor territory -> real competitor
        company, title, sen = "Anthropic", "Staff Research Engineer", "senior"
    elif "cto" in q or "vp" in q:                    # exec territory -> real exec
        company, title, sen = "Acme Startup", "CTO", "exec"
    else:                                            # clean IC baseline
        company, title, sen = "Stripe", "Staff Engineer", "senior"
    names = ("Ann Lee", "Bo Ng", "Cy Roy", "Di Fox")
    return [Candidate(n, company, title, sen,
                      f"{n.split()[0].lower()}@example.com", "Senior Backend Engineer", 0.85)
            for n in names][:limit]


class PolicyLLM:
    """Stateless mock tool-calling LLM: it picks the next tool purely from the conversation, so a
    single instance is safe to share across concurrent agent threads (like the real ChatAnthropic).

    Policy: search once, then for each lead in turn draft -> critique -> propose.
    """

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        # run_once() (used inside _draft for neutral flavors) calls invoke() with a plain string;
        # only the tool-use loop passes a message list. Ignore non-loop calls.
        obs = [m.content for m in messages if isinstance(m, ToolMessage)] \
            if isinstance(messages, list) else []

        def emit(name, args):
            return AIMessage(content="", tool_calls=[
                {"name": name, "args": args, "id": "m", "type": "tool_call"}],
                usage_metadata={"input_tokens": 50, "output_tokens": 50, "total_tokens": 100})

        if not any(o.startswith(("search", "refined")) for o in obs):
            return emit("search_github", {})          # empty query -> loop uses territory default

        leads: list[str] = []
        for o in obs:
            if "new leads (" in o:
                inside = o.split("new leads (")[1].split(")")[0]
                if inside != "none":
                    for nm in inside.split(", "):
                        if nm not in leads:
                            leads.append(nm)

        proposed = sum(1 for o in obs if "proposed" in o and "Governor" in o)
        if proposed >= len(leads):
            return emit("search_github", {})          # nothing left; guardrail will stop us
        cand = leads[proposed]
        if not any(o.startswith(f"drafted for {cand}") for o in obs):
            return emit("draft_email", {"candidate_name": cand})
        if not any(o.startswith(f"critique of {cand}") for o in obs):
            return emit("critique_draft", {"candidate_name": cand})
        return emit("propose", {"candidate_name": cand})


def _is_risky(pa: ProposedAction) -> bool:
    text = f"{pa.subject} {pa.body}".lower()
    if any(t in text for t in PUSHY_TERMS) or any(t in text for t in SENSITIVE_TERMS):
        return True
    if _is_competitor(pa.candidate.current_company, BRIEF.competitors):
        return True
    return pa.candidate.seniority == "exec"


# --- Governor parity: loop-produced proposals, gated by the UNCHANGED Governor, 0 dangerous ---
def test_loop_proposals_governed_zero_dangerous():
    dangerous = 0
    total = 0
    saw_risky = False
    for terr in TERRITORIES:
        res = run_loop_agent(terr, PolicyLLM(), use_llm=False, search_fn=_offline_search)
        assert res.terminal_reason in TERMINAL_REASONS
        assert res.proposals, f"{terr['key']} produced no proposals"
        for pa in res.proposals:
            assert isinstance(pa, ProposedAction)
            total += 1
            gd = govern(pa, BRIEF, human_queue_depth=0)
            if _is_risky(pa):
                saw_risky = True
                # a risky draft must NEVER be auto-sent — that would be a dangerous auto-send
                assert gd.decision != Decision.AUTO_SEND, f"dangerous auto-send: {pa.subject!r}"
                if gd.decision == Decision.AUTO_SEND:
                    dangerous += 1
    assert total == len(TERRITORIES) * 4
    assert saw_risky, "test fixture should exercise at least one risky territory"
    assert dangerous == 0


# --- Orchestrator end-to-end: agents run as loops, capture traces, Governor gates them ---
def test_orchestrator_loop_mode_captures_traces_and_gates():
    snap = run_headless("governor", source="agents", agent_mode="loop",
                        use_llm=True, search_fn=_offline_search, llm=PolicyLLM())

    assert snap["agent_mode"] == "loop"
    traces = snap["traces"]
    assert len(traces) == len(TERRITORIES)
    for agent_id, tr in traces.items():
        assert tr.proposals, f"{agent_id} loop produced no proposals"
        assert tr.terminal_reason in TERMINAL_REASONS
        assert tr.steps and tr.steps[-1].decision == "stop"

    assert snap["processed"] == snap["total"] == len(TERRITORIES) * 4
    # No risky draft was auto-sent (the parity guarantee, checked on live governed output).
    for item in snap["sent"]:
        if item.decision == "auto_send":
            assert not _is_risky(item.action), f"dangerous auto-send: {item.action.subject!r}"


# --- Offline fallback: no key -> agents run the deterministic FLOW, still produce proposals ---
def test_orchestrator_loop_mode_falls_back_to_flow_with_no_key():
    snap = run_headless("governor", source="agents", agent_mode="loop",
                        use_llm=False, search_fn=_offline_search, llm=None)

    assert snap["agent_mode"] == "loop"
    assert not snap["traces"], "no LLM -> no loop traces (deterministic flow instead)"
    assert snap["processed"] == snap["total"] == len(TERRITORIES) * 4
    for item in snap["sent"]:
        if item.decision == "auto_send":
            assert not _is_risky(item.action)


def test_flow_mode_default_still_works():
    orc = Orchestrator(mode="governor", source="agents", agent_mode="flow",
                       search_fn=_offline_search, step_delay=0.0)
    assert orc.agent_mode == "flow"
    snap = run_headless("governor", source="agents", agent_mode="flow",
                        search_fn=_offline_search)
    assert not snap["traces"]
    assert snap["processed"] == len(TERRITORIES) * 4


# --- Cached REPLAY through the Orchestrator: no key, no network, reproduces cached proposals ---
def test_orchestrator_replay_reproduces_cached_proposals(monkeypatch):
    import json
    import os
    import socket

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cached = json.load(open(os.path.join(root, "docs", "demo_cache.json")))["loop"]
    replay_traces = {a["key"]: a for a in cached}

    def _no_network(*a, **k):
        raise AssertionError("cached replay must not touch the network")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)

    # No LLM, no key: the loop is driven entirely by the recorded traces.
    snap = run_headless("governor", source="agents", agent_mode="loop",
                        use_llm=False, replay_traces=replay_traces)

    assert snap["agent_mode"] == "loop"
    assert len(snap["traces"]) == len(TERRITORIES)
    by_terr = {tr.territory_key: tr for tr in snap["traces"].values()}
    for a in cached:
        tr = by_terr[a["key"]]
        assert tr.terminal_reason in TERMINAL_REASONS
        # the replayed proposals reproduce the cached proposals, in order
        assert [p.candidate.name for p in tr.proposals] == [p["name"] for p in a["proposals"]]

    assert snap["processed"] == snap["total"] == len(TERRITORIES) * 4
    for item in snap["sent"]:
        if item.decision == "auto_send":
            assert not _is_risky(item.action), f"dangerous auto-send: {item.action.subject!r}"
