"""Decision-trace REPLAY tests: a persisted trace, replayed with the network hard-blocked,
reproduces the same proposals — proving the autonomous loop runs deterministically offline
(the cached trace that drives a reliable demo/video with no key and no network)."""
from __future__ import annotations

import json

import pytest

from governor.loop_agent import (
    load_traces,
    replay_trace,
    run_loop_agent,
    save_traces,
)
from governor.models import Candidate, LoopResult, ProposedAction, TERMINAL_REASONS

from test_loop_agent import ScriptedLLM, TERRITORY, _propose_flow, _search_fn, _turn


def _key(pa: ProposedAction):
    """The identity of a proposal that must survive a round-trip + replay."""
    return (pa.candidate.name, pa.candidate.email, pa.subject, pa.body,
            round(pa.draft_confidence, 6), tuple(pa.concerns))


def _record() -> LoopResult:
    """Produce a genuine loop run (mock LLM) whose trace we then persist and replay."""
    return run_loop_agent(TERRITORY, ScriptedLLM(_propose_flow()), use_llm=False, search_fn=_search_fn)


# --- The trace round-trips through JSON (persist) and rebuilds into a LoopResult ---
def test_loop_result_json_round_trip():
    original = _record()
    restored = LoopResult.from_dict(json.loads(json.dumps(original.to_dict())))

    assert [s.to_dict() for s in restored.steps] == [s.to_dict() for s in original.steps]
    assert [_key(p) for p in restored.proposals] == [_key(p) for p in original.proposals]
    assert restored.terminal_reason == original.terminal_reason
    assert restored.terminal_reason in TERMINAL_REASONS


# --- Replaying a PERSISTED trace with the network HARD-BLOCKED reproduces the proposals ---
def test_replay_reproduces_proposals_offline(monkeypatch):
    original = _record()
    persisted = json.loads(json.dumps(original.to_dict()))   # exactly what a cache file holds

    # Hard-block the network: any socket/HTTP attempt during replay must fail the test.
    import socket

    def _no_network(*a, **k):
        raise AssertionError("replay must not touch the network")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)

    # Replay drives the SAME loop from the recorded tool calls over an offline search.
    replayed = replay_trace(persisted, TERRITORY, search_fn=_search_fn)

    assert [_key(p) for p in replayed.proposals] == [_key(p) for p in original.proposals]
    assert len(replayed.proposals) == len(original.proposals) == 4
    # Every replayed item is still a valid ProposedAction over a real Candidate.
    for pa in replayed.proposals:
        assert isinstance(pa, ProposedAction) and isinstance(pa.candidate, Candidate)


# --- save_traces/load_traces persist and restore replayable traces (the cache-replay pattern) ---
def test_save_and_load_traces_then_replay(tmp_path):
    original = _record()
    path = tmp_path / "demo_traces.json"
    save_traces([original], str(path))

    restored = load_traces(str(path))
    assert len(restored) == 1
    replayed = replay_trace(restored[0], TERRITORY, search_fn=_search_fn)
    assert [_key(p) for p in replayed.proposals] == [_key(p) for p in original.proposals]


# --- A budget-terminated trace still replays to the same proposals (tokens don't gate replay) ---
def test_replay_of_guarded_trace_matches_proposals():
    from governor.loop_agent import SourcingLoop

    turns = ScriptedLLM([
        _turn([("search_github", {"query": "q"})], tokens=100),
        _turn([("draft_email", {"candidate_name": "Ann Lee"})], tokens=100),
        _turn([("propose", {"candidate_name": "Ann Lee"})], tokens=100),
        _turn([("refine_search", {"query": "expensive"})], tokens=50_000),
    ])
    original = SourcingLoop(TERRITORY, turns, use_llm=False, search_fn=_search_fn,
                            token_budget=20_000).run()
    assert original.terminal_reason == "budget"

    replayed = replay_trace(original.to_dict(), TERRITORY, search_fn=_search_fn)
    assert [_key(p) for p in replayed.proposals] == [_key(p) for p in original.proposals]
