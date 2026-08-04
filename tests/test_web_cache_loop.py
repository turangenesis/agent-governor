"""The web cache mirror carries the autonomous-loop DECISION TRACE, and that persisted trace
replays deterministically with no key and no network — this is what drives the offline demo/video.

Covers the objective's "persisted to the existing cached-replay pattern (mirror docs/demo_cache.json
+ scripts/gen_web_cache.py)" and "replaying a persisted trace with the network hard-blocked
reproduces the same proposals".
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

from governor.discover import _cache_path, save
from governor.eval_set import BRIEF
from governor.governor import govern
from governor.models import Candidate, Decision, LoopResult, ProposedAction, TERMINAL_REASONS
from governor.loop_agent import replay_trace
from governor.territories import TERRITORIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "docs", "demo_cache.json")


def _fixture_candidates(flavor: str) -> list[Candidate]:
    """4 deterministic candidates per flavor, crafted so build()/canonical_trace produce a full
    4-proposal arc offline AND exercise real risk: competitor company + exec seniority (pushy/
    comp_forward risk lives in the scripted draft copy, so any candidate triggers it)."""
    if flavor == "competitor":
        company, seniority = "OpenAI", "senior"          # a real competitor in BRIEF.competitors
    elif flavor == "exec":
        company, seniority = "Acme Corp", "exec"         # exec seniority -> escalate
    else:
        company, seniority = "Independent / open source", "senior"
    return [
        Candidate(name=f"{flavor.title()} Lead {i}", current_company=company,
                  current_title="Backend Engineer", seniority=seniority,
                  email=f"{flavor}{i}@users.noreply.github.com",
                  matched_role=BRIEF.role, match_confidence=0.8)
        for i in range(1, 5)
    ]


@pytest.fixture(autouse=True)
def _seed_territory_caches():
    """Seed a deterministic discovery cache for each territory query so these tests are HERMETIC
    (offline + order-independent) instead of depending on a live search a prior test happened to
    run. Any real cache_*.json is backed up and restored, so a developer's live caches are safe."""
    written, backups = [], []
    for terr in TERRITORIES:
        path = _cache_path(terr["query"])
        if os.path.exists(path):
            bak = path + ".pretest-bak"
            os.replace(path, bak)
            backups.append((path, bak))
        save(_fixture_candidates(terr["flavor"]), path)
        written.append(path)
    try:
        yield
    finally:
        for p in written:
            if os.path.exists(p):
                os.remove(p)
        for path, bak in backups:
            os.replace(bak, path)


def _load_gen_module():
    """Import scripts/gen_web_cache.py (not on the package path) by file location."""
    path = os.path.join(ROOT, "scripts", "gen_web_cache.py")
    spec = importlib.util.spec_from_file_location("gen_web_cache", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(pa: ProposedAction):
    return (pa.candidate.name, pa.candidate.email, pa.subject, pa.body,
            round(pa.draft_confidence, 6), tuple(pa.concerns))


# --- build() emits a well-formed loop trace per territory (tool/input/observation/decision) ---
def test_build_emits_wellformed_loop_traces():
    data = _load_gen_module().build()
    assert "loop" in data
    loop = data["loop"]
    assert len(loop) == len(TERRITORIES)
    for agent in loop:
        assert agent["terminal_reason"] in TERMINAL_REASONS
        assert agent["steps"], f"{agent['key']} has no trace steps"
        for i, s in enumerate(agent["steps"]):
            assert s["index"] == i                       # ordered
            assert s["tool"] and isinstance(s["tool_input"], dict)
            assert s["observation"]
            assert s["decision"] in ("continue", "stop")
            assert s["reason"]
        last = agent["steps"][-1]
        assert last["decision"] == "stop"
        assert last["reason"] == agent["terminal_reason"]  # ends in the explicit terminal reason
        assert len(agent["proposals"]) == 4


# --- the shipped docs/demo_cache.json actually contains those traces (regen kept in sync) ---
def test_shipped_cache_has_loop_traces():
    data = json.load(open(CACHE))
    assert "loop" in data and len(data["loop"]) == len(TERRITORIES)
    for agent in data["loop"]:
        assert agent["steps"] and agent["terminal_reason"] in TERMINAL_REASONS


# --- the persisted trace REPLAYS offline (network hard-blocked) to the same 4 proposals ---
def test_shipped_trace_replays_offline(monkeypatch):
    data = json.load(open(CACHE))
    by_key = {t["key"]: t for t in TERRITORIES}

    import socket

    def _no_network(*a, **k):
        raise AssertionError("replay must not touch the network")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)

    for agent in data["loop"]:
        terr = by_key[agent["key"]]
        # Pass the FULL cached trace (incl. its persisted `leads`): replay serves the recorded
        # candidates from the trace itself, so it never touches the discovery cache or network.
        assert agent.get("leads"), f"{agent['key']} cache has no persisted leads to replay offline"
        replayed = replay_trace(agent, terr)
        assert replayed.terminal_reason in TERMINAL_REASONS
        assert len(replayed.proposals) == len(agent["proposals"]) == 4
        # the replayed proposal identities line up with what the cache recorded, in order
        assert [p.candidate.name for p in replayed.proposals] == \
               [p["name"] for p in agent["proposals"]]


# --- Governor parity on the persisted loop proposals: no genuinely-risky draft is auto-sent ---
def test_cached_loop_proposals_zero_dangerous():
    from governor.governor import PUSHY_TERMS, SENSITIVE_TERMS, _is_competitor
    from governor.loop_agent import canonical_trace

    def _is_risky(pa: ProposedAction) -> bool:
        text = f"{pa.subject} {pa.body}".lower()
        if any(t in text for t in PUSHY_TERMS) or any(t in text for t in SENSITIVE_TERMS):
            return True
        if _is_competitor(pa.candidate.current_company, BRIEF.competitors):
            return True
        return pa.candidate.seniority == "exec"

    dangerous, saw_risky = 0, False
    for terr in TERRITORIES:
        res: LoopResult = canonical_trace(terr)
        for pa in res.proposals:
            gd = govern(pa, BRIEF)
            if _is_risky(pa):
                saw_risky = True
                if gd.decision == Decision.AUTO_SEND:
                    dangerous += 1
    assert saw_risky, "fixture should exercise at least one genuinely risky draft"
    assert dangerous == 0
