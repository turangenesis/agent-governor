"""Generate docs/demo_cache.json — the cached snapshot the static web demo loads instantly.

Runs the 5 territory agents over their cached GitHub results, scores each proposal with the
real Governor, and writes everything the page needs (candidates, drafts, decisions, reasons,
scoreboard) as JSON. Zero cost to run again; the page ships this so visitors pay nothing.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from governor.discover import search_by_query
from governor.eval_set import BRIEF
from governor.evaluate import evaluate
from governor.governor import govern
from governor.loop_agent import canonical_trace
from governor.models import ProposedAction
from governor.sourcing_agent import _draft
from governor.territories import TERRITORIES

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "demo_cache.json")


def _loop_traces() -> list[dict]:
    """Per-territory autonomous-loop decision traces, generated deterministically offline.

    Each entry mirrors the persisted LoopResult (ordered steps + terminal reason) plus the
    Governor's decision on every proposal, so the static demo can render the agent DECIDING
    step by step and show the gate — the same trace replays with no key and no network.
    """
    out = []
    for terr in TERRITORIES:
        res = canonical_trace(terr, search_fn=search_by_query)
        proposals = []
        for pa in res.proposals:
            gd = govern(pa, BRIEF)
            proposals.append({
                "name": pa.candidate.name, "company": pa.candidate.current_company,
                "seniority": pa.candidate.seniority, "subject": pa.subject,
                "decision": gd.decision.value, "risk": round(gd.risk_score, 2),
                "reasons": gd.reasons[:3],
            })
        out.append({
            "key": terr["key"], "goal": terr["goal"], "flavor": terr["flavor"],
            "terminal_reason": res.terminal_reason,
            "iterations": res.iterations, "tokens_used": res.tokens_used,
            "steps": [s.to_dict() for s in res.steps],
            "proposals": proposals,
            # Full candidate records so `--replay` rebuilds this trace with no cache/network.
            # (The static HTML ignores this key; it renders `proposals` above.)
            "leads": [asdict(c) for c in res.leads],
        })
    return out


def _eval_summary() -> dict:
    """Eval depth for the web demo: labeled + held-out scoreboards, and the LLM-as-judge result -
    including the adversarial evasions the keyword gate auto-sent but the judge flagged. Offline."""
    from governor.eval_set import build_cases
    from governor.eval_set_heldout import build_heldout_cases
    from governor.evaluate import evaluate, evaluate_heldout
    from governor.judge import StubJudge, judge_report
    from governor.models import Decision

    pri, ho = evaluate(), evaluate_heldout()
    judge = StubJudge()
    caught = []   # held-out drafts the gate AUTO-SENT but the judge catches = the value-add
    for c in build_heldout_cases():
        gate = govern(c.action, BRIEF, human_queue_depth=0)
        v = judge.judge(c.action)
        if gate.decision == Decision.AUTO_SEND and not v.passed:
            caught.append({"name": c.action.candidate.name,
                           "reason": v.reasons[0] if v.reasons else "flagged"})
    actions = [x.action for x in build_cases()] + [x.action for x in build_heldout_cases()]
    rep = judge_report(actions, judge)
    return {
        "primary": {"recall": round(pri.escalation_recall, 2),
                    "precision": round(pri.escalation_precision, 2),
                    "dangerous": pri.false_auto_send, "autonomy": round(pri.autonomy_pct)},
        "heldout": {"cases": ho.total, "recall": round(ho.escalation_recall, 2),
                    "precision": round(ho.escalation_precision, 2), "dangerous": ho.false_auto_send},
        "judge": {"total": rep["total"], "passed": rep["passed"], "pass_rate": rep["pass_rate"],
                  "avg_scores": rep["avg_scores"], "caught_evasions": caught},
    }


def build() -> dict:
    agents = []
    for terr in TERRITORIES:
        cands = search_by_query(terr["query"], limit=4)  # served from cache; no network
        rows = []
        for c in cands:
            subj, body, conf, concerns = _draft(c, None, use_llm=False, flavor=terr["flavor"])
            pa = ProposedAction("agent", c, subj, body, conf, concerns, seq=0)
            gd = govern(pa, BRIEF)
            rows.append({
                "name": c.name, "company": c.current_company, "title": c.current_title,
                "seniority": c.seniority, "match": c.match_confidence,
                "subject": subj, "body": body,
                "decision": gd.decision.value, "risk": round(gd.risk_score, 2),
                "reasons": gd.reasons[:3],
            })
        agents.append({"key": terr["key"], "goal": terr["goal"], "flavor": terr["flavor"],
                       "note": terr["note"], "candidates": rows})

    sb = evaluate()
    return {
        "brief": {"role": BRIEF.role, "must_haves": BRIEF.must_haves,
                  "hiring_company": BRIEF.hiring_company, "competitors": list(BRIEF.competitors)},
        "agents": agents,
        "loop": _loop_traces(),
        "eval": _eval_summary(),
        "scoreboard": {
            "recall": round(sb.escalation_recall, 2),
            "precision": round(sb.escalation_precision, 2),
            "dangerous": sb.false_auto_send,
            "human_load_saved_pct": round(sb.human_load_saved_pct),
        },
    }


if __name__ == "__main__":
    data = build()
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    n = sum(len(a["candidates"]) for a in data["agents"])
    print(f"wrote {OUT}: {len(data['agents'])} agents, {n} candidates")
