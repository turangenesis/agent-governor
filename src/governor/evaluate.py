"""Step 5b: run the Governor over the labeled set and score its JUDGMENT.

Headline the demo cares about is NOT "40 -> 2". It's:
    "40 -> N, with ZERO dangerous auto-sends and X% of the human's attention saved."

Confusion matrix (predicted vs ground truth):
    true=escalate, pred=escalate  -> correct escalation
    true=escalate, pred=auto      -> FALSE AUTO-SEND  (the dangerous error; target 0)
    true=auto,     pred=auto       -> correct auto-send
    true=auto,     pred=escalate   -> false escalation (wasted scarce human attention)

FIFO baseline escalates everything: 0 dangerous auto-sends, but 0% human-load saved.
The Governor's job is to keep dangerous auto-sends at 0 while saving most of the load.
"""
from __future__ import annotations

from .core import Scoreboard, evaluate_policy         # Scoreboard re-exported for back-compat
from .eval_set import build_cases, BRIEF
from .governor import recruiting_thresholds, score_signals


def _recruiting_risk_fn(policy=None):
    """The recruiting domain plugged into the reusable core: action -> named risk signals."""
    from .policy import BASELINE
    pol = policy or BASELINE
    return lambda action: score_signals(action, BRIEF, pol)


def evaluate(cases=None, policy=None) -> Scoreboard:
    """Score the recruiting Governor over the labeled set - via the reusable core engine."""
    cases = build_cases() if cases is None else cases
    pairs = [(c.action, c.ground_truth) for c in cases]
    return evaluate_policy(pairs, _recruiting_risk_fn(policy), recruiting_thresholds())


def evaluate_heldout() -> Scoreboard:
    """Score the Governor on the HELD-OUT / adversarial set (labels not used to tune the policy).

    Expected to be imperfect: it includes adversarial evasions the keyword gate misses. Reporting
    that honest gap is the point (see docs/EVAL.md); the LLM-as-judge in judge.py closes part of it.
    """
    from .eval_set_heldout import build_heldout_cases
    return evaluate(build_heldout_cases())


def fifo_baseline_stats():
    """FIFO: every action hits the human. 0 autonomy, 0 load saved, but 0 dangerous sends."""
    cases = build_cases()
    return {"human_decisions": len(cases), "autonomy_pct": 0.0,
            "human_load_saved_pct": 0.0, "false_auto_send": 0}


if __name__ == "__main__":
    sb = evaluate()
    print(sb.pretty())
    print()
    fifo = fifo_baseline_stats()
    print("=== FIFO baseline (no Governor) ===")
    print(f"  human sees ALL {fifo['human_decisions']} requests | autonomy 0% | load saved 0%")
    print(f"  Governor: human sees {sb.correct_escalation + sb.false_escalation} "
          f"| autonomy {sb.autonomy_pct:.0f}% | dangerous auto-sends {sb.false_auto_send}")
