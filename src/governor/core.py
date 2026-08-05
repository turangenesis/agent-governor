"""The reusable oversight + eval CORE - domain-free.

This is the layer, separated from the recruiting demo. Bring your own agent:

    gov = Governor(risk_fn=my_risk_fn)          # risk_fn: action -> {signal_name: weight}
    decision = gov.decide(my_action, human_queue_depth=q)   # AUTO_SEND / ESCALATE / HOLD
    scoreboard = evaluate_policy(my_labeled_cases, my_risk_fn)   # measured correctness

Nothing here knows about recruiting, candidates, or emails - it only aggregates named risk
signals, decides against thresholds (with load-shedding), and scores decisions against ground
truth. The recruiting policy (governor.py) and its eval (evaluate.py) are just one consumer;
examples/support_desk.py is a second, unrelated one that proves the core is generic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import Decision, GovernorDecision

# A risk function maps an arbitrary action to the named risk signals that fired -> their weight.
RiskFn = Callable[[object], "dict[str, float]"]


@dataclass(frozen=True)
class Thresholds:
    """The knobs an oversight operator tunes for their own risk tolerance."""
    escalate: float = 0.50       # risk >= this -> a human must look
    auto: float = 0.25           # risk <= this -> safe to act autonomously
    max_human_queue: int = 5     # human is saturated beyond this many pending escalations
    hard_escalate: float = 0.80  # above this, ALWAYS escalate - never load-shed a real risk


DEFAULT_THRESHOLDS = Thresholds()


def decide(signals: dict[str, float], human_queue_depth: int = 0,
           thresholds: Thresholds = DEFAULT_THRESHOLDS) -> GovernorDecision:
    """The judgment: aggregate signals -> AUTO_SEND / ESCALATE / HOLD. Pure, deterministic."""
    score = min(1.0, sum(signals.values()))
    reasons = [f"{k} (+{v:.2f})" for k, v in sorted(signals.items(), key=lambda kv: -kv[1])]

    # A genuinely risky action is never load-shed away.
    if score >= thresholds.hard_escalate:
        return GovernorDecision(Decision.ESCALATE, score, ["HARD-ESCALATE"] + reasons, signals)
    if score <= thresholds.auto:
        return GovernorDecision(Decision.AUTO_SEND, score, reasons or ["clearly low risk"], signals)
    if score >= thresholds.escalate:
        # Needs a human - but if the human is saturated, park it instead of piling on.
        if human_queue_depth > thresholds.max_human_queue:
            return GovernorDecision(
                Decision.HOLD, score,
                [f"load-shed: human queue {human_queue_depth} > {thresholds.max_human_queue}"] + reasons,
                signals)
        return GovernorDecision(Decision.ESCALATE, score, reasons, signals)
    # Middle band (auto < score < escalate): lean safe, auto-send.
    return GovernorDecision(Decision.AUTO_SEND, score,
                            ["mid-band, below escalate threshold"] + reasons, signals)


class Governor:
    """A reusable oversight layer parameterized by YOUR risk function and thresholds."""

    def __init__(self, risk_fn: RiskFn, thresholds: Thresholds = DEFAULT_THRESHOLDS):
        self.risk_fn = risk_fn
        self.thresholds = thresholds

    def decide(self, action, human_queue_depth: int = 0) -> GovernorDecision:
        return decide(self.risk_fn(action), human_queue_depth, self.thresholds)


@dataclass
class Scoreboard:
    total: int
    correct_escalation: int
    false_auto_send: int      # DANGEROUS
    correct_auto: int
    false_escalation: int
    autonomy_pct: float
    human_load_saved_pct: float
    escalation_precision: float
    escalation_recall: float

    def pretty(self) -> str:
        L = ["=== Eval scoreboard: judgment vs ground truth ===", f"cases: {self.total}", "",
             "                    pred=ESCALATE   pred=AUTO",
             f"  true=ESCALATE          {self.correct_escalation:>3}          {self.false_auto_send:>3}  <- false auto-sends (DANGEROUS)",
             f"  true=AUTO              {self.false_escalation:>3}          {self.correct_auto:>3}", "",
             f"  DANGEROUS auto-sends : {self.false_auto_send}   (target 0)",
             f"  false escalations    : {self.false_escalation}   (wasted human attention)",
             f"  autonomy             : {self.autonomy_pct:.0f}%  (acted without a human)",
             f"  human-load saved     : {self.human_load_saved_pct:.0f}%",
             f"  escalation precision : {self.escalation_precision:.2f}",
             f"  escalation recall    : {self.escalation_recall:.2f}  (fraction of real risks caught)"]
        return "\n".join(L)


def evaluate_policy(cases, risk_fn: RiskFn,
                    thresholds: Thresholds = DEFAULT_THRESHOLDS) -> Scoreboard:
    """Score a risk_fn's decisions against ground truth.

    `cases` is an iterable of (action, ground_truth) where ground_truth is "escalate" | "auto".
    Queue is empty here (pure policy eval), so no load-shed HOLD.
    """
    gov = Governor(risk_fn, thresholds)
    cc = ca = fa = fe = 0
    total = 0
    for action, gt in cases:
        total += 1
        d = gov.decide(action, human_queue_depth=0)
        pred = "escalate" if d.decision == Decision.ESCALATE else "auto"
        if gt == "escalate" and pred == "escalate":
            cc += 1
        elif gt == "escalate" and pred == "auto":
            fa += 1
        elif gt == "auto" and pred == "auto":
            ca += 1
        else:
            fe += 1
    n_auto_sent = ca + fa
    n_pred_escalate = cc + fe
    autonomy = 100.0 * n_auto_sent / total if total else 0.0
    precision = cc / n_pred_escalate if n_pred_escalate else 1.0
    recall = cc / (cc + fa) if (cc + fa) else 1.0
    return Scoreboard(total, cc, fa, ca, fe, autonomy, autonomy, precision, recall)
