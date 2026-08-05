"""Second consumer of the reusable core (governor.core), an ENTIRELY different domain.

A support agent wants to auto-issue refunds. Same oversight layer, brand-new domain: we only
supply an action type, a `risk_fn` (action -> named risk signals), and a few labeled cases. No
recruiting, no candidates, no emails - proof that `governor.core` is a general governance + eval
layer, not a one-off demo.

    python examples/support_desk.py
"""
from __future__ import annotations

from dataclasses import dataclass

from governor.core import Governor, evaluate_policy
from governor.models import Decision


@dataclass
class RefundRequest:
    amount: float
    message: str
    vip: bool
    confidence: float     # the agent's own confidence it should auto-refund (0..1)


_LEGAL_TERMS = ("chargeback", "attorney", "lawsuit", "sue", "fraud")


def refund_risk(a: RefundRequest) -> dict:
    """The support domain's risk function: action -> {signal: weight}."""
    s: dict[str, float] = {}
    if a.amount >= 200:
        s["large_refund"] = min(0.6, 0.2 + a.amount / 1000)
    if any(t in a.message.lower() for t in _LEGAL_TERMS):
        s["legal_language"] = 0.6
    if a.vip:
        s["vip_account"] = 0.3
    if a.confidence < 0.6:
        s["low_confidence"] = round(0.4 * (0.6 - a.confidence) / 0.6, 3)
    return s


# Labeled cases (action, ground_truth) - what a human support lead would decide.
CASES = [
    (RefundRequest(15, "Package arrived damaged, please refund.", False, 0.9), "auto"),
    (RefundRequest(120, "Wrong size shipped, requesting a refund.", False, 0.85), "auto"),
    (RefundRequest(10, "Refund please, changed my mind.", False, 0.95), "auto"),
    (RefundRequest(800, "Refund my order, it never arrived.", False, 0.9), "escalate"),
    (RefundRequest(60, "I will file a chargeback and contact my attorney.", False, 0.8), "escalate"),
    (RefundRequest(500, "Very unhappy, refund now.", True, 0.7), "escalate"),
]


def main() -> None:
    gov = Governor(refund_risk)
    print("=== Support-desk refunds on the reusable Governor core ===\n")
    for req, gt in CASES:
        d = gov.decide(req)
        mark = "OK " if (d.decision == Decision.ESCALATE) == (gt == "escalate") else "XX "
        print(f"  {mark}${req.amount:>5.0f}  {d.decision.value:<9} risk={d.risk_score:.2f}  (truth={gt})")
    print()
    print(evaluate_policy(CASES, refund_risk).pretty())


if __name__ == "__main__":
    main()
