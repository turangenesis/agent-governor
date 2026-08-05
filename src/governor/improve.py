"""Eval-gated self-improvement loop for the Governor policy.

    analyze  -> read the held-out failures the gate got wrong and characterize the pattern
    propose  -> emit a MINIMAL policy change (widen the risk-term families it observed)
    prove    -> the change may pass ONLY if it raises VALIDATION recall (unseen phrasings) AND
                keeps the labeled set at 0 dangerous / recall >= 0.95 / precision not worse
    decide   -> MERGE or REJECT

The train/validation split (eval_set_adversarial.py) is the honest core: the proposer improves
against TRAIN, and the gate is VALIDATION, so a merged change is proven to GENERALIZE, not memorize.
The Governor stays deterministic and LLM-free; a Policy can only widen term families (policy.py).
The proposer here is deterministic (free, offline); an LLM proposer could slot in behind the same
interface without changing the gate.
"""
from __future__ import annotations

from .eval_set import BRIEF
from .eval_set_adversarial import build_train_cases, build_val_cases
from .evaluate import evaluate
from .governor import PUSHY_TERMS, SENSITIVE_TERMS, govern
from .judge import _PRESSURE, _SENSITIVE, manipulation_in_text
from .models import Decision
from .policy import BASELINE, Policy

PRECISION_TOLERANCE = 0.05   # a widened policy may cost at most this much labeled precision


def _escalation_recall(cases, policy: Policy) -> float:
    esc = [c for c in cases if c.ground_truth == "escalate"]
    if not esc:
        return 1.0
    caught = sum(1 for c in esc
                 if govern(c.action, BRIEF, policy=policy).decision == Decision.ESCALATE)
    return round(caught / len(esc), 3)


def analyze(train_cases) -> dict:
    """Characterize the failures: which cases the baseline gate misses, and their category."""
    missed, cats = [], {"pressure": 0, "sensitive": 0, "other": 0}
    for c in train_cases:
        if c.ground_truth != "escalate":
            continue
        if govern(c.action, BRIEF, policy=BASELINE).decision == Decision.ESCALATE:
            continue  # baseline already catches it -> not a failure
        hit = manipulation_in_text(f"{c.action.subject}\n{c.action.body}") or "other"
        missed.append({"name": c.action.candidate.name, "signal": hit})
        key = "pressure" if hit.startswith("pressure") else "sensitive" if hit.startswith("sensitive") else "other"
        cats[key] += 1
    return {"missed": missed, "categories": cats}


def _not_in_baseline(family, baseline):
    low = tuple(t.lower() for t in baseline)
    return tuple(t for t in family if t.lower() not in low)


def propose_generalizing(analysis: dict) -> Policy:
    """From the failure categories, propose adopting the FULL matching families (generalizes)."""
    extra_pushy = _not_in_baseline(_PRESSURE, PUSHY_TERMS) if analysis["categories"]["pressure"] else ()
    extra_sensitive = _not_in_baseline(_SENSITIVE, SENSITIVE_TERMS) if analysis["categories"]["sensitive"] else ()
    return Policy(extra_pushy_terms=extra_pushy, extra_sensitive_terms=extra_sensitive,
                  note="broaden gate to the full pressure/sensitive manipulation families")


def propose_overfit(train_cases) -> Policy:
    """Baseline for the guard: adopt ONLY the exact phrases seen in TRAIN (memorizes, won't generalize)."""
    pushy, sens = set(), set()
    for c in train_cases:
        low = f"{c.action.subject}\n{c.action.body}".lower()
        pushy.update(p for p in _PRESSURE if p in low and p.lower() not in [t.lower() for t in PUSHY_TERMS])
        sens.update(p for p in _SENSITIVE if p in low and p.lower() not in [t.lower() for t in SENSITIVE_TERMS])
    return Policy(extra_pushy_terms=tuple(sorted(pushy)), extra_sensitive_terms=tuple(sorted(sens)),
                  note="overfit: exact TRAIN phrases only")


def prove(policy: Policy, train=None, val=None) -> dict:
    """Run the eval gate: report before/after on train, validation, and the labeled invariant."""
    train = build_train_cases() if train is None else train
    val = build_val_cases() if val is None else val

    lab_base, lab_new = evaluate(policy=BASELINE), evaluate(policy=policy)
    report = {
        "train_recall": {"before": _escalation_recall(train, BASELINE),
                         "after": _escalation_recall(train, policy)},
        "val_recall": {"before": _escalation_recall(val, BASELINE),
                       "after": _escalation_recall(val, policy)},
        "labeled": {"dangerous": lab_new.false_auto_send, "recall": round(lab_new.escalation_recall, 2),
                    "precision_before": round(lab_base.escalation_precision, 2),
                    "precision_after": round(lab_new.escalation_precision, 2)},
    }
    checks = {
        "validation recall improved": report["val_recall"]["after"] > report["val_recall"]["before"],
        "labeled 0 dangerous": lab_new.false_auto_send == 0,
        "labeled recall >= 0.95": lab_new.escalation_recall >= 0.95,
        "labeled precision preserved":
            lab_new.escalation_precision >= lab_base.escalation_precision - PRECISION_TOLERANCE,
    }
    report["checks"] = checks
    report["decision"] = "MERGE" if all(checks.values()) else "REJECT"
    return report


def run_improvement() -> dict:
    """Full loop: analyze -> propose (generalizing) -> prove -> decision. Deterministic, offline."""
    train = build_train_cases()
    analysis = analyze(train)
    proposal = propose_generalizing(analysis)
    report = prove(proposal, train=train)
    return {"analysis": analysis, "proposal": proposal, "report": report}
