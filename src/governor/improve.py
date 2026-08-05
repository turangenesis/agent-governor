"""Eval-gated self-improvement loop for the Governor policy - the HONEST version.

    analyze  -> the cases the baseline gate got wrong
    propose  -> DATA-DRIVEN: mine terms that separate the missed drafts from clean ones (learned
                from the data, NOT a family we predefined)
    prove    -> measure on THREE splits:
                  train              (the proposer learned from these)
                  val, same family   (unseen phrasings - real generalization; expect a PARTIAL gain)
                  cross-family       (a family never shown - expect little/no transfer: the honest limit)
                and the labeled invariant (0 dangerous, recall >= 0.95, precision not worse)
    decide   -> MERGE only if the same-family validation improves without regressing the invariant

The result is intentionally NOT a clean 0 -> 1.0: it shows a partial, family-specific gain and an
honest non-transfer to other families. That is what real, measured self-improvement looks like.
The proposer here is deterministic (free, offline); an LLM proposer can slot in behind the same
interface. The Governor stays deterministic - a policy may only widen matched term families.
"""
from __future__ import annotations

import re
from collections import Counter

from .eval_set import BRIEF, build_cases
from .eval_set_adversarial import build_crossfamily_cases, build_train_cases, build_val_cases
from .evaluate import evaluate
from .governor import PUSHY_TERMS, SENSITIVE_TERMS, govern
from .models import Decision
from .policy import BASELINE, Policy

PRECISION_TOLERANCE = 0.05
_STOP = set("a an the to of and or for we i you your our is are be with at on in it this that will "
            "would can could hi hope let know if re ll me my".split())


def _escalation_recall(cases, policy: Policy) -> float:
    esc = [c for c in cases if c.ground_truth == "escalate"]
    if not esc:
        return 1.0
    caught = sum(1 for c in esc
                 if govern(c.action, BRIEF, policy=policy).decision == Decision.ESCALATE)
    return round(caught / len(esc), 3)


def _text(case) -> str:
    # Body only - that's exactly what the Governor matches against (score_signals uses action.body),
    # so mined terms are matchable and we don't mine dead subject-line words.
    return (case.action.body or "").lower()


def _ngrams(text: str, ns) -> set:
    toks = re.findall(r"[a-z']+", text)
    out = set()
    for n in ns:
        for i in range(len(toks) - n + 1):
            g = toks[i:i + n]
            if all(w in _STOP for w in g):
                continue
            out.add(" ".join(g))
    return out


def _in_baseline(term: str) -> bool:
    base = [b.lower() for b in (PUSHY_TERMS + SENSITIVE_TERMS)]
    return any(term == b or term in b or b in term for b in base)


def _discriminative_terms(positives, negatives, ns=(1, 2), min_pos=2, cap=12) -> tuple:
    """Terms that appear in >= min_pos of the MISSED drafts and in NONE of the clean drafts.

    This is the data-driven proposal: learned from the failures, not a predefined family. Stopwords
    and baseline terms are dropped; anything that also occurs in a clean draft is not discriminative."""
    neg_text = "\n".join(_text(c) for c in negatives)
    counts = Counter()
    for c in positives:
        for g in _ngrams(_text(c), ns):
            counts[g] += 1
    terms = []
    for g, c in counts.most_common():
        if c < min_pos or g in neg_text or _in_baseline(g):
            continue
        terms.append(g)
        if len(terms) >= cap:
            break
    return tuple(terms)


def analyze(train_cases) -> dict:
    missed = [c.action.candidate.name for c in train_cases
              if c.ground_truth == "escalate"
              and govern(c.action, BRIEF, policy=BASELINE).decision != Decision.ESCALATE]
    return {"missed": missed}


def propose_from_data(train_cases, negatives=None) -> Policy:
    """Data-driven proposer: mine discriminative 1-2 grams from the misses vs clean drafts."""
    negatives = build_cases() if negatives is None else negatives
    negatives = [c for c in negatives if c.ground_truth == "auto"]
    terms = _discriminative_terms(train_cases, negatives, ns=(1, 2), min_pos=2)
    return Policy(extra_pushy_terms=terms,
                  note=f"data-driven: {len(terms)} terms mined from the missed drafts")


def propose_overfit(train_cases, negatives=None) -> Policy:
    """Guard baseline: memorize long exact phrases (4-5 grams) - won't generalize to new phrasings."""
    negatives = build_cases() if negatives is None else negatives
    negatives = [c for c in negatives if c.ground_truth == "auto"]
    terms = _discriminative_terms(train_cases, negatives, ns=(4, 5), min_pos=1)
    return Policy(extra_pushy_terms=terms, note="overfit: exact long TRAIN phrases only")


def prove(policy: Policy, train=None, val=None, cross=None) -> dict:
    """Run the gate on train / same-family val / cross-family, plus the labeled invariant."""
    train = build_train_cases() if train is None else train
    val = build_val_cases() if val is None else val
    cross = build_crossfamily_cases() if cross is None else cross

    lab_base, lab_new = evaluate(policy=BASELINE), evaluate(policy=policy)
    report = {
        "train_recall": {"before": _escalation_recall(train, BASELINE),
                         "after": _escalation_recall(train, policy)},
        "val_same_family_recall": {"before": _escalation_recall(val, BASELINE),
                                   "after": _escalation_recall(val, policy)},
        "cross_family_recall": {"before": _escalation_recall(cross, BASELINE),
                                "after": _escalation_recall(cross, policy)},
        "labeled": {"dangerous": lab_new.false_auto_send, "recall": round(lab_new.escalation_recall, 2),
                    "precision_before": round(lab_base.escalation_precision, 2),
                    "precision_after": round(lab_new.escalation_precision, 2)},
    }
    checks = {
        "same-family validation improved":
            report["val_same_family_recall"]["after"] > report["val_same_family_recall"]["before"],
        "labeled 0 dangerous": lab_new.false_auto_send == 0,
        "labeled recall >= 0.95": lab_new.escalation_recall >= 0.95,
        "labeled precision preserved":
            lab_new.escalation_precision >= lab_base.escalation_precision - PRECISION_TOLERANCE,
    }
    report["checks"] = checks
    report["decision"] = "MERGE" if all(checks.values()) else "REJECT"
    # honest note: did the gain transfer to a family the proposer never saw?
    ct = report["cross_family_recall"]
    report["cross_family_transfer"] = "none" if ct["after"] <= ct["before"] else "partial"
    return report


def run_improvement() -> dict:
    train = build_train_cases()
    analysis = analyze(train)
    proposal = propose_from_data(train)
    report = prove(proposal, train=train)
    return {"analysis": analysis, "proposal": proposal, "report": report}
