"""The eval-gated self-improvement loop: a generalizing proposal MERGES, an overfit one is REJECTED.

This is the honest core - the validation split blocks memorization, so a merge means the change
provably generalizes to unseen adversarial phrasings without regressing the labeled invariant."""
from __future__ import annotations

from governor.eval_set_adversarial import build_train_cases, build_val_cases
from governor.evaluate import evaluate
from governor.improve import (
    analyze,
    propose_generalizing,
    propose_overfit,
    prove,
    run_improvement,
)
from governor.policy import BASELINE


def test_baseline_misses_the_adversarial_evasions():
    """Precondition: the baseline gate genuinely fails these (else there's nothing to improve)."""
    a = analyze(build_train_cases())
    assert len(a["missed"]) >= 4, "baseline should miss most adversarial train cases"


def test_generalizing_proposal_merges():
    out = run_improvement()
    r = out["report"]
    assert r["decision"] == "MERGE"
    assert r["val_recall"]["after"] > r["val_recall"]["before"]       # generalized to unseen
    assert r["labeled"]["dangerous"] == 0                             # invariant preserved
    assert r["labeled"]["recall"] >= 0.95


def test_overfit_proposal_is_rejected():
    """A policy that only memorizes TRAIN's exact phrases must NOT pass the validation gate."""
    overfit = propose_overfit(build_train_cases())
    r = prove(overfit)
    # it should help train but not the unseen validation phrasings
    assert r["train_recall"]["after"] > r["train_recall"]["before"]
    assert r["val_recall"]["after"] == r["val_recall"]["before"], "overfit should NOT improve val"
    assert r["decision"] == "REJECT"


def test_merged_policy_keeps_labeled_zero_dangerous():
    proposal = propose_generalizing(analyze(build_train_cases()))
    sb = evaluate(policy=proposal)
    assert sb.false_auto_send == 0
    assert sb.escalation_recall >= 0.95


def test_loop_is_deterministic():
    assert run_improvement()["report"] == run_improvement()["report"]
