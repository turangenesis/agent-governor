"""The HONEST eval-gated self-improvement loop.

Asserts the honest properties (not a constructed 0->1.0):
  - the data-driven proposer PARTIALLY generalizes to unseen same-family phrasings,
  - it does NOT transfer to a family it was never shown (the honest limit),
  - the labeled safety invariant is preserved,
  - an overfit (memorize-exact-phrases) proposal is rejected by the validation gate."""
from __future__ import annotations

from governor.eval_set_adversarial import build_train_cases
from governor.evaluate import evaluate
from governor.improve import analyze, propose_from_data, propose_overfit, prove, run_improvement


def test_baseline_misses_the_adversarial_evasions():
    assert len(analyze(build_train_cases())["missed"]) >= 4


def test_data_driven_proposal_merges_with_partial_generalization():
    r = run_improvement()["report"]
    assert r["decision"] == "MERGE"
    v = r["val_same_family_recall"]
    assert v["after"] > v["before"]                 # generalized to unseen phrasings...
    assert v["after"] < 1.0                          # ...but only PARTIALLY (honest, not 0->1.0)
    assert r["labeled"]["dangerous"] == 0
    assert r["labeled"]["recall"] >= 0.95


def test_gain_does_not_transfer_across_families():
    """Honest limit: learning pressure terms should NOT fix a comp/visa family it never saw."""
    r = run_improvement()["report"]
    c = r["cross_family_recall"]
    assert c["after"] <= c["before"]
    assert r["cross_family_transfer"] == "none"


def test_overfit_proposal_is_rejected():
    r = prove(propose_overfit(build_train_cases()))
    v = r["val_same_family_recall"]
    assert v["after"] == v["before"], "memorized exact phrases must not improve unseen phrasings"
    assert r["decision"] == "REJECT"


def test_merged_policy_keeps_labeled_zero_dangerous():
    proposal = propose_from_data(build_train_cases())
    sb = evaluate(policy=proposal)
    assert sb.false_auto_send == 0 and sb.escalation_recall >= 0.95


def test_loop_is_deterministic():
    assert run_improvement()["report"] == run_improvement()["report"]
