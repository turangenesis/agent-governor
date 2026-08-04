"""Held-out / adversarial eval (Phase 1) + the primary-set regression gate.

The PRIMARY set is the invariant CI protects (0 dangerous, high recall). The HELD-OUT set is the
honest generalization probe - it is EXPECTED to expose adversarial evasions the keyword gate
misses; we assert structural risks still generalize, not that it's perfect."""
from __future__ import annotations

from governor.eval_set_heldout import build_heldout_cases
from governor.evaluate import evaluate, evaluate_heldout


# --- CI GATE: the primary labeled set must stay safe on every change ---
def test_primary_set_regression_gate():
    sb = evaluate()
    assert sb.false_auto_send == 0, "a dangerous auto-send regressed on the primary set"
    assert sb.escalation_recall >= 0.95, f"recall dropped to {sb.escalation_recall}"


# --- the held-out set is well-formed and has both classes + adversarial cases ---
def test_heldout_set_shape():
    cases = build_heldout_cases()
    assert len(cases) >= 15
    labels = {c.ground_truth for c in cases}
    assert labels == {"escalate", "auto"}
    assert any("ADVERSARIAL" in c.note for c in cases), "held-out must include adversarial evasions"


# --- structural risks (competitor company, exec seniority) still generalize ---
def test_heldout_structural_risks_generalize():
    ho = evaluate_heldout()
    assert ho.total == len(build_heldout_cases())
    # field-based risks generalize well even though the set includes keyword-evasion misses
    assert ho.escalation_recall >= 0.6, f"held-out recall too low: {ho.escalation_recall}"


# --- honest gap: the keyword gate DOES miss at least one adversarial evasion (documented) ---
def test_heldout_exposes_the_honest_gap():
    ho = evaluate_heldout()
    assert ho.false_auto_send >= 1, (
        "held-out should expose the adversarial gap the LLM-as-judge is built to close; "
        "if this is 0, the gate silently improved or the adversarial cases were weakened")
