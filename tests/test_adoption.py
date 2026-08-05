"""Adoption makes the self-heal REAL: writing the active policy actually changes the gate.

Proves that a proposal, once adopted (what merging a self-improvement PR does), causes a
previously-missed evasion to escalate - and that removing it reverts to baseline behavior."""
from __future__ import annotations

import os

import pytest

from governor.eval_set import BRIEF
from governor.eval_set_adversarial import build_train_cases, build_val_cases
from governor.governor import govern
from governor.improve import propose_from_data
from governor.models import Decision
from governor.policy import ACTIVE_POLICY_PATH, load_active_policy, save_active_policy


@pytest.fixture
def no_active_policy():
    """Clean slate for the test, restoring any pre-existing (e.g. merged) active policy after."""
    backup = None
    if os.path.exists(ACTIVE_POLICY_PATH):
        with open(ACTIVE_POLICY_PATH) as f:
            backup = f.read()
        os.remove(ACTIVE_POLICY_PATH)
    try:
        yield
    finally:
        if os.path.exists(ACTIVE_POLICY_PATH):
            os.remove(ACTIVE_POLICY_PATH)
        if backup is not None:                       # restore a committed/adopted policy untouched
            with open(ACTIVE_POLICY_PATH, "w") as f:
                f.write(backup)


def _a_caught_val_case():
    # "positions are filling quickly" shares the mined term "filling" -> caught after adoption
    return build_val_cases()[0].action


def test_default_is_baseline_when_no_active_policy(no_active_policy):
    assert load_active_policy().extra_pushy_terms == ()          # BASELINE
    # baseline gate misses this evasion
    assert govern(_a_caught_val_case(), BRIEF).decision != Decision.ESCALATE


def test_adoption_changes_govern_behavior(no_active_policy):
    action = _a_caught_val_case()
    before = govern(action, BRIEF)                                # None -> active (none) -> BASELINE
    assert before.decision != Decision.ESCALATE

    save_active_policy(propose_from_data(build_train_cases()))    # what merging the PR does
    after = govern(action, BRIEF)                                 # None -> adopted policy
    assert after.decision == Decision.ESCALATE, "adopted policy should now catch the evasion"

    os.remove(ACTIVE_POLICY_PATH)
    assert govern(action, BRIEF).decision != Decision.ESCALATE   # reverts cleanly
