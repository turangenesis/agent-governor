"""The tunable Governor policy - the ONLY surface the self-improvement loop may change.

The Governor stays deterministic and LLM-free. A `Policy` can only *widen the risk-term families*
the gate matches; it can never touch scoring code, thresholds, weights, or the eval labels. That
bound is what makes the self-improvement loop safe: the proposer can broaden detection, nothing more.

`BASELINE` reproduces today's shipped behavior exactly (empty extras)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    """A policy is baseline behavior + extra risk-term families to match in the draft body."""
    extra_pushy_terms: tuple = ()
    extra_sensitive_terms: tuple = ()
    note: str = "baseline"

    def to_dict(self) -> dict:
        return {"extra_pushy_terms": list(self.extra_pushy_terms),
                "extra_sensitive_terms": list(self.extra_sensitive_terms), "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        return cls(tuple(d.get("extra_pushy_terms", ())),
                   tuple(d.get("extra_sensitive_terms", ())), d.get("note", "baseline"))


BASELINE = Policy()

# The ADOPTED policy. If this file exists (e.g. merged from a self-improvement PR), the Governor
# uses it; otherwise it stays BASELINE. This is what makes "merge the PR -> the Governor improves"
# real: the eval-gated proposal, once merged here, actually changes the gate's behavior.
ACTIVE_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active_policy.json")


def load_active_policy() -> Policy:
    """The policy the Governor runs by default: the adopted one if present, else BASELINE."""
    if not os.path.exists(ACTIVE_POLICY_PATH):
        return BASELINE
    try:
        with open(ACTIVE_POLICY_PATH) as f:
            return Policy.from_dict(json.load(f))
    except Exception:
        return BASELINE


def save_active_policy(policy: Policy) -> None:
    """Adopt a policy - what a merged self-improvement PR effectively does."""
    with open(ACTIVE_POLICY_PATH, "w") as f:
        json.dump(policy.to_dict(), f, indent=2)
