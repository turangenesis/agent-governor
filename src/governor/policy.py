"""The tunable Governor policy - the ONLY surface the self-improvement loop may change.

The Governor stays deterministic and LLM-free. A `Policy` can only *widen the risk-term families*
the gate matches; it can never touch scoring code, thresholds, weights, or the eval labels. That
bound is what makes the self-improvement loop safe: the proposer can broaden detection, nothing more.

`BASELINE` reproduces today's shipped behavior exactly (empty extras)."""
from __future__ import annotations

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
