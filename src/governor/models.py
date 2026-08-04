"""Shared data models. Reused by the agent, the Governor, the eval harness, and the UI.

Deliberately plain dataclasses — no framework coupling, easy to serialize for the queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


@dataclass
class Candidate:
    """A sourcing target. `matched` = how well they fit the role (agent/data-layer supplied)."""
    name: str
    current_company: str
    current_title: str
    seniority: str            # e.g. "junior" | "mid" | "senior" | "exec"
    email: str
    matched_role: str
    match_confidence: float   # 0..1 how well the profile fits the role


@dataclass
class HiringBrief:
    role: str
    must_haves: str
    hiring_company: str
    competitors: tuple[str, ...] = ()   # companies we must NOT poach from without a human


@dataclass
class ProposedAction:
    """What an agent wants to do: send this outreach email. The Governor decides its fate."""
    agent_id: str
    candidate: Candidate
    subject: str
    body: str
    draft_confidence: float             # 0..1 the agent's own confidence in the draft
    concerns: list[str] = field(default_factory=list)  # agent-surfaced worries
    seq: int = 0                        # arrival order (for FIFO)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProposedAction":
        """Rebuild a ProposedAction from its serialized form (for cached-trace replay)."""
        d = dict(d)
        d["candidate"] = Candidate(**d["candidate"])
        return cls(**d)


@dataclass
class TraceStep:
    """One step of an agent's tool-use loop — the unit the DECISION TRACE is built from.

    Records what the LLM chose to do, what it passed, what it observed back, and whether it
    continued or stopped. This is a structured ACTION log (tool/input/observation/decision),
    never the model's hidden chain-of-thought.
    """
    index: int                 # 0-based order within the loop
    tool: str                  # the tool the LLM chose (search_github / refine_search / ...)
    tool_input: dict           # the arguments the LLM passed to that tool
    observation: str           # short summary of what the tool returned
    decision: str              # "continue" | "stop"
    reason: str                # why it continued/stopped (terminal reason when decision=="stop")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TraceStep":
        return cls(index=d["index"], tool=d["tool"], tool_input=dict(d.get("tool_input") or {}),
                   observation=d["observation"], decision=d["decision"], reason=d["reason"])


# Terminal reasons a loop can stop for (the guardrails that guarantee termination).
TERMINAL_REASONS = ("enough-candidates", "max-iterations", "budget")


@dataclass
class LoopResult:
    """The output of one territory agent's tool-use loop: its proposals + its decision trace."""
    territory_key: str
    steps: list["TraceStep"] = field(default_factory=list)
    proposals: list["ProposedAction"] = field(default_factory=list)
    terminal_reason: str = ""            # one of TERMINAL_REASONS
    tokens_used: int = 0
    iterations: int = 0
    leads: list["Candidate"] = field(default_factory=list)   # candidates the loop discovered, in order

    def to_dict(self) -> dict:
        return {
            "territory_key": self.territory_key,
            "steps": [s.to_dict() for s in self.steps],
            "proposals": [p.to_dict() for p in self.proposals],
            "terminal_reason": self.terminal_reason,
            "tokens_used": self.tokens_used,
            "iterations": self.iterations,
            # Persist the discovered candidates WITH the trace so an offline replay can rebuild
            # them itself and NEVER needs to re-hit the discovery layer (no cache, no network).
            "leads": [asdict(c) for c in self.leads],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoopResult":
        """Rebuild a LoopResult from its serialized form — the persisted decision trace."""
        return cls(
            territory_key=d.get("territory_key", ""),
            steps=[TraceStep.from_dict(s) for s in d.get("steps", [])],
            proposals=[ProposedAction.from_dict(p) for p in d.get("proposals", [])],
            terminal_reason=d.get("terminal_reason", ""),
            tokens_used=d.get("tokens_used", 0),
            iterations=d.get("iterations", 0),
            leads=[c if isinstance(c, Candidate) else Candidate(**c) for c in d.get("leads", [])],
        )


class Decision(str, Enum):
    AUTO_SEND = "auto_send"       # safe enough to fire without a human
    ESCALATE = "escalate"         # genuinely needs a human recruiter's judgment
    HOLD = "hold"                 # load-shed: human is saturated, park the low-risk marginal case


@dataclass
class GovernorDecision:
    decision: Decision
    risk_score: float             # 0..1 aggregate risk
    reasons: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)   # named risk signals -> value/weight
