"""Optional HTTP surface: run the Governor as a SERVICE, not just a library.

    pip install -e ".[api]"
    uvicorn governor.api:app        # then POST /decide (generic) or /govern (recruiting)

This is the other way to consume the reusable core: any app in any language can POST an action's
risk signals and get back the gate decision. The core never imports this module, so a plain
`pip install .` stays light.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .core import decide

app = FastAPI(
    title="The Governor",
    version="0.1.0",
    description="Oversight + eval layer for autonomous agents: gate an action as "
                "auto-send / escalate / hold, and measure the policy's correctness.",
)


# --- generic reusable layer: bring your own risk signals -----------------------------------
class DecideIn(BaseModel):
    signals: dict                 # {signal_name: weight in 0..1}
    human_queue_depth: int = 0


def _decision_dict(d) -> dict:
    return {"decision": d.decision.value, "risk_score": round(d.risk_score, 3), "reasons": d.reasons}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/decide")
def decide_endpoint(body: DecideIn) -> dict:
    """The domain-free gate: named risk signals -> AUTO_SEND / ESCALATE / HOLD."""
    return _decision_dict(decide(body.signals, body.human_queue_depth))


# --- recruiting example wired end to end ----------------------------------------------------
class CandidateIn(BaseModel):
    name: str
    current_company: str = ""
    current_title: str = ""
    seniority: str = "mid"
    email: str = ""
    matched_role: str = ""
    match_confidence: float = 0.8


class GovernIn(BaseModel):
    candidate: CandidateIn
    subject: str = ""
    body: str = ""
    draft_confidence: float = 0.8
    concerns: list = []
    human_queue_depth: int = 0


@app.post("/govern")
def govern_endpoint(body: GovernIn) -> dict:
    """The recruiting policy end to end: a proposed outreach -> the Governor's decision."""
    from .eval_set import BRIEF
    from .governor import govern
    from .models import Candidate, ProposedAction

    cand = Candidate(
        name=body.candidate.name, current_company=body.candidate.current_company,
        current_title=body.candidate.current_title, seniority=body.candidate.seniority,
        email=body.candidate.email, matched_role=body.candidate.matched_role,
        match_confidence=body.candidate.match_confidence,
    )
    action = ProposedAction("api", cand, body.subject, body.body,
                            body.draft_confidence, list(body.concerns))
    return _decision_dict(govern(action, BRIEF, human_queue_depth=body.human_queue_depth))


@app.get("/scoreboard")
def scoreboard() -> dict:
    """The measured correctness of the recruiting policy (the eval scoreboard headline)."""
    from .evaluate import evaluate
    sb = evaluate()
    return {"dangerous_auto_sends": sb.false_auto_send, "recall": round(sb.escalation_recall, 2),
            "precision": round(sb.escalation_precision, 2), "autonomy_pct": round(sb.autonomy_pct)}
