"""Decision-level tests for the deterministic Governor (no LLM, no network)."""
from __future__ import annotations

from governor.governor import (
    AUTO_THRESHOLD,
    ESCALATE_THRESHOLD,
    HARD_ESCALATE,
    _is_competitor,
    govern,
    risk_of,
)
from governor.models import Candidate, Decision, HiringBrief, ProposedAction

BRIEF = HiringBrief(
    role="Senior Backend Engineer",
    must_haves="Python, distributed systems",
    hiring_company="Acme AI",
    competitors=("OpenAI", "Anthropic", "Google DeepMind", "Cohere"),
)

CLEAN_BODY = (
    "Hi there, I came across your work and was genuinely impressed by your background. "
    "We're building reliability infrastructure at Acme AI and think the Senior Backend "
    "Engineer role could be a great fit. Would you be open to a short call next week?"
)


def _action(company="Stripe", seniority="mid", match=0.88, conf=0.9,
            body=CLEAN_BODY, concerns=None):
    cand = Candidate("Test Person", company, "Software Engineer", seniority,
                     "t@example.com", "Senior Backend Engineer", match)
    return ProposedAction("agent", cand, "A role you might like", body, conf, concerns or [])


def test_competitor_company_escalates():
    d = govern(_action(company="Anthropic"), BRIEF)
    assert d.decision == Decision.ESCALATE
    assert "competitor_poach" in d.signals


def test_exec_seniority_escalates():
    d = govern(_action(seniority="exec"), BRIEF)
    assert d.decision == Decision.ESCALATE
    assert "exec_seniority" in d.signals


def test_clean_low_risk_auto_sends():
    d = govern(_action(), BRIEF)
    assert d.decision == Decision.AUTO_SEND
    assert d.risk_score <= AUTO_THRESHOLD


def test_pushy_draft_body_escalates():
    body = "ACT NOW - this is your LAST CHANCE to join Acme AI. Urgent, reply ASAP."
    d = govern(_action(body=body), BRIEF)
    assert d.decision == Decision.ESCALATE
    assert "pushy_content" in d.signals


def test_sensitive_draft_body_escalates():
    body = ("Hi, we can beat your current salary and sort visa sponsorship fast. "
            "Heard your team had layoffs - let's talk about the role.")
    d = govern(_action(body=body), BRIEF)
    assert d.decision == Decision.ESCALATE
    assert "sensitive_content" in d.signals


def test_thresholds_are_ordered_and_respected():
    assert 0.0 < AUTO_THRESHOLD < ESCALATE_THRESHOLD < HARD_ESCALATE <= 1.0

    # A single competitor signal (weight 0.60) clears the escalate threshold on its own.
    score, _ = risk_of(_action(company="OpenAI"), BRIEF)
    assert score >= ESCALATE_THRESHOLD

    # A clean action sits at/below the auto threshold.
    score, _ = risk_of(_action(), BRIEF)
    assert score <= AUTO_THRESHOLD


def test_competitor_name_normalization():
    competitors = ("Anthropic",)
    # Messy real-world company strings still match.
    assert _is_competitor("@Anthropic", competitors)
    assert _is_competitor("anthropics", competitors)
    assert _is_competitor("  ANTHROPIC ", competitors)
    # Unrelated companies do not.
    assert not _is_competitor("Stripe", competitors)
    assert not _is_competitor("", competitors)


def test_hard_escalate_never_load_sheds():
    # Stack signals well past HARD_ESCALATE (competitor + exec + pushy + sensitive).
    body = "ACT NOW salary visa - LAST CHANCE, reply ASAP."
    d = govern(_action(company="Anthropic", seniority="exec", body=body),
               BRIEF, human_queue_depth=99)
    assert d.decision == Decision.ESCALATE
    assert d.risk_score >= HARD_ESCALATE
    assert d.reasons[0] == "HARD-ESCALATE"
