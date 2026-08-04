"""LLM-as-judge draft-quality eval (Phase 3), exercised via the free deterministic StubJudge.

The key test is the VALUE-ADD one: the judge catches manipulative drafts that the keyword-based
Governor auto-sent - proving the two layers are complementary."""
from __future__ import annotations

from governor.eval_set import BRIEF, build_cases
from governor.eval_set_heldout import build_heldout_cases
from governor.governor import govern
from governor.judge import JudgeVerdict, StubJudge, get_judge, judge_report
from governor.models import Decision


def _by_name(cases, name):
    return next(c.action for c in cases if c.action.candidate.name == name)


def test_stub_judge_passes_clean_drafts():
    j = StubJudge()
    clean = _by_name(build_cases(), "Jordan Lee")
    v = j.judge(clean)
    assert v.passed
    assert v.scores["no_manipulation"] >= 3 and v.scores["professional_tone"] >= 3


def test_stub_judge_flags_pushy_and_sensitive_drafts():
    j = StubJudge()
    pushy = _by_name(build_cases(), "Ian Wells")        # "ACT NOW ... LAST CHANCE ... ASAP"
    comp = _by_name(build_cases(), "Alex Kim")          # "beat your salary + visa ... layoffs"
    for a in (pushy, comp):
        v = j.judge(a)
        assert not v.passed, f"judge should flag {a.candidate.name}"
        assert v.scores["no_manipulation"] <= 2


def test_judge_catches_evasions_the_gate_missed():
    """The money test: at least one held-out draft the Governor AUTO-SENT is caught by the judge."""
    j = StubJudge()
    gate_missed_judge_caught = 0
    for c in build_heldout_cases():
        gate = govern(c.action, BRIEF, human_queue_depth=0)
        verdict = j.judge(c.action)
        if gate.decision == Decision.AUTO_SEND and not verdict.passed:
            gate_missed_judge_caught += 1
    assert gate_missed_judge_caught >= 1, "the judge should add value over the keyword gate"


def test_judge_report_schema():
    actions = [c.action for c in build_cases()]
    rep = judge_report(actions, StubJudge())
    assert rep["judge"] == "stub"
    assert rep["total"] == len(actions)
    assert 0.0 <= rep["pass_rate"] <= 1.0
    for k in ("relevance", "specificity", "professional_tone", "no_manipulation"):
        assert 1 <= rep["avg_scores"][k] <= 5


def test_verdict_to_dict_and_factory():
    v = StubJudge().judge(build_cases()[0].action)
    d = v.to_dict()
    assert set(d) == {"scores", "passed", "reasons"} and isinstance(d["scores"], dict)
    assert isinstance(get_judge("stub"), StubJudge)
    assert get_judge("llm").name == "llm"           # constructs without a key (lazy until .judge)
