"""The scoreboard over the 40 labeled cases must hit the demo's headline guarantees:
perfect escalation recall and zero dangerous (false) auto-sends."""
from __future__ import annotations

from governor.eval_set import build_cases
from governor.evaluate import evaluate


def test_labeled_set_is_forty_cases():
    assert len(build_cases()) == 40


def test_escalation_recall_is_perfect():
    sb = evaluate()
    assert sb.escalation_recall == 1.0


def test_zero_dangerous_auto_sends():
    sb = evaluate()
    assert sb.false_auto_send == 0


def test_scoreboard_covers_every_case():
    sb = evaluate()
    assert (sb.correct_escalation + sb.false_auto_send
            + sb.correct_auto + sb.false_escalation) == sb.total == 40
