"""The reusable core is domain-free: the same engine governs + evaluates ANY risk function.

Proves the repositioning - Governor/decide/evaluate_policy work on a toy float action and on a
second, unrelated domain (examples/support_desk.py), not just recruiting."""
from __future__ import annotations

import importlib.util
import os
import sys

from governor.core import Governor, Thresholds, decide, evaluate_policy
from governor.models import Decision


def test_decide_threshold_bands():
    assert decide({}).decision == Decision.AUTO_SEND                     # no signals
    assert decide({"x": 0.30}).decision == Decision.AUTO_SEND            # mid-band leans safe
    assert decide({"x": 0.60}).decision == Decision.ESCALATE            # over escalate threshold
    assert decide({"x": 0.90}).decision == Decision.ESCALATE            # hard-escalate
    # load-shed only in the escalate band, never for a hard risk
    assert decide({"x": 0.60}, human_queue_depth=10).decision == Decision.HOLD
    assert decide({"x": 0.90}, human_queue_depth=10).decision == Decision.ESCALATE


def test_custom_thresholds():
    strict = Thresholds(escalate=0.2, auto=0.05, hard_escalate=0.9)
    assert decide({"x": 0.25}, thresholds=strict).decision == Decision.ESCALATE


def test_governor_on_a_plain_float_action():
    gov = Governor(risk_fn=lambda a: {"r": a})     # action is just a float
    assert gov.decide(0.9).decision == Decision.ESCALATE
    assert gov.decide(0.05).decision == Decision.AUTO_SEND


def _load_support_example():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "examples", "support_desk.py")
    spec = importlib.util.spec_from_file_location("support_desk", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # a module defining @dataclass must be in sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_core_reused_on_a_different_domain():
    """The support-desk example (refunds) runs on the same core with 0 dangerous / full recall."""
    m = _load_support_example()
    sb = evaluate_policy(m.CASES, m.refund_risk)
    assert sb.total == len(m.CASES)
    assert sb.false_auto_send == 0
    assert sb.escalation_recall == 1.0
