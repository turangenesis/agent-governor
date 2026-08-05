"""Eval-gated self-improvement loop for the Governor policy - the HONEST version.

    analyze  -> the cases the baseline gate got wrong
    propose  -> DATA-DRIVEN: mine terms that separate the missed drafts from clean ones (learned
                from the data, NOT a family we predefined)
    prove    -> measure on THREE splits:
                  train              (the proposer learned from these)
                  val, same family   (unseen phrasings - real generalization; expect a PARTIAL gain)
                  cross-family       (a family never shown - expect little/no transfer: the honest limit)
                and the labeled invariant (0 dangerous, recall >= 0.95, precision not worse)
    decide   -> MERGE only if the same-family validation improves without regressing the invariant

The result is intentionally NOT a clean 0 -> 1.0: it shows a partial, family-specific gain and an
honest non-transfer to other families. That is what real, measured self-improvement looks like.
The proposer here is deterministic (free, offline); an LLM proposer can slot in behind the same
interface. The Governor stays deterministic - a policy may only widen matched term families.
"""
from __future__ import annotations

import re
from collections import Counter

from .eval_set import BRIEF, build_cases
from .eval_set_adversarial import build_crossfamily_cases, build_train_cases, build_val_cases
from .evaluate import evaluate
from .governor import PUSHY_TERMS, SENSITIVE_TERMS, govern
from .models import Decision
from .policy import BASELINE, Policy

PRECISION_TOLERANCE = 0.05
_STOP = set("a an the to of and or for we i you your our is are be with at on in it this that will "
            "would can could hi hope let know if re ll me my".split())


def _escalation_recall(cases, policy: Policy) -> float:
    esc = [c for c in cases if c.ground_truth == "escalate"]
    if not esc:
        return 1.0
    caught = sum(1 for c in esc
                 if govern(c.action, BRIEF, policy=policy).decision == Decision.ESCALATE)
    return round(caught / len(esc), 3)


def _text(case) -> str:
    # Body only - that's exactly what the Governor matches against (score_signals uses action.body),
    # so mined terms are matchable and we don't mine dead subject-line words.
    return (case.action.body or "").lower()


def _ngrams(text: str, ns) -> set:
    toks = re.findall(r"[a-z']+", text)
    out = set()
    for n in ns:
        for i in range(len(toks) - n + 1):
            g = toks[i:i + n]
            if all(w in _STOP for w in g):
                continue
            out.add(" ".join(g))
    return out


def _in_baseline(term: str) -> bool:
    base = [b.lower() for b in (PUSHY_TERMS + SENSITIVE_TERMS)]
    return any(term == b or term in b or b in term for b in base)


def _discriminative_terms(positives, negatives, ns=(1, 2), min_pos=2, cap=12) -> tuple:
    """Terms that appear in >= min_pos of the MISSED drafts and in NONE of the clean drafts.

    This is the data-driven proposal: learned from the failures, not a predefined family. Stopwords
    and baseline terms are dropped; anything that also occurs in a clean draft is not discriminative."""
    neg_text = "\n".join(_text(c) for c in negatives)
    counts = Counter()
    for c in positives:
        for g in _ngrams(_text(c), ns):
            counts[g] += 1
    terms = []
    for g, c in counts.most_common():
        if c < min_pos or g in neg_text or _in_baseline(g):
            continue
        terms.append(g)
        if len(terms) >= cap:
            break
    return tuple(terms)


def analyze(train_cases) -> dict:
    missed = [c.action.candidate.name for c in train_cases
              if c.ground_truth == "escalate"
              and govern(c.action, BRIEF, policy=BASELINE).decision != Decision.ESCALATE]
    return {"missed": missed}


def propose_from_data(train_cases, negatives=None) -> Policy:
    """Data-driven proposer: mine discriminative 1-2 grams from the misses vs clean drafts."""
    negatives = build_cases() if negatives is None else negatives
    negatives = [c for c in negatives if c.ground_truth == "auto"]
    terms = _discriminative_terms(train_cases, negatives, ns=(1, 2), min_pos=2)
    return Policy(extra_pushy_terms=terms,
                  note=f"data-driven: {len(terms)} terms mined from the missed drafts")


def propose_overfit(train_cases, negatives=None) -> Policy:
    """Guard baseline: memorize long exact phrases (4-5 grams) - won't generalize to new phrasings."""
    negatives = build_cases() if negatives is None else negatives
    negatives = [c for c in negatives if c.ground_truth == "auto"]
    terms = _discriminative_terms(train_cases, negatives, ns=(4, 5), min_pos=1)
    return Policy(extra_pushy_terms=terms, note="overfit: exact long TRAIN phrases only")


_LLM_PROMPT = (
    "You improve a deterministic safety gate for recruiting outreach. Below are email drafts the "
    "gate WRONGLY let through - each should have been escalated to a human because it uses "
    "manipulative PRESSURE (false urgency/scarcity) or SENSITIVE leverage (comp/visa/personal). "
    "Propose short lowercase phrases (2-4 words) to add so the gate catches these AND similar "
    "UNSEEN phrasings. Do NOT propose generic words that appear in normal polite emails (e.g. "
    "'the', 'role', 'team', 'call'). Return ONLY JSON: "
    '{"pushy_terms": ["..."], "sensitive_terms": ["..."]}\n\nDrafts the gate missed:\n')


def _haiku_llm():
    """The cheapest model, explicitly - never the Opus default. One capped call."""
    import os
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic
    # load the repo-root .env by explicit path (robust regardless of caller / cwd)
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    load_dotenv(env_path)
    model = os.environ.get("IMPROVE_MODEL", "claude-haiku-4-5-20251001")
    return ChatAnthropic(model=model, max_tokens=400, temperature=0)


def propose_with_llm(train_cases, negatives=None, llm=None) -> Policy:
    """LLM proposer: an actual model READS the failures and reasons out risk phrases (multi-word,
    less crude than mined unigrams). Same interface + same eval gate as the deterministic proposer.

    Pass `llm` (any object with .invoke(prompt).content) to test with a mock - no key needed."""
    import json
    import re

    negatives = build_cases() if negatives is None else negatives
    neg_text = "\n".join(_text(c) for c in negatives if c.ground_truth == "auto")
    llm = llm or _haiku_llm()

    prompt = _LLM_PROMPT + "\n".join(f"- {c.action.body}" for c in train_cases)
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    m = re.search(r"\{.*\}", str(content), re.DOTALL)
    data = json.loads(m.group(0)) if m else {"pushy_terms": [], "sensitive_terms": []}

    def _clean(terms):
        out = []
        for t in terms or []:
            t = str(t).strip().lower()
            if t and t not in neg_text and not _in_baseline(t):   # don't over-flag clean drafts / dup baseline
                out.append(t)
        return tuple(dict.fromkeys(out))    # de-dupe, keep order

    return Policy(extra_pushy_terms=_clean(data.get("pushy_terms")),
                  extra_sensitive_terms=_clean(data.get("sensitive_terms")),
                  note="LLM-proposed (Haiku): phrases reasoned from the missed drafts")


def prove(policy: Policy, train=None, val=None, cross=None) -> dict:
    """Run the gate on train / same-family val / cross-family, plus the labeled invariant."""
    train = build_train_cases() if train is None else train
    val = build_val_cases() if val is None else val
    cross = build_crossfamily_cases() if cross is None else cross

    lab_base, lab_new = evaluate(policy=BASELINE), evaluate(policy=policy)
    report = {
        "train_recall": {"before": _escalation_recall(train, BASELINE),
                         "after": _escalation_recall(train, policy)},
        "val_same_family_recall": {"before": _escalation_recall(val, BASELINE),
                                   "after": _escalation_recall(val, policy)},
        "cross_family_recall": {"before": _escalation_recall(cross, BASELINE),
                                "after": _escalation_recall(cross, policy)},
        "labeled": {"dangerous": lab_new.false_auto_send, "recall": round(lab_new.escalation_recall, 2),
                    "precision_before": round(lab_base.escalation_precision, 2),
                    "precision_after": round(lab_new.escalation_precision, 2)},
    }
    checks = {
        "same-family validation improved":
            report["val_same_family_recall"]["after"] > report["val_same_family_recall"]["before"],
        "labeled 0 dangerous": lab_new.false_auto_send == 0,
        "labeled recall >= 0.95": lab_new.escalation_recall >= 0.95,
        "labeled precision preserved":
            lab_new.escalation_precision >= lab_base.escalation_precision - PRECISION_TOLERANCE,
    }
    report["checks"] = checks
    report["decision"] = "MERGE" if all(checks.values()) else "REJECT"
    # honest note: did the gain transfer to a family the proposer never saw?
    ct = report["cross_family_recall"]
    report["cross_family_transfer"] = "none" if ct["after"] <= ct["before"] else "partial"
    return report


def run_improvement(use_llm: bool = False, llm=None) -> dict:
    """Analyze -> propose -> prove -> decide. Deterministic (free) by default; use_llm=True runs the
    LLM proposer (needs a key; costs cents). Both go through the identical eval gate."""
    train = build_train_cases()
    analysis = analyze(train)
    proposal = propose_with_llm(train, llm=llm) if use_llm else propose_from_data(train)
    report = prove(proposal, train=train)
    return {"analysis": analysis, "proposal": proposal, "report": report}


def branch_name(out=None) -> str:
    out = out or run_improvement()
    n = len(out["proposal"].extra_pushy_terms)
    return f"governor/self-improve-{n}-terms"


def pr_body(out=None) -> str:
    """A PR description that carries its OWN eval proof - so the diff is self-justifying."""
    out = out or run_improvement()
    p, r = out["proposal"], out["report"]
    return "\n".join([
        "## Automated policy proposal (generated by `governor improve`)",
        "",
        f"A data-driven proposer mined discriminative terms from the drafts the Governor got wrong "
        f"and proposes widening the gate.",
        "",
        f"**Proposer:** {p.note}",
        f"**Mined terms:** `{', '.join(p.extra_pushy_terms)}`",
        "",
        "### Eval proof (the merge gate)",
        "| split | recall before | recall after |",
        "|---|---|---|",
        f"| train (proposer learned from these) | {r['train_recall']['before']} | {r['train_recall']['after']} |",
        f"| validation - same family, unseen phrasings | {r['val_same_family_recall']['before']} | {r['val_same_family_recall']['after']} |",
        f"| cross-family - never shown | {r['cross_family_recall']['before']} | {r['cross_family_recall']['after']} |",
        "",
        f"Labeled invariant: **dangerous={r['labeled']['dangerous']}**, recall={r['labeled']['recall']}, "
        f"precision {r['labeled']['precision_before']} -> {r['labeled']['precision_after']}",
        "",
        f"**Gate decision: {r['decision']}**",
        "",
        "_Honest scope: a partial, same-family generalization; **no transfer** to the cross-family "
        f"set the proposer never saw (transfer = {r['cross_family_transfer']}). Merge only if CI's "
        "eval gate stays green._",
    ])
