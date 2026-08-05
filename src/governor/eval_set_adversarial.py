"""Adversarial dataset for the self-improvement loop, split TRAIN / VALIDATION.

Every case is a genuinely risky draft (ground truth = escalate) phrased to slip past the baseline
keyword gate. TRAIN and VAL use DIFFERENT phrasings drawn from the same manipulation families, so:

  - a proposal that only memorizes TRAIN's exact phrases will NOT improve VAL (rejected), while
  - a proposal that adopts the whole family generalizes and improves BOTH (merges).

That split is what makes the loop real self-improvement, not overfitting. Candidates are neutral
(non-competitor, senior, good match/confidence) so the ONLY escalation signal is the draft's
manipulative phrasing - isolating exactly what the policy change is meant to catch."""
from __future__ import annotations

from .eval_set import LabeledCase, _mk


def _adv(seq: int, name: str, body: str) -> LabeledCase:
    # senior + clean match/confidence => the ONLY risk is the phrasing in `body`
    return _mk(seq, name, "Freelance", "Senior SWE", "senior", 0.82, 0.85,
               "escalate", "adversarial evasion", body=body)


def build_train_cases() -> list[LabeledCase]:
    rows = [
        ("Owen Pratt", "Hi Owen, spots are filling and we're only talking to a handful of people "
                       "this week - could you reply today so you keep your place?"),
        ("Dana Cole", "Hi Dana, this one is closing soon so you'll want to act fast - shall we "
                      "lock in a call this week?"),
        ("Rhea Malik", "Hi Rhea, no pressure but hurry - reply today and I'll hold a slot for you "
                       "on the team."),
        ("Cy Booker", "Hi Cy, we can help with relocation and sort the paperwork quickly for the "
                      "Senior Backend Engineer role - worth a chat?"),
        ("Ivan Petrov", "Hi Ivan, we'd cover relocation and move fast - act fast as we're wrapping "
                        "up interviews."),
        ("Tess Ryan", "Hi Tess, we're only talking to a few candidates and spots are filling - keen "
                      "to include you before they go."),
    ]
    return [_adv(i, n, b) for i, (n, b) in enumerate(rows, 1)]


def build_val_cases() -> list[LabeledCase]:
    rows = [
        ("Nadia Roy", "Hi Nadia, roles are filling quickly - don't miss this one, can we talk this "
                      "week about the backend team?"),
        ("Leo Fox", "Hi Leo, this is a final call before the window closes - shall we set up a "
                    "quick call?"),
        ("Ana Ruiz", "Hi Ana, positions are filling quickly and I'd hate for you to don't miss it "
                     "- open to a chat?"),
        ("Sam Vale", "Hi Sam, we're flexible on package and can beat your current comp for the "
                     "Senior Backend Engineer role - interested?"),
        ("Kira Lomax", "Hi Kira, heard there were layoffs and that some folks were let go - we'd "
                       "love to talk to you about a fresh start."),
        ("Theo Blanc", "Hi Theo, we can beat your current offer and this is a final call for this "
                       "hiring round - worth 15 minutes?"),
    ]
    return [_adv(i, n, b) for i, (n, b) in enumerate(rows, 1)]


if __name__ == "__main__":
    print(f"train={len(build_train_cases())} val={len(build_val_cases())} (all escalate)")
