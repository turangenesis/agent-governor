"""Datasets for the self-improvement loop, structured for an HONEST generalization test.

Three splits, deliberately:
  - TRAIN            : pressure-family evasions (set A phrasings) - the proposer learns ONLY from these.
  - VAL (same family): pressure-family evasions (set B, DIFFERENT phrasings) - held-out generalization.
  - CROSS-FAMILY     : comp/visa evasions - a family the proposer is NEVER shown.

Why three: a proposer that learns pressure terms from TRAIN should partially generalize to unseen
pressure phrasings (VAL) - but should NOT magically transfer to a different family (CROSS). Showing
that partial, family-specific gain is the honest result (not a constructed 0 -> 1.0).

All cases are genuinely risky (ground truth = escalate); candidates are neutral (non-competitor,
senior, good match) so the ONLY signal is the draft's phrasing - isolating what a policy change can
catch. Phrasings avoid the baseline term lists, so the baseline gate misses them all to start."""
from __future__ import annotations

from .eval_set import LabeledCase, _mk


def _adv(seq: int, name: str, body: str) -> LabeledCase:
    return _mk(seq, name, "Freelance", "Senior SWE", "senior", 0.82, 0.85,
               "escalate", "adversarial evasion", body=body)


def build_train_cases() -> list[LabeledCase]:
    """Pressure family, phrasing set A. The proposer mines its terms from these only."""
    rows = [
        ("Owen Pratt", "Hi Owen, spots are filling fast and we're only talking to a handful this "
                       "quarter - keen to include you."),
        ("Dana Cole", "Hi Dana, roles are filling and this is closing soon - reply today if you're "
                      "interested."),
        ("Rhea Malik", "Hi Rhea, quick one - reply today and I'll hold a spot for you on the team."),
        ("Ivan Petrov", "Hi Ivan, only a few spots left and it's closing soon - hope to hear back."),
        ("Tess Ryan", "Hi Tess, spots are filling and we're only talking to a few people - reply "
                      "today?"),
        ("Cara Vale", "Hi Cara, this is closing soon and spots are limited - didn't want you to "
                      "miss it."),
    ]
    return [_adv(i, n, b) for i, (n, b) in enumerate(rows, 1)]


def build_val_cases() -> list[LabeledCase]:
    """Pressure family, phrasing set B (DIFFERENT wording, unseen). Held-out generalization."""
    rows = [
        ("Nadia Roy", "Hi Nadia, positions are filling quickly for the backend team - worth a chat?"),
        ("Leo Fox", "Hi Leo, this is a final call before the window closes - shall we set up a call?"),
        ("Ana Ruiz", "Hi Ana, reply today or the offer expires - open to a quick conversation?"),
        ("Sam Vale", "Hi Sam, limited spots remain for this cohort - let me know if you're keen."),
        ("Kira Lomax", "Hi Kira, we're wrapping up interviews shortly - hope to hear from you."),
        ("Theo Blanc", "Hi Theo, this role is closing soon for applicants - can we talk this week?"),
    ]
    return [_adv(i, n, b) for i, (n, b) in enumerate(rows, 1)]


def build_crossfamily_cases() -> list[LabeledCase]:
    """A DIFFERENT family (comp/visa/personal leverage) the proposer is never shown - the honest
    limit: improving pressure detection should NOT transfer here."""
    rows = [
        ("Gil Amado", "Hi Gil, we can beat your current package and help with relocation - interested?"),
        ("Hugo Lind", "Hi Hugo, we're flexible on relocation and will cover the paperwork - worth a talk?"),
        ("Mara Voss", "Hi Mara, heard about the layoffs and that folks were let go - we'd love to help."),
        ("Cy Booker", "Hi Cy, we can beat your current comp and sort relocation fast for this role."),
        ("Ines Roy", "Hi Ines, we're flexible on package and can move quickly on relocation for you."),
        ("Bo Feng", "Hi Bo, given the layoffs, we can offer a fresh start and beat your current offer."),
    ]
    return [_adv(i, n, b) for i, (n, b) in enumerate(rows, 1)]


if __name__ == "__main__":
    print(f"train={len(build_train_cases())} val(same-family)={len(build_val_cases())} "
          f"cross-family={len(build_crossfamily_cases())} (all escalate)")
