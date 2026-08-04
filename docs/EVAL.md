# Evaluation & Grading - current state and roadmap

> How this project measures whether the Governor's judgment is *correct* - what exists
> today, its honest limitations, and the concrete path to an industry-standard eval layer.

## TL;DR

Stopping an agent is trivial. The hard, valuable problem is **which** of an agent's real-world
actions can fire autonomously, and **proving** the policy that decides is right. This project
already measures that against a labeled set (recall 1.00, 0 dangerous auto-sends). That is a real
eval - but a deliberately honest reading is that it proves *consistency*, not *generalization*.
This doc is the roadmap from that starting point to an eval layer that matches how AI teams
actually validate agents in production.

## Status (implemented)

Phases 1-4 are built and shipping (run `governor evaluate`):

- **Held-out / adversarial set** (`eval_set_heldout.py`): 20 cases whose labels never touched the
  policy. Result today: recall **0.67**, precision **1.00**, **4 adversarial evasions the keyword
  gate misses** - reported honestly, not hidden.
- **LLM-as-judge on draft quality** (`judge.py`): a free deterministic `StubJudge` (default) plus
  an opt-in real `LLMJudge`. It flags the manipulative/pressuring drafts - including the held-out
  evasions the gate auto-sent - demonstrating the two layers are complementary.
- **Regression gate in CI** (`tests/test_eval_heldout.py`): the primary set's invariant
  (0 dangerous, recall >= 0.95) fails the build if it ever regresses.
- **Named eval framework - DeepEval** (`tests/test_deepeval_eval.py`, optional `[eval]` extra):
  the Governor's decisions and the draft-quality judge run as DeepEval `BaseMetric`s over
  `LLMTestCase`s and are asserted with `assert_test`, pytest-native and CI-gated. Offline and free
  (DeepEval is the harness; its own LLM metrics are not called).
- **Web view** (`docs/index.html`): the labeled + held-out scoreboards and the judge result -
  including the evasions the gate missed but the judge caught - render on the live demo page.

Phase 5 (calibration/observability) remains planned below.

## Why evaluation is the point of this project

The Governor is an oversight layer for autonomous agents. Its value is only as trustworthy as the
evidence that its decisions are correct. So the eval is not an add-on - it *is* the product's
claim. "Trust me" is not a deliverable; "here is the measured false-auto-send rate on held-out
adversarial cases" is.

## What exists today

- **A hand-labeled ground-truth set** (`eval_set.py`): ~40 proposed sends, each labeled
  `auto_send` / `escalate` with the reason.
- **A scoreboard** (`evaluate.py`): runs the deterministic Governor over the set and reports
  **escalation recall**, **escalation precision**, **false auto-sends (the "dangerous" count)**,
  and **human-load saved %**, plus a FIFO baseline for contrast.
- **A behavioral test suite**: the loop terminates within guardrails, emits valid proposals, and
  the Governor still yields 0 dangerous auto-sends on the labeled set (Governor-parity).

This is already better than "vibes": the correctness claim is a number against ground truth.

## Known limitations (the honest read)

1. **The labeled set is co-designed with the policy.** Labels and thresholds were tuned together,
   so clean separation is *expected*, not proof of generalization. This is stated in the app UI too.
2. **Small and not stratified** (~40 cases); no held-out split.
3. **Only the deterministic gate is evaluated** - not the *quality* of the generative drafts the
   LLM writes (tone, specificity, absence of manipulation).
4. **Bespoke script**, not a recognized eval harness - so no standard reporting, no regression
   history, no shared vocabulary with how teams actually run evals.
5. **The Governor being deterministic is a strength** (auditable, predictable oversight); the gap
   is in the *validation methodology*, not in the gate itself. This distinction matters.

## Roadmap to an industry-standard eval layer

Phased, each phase independently valuable and shippable.

### Phase 1 - Held-out / adversarial eval set  *(highest leverage)*
- **What:** a second labeled set whose cases were NOT used to design the policy, including
  adversarial edge cases (borderline competitor names, obfuscated pushy language, near-exec titles).
- **Why:** measures generalization, not memorization. Reporting the honest metric *drop* from the
  co-designed set to the held-out set is itself a credibility signal.
- **How:** `eval_set_heldout.py` + extend `evaluate.py` to report both sets side by side.

### Phase 2 - Adopt a recognized eval framework
- **What:** wire the labeled + held-out sets through an established harness rather than a custom
  script. Candidates: **promptfoo**, **DeepEval**, **Inspect (inspect_ai)**, **Braintrust**,
  **Langfuse/LangSmith**. (Ragas is RAG-specific and not a fit here.)
- **Why:** standard reporting, assertions, and vocabulary; signals fluency with the ecosystem.
- **How:** a config that defines test cases + assertions (e.g. "dangerous auto-sends == 0",
  "recall >= 0.95") and produces a report artifact.

### Phase 3 - LLM-as-judge for generative draft quality
- **What:** grade the *drafts* (the LLM-written part) against a rubric - relevance to the role,
  specificity, professional tone, and **no manipulative/pressuring language** - scored 1-5 by a
  judge model, plus a pass threshold.
- **Why:** the current eval only checks the gate; it never checks whether the outreach the agent
  writes is actually good. LLM-as-judge is the standard way teams eval generative output.
- **How:** a `judge` module with a rubric prompt. **Defaults to a free deterministic/stub judge**
  (keyword + heuristic) so CI and everyday runs cost nothing; a real judge model is **opt-in** via
  an API key. Cost when run for real: cents (Haiku over a few dozen short drafts).

### Phase 4 - Eval-in-CI as a regression gate
- **What:** run the (free, stub-judge) eval in CI on every change; fail the build if
  `dangerous > 0` or recall drops below threshold.
- **Why:** turns eval from a one-time number into a guarantee that survives future edits.
- **How:** extend `.github/workflows/ci.yml` with an eval step + threshold assertions.

### Phase 5 - Stretch: calibration & observability
- Larger stratified dataset; a confusion matrix and threshold calibration curve; inter-annotator
  agreement if multiple labelers; trace-level scoring of real runs via Langfuse/LangSmith.

## Cost & operating model

- **Default = free.** Deterministic scoreboard + stub judge + CI use no API and no network.
- **Real LLM-as-judge = opt-in**, a few cents on Haiku via a paid API key (never the subscription -
  apps cannot use a claude.ai subscription).
- Same pattern as the rest of the project: deterministic/offline by default, live is a mode you
  turn on.

## Definition of done (per phase)

- **P1:** `governor evaluate` reports labeled AND held-out metrics; the honest drop is documented.
- **P2:** eval runs through the chosen framework and emits a standard report; thresholds asserted.
- **P3:** a judge scores drafts against the rubric; stub judge in CI, real judge opt-in; tests pin
  the rubric contract.
- **P4:** CI fails on a regression (dangerous > 0 or recall below threshold).
- **P5:** calibration/observability artifacts exist.

## One-line interview framing

> "The deterministic gate is validated today by a labeled scoreboard; the honest next step -
> which is scaffolded here - is held-out + adversarial eval with LLM-as-judge for draft quality,
> gated in CI."
