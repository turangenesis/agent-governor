# 🛡️ The Governor

**The oversight & eval layer for autonomous recruiting agents.**

![CI](https://github.com/turangenesis/agent-governor/actions/workflows/ci.yml/badge.svg)
&nbsp;![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
&nbsp;![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

### ▶ Live demo: **https://turangenesis.github.io/agent-governor/**

A recorded run of the five agents plays instantly in your browser - fully offline, no key, no
network. Toggle **Flow** vs **🔁 Loop** to watch the agents either follow a fixed pipeline or
**decide their own next step**. The full system (real GitHub sourcing + optional live Claude
drafting) runs from the source in this repo.

## Architecture

```mermaid
flowchart TD
  B["Hiring brief: role, must-haves, competitors"] --> S["split into 5 territories"]
  S --> A["5 autonomous agents - each runs its own tool-use loop:<br/>search -> draft -> critique -> propose"]
  A --> G["Governor (deterministic, no LLM):<br/>score risk -> auto-send / escalate / hold"]
  G --> H["Human queue: only the calls that need judgment"]
  G --> E["Eval scoreboard: recall 1.00 · 0 dangerous · 62% autonomy"]
```

Five Fillmore-style sourcing agents each want to send cold outreach. The **Governor** — a
plain-Python judgment layer (no LLM) — decides which sends fire autonomously and which a human
recruiter must see, reserving scarce human attention for the calls that need judgment.

Stopping an agent is trivial (an `if`). The hard, valuable problem is the **judgment layer**:
*which* of an agent's real-world actions deserve a bounded human's attention — and how do you
*measure* that the policy is right. That's this demo.

## The result

```
40 proposed sends →  FIFO:      human sees all 40   | autonomy 0%
                     GOVERNOR:  human sees ~7       | autonomy 62% | held (deferred) 8
  DANGEROUS auto-sends: 0   (recall 1.00 — never let a risky email through)
  false escalations:   2   (deliberate over-caution: a wasted glance << a bad send)
```

Under live load the Governor **degrades by deferring (HOLD), never by unsafely auto-sending** —
the 0-dangerous invariant holds even when the human is saturated.

Beyond that headline, `governor evaluate` also runs a **held-out / adversarial set** (measures
generalization - and honestly surfaces the evasions the keyword gate misses) and an
**LLM-as-judge** pass that grades the *drafts* themselves for tone and manipulation (free
deterministic judge by default; real judge opt-in). See [`docs/EVAL.md`](docs/EVAL.md) for the
methodology and roadmap.

## Stack

- **Python 3.12** - deterministic policy core (no LLM); `langgraph` + `langchain` power the agent tool-use loop.
- **AWS Bedrock / Anthropic API** - optional live Claude drafting (`ChatBedrockConverse` / `ChatAnthropic`).
- **Zero.xyz** - the real-world action layer the Governor gates (the send is stubbed in the demo).
- **Streamlit** (local dashboard) + **GitHub Pages** (hosted, offline demo).
- **pytest** - test suite; a dedicated eval scoreboard measures the policy's correctness.

## Run locally

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -e ".[demo]"   # core + Streamlit/LangGraph/Bedrock stack
cp .env.example .env                    # optional: fill in AWS + Zero for live services

# deterministic core via the `governor` CLI (no creds, no network):
./.venv/bin/governor evaluate           # the eval scoreboard (FIFO vs Governor)
./.venv/bin/governor discover           # replay cached discovered candidates

# autonomous LLM tool-use loop — each agent decides its own next action:
./.venv/bin/governor run --source agents --mode loop           # live (needs a key)
./.venv/bin/governor run --source agents --mode loop --replay  # cached, no key/network

# individual component checks (run as package modules):
./.venv/bin/python -m governor.agent --dry   # agent graph wiring
./.venv/bin/python -m governor.governor      # judgment policy on hand cases
./.venv/bin/python -m governor.runner        # FIFO vs Governor, headless

# the demo:
./.venv/bin/streamlit run app.py
```

The deterministic core (`governor evaluate` / `discover`) needs no third-party stack —
`pip install -e .` alone is enough for it. The heavier agent/UI stack lives under the
optional `demo` extra shown above.

In the UI: hit **Start** in **FIFO** first (human drowns in 40), then **Governor**
(judgment layer + load-shedding). Work the queue with Approve/Deny.

## Autonomous loop mode (agents decide their own steps)

The default agent flow is a fixed pipeline (Python decides the steps; the LLM only writes
text). **Loop mode** turns each territory agent into a genuine LLM tool-use loop: the model
itself chooses the next action each step — `search_github` → `draft_email` → `critique_draft`
→ `propose` — observing the result between steps and continuing until it has ~4 candidates,
then stopping. Guardrails (max-iterations, `max_candidates=4`, token budget) guarantee it
always terminates. The **Governor stays deterministic and unchanged** — it gates whatever the
loop proposes, so the 0-dangerous invariant holds regardless of how the candidates were found.

Every agent emits a first-class **decision trace**: an ordered log of each step's tool, input,
observed result, and the continue/stop decision — ending in an explicit terminal reason
(`enough-candidates` / `max-iterations` / `budget`). This is a structured action log, not the
model's hidden reasoning, so the autonomy is visible and auditable.

```bash
# a genuine autonomous loop per agent (needs a key; falls back to the fixed flow offline):
./.venv/bin/governor run --source agents --mode loop

# replay the recorded traces deterministically — no key, no network, reliable for a demo/video:
./.venv/bin/governor run --source agents --mode loop --replay
```

Traces surface step by step in the Streamlit app, in the [live web demo](https://turangenesis.github.io/agent-governor/)
(toggle **🔁 Loop**), and at the CLI above — all replaying from `docs/demo_cache.json` with no
credentials. A live key produces a fresh trace; the cached trace drives the offline demo.

## Layout

The importable package lives under `src/governor/` (installed via `pip install -e .`);
`app.py` is the Streamlit entry point and `scripts/` holds the web-cache generator.

For the module-by-module map (which file owns which concern) and the full system flow, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Runs fully offline

The default demo needs **no credentials and no network** — candidate search replays from a
local cache, drafting uses templates (no LLM), and sends are stubbed. Live services (real
Claude drafting, real Zero.xyz sends) are strictly opt-in and never required.

**To have agents draft with real Claude:** add an `ANTHROPIC_API_KEY` (a paid key from
[console.anthropic.com](https://console.anthropic.com) — not a claude.ai subscription, which
apps can't use) to `.env`, then toggle **Live LLM drafting** on in the UI. AWS Bedrock works
too — see `.env.example`. On any error it silently falls back to templates.

## Deploy a public demo (Streamlit Community Cloud)

The app runs with zero secrets, so a public deploy is straightforward:

1. Push this repo to a **public GitHub repo** (secrets and private notes are already
   `.gitignore`d — see the file list below).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → point it at your
   repo and `app.py` → **Deploy**. You get a public URL in ~2 minutes.
3. In the deployed app, use the **Labeled set** source (synthetic candidates) — it tells the
   full governance story with no real individuals shown and needs no cache files.

No environment variables are required. To also enable live GitHub discovery on the deploy,
add a `GITHUB_TOKEN` in Streamlit **Secrets** (optional; the labeled demo doesn't need it).
