"""Step 6: single-file Streamlit UI for The Governor.

Left  = 5 Fillmore-style sourcing agents, live status.
Right = the shared approval queue a human works.
Top   = counters (autonomy %, human-facing, queue depth, DANGEROUS auto-sends).
Bottom= the eval scoreboard (static proof the judgment is correct).

The UI only polls Orchestrator.snapshot() — it never drives the agent threads.

Run:  ./.venv/bin/streamlit run app.py
"""
from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from governor.runner import Orchestrator
from governor.evaluate import evaluate

load_dotenv()

_DEMO_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "demo_cache.json")


@st.cache_data
def _cached_loop_traces() -> list[dict]:
    """The persisted autonomous-loop decision traces (docs/demo_cache.json).

    Lets loop mode SHOW the step-by-step trace with no key and no network — the same recorded
    run that drives the offline demo/video. Empty list if the cache hasn't been generated yet.
    """
    try:
        with open(_DEMO_CACHE) as f:
            return json.load(f).get("loop", [])
    except (OSError, ValueError):
        return []

st.set_page_config(page_title="The Governor", layout="wide", page_icon="🛡️")

# ---- session state ----
if "orc" not in st.session_state:
    st.session_state.orc = None

# ---- header ----
st.title("🛡️ The Governor")
st.caption(
    "The oversight & eval layer for autonomous recruiting agents. "
    "5 Fillmore-style sourcing agents propose outreach; the Governor decides which sends "
    "a human must see — reserving scarce human attention for the calls that need judgment."
)

# ---- sidebar controls ----
with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["governor", "fifo"],
                    format_func=lambda m: "🛡️ Governor (judgment layer)" if m == "governor"
                    else "📥 FIFO (every send hits the human)")
    source = st.radio(
        "Candidate source",
        ["labeled", "discovered", "agents"],
        format_func=lambda s: {
            "labeled": "🏷️ Labeled set (scoreboard proof)",
            "discovered": "🌐 Discovered (one shared search)",
            "agents": "🤖 Autonomous agents (each its own territory)",
        }[s],
        help="labeled = the 40 hand-labeled cases the eval scoreboard is built on. "
             "discovered = one shared GitHub search feeding all agents (a pipeline). "
             "agents = Level 2: each of the 5 agents autonomously searches its OWN "
             "territory, drafts, self-critiques, and proposes. Discovered/agent leads have "
             "no ground-truth labels, so they don't feed the correctness metrics.",
    )
    if source == "discovered":
        if st.button("🔎 Discover now (live GitHub search)", use_container_width=True):
            with st.spinner("Searching GitHub for real candidates matching the brief…"):
                from governor.discover import search_candidates, save, BRIEF
                try:
                    cands = search_candidates(BRIEF, limit=10)
                    save(cands)
                    st.session_state.discovered_preview = [
                        (c.name, c.current_company, c.match_confidence) for c in cands]
                    st.success(f"Discovered {len(cands)} real candidates → cached.")
                except Exception as e:
                    st.error(f"Discovery failed ({str(e)[:100]}). Cached set (if any) still used.")
        for name, co, m in st.session_state.get("discovered_preview", [])[:10]:
            st.caption(f"• {name} @ {co} — match {m:.2f}")
    agent_mode = "flow"
    if source == "agents":
        agent_mode = st.radio(
            "Agent execution",
            ["flow", "loop"],
            format_func=lambda a: {
                "flow": "⛓️ Fixed flow (deterministic pipeline)",
                "loop": "🔁 Autonomous loop (LLM decides + DECISION TRACE)",
            }[a],
            help="flow = the fixed search→draft→critique→propose pipeline (no visible reasoning). "
                 "loop = each agent runs an LLM tool-use loop and decides its OWN next action; its "
                 "decision trace is shown step by step below. With a key it's a live loop; with no "
                 "key it replays the cached trace (offline, deterministic).",
        )
    if source == "agents":
        st.caption("Each agent owns a territory and searches it live. Pre-warm caches so the "
                   "stage run replays offline:")
        if st.button("🔎 Discover all territories (live)", use_container_width=True):
            from governor.territories import TERRITORIES
            from governor.discover import search_by_query
            prev = []
            with st.spinner("5 agents searching their territories on GitHub…"):
                for terr in TERRITORIES:
                    try:
                        cands = search_by_query(terr["query"], limit=4)
                        prev.append(f"✅ {terr['goal']}: {len(cands)} leads")
                    except Exception as e:
                        prev.append(f"⚠ {terr['goal']}: {str(e)[:40]}")
            st.session_state.territory_preview = prev
            st.success("Territories cached — offline replay ready.")
        for line in st.session_state.get("territory_preview", []):
            st.caption(line)
    zero_mode = st.radio("Zero.xyz send", ["stub", "real"],
                         help="stub = demo-safe, no network. real = actually send via Zero.xyz.")
    use_llm = st.toggle(
        "🧠 Live LLM drafting (AWS Bedrock)",
        value=False,
        help="ON = each agent drafts the email via real Claude on Bedrock (needs AWS creds in .env). "
             "OFF = prewritten drafts. Flip OFF instantly if creds/network flake on stage — "
             "it's a demo-control switch, not an error safety net (that fallback is automatic).",
    )
    speed = st.slider("Agent step delay (s)", 0.05, 1.0, 0.25, 0.05)
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", use_container_width=True, type="primary"):
        orc = Orchestrator(mode=mode, zero_mode=zero_mode, step_delay=speed,
                           use_llm=use_llm, source=source, agent_mode=agent_mode)
        orc.start()
        st.session_state.orc = orc
    if c2.button("⟲ Reset", use_container_width=True):
        if st.session_state.orc:
            st.session_state.orc.stop()
        st.session_state.orc = None
    st.divider()
    st.markdown("**Sponsors:** AWS Bedrock · Zero.xyz · Akash")
    if use_llm:
        if os.environ.get("ANTHROPIC_API_KEY"):
            st.success("🧠 Live drafting ON via **Anthropic API** — agents draft with real "
                       "Claude (auto-falls back to templates on any error).")
        elif os.environ.get("AWS_ACCESS_KEY_ID"):
            st.success("🧠 Live drafting ON via **AWS Bedrock** — agents draft with real Claude.")
        else:
            st.warning("Live LLM drafting is ON but no key found. Add ANTHROPIC_API_KEY "
                       "(or AWS creds) to .env — otherwise agents fall back to templates.")
    if zero_mode == "real" and not os.environ.get("ZERO_API_URL"):
        st.warning("Zero real mode set but ZERO_API_URL missing — sends fall back to stub.")


def _badge(decision: str) -> str:
    return {"auto_send": "🟢 auto-sent", "escalate": "🟠 escalated",
            "hold": "🔵 held (load-shed)"}.get(decision, decision)


_TOOL_ICON = {"search_github": "🔎", "refine_search": "♻️", "draft_email": "✍️",
              "critique_draft": "🧐", "propose": "📤"}
_TERMINAL_LABEL = {
    "enough-candidates": "✅ enough candidates (4) — stop",
    "max-iterations": "⛔ max iterations hit — stop",
    "budget": "⛔ token budget spent — stop",
}


def _render_trace(label: str, goal: str, steps, terminal_reason: str, meta: str = ""):
    """Render ONE agent's decision trace: the ordered tool/input/observation/decision steps.

    This is the demo-critical view — a viewer watches the agent DECIDE its own next action,
    visibly distinct from the fixed flow (which shows no such reasoning)."""
    header = f"🤖 **{label}** — {goal}"
    if meta:
        header += f"  ·  {meta}"
    with st.expander(header, expanded=False):
        for s in steps:
            # steps may be TraceStep objects (live) or plain dicts (cached replay)
            g = (lambda k: getattr(s, k)) if not isinstance(s, dict) else s.get
            tool, tinput = g("tool"), g("tool_input")
            obs, decision, reason = g("observation"), g("decision"), g("reason")
            icon = _TOOL_ICON.get(tool, "•")
            arg = ""
            if isinstance(tinput, dict) and tinput:
                arg = " ".join(f"`{v}`" for v in tinput.values())
            flag = "🟢 continue" if decision == "continue" else "🔴 stop"
            st.markdown(f"**{g('index') + 1}. {icon} {tool}** {arg}")
            st.caption(f"observed: {obs}")
            st.caption(f"decision: {flag} — {reason}")
        st.markdown(f"**terminal reason:** {_TERMINAL_LABEL.get(terminal_reason, terminal_reason)}")


def _render_decision_traces(snap: dict):
    """Show the loop's decision traces — live if the loop ran with a key, else the cached replay."""
    st.subheader("🧠 Autonomous loop — decision trace")
    live = snap.get("traces") or {}
    if live:
        st.caption("Live loop: each agent chose its own next tool until it had 4 candidates.")
        for aid, tr in live.items():
            meta = f"{tr.iterations} steps · {tr.tokens_used} tokens"
            _render_trace(aid, tr.territory_key, tr.steps, tr.terminal_reason, meta)
        return
    cached = _cached_loop_traces()
    if cached:
        st.caption("Replaying the cached decision trace (no key / no network) — the same recorded "
                   "run that drives the offline demo. Add a key for a live loop.")
        for tr in cached:
            meta = f"{tr.get('iterations', len(tr['steps']))} steps · flavor {tr.get('flavor', '')}"
            _render_trace(tr["key"], tr.get("goal", ""), tr["steps"], tr["terminal_reason"], meta)
    else:
        st.info("No decision trace yet — run `python scripts/gen_web_cache.py` to seed the cached "
                "trace, or start a loop-mode run with a key.")


@st.fragment(run_every=1.0)
def live_dashboard():
    orc: Orchestrator | None = st.session_state.orc
    if orc is None:
        st.info("Pick a mode in the sidebar and hit **Start**. "
                "Try **FIFO** first (human drowns), then **Governor** (judgment layer).")
        return

    snap = orc.snapshot()

    # ---- counters ----
    m = st.columns(5)
    m[0].metric("Autonomy", f"{snap['autonomy_pct']:.0f}%")
    m[1].metric("Human-facing", f"{snap['human_facing']}", help="requests that reached a human")
    m[2].metric("Queue depth", snap["queue_depth"])
    m[3].metric("Auto-sent", snap["auto_sent"])
    dang = snap["dangerous_auto_sends"]
    m[4].metric("⚠ Dangerous auto-sends", dang, delta="target 0",
                delta_color="inverse" if dang else "off")
    st.progress(snap["processed"] / max(1, snap["total"]),
                text=f"processed {snap['processed']}/{snap['total']}")
    if dang:
        st.error(f"{dang} risky email(s) were auto-sent without a human — this is what FIFO-less "
                 f"autonomy risks. The Governor's job is to keep this at 0.")

    left, right = st.columns([1, 1.4])

    # ---- agents ----
    with left:
        st.subheader("Sourcing agents")
        for aid, status in snap["agent_status"].items():
            icon = "✅" if status == "done" else ("💤" if status == "idle" else "⚙️")
            st.write(f"{icon} **{aid}** — {status}")
        st.divider()
        st.subheader("Auto-handled")
        st.write(f"🟢 auto-sent: **{snap['auto_sent']}**   🔵 held: **{len(snap['held'])}**")
        with st.expander("held (deferred to protect human capacity)"):
            for it in snap["held"]:
                st.caption(f"{it.action.candidate.name} @ {it.action.candidate.current_company} "
                           f"— risk {it.risk:.2f} — {it.reasons[0] if it.reasons else ''}")

    # ---- queue ----
    with right:
        st.subheader(f"Approval queue ({snap['queue_depth']})")
        if not snap["human_queue"]:
            st.caption("empty — nothing needs a human right now.")
        for it in snap["human_queue"]:
            c = it.action.candidate
            with st.container(border=True):
                top = st.columns([3, 1])
                top[0].markdown(f"**{c.name}** — {c.current_title} @ {c.current_company}")
                top[1].markdown(f"risk **{it.risk:.2f}**")
                st.caption(f"{_badge(it.decision)} · {', '.join(it.reasons[:2])}")
                st.text(f"“{it.action.subject}” — {it.action.body[:90]}…")
                b = st.columns(2)
                if b[0].button("✅ Approve & send", key=f"a{it.action.seq}", use_container_width=True):
                    orc.approve(it.action.seq)
                    st.rerun()
                if b[1].button("🚫 Deny", key=f"d{it.action.seq}", use_container_width=True):
                    orc.deny(it.action.seq)
                    st.rerun()

    if snap.get("agent_mode") == "loop":
        st.divider()
        _render_decision_traces(snap)

    with st.expander("recent events"):
        for e in reversed(snap["events"]):
            st.caption(e)


live_dashboard()

# ---- eval scoreboard (static proof) ----
st.divider()
st.subheader("📊 Eval scoreboard — is the judgment correct?")
st.caption("The Governor's policy run against a hand-labeled ground-truth set. "
           "Not 'fewer approvals' — *measured* correctness.")
sb = evaluate()
e = st.columns(4)
e[0].metric("Dangerous auto-sends", sb.false_auto_send, delta="target 0",
            delta_color="inverse" if sb.false_auto_send else "off")
e[1].metric("Escalation recall", f"{sb.escalation_recall:.2f}", help="fraction of real risks caught")
e[2].metric("Escalation precision", f"{sb.escalation_precision:.2f}")
e[3].metric("Human-load saved", f"{sb.human_load_saved_pct:.0f}%")
st.code(sb.pretty(), language="text")
st.caption("Note: the labeled set is co-designed with the policy, so separation is expected — "
           "the honest next step is held-out / adversarial eval. (A good thing to say out loud.)")
