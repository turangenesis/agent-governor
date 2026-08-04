"""Feature B: the AUTONOMOUS tool-use LOOP (as opposed to the fixed FLOW in sourcing_agent.py).

Here the LLM itself decides, each step, which tool to call — search / refine / draft / critique
/ propose — observing the result between steps until it has ~4 candidates. The control flow is
NOT fixed Python; the model drives it. That is the difference between a flow and a loop.

Loop engine: LangChain `bind_tools` + a bounded while-loop (no new heavy dependency; langchain
is already in the demo stack). Governance stays OUTSIDE and downstream: every proposal this loop
emits is still gated by the unchanged deterministic Governor.

Guardrails guarantee termination (a loop that can't stop is a bug, not a feature):
  - max_candidates (=4)  : once it has proposed enough, it stops (terminal reason: enough-candidates)
  - max_iterations       : hard cap on LLM turns                (terminal reason: max-iterations)
  - token_budget         : hard cap on tokens spent             (terminal reason: budget)

DECISION TRACE: every tool call becomes a TraceStep (tool, input, observation summary, and the
continue/stop decision + reason). The ordered list of steps is the replayable proof that the
agent decided its own path. Offline/no-key runs use the deterministic flow instead (see
sourcing_agent.py); this module is only used when a real tool-calling LLM is available.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .eval_set import BRIEF
from .models import Candidate, LoopResult, ProposedAction, TERMINAL_REASONS, TraceStep
from .sourcing_agent import PER_TERRITORY, _critique, _draft
from .discover import search_by_query

MAX_ITERATIONS = 20      # hard cap on LLM turns before force-terminate (fits ~4 candidates
                         # at one tool/turn: 1 search + 4x(draft+critique+propose) + slack)
MAX_CANDIDATES = 4       # proposals per territory (== PER_TERRITORY)
TOKEN_BUDGET = 40_000    # hard cap on tokens spent per agent loop. Sized so a LIVE run reaches
                         # ~4 candidates (measured ~24-29k over the 13-16 turns a full arc needs,
                         # as summed input+output grows each turn) and terminates on enough-candidates
                         # rather than tripping this wall — while still bounding a runaway loop
                         # (MAX_ITERATIONS caps turns first; this catches pathological token growth).
_EST_TOKENS_PER_TURN = 500   # fallback per-turn cost when the LLM reports no usage metadata


# --- tool schemas (bound to the LLM via bind_tools; the class name IS the tool name) ---
class search_github(BaseModel):
    """Search GitHub for candidate engineers matching a query string."""
    query: str = Field(description="a GitHub user search query, e.g. 'language:python type:user'")


class refine_search(BaseModel):
    """Broaden or change the query to surface more/different leads when too few were found."""
    query: str = Field(description="the new, broader GitHub user search query to try")


class draft_email(BaseModel):
    """Draft an outreach email to one already-found candidate (referenced by name)."""
    candidate_name: str = Field(description="the exact name of a lead returned by a prior search")


class critique_draft(BaseModel):
    """Self-critique the current draft for a candidate to decide if it is specific enough to send."""
    candidate_name: str = Field(description="the candidate whose draft to critique")


class propose(BaseModel):
    """Propose the drafted outreach for a candidate to the Governor for a send/escalate decision."""
    candidate_name: str = Field(description="the candidate whose drafted outreach to propose")


_TOOLS = [search_github, refine_search, draft_email, critique_draft, propose]

SYSTEM_PROMPT = (
    "You are an autonomous sourcing agent that owns ONE recruiting territory. Reach your GOAL by "
    "choosing one tool at a time: search_github to find leads, refine_search to broaden when a "
    "search returns too few, draft_email then critique_draft to write and check outreach, and "
    "propose to send a finished draft to the Governor. Work one candidate at a time. STOP once you "
    "have proposed {max_candidates} candidates. Always call a tool; never answer in prose."
)


class SourcingLoop:
    """One territory agent running as an LLM tool-use loop, emitting a full decision trace."""

    def __init__(self, territory: dict, llm, *, use_llm: bool = True,
                 search_fn=search_by_query, brief=BRIEF,
                 max_iterations: int = MAX_ITERATIONS, max_candidates: int = MAX_CANDIDATES,
                 token_budget: int = TOKEN_BUDGET, on_proposal=None):
        self.territory = territory
        self.llm = llm
        self.use_llm = use_llm
        self.search_fn = search_fn
        self.brief = brief
        self.max_iterations = max_iterations
        self.max_candidates = max_candidates
        self.token_budget = token_budget
        self.on_proposal = on_proposal

        self.flavor = territory.get("flavor", "clean")
        self.leads: dict[str, object] = {}          # name -> Candidate (insertion-ordered)
        self.drafts: dict[str, tuple] = {}          # name -> (subject, body, conf, concerns)
        self.proposals: list[ProposedAction] = []
        self.steps: list[TraceStep] = []
        self.tokens_used = 0

    # --- tool implementations ---
    def _do_search(self, query: str, *, refine: bool) -> str:
        query = query or (self.territory["broaden"] if refine else self.territory["query"])
        found = self.search_fn(query, limit=self.max_candidates)
        new = [c for c in found if c.name not in self.leads]
        for c in new:
            self.leads[c.name] = c
        names = ", ".join(c.name for c in new) or "none"
        verb = "refined search" if refine else "search"
        return f"{verb} '{query[:48]}' -> {len(new)} new leads ({names}); {len(self.leads)} leads total"

    def _do_draft(self, name: str) -> str:
        cand = self.leads.get(name)
        if cand is None:
            return f"no lead named '{name}'; search first"
        subj, body, conf, concerns = _draft(cand, self.llm, self.use_llm, flavor=self.flavor)
        self.drafts[name] = (subj, body, conf, concerns)
        return f"drafted for {name}: subject '{subj[:40]}' ({len(body)} chars, conf {conf:.2f})"

    def _do_critique(self, name: str) -> str:
        cand = self.leads.get(name)
        draft = self.drafts.get(name)
        if cand is None or draft is None:
            return f"nothing to critique for '{name}'; draft first"
        weak, why = _critique(cand, draft[1])
        return f"critique of {name}: {'WEAK - ' + why if weak else 'specific and sendable'}"

    def _do_propose(self, name: str) -> str:
        cand = self.leads.get(name)
        if cand is None:
            return f"no lead named '{name}'; cannot propose"
        if len(self.proposals) >= self.max_candidates:
            return "already have enough candidates; not proposing more"
        if name not in self.drafts:                       # allow propose without an explicit draft step
            self._do_draft(name)
        subj, body, conf, concerns = self.drafts[name]
        pa = ProposedAction("", cand, subj, body, conf, list(concerns))
        self.proposals.append(pa)
        if self.on_proposal:
            self.on_proposal(pa)
        return f"proposed {name} to the Governor ({len(self.proposals)}/{self.max_candidates})"

    def _dispatch(self, name: str, args: dict) -> str:
        args = args or {}
        if name == "search_github":
            return self._do_search(args.get("query", ""), refine=False)
        if name == "refine_search":
            return self._do_search(args.get("query", ""), refine=True)
        if name == "draft_email":
            return self._do_draft(args.get("candidate_name", ""))
        if name == "critique_draft":
            return self._do_critique(args.get("candidate_name", ""))
        if name == "propose":
            return self._do_propose(args.get("candidate_name", ""))
        return f"unknown tool '{name}'"

    # --- guardrails: decide continue vs stop after each executed tool call ---
    def _guardrail(self, iteration: int) -> tuple[str, str, bool]:
        if len(self.proposals) >= self.max_candidates:
            return "stop", "enough-candidates", True
        if self.tokens_used >= self.token_budget:
            return "stop", "budget", True
        if iteration + 1 >= self.max_iterations:
            return "stop", "max-iterations", True
        return "continue", f"{len(self.proposals)}/{self.max_candidates} candidates, keep going", False

    def _account_tokens(self, resp) -> None:
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            self.tokens_used += int(usage.get("total_tokens")
                                    or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
        else:
            self.tokens_used += _EST_TOKENS_PER_TURN

    # --- the bounded loop ---
    def run(self) -> LoopResult:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        except Exception as e:                            # pragma: no cover - langchain always present in demo stack
            raise RuntimeError("langchain-core is required for loop mode") from e

        bound = self.llm.bind_tools(_TOOLS) if hasattr(self.llm, "bind_tools") else self.llm
        goal = (f"GOAL: {self.territory['goal']}. Suggested first query: {self.territory['query']}. "
                f"Hiring for {self.brief.role} at {self.brief.hiring_company}.")
        messages = [
            SystemMessage(SYSTEM_PROMPT.format(max_candidates=self.max_candidates)),
            HumanMessage(goal),
        ]

        terminal = ""
        iteration = 0
        while iteration < self.max_iterations:
            resp = bound.invoke(messages)
            self._account_tokens(resp)
            messages.append(resp)

            tool_calls = getattr(resp, "tool_calls", None) or []
            if not tool_calls:
                # No tool chosen: nudge once and count the turn so max_iterations still bounds us.
                messages.append(HumanMessage("Call one of the available tools to make progress."))
                iteration += 1
                if self.tokens_used >= self.token_budget:
                    terminal = "budget"
                    self.steps.append(TraceStep(len(self.steps), "(none)", {},
                                                "no tool call", "stop", "budget"))
                    break
                continue

            stop = False
            for tc in tool_calls:
                obs = self._dispatch(tc.get("name", ""), tc.get("args", {}))
                messages.append(ToolMessage(content=obs, tool_call_id=tc.get("id", "")))
                decision, reason, stop = self._guardrail(iteration)
                self.steps.append(TraceStep(len(self.steps), tc.get("name", ""),
                                            dict(tc.get("args", {})), obs, decision, reason))
                if stop:
                    terminal = reason
                    break
            if stop:
                break
            iteration += 1
        else:
            # while-loop exhausted its iteration budget without an inner stop.
            terminal = "max-iterations"

        assert terminal in TERMINAL_REASONS, f"invalid terminal reason {terminal!r}"
        return LoopResult(
            territory_key=self.territory.get("key", ""),
            steps=self.steps,
            proposals=self.proposals,
            terminal_reason=terminal,
            tokens_used=self.tokens_used,
            iterations=iteration + 1,
            leads=list(self.leads.values()),   # persisted with the trace for self-contained replay
        )


def run_loop_agent(territory: dict, llm, *, use_llm: bool = True,
                   search_fn=search_by_query, on_proposal=None, **kw) -> LoopResult:
    """Run one territory agent as an autonomous tool-use loop; returns proposals + decision trace."""
    return SourcingLoop(territory, llm, use_llm=use_llm, search_fn=search_fn,
                        on_proposal=on_proposal, **kw).run()


# --- deterministic offline REPLAY of a persisted decision trace -----------------------------
#
# A live key produces a fresh trace; that trace is persisted (see save_traces). To drive a
# reliable demo/video with NO key and NO network, we replay the recorded tool-call sequence
# through the very same SourcingLoop. Because the loop re-dispatches the identical tools in the
# identical order over the offline (cached) search + deterministic (use_llm=False) drafting, it
# reproduces the SAME proposals — proving the autonomous run is deterministically replayable.
class ReplayLLM:
    """A no-network LLM stand-in that replays a recorded trace's tool calls, one per invoke().

    It ignores the conversation and simply re-emits the tools the real model chose, in order.
    After the recorded calls are exhausted it returns no tool call, so the loop's own guardrails
    terminate it. Tokens are reported as zero: replay reproduces PROPOSALS (which depend only on
    tool dispatch), never re-charging a token budget.
    """

    def __init__(self, steps):
        self._calls = []
        for s in steps:
            tool = getattr(s, "tool", None) if not isinstance(s, dict) else s.get("tool")
            args = getattr(s, "tool_input", None) if not isinstance(s, dict) else s.get("tool_input")
            if tool and tool != "(none)":
                self._calls.append((tool, dict(args or {})))
        self._i = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        zero = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if self._i >= len(self._calls):
            return AIMessage(content="", tool_calls=[], usage_metadata=zero)
        name, args = self._calls[self._i]
        self._i += 1
        return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                                  "id": f"replay-{self._i}", "type": "tool_call"}],
                         usage_metadata=zero)


def replay_trace(trace, territory, *, search_fn=None, on_proposal=None,
                 max_iterations: int = MAX_ITERATIONS, max_candidates: int = MAX_CANDIDATES,
                 token_budget: int = TOKEN_BUDGET) -> LoopResult:
    """Re-run a territory loop from a persisted trace with NO key and NO network.

    `trace` may be a LoopResult, its serialized dict, or a raw list of trace steps/step-dicts.
    Returns a fresh LoopResult whose proposals reproduce the recorded run's proposals.

    Self-contained by default: when no `search_fn` is given, the recorded search tool is served
    from the trace's OWN persisted `leads` — replay never touches the discovery cache or network.
    A caller may still inject a `search_fn` (the tests do, to exercise a fake search offline).
    If neither a `search_fn` nor persisted `leads` are available, this raises instead of silently
    falling through to a live search and truncating the replay to zero proposals.
    """
    raw_leads = []
    if isinstance(trace, LoopResult):
        steps = trace.steps
        raw_leads = trace.leads
    elif isinstance(trace, dict):
        steps = trace.get("steps", [])
        raw_leads = trace.get("leads", [])
    else:
        steps = trace

    if search_fn is None:
        leads = [c if isinstance(c, Candidate) else Candidate(**c) for c in raw_leads]
        if not leads:
            raise ValueError(
                "replay_trace: this trace has no persisted 'leads', so it cannot replay offline "
                "without the network. Regenerate the trace cache (LoopResult now embeds 'leads'), "
                "or pass an explicit search_fn.")

        def search_fn(query, limit=max_candidates, **_kw):   # offline: serve the recorded leads
            return list(leads)

    return SourcingLoop(
        territory, ReplayLLM(steps), use_llm=False, search_fn=search_fn, on_proposal=on_proposal,
        max_iterations=max_iterations, max_candidates=max_candidates, token_budget=token_budget,
    ).run()


# --- deterministic OFFLINE PLANNER: seed a canonical trace with no key and no network ---------
#
# A live key produces a genuine trace by the model's own reasoning. But the cached demo/video
# must show a decision trace with NO key at all. This planner is a deterministic stand-in that
# drives the SAME loop the real model would: it picks the next tool purely from the conversation
# so far (stateless, thread-safe), following the canonical path search -> for each lead
# draft -> critique -> propose, until the loop's own guardrail stops at enough-candidates.
#
# It is NOT the no-key runtime fallback (that stays the deterministic FLOW). It exists only to
# GENERATE the persisted trace for the offline demo cache, which then replays via replay_trace.
class OfflinePlannerLLM:
    """Deterministic, stateless tool-call planner for seeding a canonical decision trace offline.

    Reads only the ToolMessage observations already in the conversation to decide the next tool,
    so one instance is safe to reuse across territories/threads (like the real shared ChatAnthropic).
    """

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage, ToolMessage

        obs = [m.content for m in messages if isinstance(m, ToolMessage)] \
            if isinstance(messages, list) else []

        def emit(name, args):
            return AIMessage(content="", tool_calls=[
                {"name": name, "args": args, "id": "planner", "type": "tool_call"}],
                usage_metadata={"input_tokens": 60, "output_tokens": 40, "total_tokens": 100})

        # 1) no search yet -> search the territory (empty query -> loop uses the territory default)
        if not any(o.startswith(("search", "refined")) for o in obs):
            return emit("search_github", {})

        # 2) collect leads discovered so far, in order
        leads: list[str] = []
        for o in obs:
            if "new leads (" in o:
                inside = o.split("new leads (")[1].split(")")[0]
                if inside != "none":
                    for nm in inside.split(", "):
                        if nm not in leads:
                            leads.append(nm)

        # 3) advance the current lead through draft -> critique -> propose
        proposed = sum(1 for o in obs if "proposed" in o and "Governor" in o)
        if proposed >= len(leads):
            return emit("search_github", {})          # nothing left; guardrail terminates us
        cand = leads[proposed]
        if not any(o.startswith(f"drafted for {cand}") for o in obs):
            return emit("draft_email", {"candidate_name": cand})
        if not any(o.startswith(f"critique of {cand}") for o in obs):
            return emit("critique_draft", {"candidate_name": cand})
        return emit("propose", {"candidate_name": cand})


def canonical_trace(territory: dict, *, search_fn=search_by_query, on_proposal=None,
                    max_candidates: int = MAX_CANDIDATES) -> LoopResult:
    """Produce one territory's canonical decision trace deterministically, with no key/network.

    Runs the real SourcingLoop driven by the OfflinePlannerLLM over the offline (cached) search,
    so the trace is genuine loop output that also replays identically via replay_trace.
    """
    return SourcingLoop(territory, OfflinePlannerLLM(), use_llm=False, search_fn=search_fn,
                        on_proposal=on_proposal, max_candidates=max_candidates).run()


def save_traces(results, path: str) -> None:
    """Persist decision traces to the cached-replay JSON (mirrors the discover cache pattern)."""
    import json

    payload = {"traces": [r.to_dict() for r in results]}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_traces(path: str) -> list[LoopResult]:
    """Load persisted decision traces back into LoopResult objects for offline replay."""
    import json

    with open(path) as f:
        payload = json.load(f)
    return [LoopResult.from_dict(t) for t in payload.get("traces", [])]


def load_demo_traces(path: str) -> dict:
    """Load the per-territory loop decision traces from the web demo cache (docs/demo_cache.json).

    The web mirror stores traces under a top-level 'loop' key (one entry per territory, keyed by
    territory 'key'). Returns {territory_key: trace_dict} ready to drive replay_trace() offline —
    the CLI's no-key/no-network demo path.
    """
    import json

    with open(path) as f:
        payload = json.load(f)
    return {a["key"]: a for a in payload.get("loop", [])}
