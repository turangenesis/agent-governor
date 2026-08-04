"""`governor` console entry point — stdlib argparse only, no LLM, no network.

Subcommands:
  governor evaluate        run the Governor over the labeled set, print the scoreboard
  governor discover        show cached discovered candidates (offline replay)

The deterministic core is untouched; this is a thin wrapper so the demo is usable
without Streamlit.
"""
from __future__ import annotations

import argparse
import sys


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, evaluate_heldout, fifo_baseline_stats

    sb = evaluate()
    print(sb.pretty())
    print()
    fifo = fifo_baseline_stats()
    print("=== FIFO baseline (no Governor) ===")
    print(f"  human sees ALL {fifo['human_decisions']} requests | autonomy 0% | load saved 0%")
    print(f"  Governor: human sees {sb.correct_escalation + sb.false_escalation} "
          f"| autonomy {sb.autonomy_pct:.0f}% | dangerous auto-sends {sb.false_auto_send}")

    # --- Held-out / adversarial generalization (honest: expected to be imperfect) ---
    ho = evaluate_heldout()
    print()
    print("=== Held-out / adversarial set (generalization - labels NOT used to tune the policy) ===")
    print(f"  cases: {ho.total} | recall {ho.escalation_recall:.2f} | precision {ho.escalation_precision:.2f} "
          f"| DANGEROUS (missed risks) {ho.false_auto_send}")
    if ho.false_auto_send:
        print(f"  ^ {ho.false_auto_send} adversarial evasion(s) slipped the keyword gate - "
              f"the honest gap the LLM-as-judge is built to close (see docs/EVAL.md).")

    # --- LLM-as-judge on draft QUALITY (stub by default = free; --judge llm is opt-in, costs $) ---
    from .eval_set import build_cases
    from .eval_set_heldout import build_heldout_cases
    from .judge import get_judge, judge_report

    actions = [c.action for c in build_cases()] + [c.action for c in build_heldout_cases()]
    try:
        judge = get_judge(getattr(args, "judge", "stub"))
        rep = judge_report(actions, judge)
    except Exception as e:
        print(f"\n  (LLM judge unavailable: {str(e)[:60]} - run without --judge llm for the free stub)")
        return 0
    print()
    print(f"=== Draft-quality eval: LLM-as-judge ({rep['judge']}) ===")
    print(f"  drafts: {rep['total']} | passed: {rep['passed']} ({rep['pass_rate']*100:.0f}%) "
          f"| avg {rep['avg_scores']}")
    for f in rep["failures"][:6]:
        print(f"    ✗ {f['name']}: {f['reasons'][0] if f['reasons'] else ''}")
    return 0


def _cmd_discover(_args: argparse.Namespace) -> int:
    from .discover import load

    try:
        cands = load()
    except FileNotFoundError:
        print("no cached candidates found — run discovery live first.", file=sys.stderr)
        return 1
    print(f"=== {len(cands)} cached candidates (replay, no network) ===")
    for c in cands:
        print(f"  {c.name:22} | {c.current_company:24} | {c.seniority:6} | "
              f"match {c.match_confidence:.2f} | {c.current_title[:40]}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import os

    from .runner import run_headless

    agent_mode = "loop" if args.mode == "loop" else "flow"

    replay = getattr(args, "replay", False)
    replay_traces = None
    if replay:
        if agent_mode != "loop" or args.source != "agents":
            print("--replay only applies to '--source agents --mode loop'.", file=sys.stderr)
            return 2
        from .loop_agent import load_demo_traces
        cache = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "docs", "demo_cache.json")
        try:
            replay_traces = load_demo_traces(cache)
        except FileNotFoundError:
            print("no cached loop traces found — run scripts/gen_web_cache.py first.", file=sys.stderr)
            return 1
        if not replay_traces:
            print("cached demo has no loop traces — run scripts/gen_web_cache.py first.", file=sys.stderr)
            return 1

    # loop mode needs a tool-calling LLM; request one (falls back to the deterministic flow
    # with no key). --replay drives the loop from the cached trace instead, so it needs no key.
    # The deterministic flow is the offline path for every other case.
    use_llm = agent_mode == "loop" and not replay
    snap = run_headless("governor", source=args.source, agent_mode=agent_mode, use_llm=use_llm,
                        replay_traces=replay_traces)

    mode_label = f"{args.source} / {agent_mode}" + (" / replay (offline)" if replay else "")
    print(f"=== governor run — {mode_label} ===")
    print(f"  processed {snap['processed']}/{snap['total']} proposals")
    print(f"  auto-sent {snap['auto_sent']} | escalated {snap['queue_depth']} "
          f"| held {len(snap['held'])}")
    print(f"  autonomy {snap['autonomy_pct']:.0f}% | DANGEROUS auto-sends {snap['dangerous_auto_sends']}")

    traces = snap.get("traces") or {}
    if traces:
        print(f"\n  {len(traces)} autonomous decision traces:")
        for agent_id, tr in sorted(traces.items()):
            print(f"    {agent_id}: {len(tr.steps)} steps, {len(tr.proposals)} proposals, "
                  f"stopped on {tr.terminal_reason}")
        if replay:
            _print_trace_steps(traces)
    elif agent_mode == "loop":
        print("\n  (no live LLM — agents ran the deterministic flow fallback)")
    return 0


def _print_trace_steps(traces: dict) -> None:
    """Render each agent's decision trace step by step — the terminal demo of the autonomous loop."""
    print("\n  --- decision traces (agent DECIDING, step by step) ---")
    for agent_id, tr in sorted(traces.items()):
        print(f"\n  {agent_id}  [{getattr(tr, 'territory_key', '')}]")
        for s in tr.steps:
            arg = next(iter(s.tool_input.values()), "") if s.tool_input else ""
            arg = f"({arg})" if arg else ""
            print(f"    {s.index:>2}. {s.tool}{arg}: {s.observation}")
            print(f"        -> {s.decision}: {s.reason}")
        print(f"    STOP: {tr.terminal_reason} ({len(tr.proposals)} proposals -> Governor)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="governor",
        description="The Governor — oversight & eval layer for recruiting agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="run the Governor over the labeled + held-out sets")
    p_eval.add_argument("--judge", choices=("stub", "llm"), default="stub",
                        help="draft-quality judge: 'stub' (free, deterministic, default) or "
                             "'llm' (real LLM-as-judge; needs an API key; costs cents)")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_disc = sub.add_parser("discover", help="show cached discovered candidates (offline)")
    p_disc.set_defaults(func=_cmd_discover)

    p_run = sub.add_parser("run", help="run the sourcing agents through the Governor")
    p_run.add_argument("--source", choices=("labeled", "discovered", "agents"),
                       default="labeled",
                       help="labeled scoreboard set | live-discovered leads | autonomous territory agents")
    p_run.add_argument("--mode", choices=("flow", "loop"), default="flow",
                       help="agent execution: fixed deterministic flow, or autonomous LLM tool-use loop "
                            "(loop needs a key; falls back to flow offline)")
    p_run.add_argument("--replay", action="store_true",
                       help="replay the cached loop decision traces with no key and no network "
                            "(reliable offline demo; only with '--source agents --mode loop')")
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
