"""The `governor` console entry point: it imports cleanly and `evaluate` prints
the scoreboard with the headline numbers."""
from __future__ import annotations

import importlib


def test_entry_point_imports_cleanly():
    cli = importlib.import_module("governor.cli")
    assert callable(cli.main)
    # The declared console_scripts target must resolve.
    assert cli.build_parser() is not None


def test_evaluate_subcommand_prints_scoreboard(capsys):
    from governor.cli import main

    rc = main(["evaluate"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Eval scoreboard" in out
    assert "escalation recall    : 1.00" in out
    assert "DANGEROUS auto-sends : 0" in out


def test_no_subcommand_errors(capsys):
    from governor.cli import main

    with __import__("pytest").raises(SystemExit):
        main([])


def test_run_subcommand_exists_with_source_and_mode():
    from governor.cli import build_parser

    # The acceptance form `governor run --source agents --mode loop` must parse.
    args = build_parser().parse_args(["run", "--source", "agents", "--mode", "loop"])
    assert args.command == "run"
    assert args.source == "agents"
    assert args.mode == "loop"


def test_run_labeled_prints_scoreboard(capsys):
    from governor.cli import main

    # labeled source is fully offline (no network, no key) -> deterministic.
    rc = main(["run", "--source", "labeled"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "governor run" in out
    assert "labeled / flow" in out
    assert "DANGEROUS auto-sends 0" in out


def test_run_replay_flag_parses():
    from governor.cli import build_parser

    args = build_parser().parse_args(
        ["run", "--source", "agents", "--mode", "loop", "--replay"])
    assert args.replay is True


def test_run_replay_rejects_wrong_source_or_mode(capsys):
    from governor.cli import main

    rc = main(["run", "--source", "labeled", "--replay"])
    assert rc == 2
    assert "--replay only applies" in capsys.readouterr().err


def test_run_agents_loop_replay_offline(capsys, monkeypatch):
    """Acceptance: the cached loop trace replays with NO key and NO network, showing each
    agent's decision trace step by step and gating every proposal through the Governor."""
    import socket

    from governor.cli import main
    from governor.territories import TERRITORIES

    def _no_network(*a, **k):
        raise AssertionError("replay must not touch the network")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)

    rc = main(["run", "--source", "agents", "--mode", "loop", "--replay"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "replay (offline)" in out
    assert "processed 20/20" in out
    assert "DANGEROUS auto-sends 0" in out
    # the step-by-step decision trace is rendered per agent (the terminal demo of autonomy)
    assert "decision traces (agent DECIDING" in out
    assert "STOP: enough-candidates" in out
    for terr in TERRITORIES:
        assert terr["key"] in out
