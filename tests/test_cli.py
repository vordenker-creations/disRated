import pytest
from disrated import HELP_TEXT, __version__, main as cli_main


def test_cli_help(capsys):
    cli_main(["--help"])
    captured = capsys.readouterr()
    assert "disRated: Evaluation harness" in captured.out
    assert "evalrun, run" in captured.out
    assert "score" in captured.out
    assert "val-gate" in captured.out
    assert "risk-coverage" in captured.out


def test_cli_version(capsys):
    cli_main(["--version"])
    captured = capsys.readouterr()
    assert captured.out.strip() == f"disrated {__version__}"


def test_cli_no_args(capsys):
    cli_main([])
    captured = capsys.readouterr()
    assert "disRated: Evaluation harness" in captured.out


def test_cli_unknown_subcommand(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["nonexistent-command"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Unknown command: nonexistent-command" in captured.out


def test_cli_subcommand_dispatch_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["evalrun", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--splits SPLITS" in captured.out

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["score", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--threshold THRESHOLD" in captured.out

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["val-gate", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--tau-mode" in captured.out

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["risk-coverage", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--dataset DATASET" in captured.out
