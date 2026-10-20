"""Real CLI argument parsing and subprocess-independent public commands."""

import json
from unittest.mock import patch

from trading_agent.cli import main


def test_doctor_checks_tensor_and_reports_missing_cuda_honestly(capsys):
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["cpu_tensor_check"] == 14.0 and report["paper_only"]
    with patch("trading_agent.cli.doctor", return_value={"cuda_verified": False}):
        assert main(["doctor", "--require-cuda"]) == 2


def test_generate_cli_and_immutable_file_error(tmp_path, capsys):
    path = str(tmp_path / "synthetic.csv")
    assert main(["generate", "--sessions", "4", "--output", path]) == 0
    assert json.loads(capsys.readouterr().out)["provenance"] == "SYNTHETIC"
    assert main(["generate", "--sessions", "4", "--output", path]) == 1
    assert json.loads(capsys.readouterr().err)["type"] == "FileExistsError"


def test_missing_provider_secrets_are_not_requested_as_arguments_or_logged(capsys):
    with patch.dict("os.environ", {}, clear=True):
        assert main(["quote", "--symbol", "EXAMPLE", "--instrument-token", "1"]) == 1
    assert "locally" in json.loads(capsys.readouterr().err)["message"]


def test_continuous_paper_stream_cli_replays_and_resumes_a_snapshot(tmp_path, capsys):
    data = str(tmp_path / "synthetic.csv")
    db = str(tmp_path / "stream.sqlite")
    assert main(["generate", "--sessions", "2", "--output", data]) == 0
    capsys.readouterr()

    assert main([
        "paper-stream", "--data", data, "--db", db, "--strategy", "cash", "--max-bars", "2",
    ]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["mode"] == "CONTINUOUS_PAPER_REPLAY"
    assert first["events_processed"] == 2
    assert first["paper_only"] and first["provider"]["provenance"] == "SYNTHETIC"

    assert main([
        "paper-stream", "--data", data, "--db", db, "--strategy", "cash", "--resume", "--max-bars", "2",
    ]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["events_processed"] == 2
    assert resumed["status"]["transitions"] == 3
