"""Exercise actual subprocess exclusion and failure preservation without accounting."""

import fcntl
import json
from pathlib import Path
import sys

import pytest


def launch(*args, **kwargs):
    from adaptive_drone_patrol.payload_solver_jobs import run_job

    return run_job(*args, **kwargs)


@pytest.mark.parametrize("historical", [None, {"used_s": 999999999, "jobs": [], "active_job": "historical"}])
def test_no_historical_budget_can_block_or_change_execution(tmp_path, monkeypatch, historical):
    monkeypatch.chdir(tmp_path)
    ledger = Path("outputs/gpu_ledger.json")
    if historical is not None:
        ledger.parent.mkdir()
        ledger.write_text(json.dumps(historical))
    before = ledger.read_bytes() if ledger.exists() else None
    result = launch(
        [sys.executable, "-c", 'print("actual child")'],
        output="outputs/new-job",
        timeout_s=3,
        lock_path=tmp_path / "gpu.lock",
    )
    assert result["status"] == "SUCCEEDED"
    assert (ledger.read_bytes() if ledger.exists() else None) == before
    assert (Path("outputs/new-job") / "job.log").read_text().strip() == "actual child"
    assert not {"used_s", "elapsed_s", "remaining_s", "budget_s"} & result.keys()


def test_competing_owner_prevents_launch(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="GPU job"):
            launch(
                [sys.executable, "-c", "raise AssertionError()"],
                output=tmp_path / "rejected",
                timeout_s=3,
                lock_path=lock_path,
            )
    assert not (tmp_path / "rejected").exists()


def test_timeout_preserves_raw_and_heartbeat_and_releases_lock(tmp_path):
    out = tmp_path / "timeout"
    command = [
        sys.executable,
        "-u",
        "-c",
        "from pathlib import Path; import time; "
        f'Path({str(out / "raw.jsonl")!r}).write_text("raw before timeout\\n"); '
        'print("abnormal window"); time.sleep(10)',
    ]
    result = launch(
        command, output=out, timeout_s=0.3, grace_s=0.1, heartbeat_s=0.05, lock_path=tmp_path / "gpu.lock"
    )
    assert result["status"] == "TIMED_OUT"
    assert (out / "raw.jsonl").read_text() == "raw before timeout\n"
    assert (out / "job.log").read_text().strip() == "abnormal window"
    assert json.loads((out / "status.json").read_text()) == result
    assert (out / "heartbeat.json").is_file()
    assert (
        launch(
            [sys.executable, "-c", "pass"],
            output=tmp_path / "next",
            timeout_s=3,
            lock_path=tmp_path / "gpu.lock",
        )["status"]
        == "SUCCEEDED"
    )


def test_failed_launch_is_retained_and_output_never_overwritten(tmp_path):
    out = tmp_path / "failed"
    with pytest.raises(FileNotFoundError):
        launch(["/nonexistent/solver-executable"], output=out, timeout_s=3, lock_path=tmp_path / "gpu.lock")
    assert json.loads((out / "status.json").read_text())["status"] == "FAILED"
    with pytest.raises(FileExistsError):
        launch([sys.executable, "-c", "pass"], output=out, timeout_s=3, lock_path=tmp_path / "gpu.lock")
