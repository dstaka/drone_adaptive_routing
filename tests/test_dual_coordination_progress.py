"""Timing provenance, retry preservation, stale state and real worker execution."""

import json
import sys
from datetime import datetime, timezone
import pytest


def test_overlap_union_deduplicates_parents_and_excludes_busy_waits():
    from adaptive_drone_patrol.dual_coordination_progress import interval_totals

    intervals = [
        {"start": 0, "end": 10, "kind": "execution"},
        {"start": 5, "end": 15, "kind": "execution"},
        {"start": 0, "end": 20, "kind": "waiting"},
        {"start": 19, "end": 25, "kind": "blocked"},
    ]
    assert interval_totals(intervals) == {"execution": 15, "waiting": 5, "blocked": 5}


def test_failed_attempt_and_retry_are_append_only_and_stale_is_explicit(tmp_path):
    from adaptive_drone_patrol.dual_coordination_progress import RunJournal, read_progress

    journal = RunJournal(tmp_path)
    first = journal.start("qualification", "bad first probe", category="qualification")
    journal.finish(first, status="failed", reason="unsafe initial pose")
    second = journal.start("qualification", "corrected probe", category="qualification", retry_of=first)
    journal.heartbeat(second, completed=4, total=36)
    data = read_progress(tmp_path, now=datetime(2100, 1, 1, tzinfo=timezone.utc))
    assert data["stale"] is True
    assert data["jobs"][0]["status"] == "failed"
    assert data["jobs"][1]["completed_units"] == 4
    assert data["jobs"][1]["ended_at"] is None
    assert data["jobs"][1]["elapsed_wall_seconds"] is None
    assert data["jobs"][1]["retry_of"] == first
    assert "stale" in (tmp_path / "progress-ja.md").read_text()
    journal.finish(second, status="passed")
    events = [json.loads(line) for line in (tmp_path / "timing/events.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "start" for e in events) == 2
    assert sum(e["event"] == "end" for e in events) == 2


def test_real_subprocess_exit_and_log_are_retained(tmp_path):
    from adaptive_drone_patrol.dual_coordination_progress import RunJournal

    journal = RunJournal(tmp_path)
    rc = journal.run_command(
        [sys.executable, "-c", 'print("retained failure"); raise SystemExit(7)'],
        "tests",
        category="test",
        heartbeat_seconds=0.01,
    )
    assert rc == 7
    data = json.loads((tmp_path / "progress.json").read_text())
    job = data["jobs"][0]
    assert job["status"] == "failed"
    assert job["elapsed_wall_seconds"] >= 0
    assert job["monotonic_end"] >= job["monotonic_start"]
    assert "retained failure" in open(job["log_path"]).read()
    assert data["current_state"] == "failed"


def test_unknown_crash_end_is_never_invented(tmp_path):
    from adaptive_drone_patrol.dual_coordination_progress import RunJournal

    journal = RunJournal(tmp_path)
    attempt = journal.start("physics", "unknown crash", category="qualification")
    journal.abandon(attempt, reason="worker disappeared; exact end unknown")
    item = json.loads((tmp_path / "timing/work-items.json").read_text())["items"][0]
    assert item["ended_at"] == "not recorded"
    assert item["elapsed_wall_seconds"] == "not recorded"
    assert item["elapsed_lower_bound_seconds"] >= 0


def test_each_open_job_becomes_stale_despite_another_workers_heartbeat(tmp_path, monkeypatch):
    import adaptive_drone_patrol.dual_coordination_progress as progress

    monkeypatch.setattr(progress, "utc_now", lambda: "2026-10-28T00:00:00+00:00")
    journal = progress.RunJournal(tmp_path)
    abandoned = journal.start("silent", "worker disappeared", category="test")
    closed = journal.start("closed", "already finished", category="test")
    journal.finish(closed, status="done")
    monkeypatch.setattr(progress, "utc_now", lambda: "2026-10-28T00:20:00+00:00")
    live = journal.start("live", "independent worker", category="test")
    saved = json.loads((tmp_path / "progress.json").read_text())
    assert saved["jobs"][0]["stale"] is True
    assert saved["jobs"][1]["stale"] is False
    assert saved["jobs"][2]["stale"] is False
    assert saved["worker_stale"] is True
    assert f"|{abandoned}|stale／未確認" in (tmp_path / "progress-ja.md").read_text()
    inspected = progress.read_progress(tmp_path, now=datetime(2026, 10, 28, 0, 31, tzinfo=timezone.utc))
    assert inspected["jobs"][2]["stale"] is True
    assert inspected["current_attempt"] == live
    assert inspected["current_state"] == "stale/unconfirmed"


@pytest.mark.parametrize("operation", ["finish", "heartbeat", "abandon"])
@pytest.mark.parametrize("terminal", ["failed", "done", "blocked", "unknown"])
def test_closed_attempt_cannot_be_modified(tmp_path, operation, terminal):
    from adaptive_drone_patrol.dual_coordination_progress import RunJournal

    journal = RunJournal(tmp_path)
    attempt = journal.start("job", "one immutable attempt", category="test")
    if terminal == "unknown":
        journal.abandon(attempt, reason="exact end unknown")
    else:
        journal.finish(attempt, status=terminal, reason="first terminal result")
    before = journal.path.read_bytes()
    arguments = {"status": "passed"} if operation == "finish" else {"reason": "late update"}
    with pytest.raises(ValueError, match="closed"):
        getattr(journal, operation)(attempt, **arguments)
    assert journal.path.read_bytes() == before


def test_torn_last_event_is_preserved_and_recovery_is_explicit(tmp_path):
    from hashlib import sha256
    from adaptive_drone_patrol.dual_coordination_progress import RunJournal

    journal = RunJournal(tmp_path)
    attempt = journal.start("job", "surviving attempt", category="test")
    complete = journal.path.read_bytes()
    damaged = complete + b'{"event":"heartbeat","reason":"partial\xe6'
    journal.path.write_bytes(damaged)
    journal.heartbeat(attempt, completed=1, total=2)
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    recovery = next(row for row in events if row["event"] == "journal_recovery")
    assert open(recovery["preserved_path"], "rb").read() == damaged
    assert recovery["preserved_sha256"] == sha256(damaged).hexdigest()
    assert recovery["retained_prefix_bytes"] == len(complete)
    assert events[0] == json.loads(complete)
    snapshot = json.loads((tmp_path / "progress.json").read_text())
    assert snapshot["journal_recoveries"] == [recovery]
    assert snapshot["jobs"][0]["ended_at"] is None
    assert snapshot["jobs"][0]["completed_units"] == 1
    journal.finish(attempt, status="done")


def test_valid_last_event_without_newline_is_retained_during_recovery(tmp_path):
    from adaptive_drone_patrol.dual_coordination_progress import RunJournal

    journal = RunJournal(tmp_path)
    attempt = journal.start("job", "complete JSON with missing delimiter", category="test")
    journal.path.write_bytes(journal.path.read_bytes().rstrip(b"\n"))
    journal.finish(attempt, status="done")
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["start", "journal_recovery", "end"]


def test_corrupt_interior_record_is_not_silently_recovered(tmp_path):
    from adaptive_drone_patrol.dual_coordination_progress import RunJournal

    journal = RunJournal(tmp_path)
    attempt = journal.start("job", "history must stay intact", category="test")
    damaged = b"not-json\n" + journal.path.read_bytes() + b'{"torn":'
    journal.path.write_bytes(damaged)
    with pytest.raises(ValueError):
        journal.heartbeat(attempt)
    assert journal.path.read_bytes() == damaged


def test_remote_snapshot_retains_source_time_and_failed_sync(tmp_path):
    from adaptive_drone_patrol.dual_coordination_progress import RunJournal, atomic_json
    import json

    journal = RunJournal(tmp_path)
    attempt = journal.start("remote", "remote", category="test")
    atomic_json(
        tmp_path / "remote-progress.json",
        dict(
            source_timestamp="2026-10-27T00:00:00+00:00",
            latest_successful_sync_at="2026-10-27T00:00:10+00:00",
            sync_status="failed",
            error="connection lost",
        ),
    )
    journal.heartbeat(attempt)
    result = json.loads((tmp_path / "progress.json").read_text())
    assert result["remote_source_timestamp"] == "2026-10-27T00:00:00+00:00"
    assert result["remote_sync_status"] == "failed"
    assert result["remote_progress_unconfirmed"] is True
