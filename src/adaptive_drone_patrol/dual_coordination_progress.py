"""Append-only run timing and atomic sidechat snapshots, independent of chat.

UTC is the cross-process interval source. Monotonic readings are subtracted only
inside the originating process. Incomplete/crashed attempts retain unknown ends.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from datetime import datetime, timezone
from contextlib import contextmanager


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _seconds(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def interval_totals(intervals):
    """Union execution first; waiting/blocked exclude already accounted spans."""
    valid = [r for r in intervals if isinstance(r.get("end"), (int, float))]
    edges = sorted({r[k] for r in valid for k in ("start", "end")})
    totals = dict(execution=0.0, waiting=0.0, blocked=0.0)
    for a, b in zip(edges, edges[1:]):
        for kind in totals:
            if any(r["kind"] == kind and r["start"] <= a and r["end"] >= b for r in valid):
                totals[kind] += b - a
                break
    return totals


def read_progress(root, *, now=None):
    result = json.loads((Path(root) / "progress.json").read_text())
    current = (now or datetime.now(timezone.utc)).timestamp()
    _freshness(result, current)
    result["inspection_timestamp"] = datetime.fromtimestamp(current, timezone.utc).isoformat()
    return result


def _freshness(snapshot, current):
    """Freshness is per open attempt; a healthy sibling cannot confirm it."""
    snapshot["stale"] = current - _seconds(snapshot["snapshot_generated_at"]) > 600
    for job in snapshot["jobs"]:
        job["stale"] = job["ended_at"] is None and current - _seconds(job["last_worker_heartbeat_at"]) > 600
        job["confirmation_state"] = "stale/unconfirmed" if job["stale"] else "recorded"
    snapshot["worker_stale"] = any(job["stale"] for job in snapshot["jobs"])
    remote_time = snapshot.get("remote_source_timestamp")
    snapshot["remote_progress_unconfirmed"] = bool(
        snapshot.get("remote_sync_status") == "failed"
        or (remote_time is not None and current - _seconds(remote_time) > 600)
    )
    current_job = next(job for job in snapshot["jobs"] if job["attempt_id"] == snapshot["current_attempt"])
    snapshot["current_state"] = "stale/unconfirmed" if current_job["stale"] else current_job["status"]


class RunJournal:
    def __init__(self, root):
        self.root = Path(root).resolve()
        (self.root / "timing").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)
        (self.root / "progress-history").mkdir(exist_ok=True)
        self.path = self.root / "timing/events.jsonl"
        self.identity = f"{socket.gethostname()}:{os.getpid()}:{time.monotonic_ns()}"
        self._monotonic = {}

    @contextmanager
    def _lock(self):
        with (self.root / "timing/.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._recover_torn_tail()
                self._events()  # Reject corrupt complete history before any mutation.
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _recover_torn_tail(self):
        """Recover only an unterminated tail, preserving the entire damaged file.

        Completed event records remain byte-identical. Before replacing a torn
        suffix, fsync its original bytes into a content-addressed evidence file.
        The replacement atomically contains the complete prefix and an explicit
        recovery event. A complete JSON record missing only its newline is kept.
        Corrupt newline-terminated history is never silently skipped.
        """
        if not self.path.exists():
            return
        original = self.path.read_bytes()
        if not original or original.endswith(b"\n"):
            return
        boundary = original.rfind(b"\n") + 1
        prefix, tail = original[:boundary], original[boundary:]
        for line in prefix.splitlines():
            json.loads(line)  # Validate all preceding records before writing anything.
        try:
            json.loads(tail)
        except (ValueError, UnicodeError):
            retained = prefix
            reason = "incomplete final event; exact interrupted event and end not recorded"
        else:
            retained = original + b"\n"
            reason = "complete final event retained; missing newline repaired"
        digest = hashlib.sha256(original).hexdigest()
        evidence = self.root / "timing" / f"journal-before-recovery-{digest}.bin"
        if evidence.exists():
            if evidence.read_bytes() != original:
                raise ValueError("Journal recovery evidence differs from its digest")
        else:
            with evidence.open("xb") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
        event = dict(
            event="journal_recovery",
            observed_at=utc_now(),
            reason=reason,
            preserved_path=str(evidence),
            preserved_sha256=digest,
            preserved_bytes=len(original),
            retained_prefix_bytes=len(retained),
            process_identity=self.identity,
        )
        temporary = self.path.with_name(f"events.{os.getpid()}.recovery.tmp")
        with temporary.open("wb") as stream:
            stream.write(retained)
            stream.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def _events(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def _append(self, event):
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._snapshot()

    def _snapshot(self):
        events = self._events()
        items = {}
        for e in events:
            if e["event"] == "journal_recovery":
                continue
            aid = e["attempt_id"]
            if e["event"] == "start":
                items[aid] = dict(
                    ended_at=None,
                    elapsed_wall_seconds=None,
                    completed_units=None,
                    total_units=None,
                    last_worker_heartbeat_at=e["observed_at"],
                    **e,
                )
            else:
                items[aid].update({k: v for k, v in e.items() if k not in ("event", "attempt_id")})
        now = utc_now()
        for item in items.values():
            if item["ended_at"] is None:
                item["elapsed_so_far_seconds"] = _seconds(now) - _seconds(item["started_at"])
                item["elapsed_as_of"] = now
                item["elapsed_so_far_source"] = "UTC timestamp span; execution utilization not inferred"
        rows = list(items.values())
        active = [r for r in rows if r["ended_at"] is None]
        current = active[-1] if active else rows[-1]
        intervals = [
            dict(
                start=_seconds(r["started_at"]),
                end=_seconds(r["ended_at"]),
                kind=r.get("interval_kind", "execution"),
            )
            for r in rows
            if r["ended_at"] not in (None, "not recorded")
        ]
        data = dict(
            schema="dual-progress-v1",
            campaign_started_at=rows[0]["started_at"],
            snapshot_generated_at=now,
            last_worker_heartbeat_at=max(r["last_worker_heartbeat_at"] for r in rows),
            expected_update_seconds=300,
            stale_after_seconds=600,
            current_item=current["work_item_id"],
            current_attempt=current["attempt_id"],
            current_state=current["status"],
            jobs=rows,
            campaign_elapsed_seconds=_seconds(now) - _seconds(rows[0]["started_at"]),
            elapsed_source="UTC timestamp span from first recorded clock; preceding duration not recorded",
            completed_units=current.get("completed_units"),
            total_units=current.get("total_units"),
            recent_failures=[r["attempt_id"] for r in rows if r["status"] in ("failed", "unknown")],
            blocking_reason=current.get("reason"),
            next_action=current.get("next_action"),
            timing_events=str(self.path),
            timing_work_items=str(self.root / "timing/work-items.json"),
            logs=str(self.root / "logs"),
            artifacts=str(self.root),
            closed_interval_union_seconds=interval_totals(intervals),
            union_excludes_open_or_unknown_intervals=True,
            remote_source_timestamp=None,
            latest_successful_sync_at=None,
            journal_recoveries=[e for e in events if e["event"] == "journal_recovery"],
        )
        mirror = self.root / "remote-progress.json"
        if mirror.exists():
            try:
                remote = json.loads(mirror.read_text())
                data.update(
                    remote_source_timestamp=remote.get("source_timestamp"),
                    latest_successful_sync_at=remote.get("latest_successful_sync_at"),
                    remote_sync_status=remote.get("sync_status"),
                    remote_snapshot_path=str(mirror),
                    remote_sync_error=remote.get("error"),
                )
            except (OSError, ValueError) as error:
                data.update(remote_sync_status="failed", remote_sync_error=str(error))
        _freshness(data, _seconds(now))
        atomic_json(
            self.root / "timing/work-items.json", dict(campaign_started_at=rows[0]["started_at"], items=rows)
        )
        atomic_json(self.root / "progress.json", data)
        atomic_json(self.root / "progress-history" / f"{len(events):08d}.json", data)
        lines = [
            "# 2機協調run 進捗",
            "",
            f"開始: {data['campaign_started_at']}",
            f"最終更新: {now}",
            f"worker heartbeat: {data['last_worker_heartbeat_at']}",
            f"全体経過: {data['campaign_elapsed_seconds']:.1f} 秒（UTC差）",
            "",
            "更新またはworker heartbeatから600秒超は stale／未確認。長時間jobは300秒以内に更新。",
            "未計測値は not recorded。学習・物理合格は各receiptによる。",
            "",
            "|工程/試行|状態|完了/総数|経過秒|理由|",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            elapsed = r["elapsed_wall_seconds"] if r["ended_at"] is not None else r["elapsed_so_far_seconds"]
            status = f"stale／未確認（最終記録: {r['status']}）" if r["stale"] else r["status"]
            lines.append(
                f"|{r['attempt_id']}|{status}|{r.get('completed_units')}/{r.get('total_units')}|{elapsed}|{r.get('reason', '')}|"
            )
        lines.extend(
            [
                "",
                f"時間記録: {self.path}",
                f"工程記録: {self.root / 'timing/work-items.json'}",
                f"次: {current.get('next_action', '各工程の前提条件を確認')}",
                f"journal復旧記録: {len(data['journal_recoveries'])} 件（元の破損bytesはtiming内に保存）",
                f"remote元時刻: {data['remote_source_timestamp']}／最終同期: {data['latest_successful_sync_at']}／未確認: {data['remote_progress_unconfirmed']}",
            ]
        )
        target = self.root / "progress-ja.md"
        temporary = target.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines) + "\n")
        os.replace(temporary, target)

    def start(
        self,
        item_id,
        description,
        *,
        category,
        kind="execution",
        retry_of=None,
        dependencies=(),
        parent=None,
        log_path=None,
    ):
        if kind not in ("execution", "waiting", "blocked"):
            raise ValueError("Unknown interval kind")
        with self._lock():
            n = 1 + sum(e["event"] == "start" and e["work_item_id"] == item_id for e in self._events())
            aid = f"{item_id}-{n:03d}"
            monotonic = time.monotonic()
            now = utc_now()
            self._monotonic[aid] = monotonic
            self._append(
                dict(
                    event="start",
                    work_item_id=item_id,
                    attempt_id=aid,
                    category=category,
                    description=description,
                    interval_kind=kind,
                    started_at=now,
                    observed_at=now,
                    status="running" if kind == "execution" else kind,
                    host=socket.gethostname(),
                    process_identity=self.identity,
                    monotonic_start=monotonic,
                    monotonic_end=None,
                    measurement_source="process monotonic duration and UTC timestamps; wall adjustment not recorded",
                    log_path=str(log_path or self.path),
                    retry_of=retry_of,
                    dependency_ids=list(dependencies),
                    parent_id=parent,
                )
            )
        return aid

    def _open_attempt(self, attempt):
        matching = [e for e in self._events() if e.get("attempt_id") == attempt]
        if not matching or matching[0]["event"] != "start":
            raise ValueError("Unknown journal attempt")
        if any(e["event"] == "end" for e in matching):
            raise ValueError("Journal attempt is closed; create a separate retry")
        return matching

    def heartbeat(self, attempt, *, completed=None, total=None, reason=None, next_action=None):
        with self._lock():
            self._open_attempt(attempt)
            self._append(
                dict(
                    event="heartbeat",
                    attempt_id=attempt,
                    observed_at=utc_now(),
                    last_worker_heartbeat_at=utc_now(),
                    completed_units=completed,
                    total_units=total,
                    reason=reason,
                    next_action=next_action,
                )
            )

    def finish(self, attempt, *, status, reason=None):
        if status not in ("passed", "failed", "done", "blocked"):
            raise ValueError("Explicit attempt result required")
        now, mono = utc_now(), time.monotonic()
        # Never use another process's saved monotonic origin.
        elapsed = mono - self._monotonic[attempt] if attempt in self._monotonic else None
        with self._lock():
            start = self._open_attempt(attempt)[0]
            self._append(
                dict(
                    event="end",
                    attempt_id=attempt,
                    status=status,
                    reason=reason,
                    observed_at=now,
                    ended_at=now,
                    last_worker_heartbeat_at=now,
                    monotonic_end=mono if elapsed is not None else "not recorded",
                    elapsed_wall_seconds=elapsed
                    if elapsed is not None
                    else _seconds(now) - _seconds(start["started_at"]),
                    elapsed_source="process monotonic" if elapsed is not None else "UTC timestamp difference",
                )
            )

    def abandon(self, attempt, *, reason):
        with self._lock():
            matching = self._open_attempt(attempt)
            lower = _seconds(matching[-1]["observed_at"]) - _seconds(matching[0]["started_at"])
            self._append(
                dict(
                    event="end",
                    attempt_id=attempt,
                    status="unknown",
                    reason=reason,
                    observed_at=utc_now(),
                    ended_at="not recorded",
                    elapsed_wall_seconds="not recorded",
                    elapsed_lower_bound_seconds=max(0, lower),
                )
            )

    def run_command(self, argv, item_id, *, category, heartbeat_seconds=60):
        if not 0 < heartbeat_seconds <= 300:
            raise ValueError("Worker heartbeat interval must be in (0,300]")
        log = self.root / "logs" / f"{item_id}-{time.time_ns()}.log"
        attempt = self.start(item_id, " ".join(argv), category=category, log_path=log)
        try:
            with log.open("w") as stream:
                process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
                while True:
                    try:
                        rc = process.wait(timeout=heartbeat_seconds)
                        break
                    except subprocess.TimeoutExpired:
                        self.heartbeat(attempt, reason=f"worker pid={process.pid}; execution ongoing")
            self.finish(attempt, status="passed" if rc == 0 else "failed", reason=f"exit_code={rc}")
            return rc
        except BaseException as exc:
            self.abandon(attempt, reason=f"wrapper interrupted: {type(exc).__name__}; worker end unconfirmed")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--category", default="execution")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(RunJournal(args.run).run_command(command, args.item, category=args.category))
