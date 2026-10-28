"""Run the prospective native81-case declaration with at most seven processes.

Every child owns its stdout log, case outputs and native-progress journal. The
parent records orchestration in the requested run journal. A failure stops new
launches; already running children finish normally. This is case execution, not
an overall physical qualification/admission receipt. Existing outputs are never
reused. No production learning or held-out generation occurs here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from adaptive_drone_patrol.dual_coordination_progress import RunJournal, atomic_json
from adaptive_drone_patrol.dual_coordination_qualify import (
    MANIFEST,
    qualification_cases,
    qualification_source_freeze,
    verify_qualification_source_freeze,
)
from adaptive_drone_patrol.util import sha256


def _now():
    return datetime.now(timezone.utc).isoformat()


def _record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def _case_command(*, case, output, original, design_path, source_freeze, no_progress_timeout_seconds):
    # No shell and no pilot override: each child must run its complete declared
    # duration and retain its own journal beneath the fresh case directory.
    return [
        sys.executable,
        "-m",
        "adaptive_drone_patrol.dual_coordination_qualify",
        "--original",
        str(original),
        "--design",
        str(design_path),
        "--output",
        str(output),
        "--case-id",
        case["case_id"],
        "--source-freeze",
        str(source_freeze),
        "--no-progress-timeout",
        str(no_progress_timeout_seconds),
    ]


def _preflight_command(**kwargs):
    return _case_command(**kwargs) + ["--pilot-duration", "0.01"]


def _preflight_result(output, exit_code, frozen, case):
    if exit_code != 0:
        raise ValueError(f"Native one-tick preflight exited with code {exit_code}")
    output = Path(output)
    result = json.loads((output / "result.json").read_text())
    inputs = json.loads((output / "inputs.json").read_text())
    if (
        result.get("schema") != "dual-qualification-case-v1"
        or result.get("case") != case
        or result.get("pilot") is not True
        or result.get("passed") is not False
        or result.get("admission_granted") is not False
        or result.get("failure") is not None
        or result.get("completed_declared_duration") is not False
        or result.get("completed_requested_duration") is not True
        or type(result.get("completed_control_steps")) is not int
        or result["completed_control_steps"] != 1
        or result.get("source_integrity_passed") is not True
        or result.get("online_audit", {}).get("safe") is not True
        or result.get("numerical", {}).get("passed") is not True
    ):
        raise ValueError("Native preflight requires exactly one safe diagnostic tick, never a formal pass")
    if (
        inputs.get("source_hashes") != frozen["source_hashes"]
        or result.get("source_hashes_end") != frozen["source_hashes"]
        or inputs.get("dependency_versions") != frozen["dependency_versions"]
        or result.get("dependency_versions_end") != frozen["dependency_versions"]
    ):
        raise ValueError("Native preflight dependency closure differs before/after actual step")
    return _record(output / "result.json")


def _run_preflight(
    *,
    original,
    design_path,
    output,
    source_freeze,
    frozen,
    no_progress_timeout_seconds,
    journal,
    parent_attempt,
):
    # Fixed diagnostic fixture only. This pilot is never among the81 formal
    # case rows and never repairs or salvages a prior failed batch.
    case = qualification_cases()[0]
    command = _preflight_command(
        case=case,
        output=output / "preflight",
        original=original,
        design_path=design_path,
        source_freeze=source_freeze,
        no_progress_timeout_seconds=no_progress_timeout_seconds,
    )
    log_path = output / "logs/preflight.log"
    attempt = journal.start(
        "qualification-batch-preflight",
        "One native diagnostic control tick and dependency closure verification",
        category="physical qualification",
        parent=parent_attempt,
        log_path=log_path,
    )
    started = time.monotonic()
    process = None
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            next_heartbeat = started + 30
            while process.poll() is None:
                now = time.monotonic()
                if now - started > no_progress_timeout_seconds + 60:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise ValueError("Native preflight exceeded bounded startup/one-tick timeout")
                if now >= next_heartbeat:
                    journal.heartbeat(attempt, reason="Diagnostic native startup/one control tick running")
                    journal.heartbeat(
                        parent_attempt, reason="Formal cases wait for native one-tick preflight"
                    )
                    next_heartbeat = now + 30
                time.sleep(0.05)
        evidence = _preflight_result(output / "preflight", process.returncode, frozen, case)
        verify_qualification_source_freeze(frozen)
        journal.finish(
            attempt,
            status="passed",
            reason="One native diagnostic tick complete; source closure stable; no formal qualification",
        )
        return dict(
            status="passed",
            diagnostic_only=True,
            admission_granted=False,
            command=command,
            result=evidence,
            stdout_log=_record(log_path),
            elapsed_wall_seconds=time.monotonic() - started,
        )
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        journal.finish(attempt, status="failed", reason=f"Native preflight: {type(error).__name__}: {error}")
        raise


def _case_result(row, exit_code, frozen):
    if exit_code != 0:
        raise ValueError(f"Qualification child exited with code {exit_code}")
    output = Path(row["output"])
    result = json.loads((output / "result.json").read_text())
    inputs = json.loads((output / "inputs.json").read_text())
    if (
        result.get("schema") != "dual-qualification-case-v1"
        or result.get("case") != row["case"]
        or result.get("passed") is not True
        or result.get("pilot") is not False
        or result.get("completed_declared_duration") is not True
        or result.get("admission_granted") is not False
        or result.get("failure") is not None
        or result.get("source_integrity_passed") is not True
    ):
        raise ValueError("Child lacks a complete passing declared case result")
    if type(result.get("completed_control_steps")) is not int or result["completed_control_steps"] != round(
        row["case"]["duration_s"] * 100
    ):
        raise ValueError("Child control count differs from the full declared duration")
    if (
        inputs.get("source_hashes") != frozen["source_hashes"]
        or result.get("source_hashes_end") != frozen["source_hashes"]
        or inputs.get("dependency_versions") != frozen["dependency_versions"]
        or result.get("dependency_versions_end") != frozen["dependency_versions"]
    ):
        raise ValueError("Child source/dependency evidence differs from prospective batch freeze")
    return _record(output / "result.json")


def run_batch(*, original, design_path, output, journal_root, workers=7, no_progress_timeout_seconds=600.0):
    if type(workers) is not int or not 1 <= workers <= 7:
        raise ValueError("Native qualification requires 1..7 worker processes")
    if (
        type(no_progress_timeout_seconds) not in (int, float)
        or not math.isfinite(no_progress_timeout_seconds)
        or no_progress_timeout_seconds <= 0
    ):
        raise ValueError("Positive finite native-progress watchdog timeout required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    cases = qualification_cases()
    journal = RunJournal(journal_root)
    attempt = journal.start(
        "native-qualification-batch",
        "Finite prospective native qualification cases",
        category="physical qualification",
        log_path=output / "batch.json",
    )
    started = time.monotonic()
    report = dict(
        schema="dual-qualification-batch-v1",
        status="running",
        admission_granted=False,
        started_at=_now(),
        ended_at=None,
        workers=workers,
        no_progress_timeout_seconds=no_progress_timeout_seconds,
        cases=[
            dict(
                case=case,
                status="pending",
                started_at=None,
                ended_at=None,
                exit_code=None,
                output=str(output / "cases" / case["case_id"]),
                stdout_log=str(output / "logs" / f"{case['case_id']}.log"),
                reason=None,
            )
            for case in cases
        ],
        failures=[],
        inherited_manifest=_record(MANIFEST),
        launcher=_record(__file__),
    )
    running = {}
    stopped = False
    frozen = None

    def save():
        report["observed_at"] = _now()
        report["elapsed_wall_seconds"] = time.monotonic() - started
        report["counts"] = {
            key: sum(row["status"] == key for row in report["cases"])
            for key in ("pending", "running", "passed", "failed", "skipped")
        }
        atomic_json(output / "batch.json", report)

    def stop(reason):
        nonlocal stopped
        stopped = True
        report["failures"].append(reason)
        for row in report["cases"]:
            if row["status"] == "pending":
                row.update(status="skipped", ended_at=_now(), reason="not launched after an earlier failure")

    save()
    try:
        original, design_path = Path(original).resolve(strict=True), Path(design_path).resolve(strict=True)
        report["original"], report["design"] = _record(original), _record(design_path)
        frozen = qualification_source_freeze()
        source_path = output / "source-freeze.json"
        atomic_json(source_path, frozen)
        report["source_freeze"] = _record(source_path)
        atomic_json(
            output / "declaration.json",
            dict(
                schema="dual-native-qualification-declaration-v2",
                cases=cases,
                admission_granted=False,
                inherited_manifest_sha256=sha256(MANIFEST),
            ),
        )
        save()
        report["preflight"] = dict(status="running", diagnostic_only=True, admission_granted=False)
        save()
        try:
            report["preflight"] = _run_preflight(
                original=original,
                design_path=design_path,
                output=output,
                source_freeze=source_path,
                frozen=frozen,
                no_progress_timeout_seconds=no_progress_timeout_seconds,
                journal=journal,
                parent_attempt=attempt,
            )
        except BaseException as error:
            report["preflight"].update(status="failed", reason=f"{type(error).__name__}: {error}")
            raise
        save()
        heartbeat_at = time.monotonic() + 30
        while running or any(row["status"] == "pending" for row in report["cases"]):
            # Reap ALL completed children before refilling even one worker slot.
            for index, (process, log, child_attempt) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                row = report["cases"][index]
                log.close()
                row.update(exit_code=code, ended_at=_now())
                try:
                    row["result"] = _case_result(row, code, frozen)
                    row.update(
                        status="passed",
                        reason="Full declared case complete; overall admission remains separate",
                    )
                except (ValueError, OSError, KeyError, TypeError) as error:
                    row.update(status="failed", reason=str(error))
                    stop(f"{row['case']['case_id']}: {error}")
                journal.finish(
                    child_attempt,
                    status="passed" if row["status"] == "passed" else "failed",
                    reason=row["reason"],
                )
                del running[index]
                save()
            if not stopped and len(running) < workers:
                pending = next(
                    (i for i, row in enumerate(report["cases"]) if row["status"] == "pending"), None
                )
                if pending is not None:
                    row = report["cases"][pending]
                    log = None
                    child_attempt = None
                    try:
                        verify_qualification_source_freeze(frozen)
                        if (
                            _record(original) != report["original"]
                            or _record(design_path) != report["design"]
                        ):
                            raise ValueError("Physical input changed between case launches")
                        command = _case_command(
                            case=row["case"],
                            output=Path(row["output"]),
                            original=original,
                            design_path=design_path,
                            source_freeze=source_path,
                            no_progress_timeout_seconds=no_progress_timeout_seconds,
                        )
                        log = Path(row["stdout_log"]).open("xb")
                        child_attempt = journal.start(
                            f"qualification-batch-{row['case']['case_id']}",
                            "Full native case child process",
                            category="physical qualification",
                            parent=attempt,
                            log_path=row["stdout_log"],
                        )
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        row.update(
                            status="running",
                            started_at=_now(),
                            pid=process.pid,
                            command=command,
                            journal_attempt=child_attempt,
                        )
                        running[pending] = (process, log, child_attempt)
                    except (OSError, ValueError, RuntimeError) as error:
                        if log is not None:
                            log.write((repr(error) + "\n").encode())
                            log.close()
                        row.update(status="failed", ended_at=_now(), reason=repr(error))
                        if child_attempt is not None:
                            journal.finish(child_attempt, status="failed", reason=repr(error))
                        stop(f"Launch {row['case']['case_id']}: {error}")
                    save()
                    continue  # Poll again before launching the next pending case.
            if time.monotonic() >= heartbeat_at:
                for index, (_, _, child_attempt) in running.items():
                    journal.heartbeat(
                        child_attempt, reason="Child running; native progress tracked in its case journal"
                    )
                journal.heartbeat(
                    attempt,
                    completed=sum(r["status"] in ("passed", "failed", "skipped") for r in report["cases"]),
                    total=len(cases),
                    reason=f"{len(running)} children running; new launches {'stopped' if stopped else 'enabled'}",
                )
                save()
                heartbeat_at = time.monotonic() + 30
            if running:
                time.sleep(0.05)
        verify_qualification_source_freeze(frozen)
        if _record(original) != report["original"] or _record(design_path) != report["design"]:
            raise ValueError("Physical batch inputs changed during execution")
    except BaseException as error:
        stop(f"Batch infrastructure: {type(error).__name__}: {error}")
        # Preserve outstanding work even if orchestration fails. No new child is
        # launched, and every already running process is awaited and recorded.
        for index, (process, log, child_attempt) in list(running.items()):
            next_drain_heartbeat = 0.0
            while process.poll() is None:
                if time.monotonic() >= next_drain_heartbeat:
                    journal.heartbeat(child_attempt, reason="Draining existing child after batch failure")
                    journal.heartbeat(
                        attempt, reason="Waiting for already running native cases after batch failure"
                    )
                    save()
                    next_drain_heartbeat = time.monotonic() + 30
                time.sleep(0.1)
            row = report["cases"][index]
            log.close()
            row.update(exit_code=process.returncode, ended_at=_now())
            try:
                row["result"] = _case_result(row, process.returncode, frozen)
                row.update(status="passed", reason="Completed while draining after batch failure")
            except (ValueError, OSError, KeyError, TypeError) as child_error:
                row.update(status="failed", reason=str(child_error))
            journal.finish(
                child_attempt,
                status="passed" if row["status"] == "passed" else "failed",
                reason=row["reason"],
            )
            del running[index]
            save()
    report.update(status="failed" if report["failures"] else "completed", ended_at=_now())
    save()
    journal.finish(
        attempt,
        status="failed" if report["failures"] else "done",
        reason="; ".join(report["failures"])
        or "All declared cases completed; qualification reducer still required",
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal-root", required=True)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--no-progress-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    result = run_batch(
        original=args.original,
        design_path=args.design,
        output=args.output,
        journal_root=args.journal_root,
        workers=args.workers,
        no_progress_timeout_seconds=args.no_progress_timeout,
    )
    print(
        json.dumps(
            dict(status=result["status"], counts=result["counts"], output=str(Path(args.output).resolve()))
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
