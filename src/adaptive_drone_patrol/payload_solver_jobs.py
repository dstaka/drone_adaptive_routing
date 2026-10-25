"""Exclusive solver subprocesses with heartbeats and failure preservation.

The lock is host-wide across solver workspaces. Historical accounting files are
never read or written. Deadlines bound individual jobs, not cumulative use.
"""

import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import subprocess
import threading
import time

from .jobs import terminate_and_reap, write_json

DEFAULT_LOCK = Path("/tmp/adaptive-drone-payload-gpu.lock")


def run_job(command, *, output, timeout_s, lock_path=DEFAULT_LOCK, grace_s=3.0, heartbeat_s=2.0):
    """Run one owned process group; preserve raw files on every exit path."""
    for name, value in [("timeout_s", timeout_s), ("grace_s", grace_s), ("heartbeat_s", heartbeat_s)]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    if not command or isinstance(command, (str, bytes)) or any(not isinstance(c, str) for c in command):
        raise ValueError("Nonempty executable argument list required")
    output, lock_path = Path(output), Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another GPU job owns the process lock") from exc
        output.mkdir(parents=True, exist_ok=False)
        result = dict(
            schema="payload-solver-job-v1",
            status="RUNNING",
            run_id=output.name,
            command=list(command),
            timeout_s=float(timeout_s),
            pid=None,
            physical_training_admitted=False,
        )
        proc, cleaned, previous = None, False, {}

        def interrupted(signum, _frame):
            raise SystemExit(f"Solver supervisor interrupted by signal {signum}")

        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, interrupted)
        try:
            write_json(output / "status.json", result)
            with (output / "job.log").open("w") as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result["pid"] = proc.pid
                write_json(output / "status.json", result)
                deadline = time.monotonic() + timeout_s
                sequence = 0
                while True:
                    write_json(
                        output / "heartbeat.json",
                        dict(sequence=sequence, pid=proc.pid, observed_unix_s=time.time(), status="RUNNING"),
                    )
                    sequence += 1
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        code = terminate_and_reap(proc, grace_s)
                        result.update(status="TIMED_OUT", exit_code=code)
                        cleaned = True
                        break
                    try:
                        code = proc.wait(timeout=min(heartbeat_s, remaining))
                    except subprocess.TimeoutExpired:
                        continue
                    # A child may outlive its parent; do not release ownership first.
                    terminate_and_reap(proc, grace_s)
                    cleaned = True
                    result.update(status="SUCCEEDED" if code == 0 else "FAILED", exit_code=code)
                    break
        except BaseException as exc:
            result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            try:
                if proc is not None and not cleaned:
                    terminate_and_reap(proc, grace_s)
                write_json(output / "status.json", result)
                write_json(output / "heartbeat.json", dict(pid=result["pid"], status=result["status"]))
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Specify an executable after --")
    result = run_job(command, output=args.out, timeout_s=args.timeout)
    print(json.dumps(result))
    return 0 if result["status"] == "SUCCEEDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
