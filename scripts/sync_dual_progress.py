"""Read-only bounded remote progress mirror; never starts/stops remote jobs."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from adaptive_drone_patrol.dual_coordination_progress import RunJournal, atomic_json, utc_now

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", required=True)
p.add_argument("--host", required=True)
p.add_argument("--port", required=True)
p.add_argument("--remote-run", required=True)
p.add_argument("--duration", type=float, default=10800)
a = p.parse_args()
root = Path(a.run)
journal = RunJournal(root)
attempt = journal.start(
    "remote-progress-sync", "Read-only remote progress mirror every60s", category="observability"
)
end = time.monotonic() + a.duration
try:
    while time.monotonic() < end:
        target = root / "remote-progress.json"
        previous = json.loads(target.read_text()) if target.exists() else {}
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-p",
                    a.port,
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "ServerAliveInterval=10",
                    "-o",
                    "ServerAliveCountMax=1",
                    a.host,
                    "cat " + shlex.quote(a.remote_run + "/progress.json"),
                ],
                text=True,
                capture_output=True,
                timeout=45,
                check=True,
            )
            remote = json.loads(result.stdout)
            if remote.get("schema") != "dual-progress-v1":
                raise ValueError("Unexpected remote progress schema")
            atomic_json(
                target,
                dict(
                    schema="dual-remote-progress-mirror-v1",
                    source_timestamp=remote["snapshot_generated_at"],
                    latest_successful_sync_at=utc_now(),
                    last_sync_attempt_at=utc_now(),
                    sync_status="succeeded",
                    snapshot=remote,
                ),
            )
            journal.heartbeat(
                attempt, reason="Remote snapshot synced; inspect per-job age rather than infer activity"
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            previous.update(
                schema="dual-remote-progress-mirror-v1",
                last_sync_attempt_at=utc_now(),
                sync_status="failed",
                error=str(error),
            )
            atomic_json(target, previous)
            journal.heartbeat(
                attempt, reason="Remote sync failed; previous snapshot is unconfirmed: " + str(error)
            )
        time.sleep(min(60, max(0, end - time.monotonic())))
    journal.finish(
        attempt, status="done", reason="Bounded read-only mirror duration ended; remote jobs untouched"
    )
except BaseException as error:
    journal.finish(attempt, status="failed", reason=f"{type(error).__name__}: {error}")
    raise
