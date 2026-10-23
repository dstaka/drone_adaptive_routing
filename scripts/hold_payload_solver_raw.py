#!/usr/bin/env python
"""Hold already delivered closed learning bytes in an owned RAM caretaker.

Run only after independently verifying the persistent external archive and all
extracted file sizes/SHA-256. This is a storage operation, never admission. The
caretaker must remain alive until original regular paths have been restored.
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from adaptive_drone_patrol.payload_solver_storage import (
    ShadowHold,
    ShadowInstallError,
    closed_inventory,
    hold_closed_files,
    measured_memory_available,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "job-status", "delivery", "ready"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.ready.exists():
        raise FileExistsError("Preserve original caretaker evidence")
    requested_exit = False

    def stop_after_restore(signum, frame):
        nonlocal requested_exit
        requested_exit = True

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, stop_after_restore)
    signal.signal(signal.SIGINT, stop_after_restore)
    plan = closed_inventory(
        args.root, json.loads(args.job_status.read_text()), json.loads(args.delivery.read_text())
    )
    available = measured_memory_available()
    # Establish top-level ownership before any function can substitute a leaf.
    held = ShadowHold(args.root, plan["files"])
    try:
        try:
            hold_closed_files(args.root, plan, available_bytes=available, owner=held)
            proof = held.proof()
        except ShadowInstallError as exc:
            held = exc.hold
            proof = dict(
                status="PARTIAL_STORAGE_INSTALL_REQUIRES_RESTORATION",
                root=str(args.root),
                pid=os.getpid(),
                installed=held.installed,
                error=str(exc),
                physical_acceptance=False,
            )
        proof["measured_available_memory_before_copy_bytes"] = available
        with args.ready.open("x") as stream:
            json.dump(proof, stream, indent=2)
        print(json.dumps({k: v for k, v in proof.items() if k != "files"}), flush=True)
    except BaseException as exc:
        try:
            print(
                f"Storage reporting failed; retain live descriptors until exact restore: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
        except BaseException:
            pass
        try:
            sys.stdout.close()
        except BaseException:
            pass
        sys.stdout = open(os.devnull, "w")
        if not held.installed:
            raise
    finally:
        if held is not None:
            # Reporting, broken pipes and shutdown requests cannot release a
            # descriptor while any original leaf still depends on it.
            while held.installed:
                try:
                    if requested_exit and held.release_if_restored():
                        break
                    signal.pause()
                except BaseException:
                    requested_exit = True
            held.close_uninstalled()


if __name__ == "__main__":
    main()
