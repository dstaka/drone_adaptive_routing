"""Read-only completed-update counters for an existing finite CPU campaign.

This observer never loads checkpoints, changes training inputs, starts a trainer,
or opens evaluation cases. Curves give a lower bound, excluding a partial rollout.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from adaptive_drone_patrol.dual_coordination_progress import RunJournal, atomic_json, utc_now


def read_training_progress(root, *, previous=None):
    root = Path(root)
    seeds = []
    for seed in (42, 43, 44):
        path = root / f"seed-{seed}" / "curves.json"
        row = dict(seed=seed, completed_updates=0, completed_transitions=0, last_completed_stage=None)
        if path.exists():
            payload = path.read_bytes()
            data = json.loads(payload)
            if data.get("seed") != seed or data.get("selectable") is not False:
                raise ValueError("Unexpected seed or selectable curves")
            updates = data.get("updates")
            if not isinstance(updates, list) or len(updates) > 256:
                raise ValueError("Finite update budget differs")
            for i, curve in enumerate(updates, 1):
                stage = 1 if i <= 64 else 2 if i <= 240 else 3
                if any(
                    type(curve.get(key)) is not int or curve[key] != expected
                    for key, expected in [("update", i), ("transitions", i * 4096), ("stage", stage)]
                ):
                    raise ValueError("Completed update counter differs from finite plan")
            row.update(
                completed_updates=len(updates),
                completed_transitions=len(updates) * 4096,
                last_completed_stage=updates[-1]["stage"] if updates else None,
                curves_sha256=hashlib.sha256(payload).hexdigest(),
                curves_bytes=len(payload),
                source=str(path.resolve()),
            )
        if previous is not None:
            prior = next(x for x in previous["seeds"] if x["seed"] == seed)
            if row["completed_updates"] < prior["completed_updates"]:
                raise ValueError("Durable update counters regressed")
        seeds.append(row)
    return dict(
        schema="dual-training-observation-v1",
        observed_at=utc_now(),
        seeds=seeds,
        completed_updates=sum(x["completed_updates"] for x in seeds),
        total_updates=768,
        completed_transitions_lower_bound=sum(x["completed_transitions"] for x in seeds),
        total_transitions=3145728,
        partial_rollout_transitions="not recorded",
        terminal_models_verified=False,
        acceptance_verified=False,
        scope="Atomic curves only; complete PPO updates, not partial rollout or acceptance",
    )


def _start_ticks(pid, proc_root):
    # comm can contain spaces and parentheses; field22 follows the last ')'.
    text = (Path(proc_root) / str(pid) / "stat").read_text()
    return text.rsplit(")", 1)[1].split()[19]


def training_process_matches(pid, start_ticks, *, proc_root=Path("/proc")):
    try:
        args = (Path(proc_root) / str(pid) / "cmdline").read_bytes().split(b"\0")
        return (
            len(args) > 2
            and args[1:3] == [b"-m", b"adaptive_drone_patrol.dual_coordination_train"]
            and _start_ticks(pid, proc_root) == start_ticks
        )
    except (OSError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--journal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--duration", type=float, default=30 * 86400)
    args = parser.parse_args()
    if not 1 <= args.interval <= 300 or args.duration <= 0:
        raise ValueError("Bounded positive observer interval/duration required")
    training = args.training_root.resolve(strict=True)
    for path in (args.output, args.journal_root):
        if path.resolve() == training or training in path.resolve().parents:
            raise ValueError("Observer outputs must be outside training inputs")
    ticks = _start_ticks(args.process_pid, Path("/proc"))
    if not training_process_matches(args.process_pid, ticks):
        raise ValueError("Existing live training process required; never start or restart it")
    args.output.mkdir(parents=True, exist_ok=False)
    journal = RunJournal(args.journal_root)
    attempt = journal.start(
        "production-training-observer",
        "Read-only completed-update lower bounds; no checkpoint load, policy action or evaluation",
        category="observability",
        kind="waiting",
    )
    started = time.monotonic()
    previous = None
    state = {}
    try:
        while True:
            state = read_training_progress(training, previous=previous)
            previous = state
            live = training_process_matches(args.process_pid, ticks)
            receipt = training / "training-complete.json"
            completed = receipt.exists()
            if completed:
                raw = json.loads(receipt.read_text())
                if raw.get("production_transitions") != 3145728 or state["completed_updates"] != 768:
                    raise ValueError("Completion receipt and curve counters disagree")
            state.update(
                training_pid=args.process_pid,
                training_process_start_ticks=ticks,
                training_process_verified_live=live,
                completion_receipt_present=completed,
                elapsed_wall_seconds=time.monotonic() - started,
                status="observing" if live else "process-ended",
            )
            atomic_json(args.output / "state.json", state)
            counts = ", ".join(f"seed{x['seed']}:{x['completed_updates']}/256" for x in state["seeds"])
            journal.heartbeat(
                attempt,
                completed=state["completed_transitions_lower_bound"],
                total=3145728,
                reason=counts
                + "; complete-update lower bound only; partial rollout not recorded; "
                + f"PID{args.process_pid} start{ticks} live={live}; no acceptance claim",
            )
            if completed:
                journal.finish(
                    attempt,
                    status="done",
                    reason="All finite curve counters and completion receipt present; model authentication and all acceptance remain separate",
                )
                return
            if not live:
                journal.finish(
                    attempt,
                    status="unknown",
                    reason="Same training PID/start identity no longer live, no completion receipt; training untouched, no restart",
                )
                return
            remaining = args.duration - (time.monotonic() - started)
            if remaining <= 0:
                journal.finish(
                    attempt,
                    status="done",
                    reason="Finite observer duration expired; training untouched, no completion implied",
                )
                return
            time.sleep(min(args.interval, remaining))
    except BaseException as error:
        state.update(status="observer-failed", failure=repr(error), observed_at=utc_now())
        atomic_json(args.output / "state.json", state)
        journal.finish(attempt, status="failed", reason=repr(error) + "; training untouched")
        raise


if __name__ == "__main__":
    main()
