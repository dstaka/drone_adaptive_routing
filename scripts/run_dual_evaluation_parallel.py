"""CPU administrative scheduling around the unchanged frozen mission APIs.

Child processes defer only global reduction. Every original mission, physics,
online audit and retained-input check still runs. The coordinator alone invokes
the canonical full reduction after all disjoint methods succeed. No retries.
This file is deployed outside the immutable LearningLock source directory.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from adaptive_drone_patrol.dual_coordination_progress import atomic_json, utc_now


METHODS = {
    "validation": ("B-static", "B-dynamic", "N42", "N43", "N44"),
    "diagnostics": ("N-no-transfer", "N-fixed-priority"),
    "routing": ("B-static", "B-dynamic", "N-full", "N-no-transfer", "N-fixed-priority"),
}


def deferred_reduction(*args, **kwargs):
    return dict(status="deferred-to-coordinator", acceptance_verified=False)


def effective_cpu_count():
    limits = [
        float(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    ]
    maximum = Path("/sys/fs/cgroup/cpu.max")
    if maximum.exists():
        quota, period = maximum.read_text().split()
        if quota != "max":
            limits.append(int(quota) / int(period))
    else:
        base = Path("/sys/fs/cgroup/cpu")
        if (base / "cpu.cfs_quota_us").exists():
            quota = int((base / "cpu.cfs_quota_us").read_text())
            if quota > 0:
                limits.append(quota / int((base / "cpu.cfs_period_us").read_text()))
    return min(limits)


def coordinate(
    methods,
    command,
    *,
    output,
    execution,
    max_workers,
    effective_cpus,
    finalize,
    poll_seconds=1.0,
    initialization_timeout=3600.0,
    heartbeat=None,
):
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Require nonempty unique methods")
    if not math.isfinite(effective_cpus) or not 1 <= max_workers <= min(5, math.floor(effective_cpus)):
        raise ValueError("CPU workers must fit quota and maximum five")
    if poll_seconds <= 0 or initialization_timeout <= 0:
        raise ValueError("Positive polling and declaration watchdog required")
    output, execution = Path(output), Path(execution)
    if output.exists():
        raise FileExistsError("Do not rerun or reuse an existing evaluation output")
    execution.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(
        status="running",
        started_at=utc_now(),
        methods=list(methods),
        max_workers=max_workers,
        effective_cpus=effective_cpus,
        global_reduction_calls=0,
        children=[],
        acceptance_verified=False,
        scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initialization_timeout_seconds=initialization_timeout,
        scope="Method scheduling only; original mission audits/deadlines unchanged",
    )
    active = {}
    pending = list(methods)
    logs = []

    def save():
        state.update(observed_at=utc_now(), elapsed_wall_seconds=time.monotonic() - started)
        atomic_json(execution / "state.json", state)

    def launch(method):
        log = (execution / (method + ".log")).open("x")
        logs.append(log)
        args = command(method)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        child = subprocess.Popen(
            args, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        )
        row = dict(
            method=method,
            pid=child.pid,
            command=args,
            started_at=utc_now(),
            monotonic_start=time.monotonic(),
            status="running",
        )
        state["children"].append(row)
        active[method] = (child, row)
        save()

    try:
        launch(pending.pop(0))
        ready = False
        last_heartbeat = -math.inf
        while active or pending:
            for method, (child, row) in list(active.items()):
                code = child.poll()
                if code is not None:
                    row.update(
                        returncode=code,
                        status="completed" if code == 0 else "failed",
                        ended_at=utc_now(),
                        elapsed_wall_seconds=time.monotonic() - row["monotonic_start"],
                    )
                    del active[method]
                    if code:
                        raise RuntimeError(f"child {method} exited {code}; preserve slots, no retry")
            if not ready and (output / "declaration.json").is_file():
                # Original producer commits this atomically; peers check exact equality.
                json.loads((output / "declaration.json").read_text())
                ready = True
            if not ready:
                if not active:
                    raise RuntimeError("First child ended without an atomic declaration")
                if time.monotonic() - started > initialization_timeout:
                    raise RuntimeError("Declaration infrastructure watchdog; no mission deadline changed")
            while ready and pending and len(active) < max_workers:
                launch(pending.pop(0))
            now = time.monotonic()
            if now - last_heartbeat >= 30:
                save()
                if heartbeat:
                    heartbeat(sum(r["status"] == "completed" for r in state["children"]), len(methods))
                last_heartbeat = now
            if active or pending:
                time.sleep(poll_seconds)
        state.update(status="canonical-reduction", global_reduction_calls=1)
        save()
        result = finalize()
        state["result"] = result
        if result.get("status") != "complete":
            raise RuntimeError("Canonical reducer did not verify a complete campaign")
        state.update(
            status="complete",
            acceptance_verified=False,
            ended_at=utc_now(),
            meaning="All retained slots reduced; scientific acceptance remains original gates",
        )
        save()
        return state
    except BaseException as error:
        for child, row in active.values():
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                row.update(
                    status="cancelled-after-coordinator-failure",
                    returncode=child.returncode,
                    ended_at=utc_now(),
                    elapsed_wall_seconds=time.monotonic() - row["monotonic_start"],
                )
        state.update(status="failed", failure=repr(error), ended_at=utc_now())
        save()
        raise
    finally:
        for log in logs:
            log.close()


def phase_api(phase, argv):
    """Parse the existing phase arguments without permitting case/budget overrides."""
    parser = argparse.ArgumentParser()
    common = ["original", "output", "journal-root"]
    if phase == "validation":
        names = common + ["learning-lock", "terminal42", "terminal43", "terminal44", "design"]
    elif phase == "diagnostics":
        names = common + ["learning-lock", "selection", "terminal", "design-path"]
    else:
        names = common + ["freeze", "heldout-root", "cohort", "design"]
    for name in names:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    kw = vars(args).copy()
    if phase == "validation":
        from adaptive_drone_patrol import dual_coordination_evaluate as module

        if set(module.METHODS) != set(METHODS[phase]):
            raise ValueError("Frozen validation method set differs")
        kw["terminals"] = {s: kw.pop(f"terminal{s}") for s in (42, 43, 44)}
        kw["design_path"] = kw.pop("design")
        run = module.run_validation

        def reduce():
            lock = module.VerifiedLearningLock.load(args.learning_lock)
            lock.require_current()
            declaration = json.loads((Path(args.output) / "declaration.json").read_text())
            return module.aggregate_validation(args.output, declaration, root=lock.binding.root)

        reducer_name = "aggregate_validation"
    elif phase == "diagnostics":
        from adaptive_drone_patrol import dual_coordination_diagnostics as module

        if tuple(module.METHODS) != METHODS[phase]:
            raise ValueError("Frozen diagnostic method set differs")
        run = module.run_diagnostics

        # Original API skips all retained slots before constructing any selectors.
        # This final call authenticates and summarizes600 slots, without re-flying.
        def reduce():
            return run(**kw)

        reducer_name = "summarize"
    else:
        from adaptive_drone_patrol import dual_coordination_heldout as module
        from adaptive_drone_patrol.dual_coordination_campaign import VerifiedJointFreeze

        if args.cohort != "routing" or tuple(module.ROUTING_METHODS) != METHODS[phase]:
            raise ValueError("Only declared routing methods may run in parallel")
        kw["capability"] = module.VerifiedHeldout.load(
            kw.pop("heldout_root"), freeze=VerifiedJointFreeze.load(kw.pop("freeze"))
        )
        kw["design_path"] = kw.pop("design")
        run = module.run_heldout

        def reduce():
            return module.aggregate_heldout(capability=kw["capability"], cohort="routing", output=args.output)

        reducer_name = "aggregate_heldout"
    return args, module, run, kw, reducer_name, reduce


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=METHODS, required=True)
    parser.add_argument("--child-method")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    phase, module, run, kw, reducer_name, reduce = phase_api(args.phase, forwarded)
    if args.child_method:
        if args.child_method not in METHODS[args.phase]:
            raise ValueError("Method outside declared phase")
        original = getattr(module, reducer_name)
        setattr(module, reducer_name, deferred_reduction)
        try:
            result = run(**kw, method=args.child_method)
        finally:
            setattr(module, reducer_name, original)
        if result.get("status") != "deferred-to-coordinator":
            raise RuntimeError("Child must defer global selection/reduction")
        return 0
    from adaptive_drone_patrol.dual_coordination_progress import RunJournal

    journal = RunJournal(phase.journal_root)
    attempt = journal.start(
        "parallel-" + args.phase,
        "CPU methods; original per-mission audits and finite gates",
        category="evaluation scheduling",
    )
    try:
        quota = effective_cpu_count()
        state = coordinate(
            METHODS[args.phase],
            lambda method: [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phase",
                args.phase,
                "--child-method",
                method,
                "--",
                *forwarded,
            ],
            output=phase.output,
            execution=phase.output + "-parallel-execution",
            max_workers=min(5, len(METHODS[args.phase]), math.floor(quota)),
            effective_cpus=quota,
            finalize=reduce,
            heartbeat=lambda done, total: journal.heartbeat(
                attempt,
                completed=done,
                total=total,
                reason="Independent methods; no global selection before all children complete",
            ),
        )
        journal.finish(attempt, status="done", reason="Original canonical complete reducer passed")
        print(json.dumps(state["result"], allow_nan=False))
        return 0
    except BaseException as error:
        journal.finish(attempt, status="failed", reason=repr(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
