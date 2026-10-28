"""Bounded native DEVELOPMENT capture; grants no learning/acceptance capability."""

from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
from .dual_coordination_progress import RunJournal, atomic_json
from .dual_payload_mujoco import DualMuJoCoFixture
from .dual_coordination_safety import audit_geometry, SafetyLedger
from .dual_coordination_reference import ContinuousReference, KinematicState
from .payload_contract import Payload
from .payload_hybrid_controller import Controller
from .payload_solver_mujoco import FREE_MASS_KG, HOVER_RPM
from .util import sha256


def capture_probe(
    *, original, design, output, duration_s, kind="hover", sides=(0.045, 0.045), ratios=(0.2, 0.2)
):
    if (
        not np.isfinite(duration_s)
        or not 0 < duration_s <= 330
        or abs(duration_s * 100 - round(duration_s * 100)) > 1e-8
    ):
        raise ValueError("Explicit duration in (0,330] on control boundaries required")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    journal = RunJournal(root)
    attempt = journal.start(
        "native-probe", f"{kind} {duration_s} simulated seconds", category="physical qualification"
    )
    try:
        config = dict(
            kind=kind,
            duration_s=duration_s,
            sides=list(sides),
            ratios=list(ratios),
            profile="dual-payload-v1",
            scope="DEVELOPMENT_ONLY",
            formal_admission=False,
        )
        atomic_json(root / "probe.json", config)
        if kind not in ("hover", "vertical", "transfer", "brake"):
            raise ValueError("Declared development probe required")
        compile_start = time.monotonic()
        f = DualMuJoCoFixture(
            original, root / "fixture", [Payload(FREE_MASS_KG, r, s) for r, s in zip(ratios, sides)]
        )
        compile_s = time.monotonic() - compile_start
        atomic_json(root / "properties.json", f.runtime())
        atomic_json(root / "controller.json", design)
        controllers = [Controller(design["gain"], design["hover_action"]) for _ in range(2)]
        samples = f.sample()["drones"]
        refs = [ContinuousReference(s["drone_xyz"]) for s in samples]
        if kind != "hover":
            targets = [[-3.5, -3.5, 3], [3.5, 3.5, 3]] if kind == "vertical" else [[-2, -2, 2], [2, 2, 2]]
            refs = [
                r.retarget(g, 0, actual_state=KinematicState(s["drone_xyz"], s["drone_velocity"]))
                for r, g, s in zip(refs, targets, samples)
            ]
        ledger = SafetyLedger()
        ledger.observe(audit_geometry(f))
        run_start = time.monotonic()
        trace = root / "native-control.jsonl.gz"
        next_heartbeat = run_start + 60
        completed = 0
        error = None
        with gzip.open(trace, "wt", compresslevel=3) as raw:
            raw.write(
                json.dumps(
                    dict(kind="RESET", actual=f.sample(), geometry=f.geometry_state()), separators=(",", ":")
                )
                + "\n"
            )
            for tick in range(round(duration_s * 100)):
                now = tick * 0.01
                sample = f.sample()
                if kind == "brake" and tick == 200:
                    refs = [
                        r.brake(now, actual_state=KinematicState(s["drone_xyz"], s["drone_velocity"]))
                        for r, s in zip(refs, sample["drones"])
                    ]
                references = [r.sample(now) for r in refs]
                actions = np.array(
                    [
                        c.action(s, p, tick=tick, reference_velocity=v)
                        for c, s, (p, v, a) in zip(controllers, sample["drones"], references)
                    ]
                )
                rpm = (1 + 0.8 * actions) * HOVER_RPM
                f.step_rpm(rpm, observe_stages=True)
                diagnostics = audit_geometry(f)
                ledger.observe(diagnostics)
                raw.write(
                    json.dumps(
                        dict(
                            kind="STEP",
                            tick=tick + 1,
                            actual=f.sample(),
                            controls=[c.last_audit for c in controllers],
                            reference=[
                                dict(p=p.tolist(), v=v.tolist(), a=a.tolist()) for p, v, a in references
                            ],
                            stages=f.stage_rows,
                            substeps=f.substep_rows,
                            audit=diagnostics,
                        ),
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                completed = tick + 1
                if not ledger.safe:
                    error = "observed safety violation"
                    break
                if time.monotonic() - run_start > 600:
                    error = "600 s mission wall deadline exceeded"
                    break
                if time.monotonic() >= next_heartbeat:
                    journal.heartbeat(attempt, completed=completed, total=round(duration_s * 100))
                    next_heartbeat = time.monotonic() + 60
        elapsed = time.monotonic() - run_start
        if elapsed > 600:
            error = "600 s mission wall deadline exceeded including raw flush"
        result = dict(
            **config,
            completed_control_steps=completed,
            simulated_seconds=completed * 0.01,
            compilation_wall_seconds=compile_s,
            execution_and_flush_wall_seconds=elapsed,
            completed_requested_duration=completed == round(duration_s * 100),
            observed_control_boundary_safe=ledger.safe,
            safety_failures=ledger.failures,
            failure=error,
            minimum_control_boundary_clearance_m=ledger.minimum_clearance_m,
            control_boundary_floor_minima_m=ledger.floor_minima_m,
            raw_bytes=trace.stat().st_size,
            raw_sha256=sha256(trace),
            estimated_raw_bytes_per_simulated_second=trace.stat().st_size / (completed * 0.01)
            if completed
            else None,
            internal_stage_independent_audit="pending",
            continuous_safety_qualified=False,
            final_states=f.sample()["drones"],
        )
        atomic_json(root / "result.json", result)
        journal.heartbeat(attempt, completed=completed, total=round(duration_s * 100))
    except BaseException as exc:
        try:
            journal.finish(attempt, status="failed", reason=f"{type(exc).__name__}: {exc}")
        except Exception as journal_error:
            exc.add_note(f"Failure finalization could not be persisted: {journal_error}")
        raise
    journal.finish(
        attempt,
        status="done" if error is None else "failed",
        reason=error or "capture complete; independent internal audit pending",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--kind", default="hover")
    args = parser.parse_args()
    result = capture_probe(
        original=args.original,
        design=json.loads(Path(args.design).read_text()),
        output=args.output,
        duration_s=args.duration,
        kind=args.kind,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "final_states"}, indent=2))
    raise SystemExit(0 if result["failure"] is None else 1)
