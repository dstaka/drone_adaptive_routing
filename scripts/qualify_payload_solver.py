#!/usr/bin/env python
"""Record full powered/refinement evidence after independently replayed passive32.

Run in the isolated Python3.12 reference environment. This entry never imports
an optimizer or grants training admission. The raw checker is a separate step.
"""

import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import time
import traceback

import numpy as np

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_contract import Payload, payload_metrics
from adaptive_drone_patrol.payload_probe_controller import diagnostic_rpm
from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG
from adaptive_drone_patrol.payload_solver_qualification import (
    PHYSICAL_SOURCES,
    make_manifest,
    passive_cases,
    require_passive_barrier,
)
from adaptive_drone_patrol.payload_solver_work import (
    FIELD_WIDTHS,
    pack_stages,
    reconstruct_interval_work,
    total_mechanical_energy,
)
from adaptive_drone_patrol.util import sha256

SOURCES = (
    *PHYSICAL_SOURCES,
    "src/adaptive_drone_patrol/payload_solver_work.py",
    "scripts/qualify_payload_solver.py",
)


def record_case(root, case, manifest, original, *, replay=None, timeout_s=600.0):
    """Preserve full native rows and every RK stage, including failed attempts."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="solver-powered-case-v1",
        case=case,
        settings=manifest["settings"],
        source_binding={p: sha256(p) for p in SOURCES},
        diagnostic_duration_s=None,
        rpm_replay=None if replay is None else dict(path=str(replay), sha256=sha256(replay)),
        input_schedule="Prescribed replay RPM"
        if replay
        else "Frozen privileged geometric PD and inherited12s targets",
        stage_fields=FIELD_WIDTHS,
        timeout_s=timeout_s,
        physical_training_admitted=False,
    )
    write_json(root / "declaration.json", declaration)
    payload = Payload(FREE_MASS_KG, case["mass_ratio"], case["side_m"])
    fixture = MuJoCoFixture(original, root / "fixture", payload, substeps=case["substeps"])
    fixture.reset(swing_deg=case["initial_swing_deg"], direction_rad=case["initial_direction_rad"])
    runtime = fixture.runtime()
    write_json(root / "runtime.json", runtime)
    total = FREE_MASS_KG + payload.mass + 0.0001
    count = round(case["duration_s"] / 0.01)
    commands = np.load(replay, allow_pickle=False) if replay else None
    if commands is not None and (commands.shape != (count, 4) or not np.isfinite(commands).all()):
        raise ValueError("Complete identical-input replay required")
    recent = deque(maxlen=128)
    stage_buffer = []
    stage_index = []
    rpm_commands = []
    motor_work = dissipative_work = endpoint_bound = 0.0
    start = time.monotonic()
    summary = dict(status="STARTED", physical_training_admitted=False, steps=0)

    def flush_stages():
        if not stage_buffer:
            return
        path = root / f"stages-{len(stage_index):04d}.npz"
        np.savez_compressed(path, stages=np.stack(stage_buffer))
        previous = sum(x["control_intervals"] for x in stage_index)
        stage_index.append(
            dict(
                path=path.name,
                first_step=previous,
                control_intervals=len(stage_buffer),
                shape=list(np.stack(stage_buffer).shape),
                sha256=sha256(path),
            )
        )
        stage_buffer.clear()
        write_json(root / "stages-index.json", stage_index)

    try:
        with gzip.open(root / "raw.jsonl.gz", "wt") as stream:
            for step in range(count + 1):
                raw = fixture.raw_state()
                raw.update(
                    step=step,
                    motor_work_j=motor_work,
                    dissipative_work_j=dissipative_work,
                    legacy_endpoint_absolute_work_bound_j=endpoint_bound,
                )
                stream.write(json.dumps(raw, separators=(",", ":"), allow_nan=True) + "\n")
                recent.append(raw)
                if (
                    not np.isfinite(raw["qpos"]).all()
                    or not np.isfinite(raw["qvel"]).all()
                    or any(raw["warning_counts"])
                ):
                    raise ValueError("NUMERICAL_RAW_OR_BACKEND_WARNING")
                if not np.isclose(raw["time_s"], step * 0.01, atol=1e-7, rtol=0):
                    raise ValueError("INTERNAL_RESET_OR_TIME_DISCONTINUITY")
                sample = fixture.sample()
                metrics = payload_metrics(sample, [0, 0, 1], payload)
                R = fixture.data.xmat[fixture.ids["base_link"]].reshape(3, 3)
                metrics["drone_floor_m"] = sample["drone_xyz"][2] - (
                    0.06 * np.linalg.norm(R[2, :2]) + 0.0125 * abs(R[2, 2])
                )
                for body in ["drone", "rod", "package"]:
                    if (
                        np.linalg.norm(raw[body + "_velocity"]) > 20
                        or np.linalg.norm(raw[body + "_angular_velocity"]) > 200
                    ):
                        raise ValueError("FINITE_PHYSICAL_SPEED_EXCURSION")
                if (
                    metrics["drone_floor_m"] <= 0.1
                    or metrics["package_floor_m"] <= 0.1
                    or sample["contact"]
                    or abs(metrics["drone_roll_deg"]) > 60
                    or abs(metrics["drone_pitch_deg"]) > 60
                    or metrics["constraint_residual_m"] > 2e-5
                ):
                    raise ValueError("ACTUAL_PHYSICAL_SAFETY_FAILURE")
                energy = total_mechanical_energy(raw, runtime, payload.side_m)
                if step == 0:
                    initial_energy = energy
                if abs(energy - initial_energy - motor_work - dissipative_work) > 0.02:
                    raise ValueError("RK4_SIGNED_ENERGY_WORK_RESIDUAL")
                if energy - initial_energy > 0.02 + 1.1 * endpoint_bound:
                    raise ValueError("LEGACY_ENDPOINT_WORK_BOUND")
                summary["steps"] = step
                if time.monotonic() - start > timeout_s:
                    raise TimeoutError("Case progress deadline")
                if step == count:
                    break
                target = manifest["commands"][
                    int(step * 0.01 / manifest["command_interval_s"]) % len(manifest["commands"])
                ]
                rpm = (
                    commands[step]
                    if commands is not None
                    else diagnostic_rpm(
                        sample["drone_xyz"],
                        sample["drone_quaternion_wxyz"],
                        sample["drone_velocity"],
                        sample["drone_angular_velocity"],
                        target,
                        total,
                    )
                )
                rpm_commands.append(rpm.copy())
                rows = fixture.step_rpm(rpm, observe_stages=True)
                stage_buffer.append(pack_stages(rows))
                # Store actual stages before any work/safety check can fail.
                if len(stage_buffer) >= 128:
                    flush_stages()
                work = reconstruct_interval_work(
                    rows, substeps=case["substeps"], start_time_s=step * 0.01, total_mass_kg=total
                )
                motor_work += work["motor_work_j"]
                dissipative_work += work["dissipative_work_j"]
                endpoint_bound += 0.01 * max(abs(rows[0]["motor_power_w"]), abs(rows[-1]["motor_power_w"]))
                if step % 1000 == 0:
                    print(case["id"], "control-step", step, flush=True)
        summary.update(
            status="RAW_COMPLETE_UNCHECKED", motor_work_j=motor_work, dissipative_work_j=dissipative_work
        )
    except BaseException as exc:
        summary.update(status="FAILED", error=repr(exc))
        try:
            actual = fixture.raw_state()
        except Exception as getter:
            actual = dict(
                capture_error=repr(getter), qpos=fixture.data.qpos.tolist(), qvel=fixture.data.qvel.tolist()
            )
        (root / "abnormal-window.json").write_text(
            json.dumps(dict(preceding=list(recent), actual=actual), allow_nan=True) + "\n"
        )
        (root / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        flush_stages()
        np.save(root / "rpm.npy", np.asarray(rpm_commands), allow_pickle=False)
        summary.update(
            raw_sha256=sha256(root / "raw.jsonl.gz"),
            runtime_sha256=sha256(root / "runtime.json"),
            stages_index_sha256=sha256(root / "stages-index.json")
            if (root / "stages-index.json").exists()
            else None,
            rpm_sha256=sha256(root / "rpm.npy"),
        )
        write_json(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["passive", "powered", "refinement", "motor", "reset", "initial-reset"],
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--passive-probes", type=Path)
    args = parser.parse_args()
    if args.stage == "passive":
        from adaptive_drone_patrol.payload_solver_passive import record_passive_domain

        record_passive_domain(args.out, args.original)
        return 0
    if args.passive_probes is None:
        raise ValueError("Complete passive-probes must precede any powered stage")
    manifest = make_manifest(base_substeps=16)
    locations = {c["id"]: args.passive_probes / c["id"] for c in passive_cases(manifest)}
    require_passive_barrier(manifest, locations)
    if args.stage in {"motor", "reset", "initial-reset"}:
        from adaptive_drone_patrol.payload_solver_audits import (
            record_motor_audit,
            verify_motor_audit,
            record_reset_audit,
            verify_reset_audit,
        )
        from adaptive_drone_patrol.payload_solver_initial_reset import (
            record_initial_resets,
            verify_initial_resets,
        )

        record, verify = {
            "motor": (record_motor_audit, verify_motor_audit),
            "reset": (record_reset_audit, verify_reset_audit),
            "initial-reset": (record_initial_resets, verify_initial_resets),
        }[args.stage]
        record(args.out, args.original)
        write_json(args.out / "independent-check.json", verify(args.out))
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "manifest.json", manifest)
    plan = dict(
        schema="solver-powered-stage-plan-v1",
        stage=args.stage,
        manifest=manifest,
        source_binding={p: sha256(p) for p in SOURCES},
        original_sha256=sha256(args.original),
        inputs="Inherited controller schedule; refinement half uses exact base RPM array",
        numerical_metrics=manifest["limits"],
        energy=manifest["energy"],
        raw="All control samples and every four actual RK stages",
        timeout_per_case_s=600.0,
        physical_training_admitted=False,
    )
    write_json(args.out / "plan.json", plan)
    cases = [
        c for c in manifest["cases"] if c["kind"] == ("powered" if args.stage == "powered" else "refinement")
    ]
    results = {}
    for case in cases:
        if any(sha256(p) != h for p, h in plan["source_binding"].items()):
            raise ValueError("Frozen source changed")
        replay = None
        if args.stage == "refinement" and case["timestep_role"] == "half":
            match = next(
                c
                for c in cases
                if c["side_m"] == case["side_m"]
                and c["mass_ratio"] == case["mass_ratio"]
                and c["timestep_role"] == "base"
            )
            replay = args.out / match["id"] / "rpm.npy"
        results[case["id"]] = record_case(args.out / case["id"], case, manifest, args.original, replay=replay)
        write_json(args.out / "results.json", results)
        print(case["id"], results[case["id"]]["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
