#!/usr/bin/env python
"""Reconstruct measured mode/envelope/phase diagnostics from preserved raw."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_contract import rotation
from adaptive_drone_patrol.payload_hybrid_audit import independent_metrics, LIMITS, _valid
from adaptive_drone_patrol.payload_hybrid_diagnosis import modal_summary, command_phase
from adaptive_drone_patrol.payload_solver_mujoco import KF, KM, SPIN
from adaptive_drone_patrol.util import sha256


def analyse(root, out):
    case = json.loads((root / "case.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    with gzip.open(root / "control.jsonl.gz", "rt") as stream:
        controls = {r["step"]: r for r in map(json.loads, stream) if "step" in r}
    times, link_tilt, body_omega, world_omega, moments = [], [], [], [], []
    extrema = {k: float("inf") for k in LIMITS}
    max_spin = np.zeros(3)
    saturation = longest = current = 0
    out.mkdir(parents=True, exist_ok=False)
    with (
        gzip.open(root / "runtime/raw.jsonl.gz", "rt") as native,
        gzip.open(out / "history.jsonl.gz", "wt") as history,
    ):
        for raw in map(json.loads, native):
            if raw["kind"] != "STEP":
                current = 0
                continue
            s, step = raw["actual"], raw["step"]
            control = controls[step]
            metric = independent_metrics(s, control["target"], case["side_m"])
            margins = {k: v - metric[k] for k, v in LIMITS.items()}
            for k, v in margins.items():
                extrema[k] = min(extrema[k], v)
            current = current + 1 if _valid(metric, s["contact"]) else 0
            longest = max(longest, current)
            rp, rd = rotation(s["package_quaternion_wxyz"]), rotation(s["drone_quaternion_wxyz"])
            wp = np.asarray(s["package_angular_velocity"])
            body = rp.T @ wp
            max_spin = np.maximum(max_spin, np.abs(body))
            link = (
                np.asarray(s["package_xyz"])
                + rp @ [0, 0, case["side_m"] / 2]
                - np.asarray(s["drone_xyz"])
                - rd @ [0, 0, -0.02]
            )
            tilt = np.arctan2(link[:2], -link[2])
            rpm2 = np.asarray(s["rpm"]) ** 2
            # Original rotor COMs in the authored root frame; torque includes reaction spin.
            rotor_com = np.array(
                [[0.028, -0.028, 0], [-0.028, -0.028, 0], [-0.028, 0.028, 0], [0.028, 0.028, 0]]
            )
            torque = (
                rd
                @ np.r_[
                    np.sum(rotor_com[:, 1] * KF * rpm2),
                    -np.sum(rotor_com[:, 0] * KF * rpm2),
                    np.sum(SPIN * KM * rpm2),
                ]
            )
            action = np.asarray(control["control"]["motor_action"])
            saturated = bool(np.any(np.abs(action) >= 1 - 1e-12))
            saturation += saturated
            times.append(step * 0.01)
            link_tilt.append(tilt)
            body_omega.append(body)
            world_omega.append(wp)
            moments.append(torque)
            history.write(
                json.dumps(
                    dict(
                        step=step,
                        time_s=step * 0.01,
                        leg=control["leg"],
                        predicate_margins=margins,
                        box_body_omega_rad_s=body.tolist(),
                        box_world_omega_rad_s=wp.tolist(),
                        link_tilt_rad=tilt.tolist(),
                        commanded_world_moment_nm=torque.tolist(),
                        motor_saturated=saturated,
                    ),
                    separators=(",", ":"),
                )
                + "\n"
            )
    body_omega, world_omega, moments, link_tilt = map(
        np.asarray, [body_omega, world_omega, moments, link_tilt]
    )
    axes = []
    for i, axis in enumerate("xyz"):
        mode = modal_summary(times, world_omega[:, i])
        axes.append(
            dict(
                axis=axis,
                world_angular_rate=mode,
                body_angular_rate=modal_summary(times, body_omega[:, i]),
                command_to_world_motion=command_phase(moments[:, i], world_omega[:, i], mode["dominant_hz"]),
            )
        )
    result = dict(
        schema="hybrid-measured-control-diagnosis-v1",
        case_id=case["case_id"],
        backend=metadata["backend"],
        substeps=metadata["substeps"],
        vibration_feedback=metadata["vibration_feedback"],
        source_raw_sha256=sha256(root / "runtime/raw.jsonl.gz"),
        control_sha256=sha256(root / "control.jsonl.gz"),
        analysis_source_sha256=sha256(__file__),
        helper_source_sha256=sha256(
            Path(__file__).parents[1] / "src/adaptive_drone_patrol/payload_hybrid_diagnosis.py"
        ),
        minimum_predicate_margins=extrema,
        longest_consecutive_valid_samples=longest,
        motor_saturation_s=saturation * 0.01,
        maximum_absolute_box_body_omega_rad_s=max_spin.tolist(),
        angular_axes=axes,
        link_tilt=[modal_summary(times, link_tilt[:, i]) for i in range(2)],
        interpretation="Descriptive measured modes and RMS envelopes; closed-loop cross-spectral phase is not a causal plant delay. Insufficient spectral excitation remains unidentifiable.",
        qualification=False,
        learning_admitted=False,
    )
    write_json(out / "diagnosis.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    results = []
    for case in sorted(a.root.glob("C-v1-*/*/case.json")):
        tag = case.parent.parent.name + "/" + case.parent.name
        results.append(analyse(case.parent, a.out / tag))
        print("DIAGNOSIS", tag, flush=True)
    write_json(
        a.out / "summary.json",
        dict(
            cases=len(results),
            paired="Compare exact case_id across feedback on/off and s16/s32; these diagnostics do not replace gate counts.",
            results=results,
        ),
    )


if __name__ == "__main__":
    main()
