"""Continuous target recording and independent raw settling reconstruction.

Synthetic CPU records exercise rejection rules only. Physical acceptance also
requires current complete admission, actual source/model/initial-state binding,
and the independently verified full prescribed validation set.
"""

import copy
import json
import math
from pathlib import Path
import time

import numpy as np

from .jobs import write_json
from .payload_contract import SharedSettling, rotation
from .util import sha256


def measured_predicates(sample, target, side):
    """Derive geometry and every predicate from measured bodies, not completion."""
    if "contact" in sample and (type(sample["contact"]) is not bool
            or sample["contact"] != sample.get("contacts")):
        raise ValueError("Native contact and safety contact alias differ")
    values = {}
    for body in ["drone", "package"]:
        for field, width in [("xyz", 3), ("velocity", 3), ("angular_velocity", 3), ("quaternion_wxyz", 4)]:
            a = np.asarray(sample[f"{body}_{field}"], dtype=float)
            if a.shape != (width,) or not np.isfinite(a).all():
                raise ValueError("Invalid measured body state")
            values[f"{body}_{field}"] = a
    d, p = values["drone_xyz"], values["package_xyz"]
    vd, vp = values["drone_velocity"], values["package_velocity"]
    wd, wp = values["drone_angular_velocity"], values["package_angular_velocity"]
    rd, rp = [rotation(values[f"{b}_quaternion_wxyz"]) for b in ["drone", "package"]]
    a, b = rd @ [0.0, 0.0, -0.02], rp @ [0.0, 0.0, side / 2]
    r = p + b - d - a
    length = float(np.linalg.norm(r))
    if length <= 0 or type(sample.get("contacts")) is not bool:
        raise ValueError("Missing contact or articulated geometry")
    relative = vp + np.cross(wp, b) - vd - np.cross(wd, a)
    angle = math.degrees(math.acos(float(np.clip(-r[2] / length, -1, 1))))
    rate = math.degrees(float(np.linalg.norm(np.cross(r, relative))) / length**2)
    floor = p[2] - side / 2 * sum(abs(rp[2]))
    vehicle_floor = d[2] - 0.06 * np.linalg.norm(rd[2, :2]) - 0.0125 * abs(rd[2, 2])
    roll = math.degrees(math.atan2(rd[2, 1], rd[2, 2]))
    pitch = math.degrees(math.asin(float(np.clip(-rd[2, 0], -1, 1))))
    drone_error = float(np.linalg.norm(d - target))
    package_error = float(np.linalg.norm(p - (np.asarray(target) + [0, 0, -0.32 - side / 2])))
    unsafe = bool(
        sample["contacts"]
        or floor <= 0.1
        or vehicle_floor <= 0.1
        or abs(roll) > 60
        or abs(pitch) > 60
        or abs(length - 0.3) > 2e-5
    )
    if any(np.linalg.norm(v) > bound for v, bound in [(vd, 20), (vp, 20), (wd, 200), (wp, 200)]):
        raise ValueError("Numerical speed excursion")
    if sample.get("done") is not False or sample.get("reset") is not False:
        raise ValueError("Hidden reset/termination")
    valid = bool(
        not unsafe
        and drone_error <= 0.2
        and package_error <= 0.2
        and np.linalg.norm(vd) <= 0.3
        and np.linalg.norm(vp) <= 0.1
        and angle <= 5
        and rate <= 10
        and math.degrees(np.linalg.norm(wp)) <= 10
    )
    return dict(
        valid=valid,
        unsafe=unsafe,
        drone_error_m=drone_error,
        package_error_m=package_error,
        drone_floor_m=float(vehicle_floor),
        package_floor_m=float(floor),
        swing_deg=angle,
        swing_rate_deg_s=rate,
        constraint_residual_m=abs(length - 0.3),
    )


def _line(stream, row):
    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
    stream.flush()


def record_sequence(flight, targets, output, metadata, *, clock=time.monotonic):
    """Advance every target; preserve failures without resetting or relabelling."""
    kind = metadata.get("evidence_kind")
    if kind not in {"physical", "cpu-fixture"}:
        raise ValueError("Explicit evidence provenance required")
    if kind == "physical":
        from .payload_recovery_mission import require_solver_mission_admission

        require_solver_mission_admission(flight, metadata.get("physical_evidence", {}))
    declared = np.asarray(targets, dtype=float)
    if declared.ndim != 2 or declared.shape[1] != 3 or not np.isfinite(declared).all():
        raise ValueError("Explicit finite target sequence required")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    portable = {k: v for k, v in metadata.items() if k != "physical_evidence"}
    write_json(
        root / "declaration.json",
        dict(
            targets=declared.tolist(),
            metadata=portable,
            side_m=flight.runtime.payload.side_m,
            limits=dict(leg_s=30, mission_s=120, wall_s=600),
        ),
    )
    previous = copy.deepcopy(flight.sample())
    step, completed, legs = 0, [], []
    started = clock()
    status, error = "FAILED", None
    try:
        with (root / "samples.jsonl").open("x") as stream:
            _line(stream, dict(kind="INITIAL", step=0, sample=previous))
            for index, target in enumerate(declared):
                before = copy.deepcopy(flight.sample())
                flight.set_target(target.tolist(), target_id=str(index))
                after = copy.deepcopy(flight.sample())
                _line(
                    stream,
                    dict(
                        kind="RETARGET",
                        step=step,
                        target_index=index,
                        target=target.tolist(),
                        sample=after,
                        before=before,
                    ),
                )
                gate = SharedSettling(0.01)
                gate.update(step * 0.01, False, str(index))
                leg_start = step
                while True:
                    if clock() - started >= 600 or step >= 12000 or step - leg_start >= 3000:
                        raise TimeoutError("Prescribed sequence deadline")
                    sample = copy.deepcopy(flight.advance())
                    step += 1
                    _line(
                        stream,
                        dict(
                            kind="STEP", step=step, target_index=index, target=target.tolist(), sample=sample
                        ),
                    )
                    predicates = measured_predicates(sample, target, flight.runtime.payload.side_m)
                    if predicates["unsafe"]:
                        raise ValueError("Actual physical safety failure")
                    complete = gate.update(step * 0.01, predicates["valid"], str(index))
                    previous = sample
                    if complete:
                        completed.append(index)
                        legs.append(
                            dict(
                                target_index=index,
                                start_step=leg_start,
                                end_step=step,
                                valid_intervals=gate.intervals,
                            )
                        )
                        break
        status = "SUCCEEDED"
    except BaseException as exc:
        error = repr(exc)
        write_json(root / "failure-sample.json", dict(error=error, step=step, sample=flight.sample()))
    summary = dict(
        status=status,
        error=error,
        completed=completed,
        steps=step,
        legs=legs,
        wall_s=clock() - started,
        samples_sha256=sha256(root / "samples.jsonl"),
        declaration_sha256=sha256(root / "declaration.json"),
    )
    write_json(root / "summary.json", summary)
    return summary


def _same_physical(a, b):
    for body in ["drone", "package", "rod"]:
        if body + "_xyz" not in a and body + "_xyz" not in b:
            continue
        for field in ["xyz", "quaternion_wxyz", "velocity", "angular_velocity"]:
            if not np.array_equal(a[f"{body}_{field}"], b[f"{body}_{field}"]):
                raise ValueError("Retarget changed physical state")
    for key in [
        "qpos",
        "qvel",
        "qacc",
        "rpm",
        "actual_applied_generalized_force",
        "actual_passive_generalized_force",
        "actual_xfrc_applied",
    ]:
        if key in a or key in b:
            if key not in a or key not in b or not np.array_equal(a[key], b[key]):
                raise ValueError("Retarget changed native joint/input state")
    if not np.array_equal(a["actions"], b["actions"]):
        raise ValueError("Retarget changed applied action history")


def _continuous(previous, sample):
    for body in ["drone", "package"]:
        p0, p1 = [np.asarray(x[f"{body}_xyz"]) for x in [previous, sample]]
        speed = max(np.linalg.norm(x[f"{body}_velocity"]) for x in [previous, sample])
        if np.linalg.norm(p1 - p0) > speed * 0.01 + 0.005:
            raise ValueError("Discontinuous measured body motion")
        q0, q1 = [np.asarray(x[f"{body}_quaternion_wxyz"]) for x in [previous, sample]]
        turn = 2 * math.acos(float(np.clip(abs(q0 @ q1) / (np.linalg.norm(q0) * np.linalg.norm(q1)), 0, 1)))
        spin = max(np.linalg.norm(x[f"{body}_angular_velocity"]) for x in [previous, sample])
        if turn > spin * 0.01 + 0.005:
            raise ValueError("Discontinuous measured rotation")


def verify_sequence(output, targets, *, native=None, allow_incomplete=False):
    """Replay complete time/command/body/geometry history, ignoring summary PASS."""
    root = Path(output)
    summary = json.loads((root / "summary.json").read_text())
    declaration = json.loads((root / "declaration.json").read_text())
    if (
        sha256(root / "samples.jsonl") != summary["samples_sha256"]
        or sha256(root / "declaration.json") != summary["declaration_sha256"]
        or declaration["targets"] != np.asarray(targets).tolist()
    ):
        raise ValueError("Stale sequence binding")
    if (
        declaration["limits"] != dict(leg_s=30, mission_s=120, wall_s=600)
        or type(summary["wall_s"]) not in (int, float)
        or not math.isfinite(summary["wall_s"])
        or summary["wall_s"] < 0
        or (not allow_incomplete and summary["wall_s"] >= 600)
    ):
        raise ValueError("Sequence limits differ")
    step, index, leg_start, completed, previous, active = 0, 0, 0, [], None, False
    intervals, previous_valid, distance = 0, False, 0.0
    rows = list(map(json.loads, (root / "samples.jsonl").read_text().splitlines()))
    physical = declaration["metadata"]["evidence_kind"] == "physical"
    if native is not None and not physical:
        raise ValueError('Actual physical evidence required when native binding is supplied')
    native_proof = None
    terminal_extra = False
    # Safety can raise after a real native STEP but before a derived sequence
    # row is returned. Reconcile that one preserved terminal sample explicitly.
    if allow_incomplete and summary['status'] == 'FAILED' and physical and native:
        import gzip
        with gzip.open(native['native_path'], 'rt') as stream:
            actual_records = [json.loads(line) for line in stream]
        if len(actual_records) == len(rows) + 1:
            record = actual_records[-1]
            failure = json.loads((root / 'failure-sample.json').read_text())
            command = next(r for r in reversed(rows) if r['kind'] == 'RETARGET')
            if (record['kind'] != 'STEP' or record['step'] != summary['steps'] + 1
                    or failure['step'] != summary['steps']):
                raise ValueError('Unexplained extra native failure record')
            sample = copy.deepcopy(failure['sample'])
            sample.update(step=record['step'], time_s=record['step'] * .01)
            rows.append(dict(kind='STEP', step=record['step'], target_index=command['target_index'],
                             target=command['target'], sample=sample))
            terminal_extra = True
    terminal_unsafe = False
    if physical:
        if native is None:
            raise ValueError("Actual native raw/checkpoint/case binding is required")
        if (
            declaration["metadata"]["case"] != native["case"]
            or declaration["metadata"]["checkpoint_sha256"] != native["expected_hash"]
            or declaration["metadata"]["actual_binding"] != native["binding"]
        ):
            raise ValueError("Frozen physical case/model binding differs")
        from .payload_solver_trace import verify_native_sequence

        native_proof = verify_native_sequence(rows, **native)
    for row_number, row in enumerate(rows):
        sample = row["sample"]
        if row["kind"] == "INITIAL":
            if previous is not None or row["step"] != 0:
                raise ValueError("Duplicate initial/reset")
            previous = sample
            continue
        if previous is None or index >= len(targets):
            raise ValueError("Missing initial or extra samples")
        if row["target_index"] != index or not np.array_equal(row["target"], targets[index]):
            raise ValueError("Command label differs from prescribed target")
        if not np.array_equal(sample["effective_target"], targets[index]):
            raise ValueError("Applied command differs")
        if row["kind"] == "RETARGET":
            if active or row["step"] != step:
                raise ValueError("Unexpected retarget/reset")
            _same_physical(previous, row["before"])
            _same_physical(row["before"], sample)
            previous = sample
            leg_start, intervals, previous_valid, active = step, 0, False, True
            continue
        if row["kind"] != "STEP" or not active:
            raise ValueError("Step without declared retarget")
        step += 1
        if (
            row["step"] != step
            or sample["step"] != step
            or not math.isclose(sample["time_s"], step * 0.01, abs_tol=1e-8)
        ):
            raise ValueError("Missing/duplicate samples or internal reset")
        if step > 12000 or step - leg_start > 3000:
            raise ValueError("Simulation timeout")
        _continuous(previous, sample)
        predicates = measured_predicates(sample, targets[index], declaration["side_m"])
        if predicates["unsafe"]:
            if not allow_incomplete or row_number != len(rows) - 1 or summary['status'] != 'FAILED':
                raise ValueError("Measured physical safety failure")
            terminal_unsafe = True
        distance += float(np.linalg.norm(np.array(sample["drone_xyz"]) - previous["drone_xyz"]))
        previous = sample
        intervals = intervals + 1 if predicates["valid"] and previous_valid else 0
        previous_valid = predicates["valid"]
        if intervals >= 100:
            completed.append(index)
            index += 1
            active = False
    if summary['steps'] != step - int(terminal_extra) or summary['completed'] != completed:
        raise ValueError('Actual sequence prefix/sample counts differ')
    passed = (completed == list(range(len(targets))) and summary['status'] == 'SUCCEEDED'
              and summary['wall_s'] < 600 and not terminal_unsafe)
    if not passed:
        if not allow_incomplete or summary['status'] != 'FAILED':
            raise ValueError("Incomplete prescribed continuous target sequence")
        failure = json.loads((root / 'failure-sample.json').read_text())
        if (failure['step'] != summary['steps'] or failure['error'] != summary['error']
                or not (terminal_unsafe or step - leg_start == 3000 or step == 12000
                        or summary['wall_s'] >= 600)):
            raise ValueError('Incomplete evidence has no independently measured terminal cause')
    return dict(
        passed=passed,
        failure_verified=not passed,
        completed_targets=len(completed),
        steps=step,
        measured_distance_m=distance,
        evidence_kind=declaration["metadata"]["evidence_kind"],
        physical_acceptance=False,
        samples_sha256=summary["samples_sha256"],
        native_proof=native_proof,
    )


def verify_initial_sample(sample, initial):
    """Compare the actual initial bodies to the independently declared case."""
    for body in ["drone", "package"]:
        for field in ["xyz", "velocity", "angular_velocity"]:
            if not np.allclose(sample[f"{body}_{field}"], initial[f"{body}_{field}"], rtol=0, atol=1e-8):
                raise ValueError("Actual case initial body differs")
        actual = np.asarray(sample[f"{body}_quaternion_wxyz"])
        expected = np.asarray(initial[f"{body}_quaternion_wxyz"])
        if min(np.linalg.norm(actual - expected), np.linalg.norm(actual + expected)) > 1e-8:
            raise ValueError("Actual case initial quaternion differs")
