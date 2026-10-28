"""Actual native supplemental experiments and evidence reducers.

Only these explicit experiments can supply the corresponding qualification
checks. Test pass counts and caller-supplied approval booleans are not evidence.
All thresholds below come from the existing native profile/supervisor/spec.
"""

from __future__ import annotations

from copy import deepcopy
import gzip
import json
import time
from pathlib import Path

import numpy as np

from .dual_coordination_audit import NativePointAuditor, OnlineNativeAudit
from .dual_coordination_progress import atomic_json
from .dual_coordination_qualification_report import (
    _finite,
    _require,
    check_online,
    file_record,
    read_bound_json,
)
from .dual_coordination_reference import ContinuousReference, KinematicState
from .dual_coordination_safety import ReferenceSupervisor, audit_geometry
from .dual_payload_mujoco import DualMuJoCoFixture
from .payload_contract import Payload
from .payload_solver_mujoco import FREE_MASS_KG, HOVER_RPM, KF


def _array(value, shape):
    result = np.asarray(value, dtype=float)
    _require(result.shape == shape and np.isfinite(result).all(), "Complete finite measured array required")
    return result


def surface_extent(origin, position, rotation, shape):
    """Maximum original sphere/capsule/cylinder/box surface radius about origin."""
    offset = _array(position, (3,)) - _array(origin, (3,))
    matrix = _array(rotation, (3, 3))
    local = matrix.T @ offset
    size = _array(shape["size"], (3,))
    kind = shape["kind"]
    if kind == "sphere":
        return float(np.linalg.norm(offset) + size[0])
    if kind == "box":
        return float(np.linalg.norm(np.abs(local) + size))
    if kind == "capsule":
        return float(np.sqrt(local[0] ** 2 + local[1] ** 2 + (abs(local[2]) + size[1]) ** 2) + size[0])
    if kind == "cylinder":
        return float(np.hypot(np.hypot(local[0], local[1]) + size[0], abs(local[2]) + size[1]))
    raise ValueError("Uncovered collision shape for assembly envelope")


def _initials(positions):
    return [
        dict(position=list(p), quaternion_wxyz=[1, 0, 0, 0], swing_deg=0, direction_rad=0) for p in positions
    ]


def _fixture(original, output, side=0.045, ratio=0.2):
    return DualMuJoCoFixture(original, output, [Payload(FREE_MASS_KG, ratio, side)] * 2)


def measure_mechanics(original, output):
    """Eight isolated motors, asymmetric reset isolation and every proxy pair."""
    f = _fixture(original, output)
    trials = []
    for drone in range(2):
        for rotor in range(4):
            f.reset()
            rpm = np.zeros((2, 4))
            rpm[drone, rotor] = 10000.0
            f.set_rpm(rpm)
            f.mj.mj_forward(f.model, f.data)
            trials.append(
                dict(
                    drone=drone,
                    rotor=rotor,
                    rpm=rpm.tolist(),
                    actuator_force=f.data.actuator_force[f.actuator_ids].tolist(),
                    generalized_force=[
                        f.data.qfrc_actuator[a["free_dof"] : a["free_dof"] + 6].tolist() for a in f.addresses
                    ],
                )
            )
    f.step_rpm(np.full((2, 4), 15000.0))
    dirty = dict(time=float(f.data.time), qvel=f.data.qvel.tolist(), ctrl=f.data.ctrl.tolist())
    resets = []
    for positions in ([[-2, 1, 2], [3, -1, 3]], [[-1, 2, 4], [3, -1, 3]], [[-1, 2, 4], [2, -2, 2]]):
        f.reset(_initials(positions))
        resets.append(
            dict(
                requested=positions,
                actual=f.sample(),
                qpos=f.data.qpos.tolist(),
                qvel=f.data.qvel.tolist(),
                ctrl=f.data.ctrl.tolist(),
                time=float(f.data.time),
            )
        )
    f.reset(_initials([[0, 0, 2], [0, 0, 2]]))
    geometry = f.geometry_state()
    pairs = []
    for a, b in zip(*f.audit_geom_ids, strict=True):
        distance = float(f.mj.mj_geomDistance(f.model, f.data, a, b, 10.0, None))
        pairs.append(dict(geom_ids=[a, b], distance_m=distance))
    return dict(
        kind="mechanics",
        runtime=f.runtime(),
        trials=trials,
        dirty=dirty,
        resets=resets,
        collision=dict(geometry=geometry, pairs=pairs, audit=audit_geometry(f)),
    )


def _mechanics(data):
    runtime = data.get("runtime", {})
    _require(
        runtime.get("version") == "3.15.0" and runtime.get("profile") == "dual-payload-v1",
        "Wrong mechanics profile",
    )
    aids = _array(runtime.get("actuator_ids"), (2, 4))
    _require(len(set(aids.ravel())) == 8, "Aliased actuator addresses")
    addresses = runtime["addresses"]
    _require(
        len(addresses) == 2
        and len({a["free_qpos"] for a in addresses}) == 2
        and len({a["free_dof"] for a in addresses}) == 2,
        "Aliased reset/force addresses",
    )
    trials = data.get("trials", [])
    _require(
        len(trials) == 8
        and {(r["drone"], r["rotor"]) for r in trials} == {(d, r) for d in range(2) for r in range(4)},
        "All eight isolated motors required",
    )
    for row in trials:
        d, r = row["drone"], row["rotor"]
        rpm = _array(row["rpm"], (2, 4))
        expected = np.zeros((2, 4))
        expected[d, r] = 10000.0
        _require(np.array_equal(rpm, expected), "Isolated excitation changed")
        force = _array(row["actuator_force"], (2, 4))
        _require(np.allclose(force, KF * rpm**2, rtol=1e-8, atol=1e-12), "Original thrust calibration failed")
        q = _array(row["generalized_force"], (2, 6))
        _require(
            np.allclose(q[1 - d], 0, atol=1e-12, rtol=0)
            and np.isclose(q[d, 2], KF * 10000**2, rtol=1e-8, atol=1e-12)
            and np.linalg.norm(q[d, 3:]) > 0,
            "Isolated motor wrench aliases other assembly",
        )
    dirty = data["dirty"]
    _require(
        dirty["time"] > 0 and np.any(dirty["qvel"]) and np.any(dirty["ctrl"]),
        "Reset was not exercised after motion/input",
    )
    resets = data.get("resets", [])
    _require(len(resets) == 3, "Independent asymmetric reset trials required")
    for row in resets:
        _require(
            row["time"] == 0
            and not np.any(_array(row["qvel"], (24,)))
            and not np.any(_array(row["ctrl"], (8,))),
            "Reset did not clear time/velocity/inputs",
        )
        actual = [d["drone_xyz"] for d in row["actual"]["drones"]]
        _require(
            np.array_equal(_array(actual, (2, 3)), _array(row["requested"], (2, 3))),
            "Reset poses differ from requests",
        )
    for index, unchanged in ((1, 1), (2, 0)):
        a = addresses[unchanged]["free_qpos"]
        before = _array(resets[index - 1]["qpos"], (30,))[a : a + 15]
        after = _array(resets[index]["qpos"], (30,))[a : a + 15]
        _require(np.array_equal(before, after), "Reset of one assembly changed the other")
    collision = data["collision"]
    pairs = collision["pairs"]
    _require(
        {tuple(r["geom_ids"]) for r in pairs} == set(zip(*runtime["audit_geom_ids"], strict=True)),
        "Missing intentional proxy collision coverage",
    )
    _require(
        all(_finite(r.get("distance_m"), lower=-np.inf, upper=0) for r in pairs),
        "Intentional collision did not overlap full proxies",
    )
    _require(
        collision["geometry"]["contacts"]
        and collision["audit"]["contacts"]
        and collision["audit"]["min_clearance_m"] < 0.2,
        "Contact/margin detector failed intentional collision",
    )
    return {"address_actuation", "reset_isolation", "collision_detection"}


def measure_streaming(original, output, *, artifact_root, ticks=20):
    """Compare live hooks with independent detailed-state replay on one flight."""
    from .dual_coordination_env import step_native_with_audit

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    f = _fixture(original, output / "fixture")
    live_fixture = _fixture(original, output / "live-fixture")
    live = OnlineNativeAudit(live_fixture)
    offline = NativePointAuditor(f.model_path, f.runtime())
    reset = dict(kind="RESET", actual=f.sample(), geometry=f.geometry_state())
    offline.observe(reset)
    trace = output / "detailed.jsonl.gz"
    with gzip.open(trace, "wt") as stream:
        stream.write(json.dumps(reset, allow_nan=False) + "\n")
        for tick in range(ticks):
            step_native_with_audit(
                live_fixture, np.full((2, 4), HOVER_RPM), live, detailed=False, location=f"supplement:{tick}"
            )
            f.step_rpm(np.full((2, 4), HOVER_RPM), observe_stages=True)
            _require(
                np.array_equal(f.data.qpos, live_fixture.data.qpos)
                and np.array_equal(f.data.qvel, live_fixture.data.qvel),
                "Live/detailed native trajectory changed",
            )
            row = dict(
                kind="STEP",
                tick=tick + 1,
                actual=f.sample(),
                stages=deepcopy(f.stage_rows),
                substeps=deepcopy(f.substep_rows),
            )
            offline.observe(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return dict(
        kind="streaming",
        ticks=ticks,
        live=live.report(),
        independent=offline.report(),
        detailed=file_record(artifact_root, str(trace.resolve().relative_to(Path(artifact_root).resolve()))),
    )


def _streaming(data, root):
    ticks = data["ticks"]
    _require(type(ticks) is int and ticks > 0, "Positive actual measured duration required")
    live, offline = data["live"], data["independent"]
    check_online(live, ticks)
    _require(live.get("evidence_mode") == "online-live", "Actual live hooks were not compared")
    for field in ("violations", "reconstruction_failures"):
        _require(offline.get(field) == [], "Independent detailed audit failed")
    _require(
        offline.get("reconstructed_intervals") == 16 * ticks
        and offline.get("body_reconstructed_intervals") == 16 * ticks,
        "Independent interval coverage incomplete",
    )
    for key in (
        "minimum_clearance_m",
        "floor_minima_m",
        "max_swing_deg",
        "max_spin_deg_s",
        "reconstructed_clearance_lower_bound_m",
        "reconstructed_self_clearance_lower_bound_m",
        "reconstructed_constraint_upper_bound_m",
        "reconstructed_tilt_upper_bound_deg",
    ):
        _require(
            np.allclose(live[key], offline[key], rtol=0, atol=1e-10),
            "Live/detailed diagnostic mismatch: " + key,
        )
    record = data["detailed"]
    _require(file_record(root, record["path"]) == record, "Detailed comparison trace changed")
    with gzip.open(Path(root) / record["path"], "rt") as stream:
        _require(json.loads(next(stream))["kind"] == "RESET", "Detailed reset missing")
        count = 0
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            _require(
                row["tick"] == count and len(row["stages"]) == 64 and len(row["substeps"]) == 17,
                "Detailed audit states incomplete",
            )
        _require(count == ticks, "Detailed comparison duration incomplete")
    return {"streaming_audit"}


def stopping_cases():
    return [
        dict(case_id=f"stopping-extreme-{i:02d}", side=s, ratio=r, duration_s=30.0)
        for i, (s, r) in enumerate([(0.045, 0.2)] + [(s, r) for s in (0.03, 0.045, 0.06) for r in (0.1, 0.3)])
    ]


def check_envelope_sample(row):
    tracking = _array(row["tracking_errors_m"], (2,))
    radii = _array(row["shape_radii_m"], (2,))
    _require(
        np.all((tracking >= 0) & (tracking <= 0.15)), "Measured tracking exceeds existing0.15m inflation"
    )
    _require(np.all((radii >= 0) & (radii <= 0.4)), "Measured assembly exceeds existing0.40m envelope")


def reduce_measurements(data, *, root):
    kind = data.get("kind")
    if kind == "mechanics":
        return _mechanics(data)
    if kind == "streaming":
        return _streaming(data, root)
    if kind == "stopping":
        return _stopping(data, root)
    if kind == "deterministic-replay":
        return _replay(data, root)
    if kind == "compiled-adapter-parity":
        validate_compiled_parity(data, root=root)
        return set()
    raise ValueError("Unknown measured supplemental experiment")


def _decision_record(references, states, time_s, priority, decision):
    return dict(
        time_s=time_s,
        priority=priority,
        references=[r.to_dict() for r in references],
        states=[dict(position=list(s.position), velocity=list(s.velocity)) for s in states],
        reason=decision.reason,
        held=list(decision.held),
        horizon_s=decision.horizon_s,
        failure=decision.failure,
        executed=[r.to_dict() for r in decision.references],
    )


def check_decision(row):
    refs = [ContinuousReference.from_dict(r) for r in row["references"]]
    states = [KinematicState(**s) for s in row["states"]]
    now = row["time_s"]
    brakes = [r.brake(now, actual_state=s) for r, s in zip(refs, states, strict=True)]
    expected = max(2.0, max(r.duration for r in brakes) + 0.20)
    _require(row["horizon_s"] == expected, "Horizon does not cover measured-state C2 braking +0.20s latency")
    decision = ReferenceSupervisor().decide(refs, states, now, priority=row["priority"])
    _require(
        _decision_record(refs, states, now, row["priority"], decision) == row,
        "Retained supervisor decision differs from state/reference recomputation",
    )
    return row["reason"]


def measure_stopping(original, design, output, *, artifact_root, case, progress=None):
    """One declared30s paired-load stop; fails closed on any existing bound.

    This is an independent qualification probe. It never extends a mission or
    modifies its goals. All native stages run the existing online auditor.
    """
    from .dual_coordination_env import step_native_with_audit
    from .dual_coordination_qualify import _settled
    from .payload_hybrid_controller import Controller

    _require(case in stopping_cases(), "Exact prospective stopping endpoint required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    f = _fixture(original, output / "fixture", case["side"], case["ratio"])
    online = OnlineNativeAudit(f)
    rows = f.sample()["drones"]
    refs = [
        ContinuousReference(r["drone_xyz"]).retarget(
            target, 0.0, actual_state=KinematicState(r["drone_xyz"], r["drone_velocity"])
        )
        for r, target in zip(rows, [[-0.5, -3.5, 1.0], [0.5, 3.5, 1.0]])
    ]
    brake_tick = int(np.ceil(max(r.duration for r in refs) * 50))
    controllers = [Controller(design["gain"], design["hover_action"]) for _ in range(2)]
    supervisor = ReferenceSupervisor()
    dwell = 0
    previous_valid = False
    stop_time = None
    brake_start = None
    tracking_max = np.zeros(2)
    radius_max = np.zeros(2)
    displacement = np.zeros(2)
    trace = output / "stopping.jsonl.gz"
    next_heartbeat = time.monotonic() + 30
    with gzip.open(trace, "wt", compresslevel=3) as stream:
        stream.write(
            json.dumps(
                dict(
                    kind="RESET",
                    actual=f.sample(),
                    case=case,
                    runtime=f.runtime(),
                    shapes=[
                        dict(kind=s["kind"], size=list(s["size"]), owner=s["owner"])
                        for s in online._auditor.shapes
                    ],
                ),
                allow_nan=False,
            )
            + "\n"
        )
        for tick in range(3000):
            if progress is not None and time.monotonic() >= next_heartbeat:
                progress(tick, 3000)
                next_heartbeat = time.monotonic() + 30
            now = tick * 0.01
            rows = f.sample()["drones"]
            states = [KinematicState(r["drone_xyz"], r["drone_velocity"]) for r in rows]
            if tick == brake_tick:
                brake_start = np.array([r["drone_xyz"] for r in rows])
                refs = [r.brake(now, actual_state=s) for r, s in zip(refs, states)]
            requested = refs
            decision = supervisor.decide(requested, states, now, priority=1)
            _require(decision.failure is None, "Native stopping supervisor failure: " + str(decision.failure))
            refs = list(decision.references)
            values = [r.sample(now) for r in refs]
            actions = np.array(
                [
                    c.action(row, p, tick=tick, reference_velocity=v)
                    for c, row, (p, v, a) in zip(controllers, rows, values)
                ]
            )
            step_native_with_audit(f, (1 + 0.8 * actions) * HOVER_RPM, online, location=f"stopping:{tick}")
            sample = f.sample()
            post = (tick + 1) * 0.01
            errors = [
                float(np.linalg.norm(np.asarray(row["drone_xyz"]) - r.sample(post)[0]))
                for row, r in zip(sample["drones"], refs)
            ]
            radii = []
            for drone in range(2):
                center = f.data.xpos[f.body_ids[drone]["base_link"]]
                radii.append(
                    max(
                        surface_extent(
                            center,
                            f.data.geom_xpos[g],
                            f.data.geom_xmat[g].reshape(3, 3),
                            online._auditor.shapes[g],
                        )
                        for g in f.audit_geom_ids[drone]
                    )
                )
            measured = dict(tracking_errors_m=errors, shape_radii_m=radii)
            check_envelope_sample(measured)
            _require(online.report()["safe"], "Native stopping audit failed")
            tracking_max = np.maximum(tracking_max, errors)
            radius_max = np.maximum(radius_max, radii)
            if brake_start is not None:
                displacement = np.maximum(
                    displacement,
                    np.linalg.norm(
                        np.array([r["drone_xyz"] for r in sample["drones"]]) - brake_start, axis=1
                    ),
                )
                valid = _settled(sample["drones"], f.payloads, [r.target for r in refs])
                dwell = dwell + 1 if previous_valid and valid else 0
                previous_valid = valid
                if dwell >= 100 and stop_time is None:
                    stop_time = post - brake_tick * 0.01
            stream.write(
                json.dumps(
                    dict(
                        kind="STEP",
                        tick=tick + 1,
                        actual=sample,
                        geom_centers=f.data.geom_xpos.tolist(),
                        geom_rotations=f.data.geom_xmat.tolist(),
                        **measured,
                        supervisor=_decision_record(requested, states, now, 1, decision),
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
    _require(dwell >= 100 and stop_time is not None, "Stopped endpoint full continuous settling failed")
    result = dict(
        kind="stopping-case",
        case=case,
        brake_tick=brake_tick,
        settling_ticks=dwell,
        stopping_time_s=stop_time,
        maximum_displacement_m=displacement.tolist(),
        tracking_errors_m=tracking_max.tolist(),
        shape_radii_m=radius_max.tolist(),
        audit=online.report(),
        trace=file_record(artifact_root, str(trace.resolve().relative_to(Path(artifact_root).resolve()))),
    )
    atomic_json(output / "measurements.json", result)
    return result


def _stopping(data, root):
    from .dual_coordination_qualify import _settled

    cases = data.get("cases", [])
    _require(
        len(cases) == 7 and {r["case"]["case_id"] for r in cases} == {r["case_id"] for r in stopping_cases()},
        "All six paired endpoint stops plus nominal required",
    )
    reasons = set()
    for row in cases:
        _require(row["case"] in stopping_cases(), "Stopping declaration changed")
        _require(
            type(row["settling_ticks"]) is int
            and row["settling_ticks"] >= 100
            and _finite(row["stopping_time_s"], lower=1.0, upper=30.0),
            "Stopping settling evidence incomplete",
        )
        check_envelope_sample(row)
        check_online(row["audit"], 3000)
        record = row["trace"]
        _require(file_record(root, record["path"]) == record, "Stopping trace changed")
        maxima = np.zeros((2, 2))
        dwell = 0
        previous_valid = False
        stop_time = None
        start_positions = None
        displacement = np.zeros(2)
        payloads = [Payload(FREE_MASS_KG, row["case"]["ratio"], row["case"]["side"])] * 2
        with gzip.open(Path(root) / record["path"], "rt") as stream:
            reset = json.loads(next(stream))
            _require(reset["case"] == row["case"], "Stopping reset binding differs")
            prior_positions = np.array([r["drone_xyz"] for r in reset["actual"]["drones"]])
            count = 0
            for count, line in enumerate(stream, 1):
                item = json.loads(line)
                _require(item["tick"] == count, "Stopping control coverage incomplete")
                check_envelope_sample(item)
                executed = [ContinuousReference.from_dict(r) for r in item["supervisor"]["executed"]]
                positions = [np.asarray(r["drone_xyz"]) for r in item["actual"]["drones"]]
                tracking = [
                    float(np.linalg.norm(p - r.sample(count * 0.01)[0])) for p, r in zip(positions, executed)
                ]
                geometry = np.asarray(item["geom_centers"])
                radius = [
                    max(
                        surface_extent(
                            positions[d],
                            geometry[g],
                            np.asarray(item["geom_rotations"][g]).reshape(3, 3),
                            shape,
                        )
                        for g, shape in enumerate(reset["shapes"])
                        if shape["owner"] == d
                    )
                    for d in range(2)
                ]
                _require(
                    np.array_equal(tracking, item["tracking_errors_m"])
                    and np.array_equal(radius, item["shape_radii_m"]),
                    "Envelope summary differs from native poses/reference/proxy shapes",
                )
                maxima = np.maximum(maxima, [item["tracking_errors_m"], item["shape_radii_m"]])
                if count - 1 == row["brake_tick"]:
                    start_positions = prior_positions
                if start_positions is not None:
                    displacement = np.maximum(
                        displacement, np.linalg.norm(np.asarray(positions) - start_positions, axis=1)
                    )
                    valid = _settled(item["actual"]["drones"], payloads, [r.target for r in executed])
                    dwell = dwell + 1 if previous_valid and valid else 0
                    previous_valid = valid
                    if dwell >= 100 and stop_time is None:
                        stop_time = count * 0.01 - row["brake_tick"] * 0.01
                prior_positions = np.asarray(positions)
                reasons.add(check_decision(item["supervisor"]))
            _require(count == 3000, "Full30s stopping experiment required")
            _require(
                dwell == row["settling_ticks"]
                and stop_time == row["stopping_time_s"]
                and np.array_equal(displacement, row["maximum_displacement_m"]),
                "Stopping metrics do not reconstruct from actual measured trajectory",
            )
        _require(
            np.array_equal(maxima, [row["tracking_errors_m"], row["shape_radii_m"]]),
            "Stopping envelope summary differs from actual trace",
        )
    # Branch challenge diagnostics are explicit declared state/reference inputs;
    # they cannot be substituted for the seven actual paired-load flights.
    source_rows = [read_bound_json(root, record) for record in data.get("source_measurements", [])]
    _require(
        len(source_rows) == 7 and source_rows == cases, "Stopping cases differ from bound measured files"
    )
    branch_data = read_bound_json(root, data["branch_measurements"])
    _require(
        branch_data.get("kind") == "branch-initial-state-diagnostics",
        "Native branch measurement set required",
    )
    branch_rows = branch_data.get("branches", [])
    _require(len(branch_rows) == 100, "Complete declared native initial-state diagnostics required")
    branch_reasons = set()
    for measured in branch_rows:
        row = measured["decision"]
        states = [
            dict(position=r["drone_xyz"], velocity=r["drone_velocity"]) for r in measured["actual"]["drones"]
        ]
        _require(row["states"] == states, "Supervisor input differs from native read-back state")
        branch_reasons.add(check_decision(row))
    _require(
        branch_reasons == {"clear", "priority-yield", "brake-both", "brake-both-unestablished"},
        "All four supervisor branch horizon checks required",
    )
    return {"stopping_extremes"}


def measure_replay_evidence(source, replay, *, artifact_root, native_audit_library=None):
    """Independently bind and compare the recovered replay artifacts on disk.

    Does not rerun physics or imply the old source version equals the new one.
    The native replay driver's report and both executable-profile hashes remain
    explicit evidence; admission separately binds the currently qualified source.
    """
    root = Path(artifact_root).resolve()
    source = Path(source).resolve()
    replay = Path(replay).resolve()

    def record(path):
        return file_record(root, str(path.relative_to(root)))

    return dict(
        kind="deterministic-replay",
        source_directory=str(source.relative_to(root)),
        replay_directory=str(replay.relative_to(root)),
        source_files=[record(p) for p in sorted(source.rglob("*")) if p.is_file()],
        replay_files=[record(p) for p in sorted(replay.rglob("*")) if p.is_file()],
        verification=record(replay / "replay-verification.json"),
        audit_library=record(Path(native_audit_library).resolve())
        if native_audit_library is not None
        else None,
        source_version_difference="Original and replay executable profile hashes are retained separately; replay adds interval self/tilt/constraint audit and stride20 retention.",
    )


def _replay(data, root):
    replay_measurement_summary(data, root=root)
    return {"deterministic_replay"}


def replay_measurement_summary(data, *, root):
    from .dual_coordination_raw import read_raw
    from .dual_coordination_campaign import _path

    source = Path(data["source_directory"])
    replay = Path(data["replay_directory"])
    for record in data["source_files"] + data["replay_files"]:
        _require(file_record(root, record["path"]) == record, "Replay/source file binding changed")
    report = read_bound_json(root, data["verification"])
    _require(
        report.get("schema") == "dual-development-replay-verification-v1",
        "Actual native replay verification required",
    )

    def load(directory, name):
        return json.loads(_path(Path(root), str(directory / name)).read_text())

    actions = load(source, "proposed-actions.json")
    expected = load(source, "pilot-result.json")
    original_meta = load(source, "environment.json")
    replay_meta = load(replay, "environment.json")
    _require(
        original_meta["case"] == replay_meta["case"] and original_meta["design"] == replay_meta["design"],
        "Replay inputs differ",
    )
    _require(
        expected["terminal_reason"] == "success" and expected["completed_counts"] == [5, 5],
        "Closed development mission required",
    )
    ticks = round(expected["team_time"] * 100)
    _require(
        actions and [r["tick"] for r in actions] == list(range(0, ticks, 20)),
        "Missing original policy decisions",
    )
    action_record = file_record(root, str(source / "proposed-actions.json"))
    _require(
        action_record["sha256"] == report["source_action_sha256"] and len(actions) == expected["decisions"],
        "Action identity/count mismatch",
    )
    original = {
        r["tick"]: r for p in sorted((Path(root) / source).glob("replay-*.draw")) for r in read_raw(p)
    }
    rows = [r for p in sorted((Path(root) / replay).glob("replay-*.draw")) for r in read_raw(p)]
    expected_ticks = {0, ticks} | set(range(1, ticks + 1, 20)) | set(range(20, ticks + 1, 20))
    _require(
        {r["tick"] for r in rows} == expected_ticks and len(rows) == len(expected_ticks),
        "Replay retained boundary coverage mismatch",
    )
    maximum = 0.0
    for row in rows:
        old = original[row["tick"]]
        for key in ("qpos", "qvel", "rpm", "ctrl"):
            a = np.asarray(row["actual"][key])
            b = np.asarray(old["actual"][key])
            _require(
                a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                "Invalid replay state shape/values",
            )
            maximum = max(maximum, float(np.max(np.abs(a - b))))
        for key in ("action", "ownership", "event", "reward", "supervisor"):
            if key in old or key in row:
                _require(old.get(key) == row.get(key), "Replay causal decision differs: " + key)
    _require(
        maximum <= 1e-12
        and maximum == report["maximum_state_absolute_error"]
        and len(rows) == report["checked_states"],
        "Native state replay mismatch",
    )
    for key in (
        "team_time",
        "completed_counts",
        "terminal_reason",
        "minimum_control_boundary_clearance_m",
        "floor_minima_m",
    ):
        actual = report["metrics"][key]
        target = expected[key]
        _require(
            actual == target if isinstance(target, str) else np.allclose(actual, target, rtol=0, atol=1e-9),
            "Replay terminal metric mismatch: " + key,
        )
    library = data.get("audit_library")
    library_hash = None
    if library is not None:
        _require(file_record(root, library["path"]) == library, "Replay compiled library changed")
        library_hash = library["sha256"]
        _require(
            report.get("audit_library_sha256") == library_hash
            and replay_meta.get("audit_backend") == "compiled-native"
            and replay_meta["binding"].get("audit_library_sha256") == library_hash,
            "Replay compiled backend/library mismatch",
        )
    else:
        _require(
            report.get("audit_library_sha256") is None
            and replay_meta["binding"].get("audit_library_sha256") is None,
            "Compiled replay requires actual library file binding",
        )
    check_online(report["online_audit"], ticks, compiled_library_sha256=library_hash)
    _require(data.get("source_version_difference"), "Executable version difference must be disclosed")
    return dict(
        decision_count=len(actions),
        checked_states=len(rows),
        maximum_state_absolute_error=maximum,
        native_reexecution_performed=True,
        decisions_match=True,
        terminal_match=True,
        metrics_within_tolerance=True,
        passed=True,
        audit_library_sha256=library_hash,
        profile_sha256=replay_meta["binding"]["profile_sha256"],
        measured_report=data["verification"],
        original_profile_sha256=original_meta["binding"]["profile_sha256"],
        source_version_difference=data["source_version_difference"],
    )


def measure_branch_horizons(original, output):
    """Native read-back initial-state diagnostics for all four decision branches.

    Each row is a fresh declared initial condition, never a teleported state in
    a flown mission. These checks only qualify branch/horizon arithmetic; the
    separate seven closed-loop load trials qualify physical tracking/stopping.
    """
    f = _fixture(original, output)
    scenarios = [
        (((-3, -3, 2), (3, 3, 2)), ((1, -3, 2), (-1, 3, 2))),
        (((-3, 0, 2), (0, -3, 2)), ((3, 0, 2), (0, 3, 2))),
        (((-3, 0, 2), (3, 0, 2)), ((3, 0, 2), (-3, 0, 2))),
        (((-3, 0, 2), (0, 0, 2)), ((3, 0, 2), (0, 0, 2))),
    ]
    rows = []
    for starts, goals in scenarios:
        refs = [
            ContinuousReference(p).retarget(g, 0.0, actual_state=KinematicState(p, (0, 0, 0)))
            for p, g in zip(starts, goals)
        ]
        for i in range(25):
            now = 0.2 + i * 0.1
            positions = [r.sample(now)[0] + [0.01, -0.02, 0.005] for r in refs]
            f.reset(_initials(positions))
            for r, address in zip(refs, f.addresses):
                offset = address["free_dof"]
                f.data.qvel[offset : offset + 3] = r.sample(now)[1]
            f.mj.mj_forward(f.model, f.data)
            states = [KinematicState(r["drone_xyz"], r["drone_velocity"]) for r in f.sample()["drones"]]
            decision = ReferenceSupervisor().decide(refs, states, now, priority=1 + i % 2)
            row = _decision_record(refs, states, now, 1 + i % 2, decision)
            rows.append(dict(decision=row, actual=f.sample()))
    return dict(
        kind="branch-initial-state-diagnostics",
        branches=rows,
        scope="100 separate native read-back initial states; no physical closed-loop success claim",
    )


def combine_stopping(root, case_measurement_paths, branch_measurement_path):
    records = [file_record(root, path) for path in case_measurement_paths]
    return dict(
        kind="stopping",
        cases=[read_bound_json(root, r) for r in records],
        source_measurements=records,
        branch_measurements=file_record(root, branch_measurement_path),
    )


def measure_compiled_parity(original, design, output, *, artifact_root, native_audit_library):
    """Record the existing native adapter test's three decisions and resume step.

    This is a short integration contract, not a replacement for the 81 native
    physical trials or the complete development mission replay.
    """
    from .dual_coordination_env import DualCoordinationEnv
    from .dual_coordination_cases import generate_case
    from .dual_coordination_schema import encode_action, WAIT
    from .util import sha256

    root, output = Path(artifact_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = Path(native_audit_library).resolve()
    case = generate_case(15, 0, "normal")
    case["starts"] = [[-3.5, -3.5, 1.0], [3.5, 3.5, 1.0]]
    case["initials"] = [
        dict(position=p, quaternion_wxyz=[1.0, 0.0, 0.0, 0.0], swing_deg=0.0, direction_rad=0.0)
        for p in case["starts"]
    ]
    case["goals"][0] = [0.5, -3.5, 1.0]
    common = dict(case=case, design=design, replay_stride_ticks=20)
    python = compiled = resumed = None

    def state(env):
        signature = int(env.fixture.mj.mjtState.mjSTATE_INTEGRATION)
        values = np.empty(env.fixture.mj.mj_stateSize(env.fixture.model, signature))
        env.fixture.mj.mj_getState(env.fixture.model, env.fixture.data, values, signature)
        return dict(
            integration=values.tolist(),
            qpos=env.fixture.data.qpos.tolist(),
            qvel=env.fixture.data.qvel.tolist(),
            ctrl=env.fixture.data.ctrl.tolist(),
            rpm=env.fixture.rpm.tolist(),
        )

    def continuation(env):
        saved = env.to_dict()
        return {
            key: saved[key]
            for key in (
                "backend",
                "controllers",
                "mission",
                "desired_references",
                "executed_references",
                "reference_goals",
                "last_held",
                "last_prediction",
                "last_supervisor_reason",
                "event_applied",
                "last_action_mask",
                "ledger",
                "rng",
                "last_finite_observation",
                "online_auditor_state",
            )
        }

    try:
        python = DualCoordinationEnv(original, output / "python", **common)
        compiled = DualCoordinationEnv(original, output / "compiled", native_audit_library=library, **common)
        rows = []
        action = encode_action(0, WAIT, 1)
        for tick in (20, 40, 60):
            left, right = python.step(action), compiled.step(action)
            _require(not left[2] and not right[2], "Compiled parity trial terminated early")
            rows.append(
                dict(
                    tick=tick,
                    action=action,
                    python=state(python),
                    compiled=state(compiled),
                    copyback_shadow_state=compiled.online_auditor.shadow.snapshot()["state"],
                    python_audit=python.online_auditor.report(),
                    compiled_audit=compiled.online_auditor.report(),
                )
            )
        saved = compiled.to_dict()
        atomic_json(output / "checkpoint.json", saved)
        resumed = DualCoordinationEnv(original, output / "resumed", native_audit_library=library, **common)
        resumed.restore(saved)
        for env in (compiled, resumed):
            _require(not env.step(action)[2], "Checkpoint continuation terminated early")
        result = dict(
            kind="compiled-adapter-parity",
            case=case,
            design=design,
            original_sha256=sha256(original),
            source_hashes={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
            library=file_record(root, str(library.relative_to(root))),
            build_receipt=file_record(root, str(library.with_suffix(".json").relative_to(root))),
            comparison_steps=rows,
            python_profile=python.binding,
            compiled_profile=compiled.binding,
            checkpoint=dict(
                saved=file_record(root, str((output / "checkpoint.json").relative_to(root))),
                continued=continuation(compiled),
                resumed=continuation(resumed),
            ),
            final_audit=compiled.online_auditor.report(),
        )
        validate_compiled_parity(result, root=root)
        return result
    finally:
        for env in (python, compiled, resumed):
            if env is not None:
                env.close()


def validate_compiled_parity(data, *, root):
    library = data["library"]
    _require(file_record(root, library["path"]) == library, "Compiled parity library changed")
    build = read_bound_json(root, data["build_receipt"])
    _require(
        build.get("schema") == "dual-native-audit-build-v1"
        and build.get("returncode") == 0
        and build["artifact"]["sha256"] == library["sha256"]
        and build.get("mujoco_version") == "3.15.0",
        "Measured compiled build identity mismatch",
    )
    rows = data["comparison_steps"]
    _require(
        [row["tick"] for row in rows] == [20, 40, 60],
        "Three actual existing adapter parity decisions required",
    )
    maximum = 0.0
    for row in rows:
        tick = row["tick"]
        check_online(row["python_audit"], tick)
        check_online(row["compiled_audit"], tick, compiled_library_sha256=library["sha256"])
        for key in ("qpos", "qvel"):
            a, b = np.asarray(row["python"][key]), np.asarray(row["compiled"][key])
            _require(
                a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all(),
                "Finite paired native state measurements required",
            )
            error = float(np.max(np.abs(a - b)))
            _require(error <= 1e-12, "Python/compiled state parity exceeds existing1e-12 criterion")
            maximum = max(maximum, error)
        for key in ("ctrl", "rpm"):
            _require(row["python"][key] == row["compiled"][key], "Native motor controls differ")
        _require(
            row["compiled"]["integration"] == row["copyback_shadow_state"],
            "Actual fixture copyback differs from complete compiled integration state",
        )
    saved = read_bound_json(root, data["checkpoint"]["saved"])
    _require(
        saved["mission"]["state"]["tick"] == 60
        and saved["binding"]["audit_library_sha256"] == library["sha256"],
        "Actual checkpoint clock/library mismatch",
    )
    continued, resumed = data["checkpoint"]["continued"], data["checkpoint"]["resumed"]
    required = {
        "backend",
        "controllers",
        "mission",
        "desired_references",
        "executed_references",
        "reference_goals",
        "last_held",
        "last_prediction",
        "last_supervisor_reason",
        "event_applied",
        "last_action_mask",
        "ledger",
        "rng",
        "last_finite_observation",
        "online_auditor_state",
    }
    _require(
        set(continued) == required and continued == resumed and continued["mission"]["state"]["tick"] == 80,
        "Exact native/controller/mission/reference/RNG/audit continuation differs",
    )
    _require(
        continued["backend"]["integration"] == continued["online_auditor_state"]["shadow"]["state"],
        "Continued fixture/compiled integration copyback differs",
    )
    check_online(data["final_audit"], 80, compiled_library_sha256=library["sha256"])
    _require(
        data["compiled_profile"].get("audit_library_sha256") == library["sha256"]
        and data["python_profile"]["profile_sha256"] == data["compiled_profile"]["profile_sha256"],
        "Paired adapter source/profile differs",
    )
    return dict(
        library_sha256=library["sha256"],
        maximum_state_absolute_error=maximum,
        parity_control_steps=60,
        checkpoint_continuation_control_steps=20,
        copyback_states=3,
        exact_checkpoint_resume=True,
    )


def main(argv=None):
    import argparse
    from .dual_coordination_campaign import BoundFiles, EXPERIMENT_ROLES
    from .dual_coordination_progress import RunJournal
    from .util import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument(
        "--roles", required=True, help="JSON mapping of frozen experiment roles to relative paths"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--original")
    parser.add_argument("--design")
    parser.add_argument(
        "--kind",
        required=True,
        choices=["mechanics", "streaming", "stopping-case", "branches", "replay", "compiled-parity"],
    )
    parser.add_argument("--case-id")
    parser.add_argument("--source-mission")
    parser.add_argument("--replay-mission")
    parser.add_argument("--native-audit-library")
    args = parser.parse_args(argv)
    root = Path(args.experiment_root).resolve()
    roles = json.loads(Path(args.roles).read_text())
    _require(set(roles) == EXPERIMENT_ROLES, "Exact experiment roles required before measurement")
    binding = BoundFiles.capture(root, roles)
    source_hashes = {r["sha256"] for r in binding.files["source"]}
    _require(
        all(sha256(p) in source_hashes for p in Path(__file__).parent.glob("*.py")),
        "All actual executing Python sources must be frozen experiment inputs",
    )
    output = Path(args.output).resolve()
    _require(root in output.parents, "Supplement output must be inside experiment root")
    output.mkdir(parents=True, exist_ok=False)
    journal = RunJournal(root)
    attempt = journal.start("native-qualification-supplement", args.kind, category="physical qualification")
    try:
        if args.kind != "replay":
            _require(args.original is not None, "Original CF2X asset required")
            _require(
                sha256(Path(args.original)) in {r["sha256"] for r in binding.files["assets"]},
                "Original asset not frozen",
            )
        if args.kind == "mechanics":
            data = measure_mechanics(args.original, output / "fixture")
        elif args.kind == "streaming":
            data = measure_streaming(args.original, output / "streaming", artifact_root=root)
        elif args.kind == "stopping-case":
            _require(args.design is not None, "Frozen controller design required")
            design = json.loads(Path(args.design).read_text())
            _require(
                design in [read_bound_json(root, r) for r in binding.files["controller"]],
                "Controller differs from frozen experiment",
            )
            cases = [r for r in stopping_cases() if r["case_id"] == args.case_id]
            _require(len(cases) == 1, "One exact predeclared stopping case required")
            data = measure_stopping(
                args.original,
                design,
                output / "stopping",
                artifact_root=root,
                case=cases[0],
                progress=lambda completed, total: journal.heartbeat(
                    attempt, completed=completed, total=total
                ),
            )
        elif args.kind == "compiled-parity":
            _require(
                args.design is not None and args.native_audit_library is not None,
                "Actual frozen controller and compiled library required",
            )
            design = json.loads(Path(args.design).read_text())
            _require(
                design in [read_bound_json(root, r) for r in binding.files["controller"]],
                "Compiled parity controller not frozen",
            )
            _require(
                sha256(args.native_audit_library) in {r["sha256"] for r in binding.files["audit"]},
                "Compiled parity library not frozen",
            )
            data = measure_compiled_parity(
                args.original,
                design,
                output / "compiled-parity",
                artifact_root=root,
                native_audit_library=args.native_audit_library,
            )
        elif args.kind == "branches":
            data = measure_branch_horizons(args.original, output / "branch-fixture")
        else:
            data = measure_replay_evidence(
                args.source_mission,
                args.replay_mission,
                artifact_root=root,
                native_audit_library=args.native_audit_library,
            )
        binding.verify()
        path = output / "measurements.json"
        atomic_json(path, data)
        value = dict(
            schema="dual-native-supplement-v1",
            experiment_inputs=binding.files,
            measurements=file_record(root, str(path.relative_to(root))),
        )
        atomic_json(output / "supplement.json", value)
        if args.kind not in ("stopping-case", "branches"):
            reduce_measurements(data, root=root)
        journal.finish(
            attempt,
            status="passed",
            reason="Measured supplemental artifact retained; full admission requires all case and supplement checks",
        )
    except BaseException as error:
        atomic_json(
            output / "failure.json", dict(failure=f"{type(error).__name__}: {error}", admission_granted=False)
        )
        journal.finish(attempt, status="failed", reason=str(error))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
