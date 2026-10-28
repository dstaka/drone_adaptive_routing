"""Prospective bounded native qualification experiments with compact replay.

Seven duration cases are EACH330s, inherited from payload_hybrid_duration.
Mixed36 contains the nine ordered box pairs for hover/vertical/crossing/braking.
Passive anchored probes retain every predecessor anchor/large-angle declaration;
refinement uses identical held motor inputs at16/32 actual RK4 substeps.

A completed diagnostic is not overall production admission. This driver never
emits dual-qualification-v1 and never relaxes a failing safety/numerical check.
Special anchored/refinement monitors report numerical scope explicitly; the
standard live online auditor owns the duration/mixed geometric reconstruction.
Short pilot overrides are labeled and cannot satisfy declared-duration checks.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import importlib.metadata
import importlib.util
import json
import math
import signal
import sys
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from .dual_coordination_audit import OnlineNativeAudit
from .dual_coordination_progress import RunJournal, atomic_json
from .dual_coordination_reference import ContinuousReference, KinematicState
from .dual_coordination_safety import ReferenceSupervisor, audit_geometry
from .dual_payload_mujoco import DualMuJoCoFixture
from .payload_contract import Payload, payload_metrics
from .payload_hybrid_controller import Controller
from .payload_passive_check import analytical_period, crossing_period
from .payload_solver_mujoco import FREE_MASS_KG, HOVER_RPM
from .util import sha256

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/payload_solver_probe_manifest.json"
SIZES = (0.03, 0.045, 0.06)


def qualification_source_freeze():
    """Freeze the imported repository closure in this dedicated qualifier process.

    Load the native stepping helper before taking the snapshot: its env module
    imports are real dependencies even though the helper used to load lazily in
    the first control tick. Unimported trainer/evaluator/media files are excluded.
    sys.modules is intentionally conservative for callers embedding the runner:
    any already-imported repository module remains bound, never silently omitted.
    """
    from .dual_coordination_env import step_native_with_audit

    # The helper always imports this class for its dispatch type check, even
    # when Python audit is selected. Its native module is a transitive dependency.
    from .dual_coordination_compiled_audit import CompiledOnlineNativeAudit

    if not callable(step_native_with_audit) or not callable(CompiledOnlineNativeAudit):
        raise ValueError("Native qualification step dependency unavailable")
    source_root = (ROOT / "src").resolve()
    paths = set()
    for loaded in tuple(sys.modules.values()):
        filename = getattr(loaded, "__file__", None)
        if not filename:
            continue
        path = Path(filename)
        if path.suffix == ".pyc":
            try:
                path = Path(importlib.util.source_from_cache(str(path)))
            except ValueError:
                pass
        path = path.resolve()
        if source_root in path.parents and path.is_file():
            paths.add(path)
    if not paths:
        raise ValueError("Imported qualification repository source closure is empty")
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "mujoco", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(
        schema="dual-qualification-source-freeze-v1",
        source_hashes={str(p.relative_to(ROOT.resolve())): sha256(p) for p in sorted(paths)},
        dependency_versions=versions,
    )


def verify_qualification_source_freeze(frozen):
    current = qualification_source_freeze()
    if current != frozen:
        raise ValueError("Qualification source closure or dependency versions changed")
    return current


def qualification_cases(manifest_path=MANIFEST):
    manifest = json.loads(Path(manifest_path).read_text())
    result = []
    for i, (side, ratio) in enumerate([(0.045, 0.2)] + [(s, r) for s in SIZES for r in (0.1, 0.3)]):
        result.append(
            dict(
                case_id=f"dual-duration330-{i:02d}",
                group="duration",
                kind="hover",
                duration_s=330.0,
                sides=[side, side],
                ratios=[ratio, ratio],
                substeps=16,
            )
        )
    for kind in ("hover", "vertical", "crossing", "brake"):
        for i, (a, b) in enumerate((a, b) for a in SIZES for b in SIZES):
            result.append(
                dict(
                    case_id=f"dual-mixed-{kind}-{i:02d}",
                    group="mixed",
                    kind=kind,
                    duration_s=30.0,
                    sides=[a, b],
                    ratios=[0.1, 0.3] if i % 2 == 0 else [0.3, 0.1],
                    substeps=16,
                )
            )
    for old in manifest["cases"]:
        if old["kind"] in ("anchor", "large_angle"):
            result.append(
                dict(
                    case_id="dual-passive-" + old["id"],
                    group="passive",
                    kind=old["kind"],
                    duration_s=old["duration_s"],
                    sides=[old["side_m"]] * 2,
                    ratios=[old["mass_ratio"]] * 2,
                    substeps=old["substeps"],
                    initial_swing_deg=old["initial_swing_deg"],
                    initial_direction_rad=old["initial_direction_rad"],
                    inherited_case=deepcopy(old),
                )
            )
    for side in SIZES:
        for ratio in (0.1, 0.3):
            result.append(
                dict(
                    case_id=f"dual-refinement-{side:.3f}-{ratio:.2f}",
                    group="refinement",
                    kind="vertical",
                    duration_s=30.0,
                    sides=[side] * 2,
                    ratios=[ratio] * 2,
                    substeps=16,
                    substeps_pair=[16, 32],
                )
            )
    for case in result:
        if case["kind"] == "crossing":
            case.update(
                layout="perpendicular-qualification-v2",
                starts=[[-2.0, 0.0, 3.0], [0.0, -2.0, 3.0]],
                targets=[[2.0, 0.0, 3.0], [0.0, 2.0, 3.0]],
                priority=1,
                resumption="C2-from-executed-reference-after-supervisor-hold",
            )
    return result


def qualification_initials(case):
    """Only the prospective perpendicular probe overrides canonical reset."""
    if case["kind"] != "crossing":
        return None
    return [
        dict(position=list(p), quaternion_wxyz=[1, 0, 0, 0], swing_deg=0, direction_rad=0)
        for p in case["starts"]
    ]


def crossing_requests(executed, targets, previous_held, states, time_s):
    """Propose held goals again from executed p/v/a; supervisor still decides.

    A brake is a temporary safety intervention, not a replacement destination.
    Reusing the original moving curve after a hold would introduce a reference
    jump; retarget instead preserves C2 and retains measured tracking error.
    """
    return tuple(
        ref.retarget(target, time_s, actual_state=state) if held else ref
        for ref, target, held, state in zip(executed, targets, previous_held, states, strict=True)
    )


def passive_energy_report(energies):
    values = np.asarray(energies, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or values[0] <= 1e-12:
        raise ValueError("Complete finite positive-excitation passive energy required")
    drift = float(np.max(np.abs(values - values[0])) / values[0])
    return dict(
        initial_excitation_energy_j=float(values[0]),
        maximum_absolute_energy_fraction=drift,
        positive_energy_fraction=max(0.0, float(values.max() / values[0] - 1)),
        minimum_energy_fraction=float(values.min() / values[0]),
        passed=drift <= 0.05 or math.isclose(drift, 0.05, rel_tol=1e-12, abs_tol=1e-12),
    )


def refinement_difference(base, half):
    if len(base.get("drones", [])) != 2 or len(half.get("drones", [])) != 2:
        raise ValueError("Two complete paired drone poses required")
    maximum_position = maximum_angle = 0.0
    for a, b in zip(base["drones"], half["drones"], strict=True):
        for body in ("drone", "rod", "package"):
            p, q = np.asarray(a[body + "_xyz"], float), np.asarray(b[body + "_xyz"], float)
            qa, qb = (
                np.asarray(a[body + "_quaternion_wxyz"], float),
                np.asarray(b[body + "_quaternion_wxyz"], float),
            )
            if (
                p.shape != (3,)
                or q.shape != (3,)
                or qa.shape != (4,)
                or qb.shape != (4,)
                or not all(np.isfinite(x).all() for x in (p, q, qa, qb))
            ):
                raise ValueError("Finite matched physical poses required")
            if abs(np.linalg.norm(qa) - 1) > 1e-4 or abs(np.linalg.norm(qb) - 1) > 1e-4:
                raise ValueError("Normalized physical quaternions required")
            maximum_position = max(maximum_position, float(np.linalg.norm(p - q)))
            cosine = float(np.clip(abs(np.dot(qa, qb) / (np.linalg.norm(qa) * np.linalg.norm(qb))), 0.0, 1.0))
            maximum_angle = max(maximum_angle, math.degrees(2 * math.acos(cosine)))
    return dict(
        maximum_position_difference_m=maximum_position,
        maximum_angle_difference_deg=maximum_angle,
        within_limits=maximum_position <= 0.02 and maximum_angle <= 2.0,
    )


def write_anchored_variant(source, target):
    tree = ET.parse(source)
    for drone in (1, 2):
        body = tree.find(f".//body[@name='drone-{drone}/base_link']")
        if body is None:
            raise ValueError("Both exact namespaced base bodies required")
        joints = [
            j for j in list(body) if j.tag == "freejoint" or (j.tag == "joint" and j.get("type") == "free")
        ]
        if len(joints) != 1 or joints[0].get("name") != f"drone-{drone}/vehicle_free":
            raise ValueError("Remove exactly the declared free root joint for each anchored diagnostic")
        body.remove(joints[0])
    target = Path(target)
    with target.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)


def _body_energy(fixture):
    f = fixture
    energies = [0.0, 0.0]
    for drone, ids in enumerate(f.body_ids):
        for body in ids.values():
            jp = np.zeros((3, f.model.nv))
            jr = np.zeros_like(jp)
            f.mj.mj_jacBodyCom(f.model, f.data, jp, jr, body)
            v, w = jp @ f.data.qvel, jr @ f.data.qvel
            rotation = f.data.ximat[body].reshape(3, 3)
            inertia = rotation @ np.diag(f.model.body_inertia[body]) @ rotation.T
            energies[drone] += float(
                0.5 * f.model.body_mass[body] * np.dot(v, v)
                + 0.5 * w @ inertia @ w
                - f.model.body_mass[body] * np.dot(f.model.opt.gravity, f.data.xipos[body])
            )
    return np.array(energies)


def _snapshot(f):
    signature = int(f.mj.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(f.mj.mj_stateSize(f.model, signature))
    f.mj.mj_getState(f.model, f.data, state, signature)
    return dict(
        signature=signature,
        integration=state.tolist(),
        rpm=f.rpm.tolist(),
        model_sha256=sha256(f.model_path),
        world_time=float(f.data.time),
    )


def probe_runtime(fixture, *, substeps, anchored=False):
    """Describe the actual diagnostic profile rather than its source fixture."""
    result = fixture.runtime()
    result.update(
        substeps=substeps,
        timestep_s=float(fixture.model.opt.timestep),
        nq=int(fixture.model.nq),
        nv=int(fixture.model.nv),
    )
    if anchored:
        result["profile"] = "dual-anchored-passive-probe-v1"
        # Original free-root addresses become invalid when those joints are
        # removed. The retained exact anchored XML defines its own addresses.
        result.pop("addresses", None)
    elif substeps != 16:
        result["profile"] = "dual-half-dt-refinement-probe-v1"
    return result


def _anchor(f, case):
    previous_mass = f.model.body_mass.copy()
    previous_inertia = f.model.body_inertia.copy()
    target = f.output / "anchored-probe.xml"
    write_anchored_variant(f.model_path, target)
    f.model = f.mj.MjModel.from_xml_path(str(target))
    f.data = f.mj.MjData(f.model)
    f.model_path = target
    if not np.array_equal(previous_mass, f.model.body_mass) or not np.array_equal(
        previous_inertia, f.model.body_inertia
    ):
        raise ValueError("Anchoring changed physical masses or inertia")
    f.mj.mj_forward(f.model, f.data)
    equilibrium = _body_energy(f)
    angle = math.radians(case["initial_swing_deg"]) / 2
    direction = case["initial_direction_rad"]
    q = np.array(
        [math.cos(angle), math.sin(direction) * math.sin(angle), -math.cos(direction) * math.sin(angle), 0.0]
    )
    for drone in (1, 2):
        for joint, quaternion in (
            ("suspension_ball", q),
            ("payload_ball", q * np.array([1.0, -1.0, -1.0, -1.0])),
        ):
            j = f.mj.mj_name2id(f.model, f.mj.mjtObj.mjOBJ_JOINT, f"drone-{drone}/{joint}")
            address = f.model.jnt_qposadr[j]
            f.data.qpos[address : address + 4] = quaternion
    f.rpm = np.zeros((2, 4))
    f.data.ctrl[:] = 0.0
    f.mj.mj_forward(f.model, f.data)
    return equilibrium


class NativeProgressWatchdog:
    """Infrastructure stall watchdog; never a qualification mission deadline."""

    def __init__(self, timeout_seconds=600.0):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Positive no-progress infrastructure timeout required")
        self.timeout_seconds = timeout_seconds
        self.active = False

    def __enter__(self):
        if not hasattr(signal, "setitimer"):
            raise RuntimeError("Native no-progress watchdog requires POSIX interval signals")
        if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
            raise RuntimeError("Refusing to replace an existing process interval timer")
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._expired)
        self.active = True
        self.progress()
        return self

    def _expired(self, signum, frame):
        raise TimeoutError("Infrastructure watchdog: no native progress within configured interval")

    def progress(self):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)

    def __exit__(self, *args):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self.previous_handler)
            self.active = False


def _safe_diagnostic(value):
    if isinstance(value, np.ndarray):
        return _safe_diagnostic(value.tolist())
    if isinstance(value, dict):
        return {str(k): _safe_diagnostic(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_diagnostic(v) for v in value]
    if isinstance(value, np.generic):
        return _safe_diagnostic(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_diagnostic": repr(value)}
    return value


class NumericalMonitor:
    """Stage constraint/finite checks and actual RK signed-work accumulation.

    Special probe profiles are numerical diagnostics, not a full mission swept
    safety audit. Standard free16 profiles additionally run OnlineNativeAudit.
    """

    def __init__(self, fixture, substeps, *, equilibrium=None):
        self.f, self.substeps = fixture, substeps
        self.h = 0.01 / substeps
        self.rk_states = self.native_states = 0
        self.motor_work = self.passive_work = self.absolute_motor_work = 0.0
        self.max_constraint = self.max_energy_residual = 0.0
        self.initial = _body_energy(fixture)
        self.equilibrium = equilibrium
        self.minimum = self.initial.copy()
        self.maximum = self.initial.copy()
        self.failures = []
        self.watchdog = None

    def _check(self, data):
        f = self.f
        if not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc, data.ctrl)):
            raise ValueError("Nonfinite native numerical state")
        if (
            np.any(data.ctrl < 0)
            or np.any(data.ctrl > f.model.actuator_ctrlrange[:, 1])
            or any(w.number for w in data.warning)
        ):
            raise ValueError("Motor authority or native numerical warning")
        if data.ncon:
            raise ValueError("Forbidden contact in numerical qualification probe")
        for joint in range(f.model.njnt):
            kind = int(f.model.jnt_type[joint])
            address = int(f.model.jnt_qposadr[joint])
            if kind == int(f.mj.mjtJoint.mjJNT_FREE):
                q = data.qpos[address + 3 : address + 7]
            elif kind == int(f.mj.mjtJoint.mjJNT_BALL):
                q = data.qpos[address : address + 4]
            else:
                continue
            if abs(np.linalg.norm(q) - 1) > 1e-4:
                raise ValueError("Native quaternion normalization threshold")
        for ids, payload in zip(f.body_ids, f.payloads):
            a, b = ids["base_link"], ids["payload"]
            anchor = data.xpos[a] + data.xmat[a].reshape(3, 3) @ np.array([0.0, 0.0, -0.02])
            endpoint = data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.array([0.0, 0.0, payload.side_m / 2])
            residual = float(abs(np.linalg.norm(endpoint - anchor) - 0.30))
            self.max_constraint = max(self.max_constraint, residual)
            if residual > 2e-5:
                raise ValueError("Native attachment constraint residual exceeded2e-5m")

    def stage(self, model, data):
        try:
            self._check(data)
            expected = (self.rk_states // 4 + (0.0, 0.5, 0.5, 1.0)[self.rk_states % 4]) * self.h
            if abs(data.time - expected) > 5e-9:
                raise ValueError("Actual RK stage order/count mismatch")
            weight = (1.0, 2.0, 2.0, 1.0)[self.rk_states % 4] * self.h / 6
            # Current site-transmission velocity is available before actuation;
            # motor force equals held ctrl for these original linear motors.
            power = float(data.ctrl @ data.actuator_velocity)
            passive = float(data.qfrc_passive @ data.qvel)
            self.motor_work += weight * power
            self.passive_work += weight * passive
            self.absolute_motor_work += weight * abs(power)
            self.rk_states += 1
        except ValueError as error:
            self.failures.append(str(error))
            raise

    def native(self):
        try:
            self._native()
        except ValueError as error:
            self.failures.append(str(error))
            raise

    def _native(self):
        self._check(self.f.data)
        self.native_states += 1
        if self.rk_states != 4 * self.native_states:
            raise ValueError("Missing native/RK numerical audit coverage")
        if self.watchdog is not None:
            self.watchdog.progress()
        energy = _body_energy(self.f)
        self.minimum = np.minimum(self.minimum, energy)
        self.maximum = np.maximum(self.maximum, energy)
        residual = float(abs(np.sum(energy - self.initial) - self.motor_work - self.passive_work))
        self.max_energy_residual = max(self.max_energy_residual, residual)
        if residual > 0.02 or float(np.sum(energy - self.initial)) > 0.02 + 1.1 * self.absolute_motor_work:
            raise ValueError("Inherited signed/absolute motor energy-work bound exceeded")

    def report(self):
        result = dict(
            scope="NUMERICAL_PROBE_ONLY",
            rk_states=self.rk_states,
            native_states=self.native_states,
            maximum_constraint_residual_m=self.max_constraint,
            maximum_signed_energy_work_residual_j=self.max_energy_residual,
            motor_work_j=self.motor_work,
            passive_work_j=self.passive_work,
            failures=list(self.failures),
            passed=not self.failures,
        )
        if self.equilibrium is not None:
            result["passive"] = [
                passive_energy_report([initial - equilibrium, low - equilibrium, high - equilibrium])
                for initial, low, high, equilibrium in zip(
                    self.initial, self.minimum, self.maximum, self.equilibrium
                )
            ]
            result["passed"] &= all(r["passed"] for r in result["passive"])
        return result


class _CombinedAudit(OnlineNativeAudit):
    def __init__(self, fixture, numerics):
        self.numerics = numerics
        self.previous_native_time = 0.0
        super().__init__(fixture)

    def observe_native_state(self, model, data, *, phase, location):
        super().observe_native_state(model, data, phase=phase, location=location)
        if phase == "rk":
            self.numerics.stage(model, data)
        elif data.time > self.previous_native_time + 1e-12:
            self.numerics.native()
            self.previous_native_time = float(data.time)
        return self.report()

    def report(self):
        result = super().report()
        result["numerical"] = self.numerics.report()
        result["safe"] = result["safe"] and result["numerical"]["passed"]
        return result


def _special_step(f, rpm, monitor, substeps):
    f.set_rpm(rpm)
    f.model.opt.timestep = 0.01 / substeps
    f.mj.set_mjcb_control(monitor.stage)
    try:
        for _ in range(substeps):
            f.mj.mj_step(f.model, f.data)
            f.mj.set_mjcb_control(None)
            f.mj.mj_forward(f.model, f.data)
            monitor.native()
            f.mj.set_mjcb_control(monitor.stage)
    finally:
        f.mj.set_mjcb_control(None)


def _references(f, case):
    kind = case["kind"]
    rows = f.sample()["drones"]
    refs = [ContinuousReference(row["drone_xyz"]) for row in rows]
    if kind == "hover":
        return refs
    if kind == "vertical":
        targets = [[-3.5, -3.5, 3.0], [3.5, 3.5, 3.0]]
    elif kind == "crossing":
        targets = case["targets"]
    else:
        targets = [[-0.5, -3.5, 1.0], [0.5, 3.5, 1.0]]
    return [
        r.retarget(target, 0.0, actual_state=KinematicState(row["drone_xyz"], row["drone_velocity"]))
        for r, target, row in zip(refs, targets, rows)
    ]


def _settled(rows, payloads, targets):
    limits = dict(
        drone_error_m=0.20,
        package_error_m=0.20,
        drone_speed_m_s=0.30,
        package_speed_m_s=0.15,
        swing_angle_deg=5.0,
        swing_rate_deg_s=15.0,
        package_spin_deg_s=30.0,
    )
    metrics = [payload_metrics(row, target, payload) for row, target, payload in zip(rows, targets, payloads)]
    return all(all(math.isfinite(m[k]) and 0 <= m[k] <= v for k, v in limits.items()) for m in metrics)


def run_case(
    *,
    original,
    design,
    output,
    case,
    pilot_duration_s=None,
    journal_root=None,
    no_progress_timeout_seconds=600.0,
    expected_source_freeze=None,
):
    if case not in qualification_cases():
        raise ValueError("Exact prospective qualification case required")
    duration = case["duration_s"] if pilot_duration_s is None else pilot_duration_s
    if (
        type(duration) not in (int, float)
        or not math.isfinite(duration)
        or not 0 < duration <= case["duration_s"]
        or abs(duration * 100 - round(duration * 100)) > 1e-8
    ):
        raise ValueError("Pilot duration must be bounded by the exact declaration on100Hz boundaries")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    journal = RunJournal(output if journal_root is None else journal_root)
    attempt = journal.start(
        "native-qualification-case",
        case["case_id"],
        category="physical qualification",
        log_path=output / "result.json",
    )
    started = time.monotonic()
    f = None
    completed = 0
    failure = None
    watchdog = None
    source_freeze = None
    result = dict(
        schema="dual-qualification-case-v1",
        case=case,
        pilot=pilot_duration_s is not None,
        admission_granted=False,
        passed=False,
        completed_declared_duration=False,
    )
    try:
        source_freeze = qualification_source_freeze()
        if expected_source_freeze is not None and source_freeze != expected_source_freeze:
            raise ValueError("Qualification source differs from the prospective batch freeze")
        original = Path(original).resolve()
        inputs = dict(
            case=case,
            requested_duration_s=duration,
            pilot=pilot_duration_s is not None,
            original=str(original),
            original_sha256=sha256(original),
            design=design,
            inherited_manifest_sha256=sha256(MANIFEST),
            source_hashes=source_freeze["source_hashes"],
            dependency_versions=source_freeze["dependency_versions"],
            source_binding_scope="Imported ROOT/src closure including native step helper and its unconditional adapter dispatch dependency",
            mission_wall_limit_seconds=None,
            no_native_progress_timeout_seconds=no_progress_timeout_seconds,
            permanent_internal_raw=False,
        )
        atomic_json(output / "inputs.json", inputs)
        compiled = time.monotonic()
        payloads = [Payload(FREE_MASS_KG, r, s) for s, r in zip(case["sides"], case["ratios"])]
        f = DualMuJoCoFixture(original, output / "fixture", payloads)
        initials = qualification_initials(case)
        if initials is not None:
            f.reset(initials)
        equilibrium = _anchor(f, case) if case["group"] == "passive" else None
        f.model.opt.timestep = 0.01 / case["substeps"]
        numerics = NumericalMonitor(f, case["substeps"], equilibrium=equilibrium)
        half = None
        half_monitor = None
        online = None
        if case["group"] == "refinement":
            half = DualMuJoCoFixture(original, output / "half-dt-fixture", payloads)
            half.model.opt.timestep = 0.01 / 32
            half_monitor = NumericalMonitor(half, 32)
        elif case["group"] != "passive":
            online = _CombinedAudit(f, numerics)
        compile_seconds = time.monotonic() - compiled
        atomic_json(
            output / "initial-state.json",
            dict(base=_snapshot(f), half=None if half is None else _snapshot(half)),
        )
        atomic_json(
            output / "properties.json",
            dict(
                runtime=probe_runtime(f, substeps=case["substeps"], anchored=equilibrium is not None),
                half_runtime=None if half is None else probe_runtime(half, substeps=32),
                probe_substeps=case["substeps"],
                anchored=equilibrium is not None,
                formal_mission_profile=equilibrium is None and half is None,
            ),
        )
        controllers = (
            []
            if equilibrium is not None
            else [Controller(design["gain"], design["hover_action"]) for _ in range(2)]
        )
        refs = [] if equilibrium is not None else _references(f, case)
        desired_targets = [] if not refs else [r.target for r in refs]
        previous_held = (False, False)
        supervisor = ReferenceSupervisor()
        brake_tick = None if case["kind"] != "brake" else math.ceil(max(r.duration for r in refs) * 0.5 * 100)
        brake_positions = None
        stop_max = [0.0, 0.0]
        stop_time = None
        dwell = 0
        held_seconds = 0.0
        max_difference = dict(
            maximum_position_difference_m=0.0, maximum_angle_difference_deg=0.0, within_limits=True
        )
        passive_rows = [[], []]

        def retain_passive(sample):
            for index, row in enumerate(sample["drones"]):
                passive_rows[index].append(dict(time_s=sample["world_time"], **row))

        if equilibrium is not None:
            retain_passive(f.sample())
        watchdog = NativeProgressWatchdog(no_progress_timeout_seconds)
        watchdog.__enter__()
        numerics.watchdog = watchdog
        if half_monitor is not None:
            half_monitor.watchdog = watchdog
        next_heartbeat = time.monotonic() + 30
        trace = output / "compact-replay.jsonl.gz"
        with gzip.open(trace, "wt", compresslevel=3) as stream:
            stream.write(
                json.dumps(
                    dict(kind="RESET", actual=f.sample(), integration=_snapshot(f)),
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            for tick in range(round(duration * 100)):
                rows = f.sample()["drones"]
                now = tick * 0.01
                held = (False, False)
                if equilibrium is not None:
                    rpm = np.zeros((2, 4))
                    references = []
                else:
                    if tick == brake_tick:
                        brake_positions = [np.array(row["drone_xyz"]) for row in rows]
                        refs = [
                            r.brake(now, actual_state=KinematicState(row["drone_xyz"], row["drone_velocity"]))
                            for r, row in zip(refs, rows)
                        ]
                        desired_targets = [r.target for r in refs]
                    if case["kind"] == "crossing":
                        states = [KinematicState(row["drone_xyz"], row["drone_velocity"]) for row in rows]
                        requested = crossing_requests(refs, desired_targets, previous_held, states, now)
                        decision = supervisor.decide(
                            requested,
                            states,
                            now,
                            priority=case["priority"],
                        )
                        if decision.failure:
                            raise ValueError(decision.failure)
                        refs = list(decision.references)
                        held = decision.held
                        previous_held = held
                    references = [r.sample(now) for r in refs]
                    actions = np.array(
                        [
                            controller.action(row, p, tick=tick, reference_velocity=v)
                            for controller, row, (p, v, a) in zip(controllers, rows, references)
                        ]
                    )
                    rpm = (1 + 0.8 * actions) * HOVER_RPM
                if online is not None:
                    from .dual_coordination_env import step_native_with_audit

                    audit = step_native_with_audit(
                        f, rpm, online, detailed=False, location=f"qualification:{tick}"
                    )
                    if not audit["safe"]:
                        raise ValueError("Live online geometry/numerical qualification violation")
                else:
                    _special_step(f, rpm, numerics, case["substeps"])
                    audit = numerics.report()
                    if not audit["passed"]:
                        raise ValueError("Special-profile numerical qualification violation")
                sample = f.sample()
                comparison = None
                if half is not None:
                    _special_step(half, rpm, half_monitor, 32)
                    comparison = refinement_difference(sample, half.sample())
                    for key in ("maximum_position_difference_m", "maximum_angle_difference_deg"):
                        max_difference[key] = max(max_difference[key], comparison[key])
                    max_difference["within_limits"] &= comparison["within_limits"]
                    if not comparison["within_limits"]:
                        raise ValueError("Inherited same-input half-dt refinement threshold exceeded")
                boundary = audit_geometry(f)
                if (
                    not boundary["finite"]
                    or boundary["contacts"]
                    or boundary["wash_incursions"]
                    or boundary["min_clearance_m"] < 0.2
                    or min(boundary["floor_minima_m"]) <= 0.1
                    or boundary["constraint_max_m"] > 2e-5
                    or boundary["max_roll_pitch_deg"] > 60
                ):
                    raise ValueError("Qualification control boundary safety failure")
                if equilibrium is not None:
                    retain_passive(sample)
                else:
                    dwell = dwell + 1 if _settled(sample["drones"], payloads, desired_targets) else 0
                    if brake_positions is not None:
                        stop_max = [
                            max(old, float(np.linalg.norm(np.array(row["drone_xyz"]) - start)))
                            for old, row, start in zip(stop_max, sample["drones"], brake_positions)
                        ]
                        if dwell >= 100 and stop_time is None:
                            stop_time = sample["world_time"] - brake_tick * 0.01
                held_seconds += sum(held) * 0.01
                completed = tick + 1
                stream.write(
                    json.dumps(
                        dict(
                            kind="STEP",
                            tick=completed,
                            actual=sample,
                            rpm=rpm.tolist(),
                            reference=[
                                dict(p=p.tolist(), v=v.tolist(), a=a.tolist()) for p, v, a in references
                            ],
                            controls=[c.last_audit for c in controllers],
                            held=list(held),
                            numerical=numerics.report(),
                            refinement=comparison,
                            audit_scope="live-standard-reconstruction"
                            if online
                            else "special-numerical-probe",
                        ),
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                if time.monotonic() >= next_heartbeat:
                    journal.heartbeat(attempt, completed=completed, total=round(duration * 100))
                    next_heartbeat = time.monotonic() + 30
        watchdog.__exit__(None, None, None)
        period_results = []
        if case["kind"] == "anchor" and equilibrium is not None and pilot_duration_s is None:
            for drone, rows in enumerate(passive_rows):
                ids = f.body_ids[drone]
                names = ("suspension_rod", "payload")
                runtime = dict(
                    links=[dict(name=n, mass_kg=float(f.model.body_mass[ids[n]])) for n in names],
                    inertia_kg_m2=[np.diag(f.model.body_inertia[ids[n]]).tolist() for n in names],
                )
                expected = analytical_period(runtime, payloads[drone].side_m)
                measured, crossings, spread = crossing_period(rows, payloads[drone].side_m)
                period_results.append(
                    dict(
                        predicted_s=expected,
                        measured_s=measured,
                        crossings=crossings,
                        spread_s=spread,
                        passed=abs(measured - expected) / expected <= 0.1,
                    )
                )
        if any(not p["passed"] for p in period_results):
            raise ValueError("Inherited small-angle period10% gate failed")
        if equilibrium is None and dwell < 100 and pilot_duration_s is None:
            failure = (
                "Requested target/final reference did not satisfy the full one-second settling predicate"
            )
        final_sources = verify_qualification_source_freeze(source_freeze)
        result.update(
            source_integrity_passed=True,
            source_hashes_end=final_sources["source_hashes"],
            dependency_versions_end=final_sources["dependency_versions"],
            passed=failure is None and pilot_duration_s is None,
            completed_declared_duration=completed == round(case["duration_s"] * 100)
            and pilot_duration_s is None,
            completed_requested_duration=completed == round(duration * 100),
            compile_wall_seconds=compile_seconds,
            online_audit=None if online is None else online.report(),
            numerical=numerics.report(),
            half_numerical=None if half_monitor is None else half_monitor.report(),
            refinement=max_difference if half is not None else None,
            periods=period_results,
            brake_tick=brake_tick,
            stopping_time_s=stop_time,
            maximum_braking_displacement_m=stop_max,
            supervisor_held_drone_seconds=held_seconds,
            final_settling_ticks=dwell,
            compact_bytes=trace.stat().st_size,
            compact_sha256=sha256(trace),
            final_state=_snapshot(f),
            final_references=[r.to_dict() for r in refs],
            final_controllers=[
                dict(integral=c.integral, tick=c.tick, last_audit=c.last_audit) for c in controllers
            ],
        )
    except BaseException as error:
        if watchdog is not None:
            watchdog.__exit__(None, None, None)
        failure = f"{type(error).__name__}: {error}"
        result["passed"] = False
        if source_freeze is not None:
            try:
                current = qualification_source_freeze()
                result.update(
                    source_integrity_passed=current == source_freeze,
                    source_hashes_end=current["source_hashes"],
                    dependency_versions_end=current["dependency_versions"],
                )
            except BaseException as source_error:
                result.update(source_integrity_passed=False, source_verification_error=repr(source_error))
        if f is not None:
            try:
                result["failure_state"] = _safe_diagnostic(_snapshot(f))
                result["failure_actual"] = _safe_diagnostic(f.sample())
                result["failure_last_rpm"] = _safe_diagnostic(locals().get("rpm"))
                result["failure_control_tick"] = locals().get("tick")
                if "numerics" in locals():
                    result["failure_numerics"] = _safe_diagnostic(numerics.report())
                if "online" in locals() and online is not None:
                    result["failure_online_audit"] = _safe_diagnostic(online.report())
            except BaseException as capture_error:
                result["failure_capture_error"] = repr(capture_error)
        if "trace" in locals() and trace.exists():
            result["compact_bytes"] = trace.stat().st_size
            result["compact_sha256"] = sha256(trace)
        result.update(
            failure=failure,
            completed_control_steps=completed,
            elapsed_wall_seconds=time.monotonic() - started,
        )
        atomic_json(output / "result.json", result)
        journal.finish(attempt, status="failed", reason=failure)
        raise
    result.update(
        failure=failure, completed_control_steps=completed, elapsed_wall_seconds=time.monotonic() - started
    )
    atomic_json(output / "result.json", result)
    journal.finish(
        attempt,
        status="passed" if result["passed"] else "done" if failure is None else "failed",
        reason=failure
        or (
            "short development pilot only"
            if result["pilot"]
            else "case checks complete; overall admission remains separate"
        ),
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original")
    parser.add_argument("--design")
    parser.add_argument("--output")
    parser.add_argument("--journal-root")
    parser.add_argument("--no-progress-timeout", type=float, default=600.0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--pilot-duration", type=float)
    parser.add_argument("--declare-only", action="store_true")
    parser.add_argument("--source-freeze", help="Prospective batch source/dependency freeze JSON")
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(qualification_cases(), indent=2))
        return 0
    if not args.output:
        parser.error("--output required for declaration or execution")
    if args.declare_only:
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=False)
        atomic_json(
            destination / "declaration.json",
            dict(
                schema="dual-native-qualification-declaration-v2",
                crossing_revision="perpendicular-qualification-v2",
                revision_reason="Prospective perpendicular crossing and C2 hold resumption; unchanged durations, cohorts, safety and settling thresholds",
                cases=qualification_cases(),
                inherited_manifest_sha256=sha256(MANIFEST),
                admission_granted=False,
            ),
        )
        return 0
    if not args.original or not args.design:
        parser.error("--original and --design required for execution")
    matches = [c for c in qualification_cases() if c["case_id"] == args.case_id]
    if len(matches) != 1:
        parser.error("Choose one declared --case-id; no implicit bulk long run")
    result = run_case(
        original=args.original,
        design=json.loads(Path(args.design).read_text()),
        output=args.output,
        case=matches[0],
        pilot_duration_s=args.pilot_duration,
        journal_root=args.journal_root,
        no_progress_timeout_seconds=args.no_progress_timeout,
        expected_source_freeze=json.loads(Path(args.source_freeze).read_text())
        if args.source_freeze
        else None,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("passed", "pilot", "failure", "completed_control_steps")}, indent=2
        )
    )
    return 0 if result["failure"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
