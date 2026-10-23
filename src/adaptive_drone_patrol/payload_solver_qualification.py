"""Independent raw reconstruction for the selected RK4 articulated backend.

A declaration, receipt status, short reference, or synthetic CPU fixture never
admits an optimizer. Every stage binds actual model, source, settings and raw.
"""

import copy
import gzip
import json
import math
from pathlib import Path
import numpy as np

from .payload_recovery_qualification import make_manifest as inherited_manifest
from .payload_passive_check import mechanical_energy, analytical_period, crossing_period
from .payload_contract import rotation
from .payload_solver_reference import energy_balance
from .util import sha256

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "payload-mujoco-domain-v1"
PHYSICAL_SOURCES = (
    "src/adaptive_drone_patrol/payload_solver_mujoco.py",
    "src/adaptive_drone_patrol/payload_ball_asset.py",
    "src/adaptive_drone_patrol/payload_contract.py",
    "src/adaptive_drone_patrol/payload_probe_controller.py",
)


def make_manifest(*, base_substeps=16):
    legacy = json.loads((ROOT / "configs/payload_retry_probe_manifest.json").read_text())
    result = inherited_manifest(legacy, base_substeps=base_substeps)
    result.update(schema=SCHEMA, profile=f"cf2x-mujoco-rk4-f64-s{base_substeps}-v2")
    result["settings"].pop("genesis_version")
    result["settings"].update(
        backend="mujoco",
        mujoco_version="3.15.0",
        precision="64",
        integrator="RK4",
        execution_backend="cpu",
        base_angular_damping_mass_basis="authored-urdf-0.027kg-plus-rod-and-payload",
    )
    result["limits"]["absolute_unexplained_passive_energy_fraction"] = 0.05
    result["limits"]["unexplained_passive_energy_loss_fraction"] = 0.05
    result["energy"].update(
        substep_work_bound_method="actual-rk4-stage-wrench-power-v1",
        substep_work_source="src/adaptive_drone_patrol/payload_solver_mujoco.py",
        substep_work_declaration="Evaluate original rotor COM wrench in all four actual RK stages. "
        "Integrate signed motor and passive generalized power with h*(1,2,2,1)/6. "
        "The previous constant-world-force semiimplicit path no longer applies. Preserve endpoint .02J+1.1 bound; "
        "check signed stage energy residual and identical-input half-dt convergence separately.",
    )
    return result


def validate_manifest(manifest):
    try:
        if manifest != make_manifest(base_substeps=manifest["base_substeps"]):
            raise ValueError("Solver domain differs from complete frozen physical template")
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid solver manifest") from exc
    return True


def passive_cases(manifest):
    validate_manifest(manifest)
    return [c for c in manifest["cases"] if c["kind"] in ("anchor", "large_angle")]


def physical_errors(runtime, case, *, fixed):
    """Compare actual compiled body properties and numerical settings with SI model."""
    failures = []
    try:
        if (
            runtime["backend"] != "mujoco"
            or runtime["version"] != "3.15.0"
            or runtime["precision"] != "64"
            or runtime["integrator"] != 1
            or runtime["substeps"] != case["substeps"]
            or not math.isclose(runtime["actual_dt_s"], 0.01 / case["substeps"], abs_tol=1e-15, rel_tol=0)
        ):
            failures.append("ACTUAL_NUMERICAL_SETTINGS")
        names = [x["name"] for x in runtime["links"]]
        expected = ["base_link", *[f"prop{i}_link" for i in range(4)], "suspension_rod", "payload"]
        if names != expected:
            failures.append("ACTUAL_BODY_IDENTITIES")
            return failures
        m = 0.02700047753751278 * case["mass_ratio"]
        masses = [0.027000000700354576, *([1.1920928955078125e-7] * 4), 0.0001, m]
        inertias = [
            [1.4e-5, 1.4e-5, 2.17e-5],
            *([[0.0, 0.0, 0.0]] * 4),
            [0.0001 * (3 * 0.001**2 + 0.3**2) / 12] * 2 + [0.0001 * 0.001**2 / 2],
            [m * case["side_m"] ** 2 / 6] * 3,
        ]
        coms = [
            [0, 0, 0],
            [0.028, -0.028, 0],
            [-0.028, -0.028, 0],
            [-0.028, 0.028, 0],
            [0.028, 0.028, 0],
            [0, 0, -0.15],
            [0, 0, 0],
        ]
        for i, (mass, inertia, com) in enumerate(zip(masses, inertias, coms)):
            if not math.isclose(runtime["links"][i]["mass_kg"], mass, rel_tol=1e-5, abs_tol=0):
                failures.append("ACTUAL_MASS")
            matrix = np.asarray(runtime["inertia_kg_m2"][i])
            if matrix.shape != (3, 3) or not np.allclose(matrix, np.diag(inertia), rtol=1e-5, atol=1e-15):
                failures.append("ACTUAL_INERTIA")
            if not np.allclose(runtime["links_com_body_m"][i], com, rtol=0, atol=1e-7):
                failures.append("ACTUAL_COM")
        if not math.isclose(runtime["free_drone_mass_kg"], 0.02700047753751278, rel_tol=1e-5, abs_tol=0):
            failures.append("ACTUAL_ORIGINAL_FREE_MASS")
        if np.any(np.asarray(runtime["armature"]) != 0) or np.any(
            np.asarray(runtime["dof_frictionloss"]) != 0
        ):
            failures.append("UNDECLARED_JOINT_FORCES")
        damping = np.asarray(runtime["damping_after"])
        wanted = np.zeros(6 if fixed else 12)
        if not fixed:
            wanted[3:6] = (0.027 + 0.0001 + m) * 1e-5
        if damping.shape != wanted.shape or not np.allclose(damping, wanted, atol=1e-15, rtol=1e-8):
            failures.append("ACTUAL_DAMPING")
        if runtime.get("gravity_m_s2") != [0.0, 0.0, -9.81] or any(runtime.get("joint_limited", [True])):
            failures.append("GRAVITY_OR_JOINT_LIMITS")
        geoms = runtime["collision_geoms"]
        if len(geoms) != 3:
            failures.append("ACTUAL_COLLISION_COUNT")
        for name, kind, size in [
            ("ground", 0, [20.0, 20.0, 0.1]),
            ("vehicle_collision", 5, [0.06, 0.0125, 0]),
            ("payload_collision", 6, [case["side_m"] / 2] * 3),
        ]:
            value = next((g for g in geoms if g["name"] == name), None)
            if (
                value is None
                or value["type"] != kind
                or not np.allclose(value["size"], size, atol=1e-5, rtol=0)
            ):
                failures.append("ACTUAL_COLLISION_EXTENT")
    except (ValueError, KeyError, TypeError, IndexError, StopIteration):
        failures.append("INVALID_ACTUAL_PROPERTIES")
    return sorted(set(failures))


def screen_passive(case, rows, runtime):
    """Screen complete actual passive samples, including both signs of energy drift."""
    failures = []
    result = dict(
        passed=False, failure_codes=failures, physical_training_admitted=False, sample_count=len(rows)
    )
    expected = round(case["duration_s"] / 0.01) + 1
    if len(rows) != expected or not rows:
        failures.append("INCOMPLETE_PASSIVE_TRACE")
        return result
    energies = []
    try:
        anchor = np.asarray(rows[0]["drone_xyz"]) + rotation(rows[0]["drone_quaternion_wxyz"]) @ [0, 0, -0.02]
        maxres = 0.0
        for i, row in enumerate(rows):
            if row["step"] != i or not math.isclose(row["time_s"], i * 0.01, abs_tol=1e-7, rel_tol=0):
                failures.append("SAMPLE_CONTINUITY")
            for body in ("drone", "rod", "package"):
                q = np.asarray(row[f"{body}_quaternion_wxyz"])
                v = np.asarray(row[f"{body}_velocity"])
                w = np.asarray(row[f"{body}_angular_velocity"])
                p = np.asarray(row[f"{body}_xyz"])
                if (
                    q.shape != (4,)
                    or v.shape != (3,)
                    or w.shape != (3,)
                    or p.shape != (3,)
                    or not all(np.isfinite(x).all() for x in (q, v, w, p))
                ):
                    raise ValueError("Invalid raw")
                if abs(np.linalg.norm(q) - 1) > 1e-4 or np.linalg.norm(v) > 20 or np.linalg.norm(w) > 200:
                    failures.append("RAW_PHYSICAL_GUARD")
            actual = np.asarray(row["drone_xyz"]) + rotation(row["drone_quaternion_wxyz"]) @ [0, 0, -0.02]
            endpoint = np.asarray(row["package_xyz"]) + rotation(row["package_quaternion_wxyz"]) @ [
                0,
                0,
                case["side_m"] / 2,
            ]
            if (
                np.linalg.norm(actual - anchor) > 1e-6
                or np.linalg.norm(row["drone_velocity"]) > 1e-6
                or np.linalg.norm(row["drone_angular_velocity"]) > 1e-6
            ):
                failures.append("ANCHOR_MOVED")
            residual = abs(np.linalg.norm(endpoint - actual) - 0.3)
            maxres = max(maxres, float(residual))
            if residual > 2e-5:
                failures.append("CONSTRAINT_RESIDUAL")
            if row["contact"] is not False:
                failures.append("CONTACT")
            for key, shape in [
                ("rpm", (4,)),
                ("actual_applied_generalized_force", (6,)),
                ("actual_passive_generalized_force", (6,)),
            ]:
                value = np.asarray(row[key])
                if value.shape != shape or not np.isfinite(value).all() or np.any(value != 0):
                    failures.append("PASSIVE_ACTUAL_INPUT")
            energies.append(mechanical_energy(row, runtime, case["side_m"], float(anchor[2])))
        balance = energy_balance(
            energies, applied_work_j=np.zeros(len(rows)), dissipative_work_j=np.zeros(len(rows))
        )
        result.update(energy_balance=balance, maximum_constraint_residual_m=maxres)
        if not balance["passed"]:
            failures.append("ABSOLUTE_PASSIVE_ENERGY_BALANCE")
        if case["kind"] == "anchor":
            if case["duration_s"] < 120:
                failures.append("SHORT_ANCHOR")
            measured, crossings, _ = crossing_period(rows, case["side_m"])
            expected_period = analytical_period(runtime, case["side_m"])
            error = abs(measured - expected_period) / expected_period
            result.update(period_relative_error=error, measured_period_s=measured, crossings=crossings)
            if error > 0.1:
                failures.append("SMALL_ANGLE_PERIOD")
        elif case["kind"] != "large_angle" or case["duration_s"] < 30:
            failures.append("SHORT_OR_INVALID_PASSIVE_CASE")
    except (KeyError, ValueError, TypeError, IndexError, np.linalg.LinAlgError, ZeroDivisionError):
        failures.append("INVALID_PASSIVE_RAW")
    result["failure_codes"] = sorted(set(failures))
    result["passed"] = not failures
    return result


def verify_passive_probe(path, manifest, *, source_root=ROOT):
    """Rehash exact source/model/raw and reconstruct full physical qualification."""
    validate_manifest(manifest)
    path = Path(path)
    source_root = Path(source_root)
    declaration = json.loads((path / "declaration.json").read_text())
    case = next(c for c in passive_cases(manifest) if c["id"] == declaration["case"]["id"])
    if (
        declaration["case"] != case
        or declaration["settings"] != manifest["settings"]
        or declaration.get("diagnostic_duration_s") is not None
    ):
        raise ValueError("Full matching passive case/settings required")
    if declaration["source_binding"] != {p: sha256(source_root / p) for p in PHYSICAL_SOURCES}:
        raise ValueError("Executed physical source differs")
    summary = json.loads((path / "summary.json").read_text())
    if summary["raw_sha256"] != sha256(path / "raw.jsonl.gz") or summary["runtime_sha256"] != sha256(
        path / "runtime.json"
    ):
        raise ValueError("Actual raw/runtime bytes differ")
    runtime = json.loads((path / "runtime.json").read_text())
    if runtime["model_xml_sha256"] != sha256(path / "fixture/model.xml") or physical_errors(
        runtime, case, fixed=True
    ):
        raise ValueError("Actual physical properties/model differ")
    with gzip.open(path / "raw.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    first = rows[0]
    expected_q = [
        math.cos(math.radians(case["initial_swing_deg"]) / 2),
        0.0,
        -math.sin(math.radians(case["initial_swing_deg"]) / 2),
        0.0,
    ]
    if (
        not np.allclose(first["rod_quaternion_wxyz"], expected_q, rtol=0, atol=1e-7)
        or not np.allclose(first["package_quaternion_wxyz"], expected_q, rtol=0, atol=1e-7)
        or not np.allclose(first["drone_xyz"], [0, 0, 1], rtol=0, atol=1e-7)
        or any(
            np.any(np.asarray(first[b + "_" + field]) != 0)
            for b in ("drone", "rod", "package")
            for field in ("velocity", "angular_velocity")
        )
    ):
        raise ValueError("Actual initial state differs")
    return screen_passive(case, rows, runtime)


def require_passive_barrier(manifest, locations, *, source_root=ROOT):
    expected = {c["id"] for c in passive_cases(manifest)}
    if set(locations) != expected or len({str(Path(p).resolve()) for p in locations.values()}) != 32:
        raise ValueError("All distinct 32 passive raw traces are required")
    checks = {
        key: verify_passive_probe(path, manifest, source_root=source_root) for key, path in locations.items()
    }
    if not all(c["passed"] for c in checks.values()):
        raise ValueError("Independent passive32 qualification failed")
    return checks


def verify_powered_probe(path, manifest, case, *, source_root=ROOT):
    """Reconstruct complete stage work and every raw physical/sample guard."""
    from .payload_solver_work import total_mechanical_energy
    from .payload_solver_work_check import reconstruct_batch_work

    validate_manifest(manifest)
    path = Path(path)
    source_root = Path(source_root)
    try:
        if case not in manifest["cases"] or case["kind"] not in ("powered", "refinement"):
            raise ValueError("Declared full powered case required")
        declaration = json.loads((path / "declaration.json").read_text())
        wanted = (
            *PHYSICAL_SOURCES,
            "src/adaptive_drone_patrol/payload_solver_work.py",
            "scripts/qualify_payload_solver.py",
        )
        if (
            declaration["case"] != case
            or declaration["settings"] != manifest["settings"]
            or declaration["diagnostic_duration_s"] is not None
            or declaration["source_binding"] != {p: sha256(source_root / p) for p in wanted}
        ):
            raise ValueError("Executed powered source/case differs")
        summary = json.loads((path / "summary.json").read_text())
        runtime = json.loads((path / "runtime.json").read_text())
        for name, key in [
            ("raw.jsonl.gz", "raw_sha256"),
            ("runtime.json", "runtime_sha256"),
            ("rpm.npy", "rpm_sha256"),
            ("stages-index.json", "stages_index_sha256"),
        ]:
            if sha256(path / name) != summary[key]:
                raise ValueError("Powered actual bytes differ")
        if physical_errors(runtime, case, fixed=False) or runtime["model_xml_sha256"] != sha256(
            path / "fixture/model.xml"
        ):
            raise ValueError("Powered actual properties differ")
        count = round(case["duration_s"] / 0.01)
        with gzip.open(path / "raw.jsonl.gz", "rt") as stream:
            rows = [json.loads(x) for x in stream]
        rpm = np.load(path / "rpm.npy", allow_pickle=False)
        if (
            len(rows) != count + 1
            or rpm.shape != (count, 4)
            or not np.isfinite(rpm).all()
            or summary["status"] != "RAW_COMPLETE_UNCHECKED"
            or summary["steps"] != count
        ):
            raise ValueError("Incomplete powered trace")
        if declaration["rpm_replay"] is not None:
            replay = declaration["rpm_replay"]
            if sha256(replay["path"]) != replay["sha256"] or not np.array_equal(
                rpm, np.load(replay["path"], allow_pickle=False)
            ):
                raise ValueError("Prescribed rotor replay differs")
        energy = []
        minimum_floor = 10.0
        maximum_tilt = 0.0
        maximum_constraint = 0.0
        initial_q = [
            math.cos(math.radians(case["initial_swing_deg"]) / 2),
            0.0,
            -math.sin(math.radians(case["initial_swing_deg"]) / 2),
            0.0,
        ]
        if (
            not np.allclose(rows[0]["drone_xyz"], [0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(rows[0]["drone_quaternion_wxyz"], [1, 0, 0, 0], atol=1e-10, rtol=0)
            or not np.allclose(rows[0]["rod_quaternion_wxyz"], initial_q, atol=1e-10, rtol=0)
            or not np.allclose(rows[0]["package_quaternion_wxyz"], initial_q, atol=1e-10, rtol=0)
            or np.any(rows[0]["qvel"])
        ):
            raise ValueError("Powered actual initial state differs")
        from .payload_contract import Payload, payload_metrics

        payload = Payload(0.02700047753751278, case["mass_ratio"], case["side_m"])
        for i, row in enumerate(rows):
            if row["step"] != i or not math.isclose(row["time_s"], i * 0.01, abs_tol=1e-7, rel_tol=0):
                raise ValueError("Powered sample continuity")
            if (
                not np.isfinite(row["qpos"]).all()
                or not np.isfinite(row["qvel"]).all()
                or any(row["warning_counts"])
            ):
                raise ValueError("Powered native state/warning")
            if not np.array_equal(row["rpm"], np.zeros(4) if i == 0 else rpm[i - 1]):
                raise ValueError("Recorded actual current RPM differs")
            for body in ["drone", "rod", "package"]:
                if (
                    not np.isfinite(row[body + "_xyz"]).all()
                    or np.linalg.norm(row[body + "_velocity"]) > 20
                    or np.linalg.norm(row[body + "_angular_velocity"]) > 200
                ):
                    raise ValueError("Powered actual speed guard")
                rotation(row[body + "_quaternion_wxyz"])
            metrics = payload_metrics(row, [0, 0, 1], payload)
            R = rotation(row["drone_quaternion_wxyz"])
            drone_floor = row["drone_xyz"][2] - 0.06 * np.linalg.norm(R[2, :2]) - 0.0125 * abs(R[2, 2])
            floor = min(drone_floor, metrics["package_floor_m"])
            tilt = max(abs(metrics["drone_roll_deg"]), abs(metrics["drone_pitch_deg"]))
            minimum_floor = min(minimum_floor, float(floor))
            maximum_tilt = max(maximum_tilt, float(tilt))
            maximum_constraint = max(maximum_constraint, metrics["constraint_residual_m"])
            if floor <= 0.1 or tilt > 60 or row["contact"] or metrics["constraint_residual_m"] > 2e-5:
                raise ValueError("Powered physical safety")
            energy.append(total_mechanical_energy(row, runtime, case["side_m"]))
        index = json.loads((path / "stages-index.json").read_text())
        next_step = 0
        seen = set()
        work = []
        loss = []
        bound = []
        total = 0.02700047753751278 + 0.0001 + payload.mass
        for chunk in index:
            native = path / chunk["path"]
            if chunk["path"] in seen or chunk["first_step"] != next_step or sha256(native) != chunk["sha256"]:
                raise ValueError("Missing/repeated native stage chunk")
            seen.add(chunk["path"])
            with np.load(native, allow_pickle=False) as packed:
                array = packed["stages"]
            if list(array.shape) != chunk["shape"] or len(array) != chunk["control_intervals"]:
                raise ValueError("Stage chunk shape differs")
            value = reconstruct_batch_work(
                array, substeps=case["substeps"], first_step=next_step, total_mass_kg=total
            )
            expected = rpm[next_step : next_step + len(array)]
            if not np.array_equal(
                value["rpm_all_stages"], np.broadcast_to(expected[:, None, :], value["rpm_all_stages"].shape)
            ):
                raise ValueError("Actual stage inputs differ")
            work.extend(value["motor_work_j"])
            loss.extend(value["dissipative_work_j"])
            bound.extend(value["endpoint_absolute_work_bound_j"])
            next_step += len(array)
        if next_step != count:
            raise ValueError("Incomplete actual RK-stage coverage")
        cumulative = np.r_[0.0, np.cumsum(work)]
        dissipative = np.r_[0.0, np.cumsum(loss)]
        endpoint = np.r_[0.0, np.cumsum(bound)]
        residual = np.asarray(energy) - energy[0] - cumulative - dissipative
        for key, values in [
            ("motor_work_j", cumulative),
            ("dissipative_work_j", dissipative),
            ("legacy_endpoint_absolute_work_bound_j", endpoint),
        ]:
            if not np.allclose([r[key] for r in rows], values, atol=1e-8, rtol=1e-10):
                raise ValueError("Claimed work differs from actual stages")
        if np.max(abs(residual)) > 0.02 or np.any(np.asarray(energy) - energy[0] > 0.02 + 1.1 * endpoint):
            raise ValueError("Independent signed/endpoint energy-work bound")
        return dict(
            passed=True,
            case_id=case["id"],
            sample_count=len(rows),
            actual_stage_count=count * case["substeps"] * 4,
            maximum_absolute_energy_work_residual_j=float(np.max(abs(residual))),
            minimum_floor_m=minimum_floor,
            maximum_tilt_deg=maximum_tilt,
            maximum_constraint_residual_m=maximum_constraint,
            physical_training_admitted=False,
        )
    except (OSError, KeyError, TypeError, IndexError, StopIteration) as exc:
        raise ValueError("Complete actual powered raw/source/work evidence required") from exc


def verify_refinement_pairs(root, manifest, *, source_root=ROOT):
    validate_manifest(manifest)
    root = Path(root)
    results = {}
    try:
        cases = [c for c in manifest["cases"] if c["kind"] == "refinement"]
        if len(cases) != 12:
            raise ValueError("Twelve refinement traces required")
        for base in [c for c in cases if c["timestep_role"] == "base"]:
            half = next(
                c
                for c in cases
                if c["timestep_role"] == "half"
                and c["side_m"] == base["side_m"]
                and c["mass_ratio"] == base["mass_ratio"]
            )
            a, b = root / base["id"], root / half["id"]
            verify_powered_probe(a, manifest, base, source_root=source_root)
            verify_powered_probe(b, manifest, half, source_root=source_root)
            if not np.array_equal(
                np.load(a / "rpm.npy", allow_pickle=False), np.load(b / "rpm.npy", allow_pickle=False)
            ):
                raise ValueError("Refinement rotor commands are not identical")
            with gzip.open(a / "raw.jsonl.gz", "rt") as f:
                ra = [json.loads(x) for x in f]
            with gzip.open(b / "raw.jsonl.gz", "rt") as f:
                rb = [json.loads(x) for x in f]
            position = angle = 0.0
            for x, y in zip(ra, rb):
                for body in ["drone", "rod", "package"]:
                    position = max(
                        position, float(np.linalg.norm(np.array(x[body + "_xyz"]) - y[body + "_xyz"]))
                    )
                    qa, qb = (
                        np.asarray(x[body + "_quaternion_wxyz"]),
                        np.asarray(y[body + "_quaternion_wxyz"]),
                    )
                    dot = abs(float(qa @ qb) / (np.linalg.norm(qa) * np.linalg.norm(qb)))
                    angle = max(angle, float(np.degrees(2 * np.arccos(np.clip(dot, 0.0, 1.0)))))
            if position > 0.02 or angle > 2:
                raise ValueError("Same-input powered timestep refinement failed")
            results[base["id"]] = dict(
                passed=True, maximum_position_difference_m=position, maximum_angle_difference_deg=angle
            )
        if len(results) != 6:
            raise ValueError("Six distinct refinement pairs required")
        return results
    except (OSError, KeyError, TypeError, IndexError, StopIteration) as exc:
        raise ValueError("Complete powered refinement12 raw required") from exc


QUALIFICATION_SCHEMA = "payload-solver-complete-qualification-v2"
ORIGINAL_ASSET_SHA256 = "5923a231efe0f052c31f7b5b4d3599c3b3a7e95fa2b2772b023798a68ecafc05"
RUNTIME_SOURCES = (
    "src/adaptive_drone_patrol/payload_solver_runtime.py",
    "src/adaptive_drone_patrol/payload_retry_contract.py",
    "src/adaptive_drone_patrol/payload_ball_bank.py",
    "src/adaptive_drone_patrol/payload_solver_initial_reset.py",
    "src/adaptive_drone_patrol/payload_cases.py",
    "src/adaptive_drone_patrol/payload_solver_engine.py",
)


def current_solver_binding(*, source_root=ROOT):
    """Read current actual physics/build/runtime contracts; no supplied status."""
    import importlib.metadata as metadata
    import sys
    from .payload_retry_contract import OBSERVATION_CONTRACT

    source_root = Path(source_root)
    try:
        distribution = metadata.distribution("mujoco")
        from .payload_solver_engine import engine_hashes

        engine = engine_hashes(distribution)
        if (
            distribution.version != "3.15.0"
            or metadata.version("numpy") != "2.5.3"
            or sys.version_info[:2] != (3, 12)
        ):
            raise ValueError("Actual qualified Python/MuJoCo/NumPy versions differ")
        return dict(
            profile=make_manifest()["profile"],
            original_asset_sha256=ORIGINAL_ASSET_SHA256,
            settings=make_manifest()["settings"],
            batch_size=1,
            physical_source_binding={p: sha256(source_root / p) for p in PHYSICAL_SOURCES},
            runtime_source_binding={p: sha256(source_root / p) for p in RUNTIME_SOURCES},
            engine_source_and_build=engine,
            physics_versions=dict(mujoco="3.15.0", numpy="2.5.3", python="3.12"),
            reference_lock_sha256=sha256(source_root / "requirements/payload-reference-linux.lock.txt"),
            observation_contract=copy.deepcopy(OBSERVATION_CONTRACT),
            motor_action="Four original immediate clipped actions; RPM=(1+.8*a)*14468.429183500699",
        )
    except (OSError, metadata.PackageNotFoundError) as exc:
        raise ValueError("Actual qualified physics build/source is unavailable") from exc


def _raw_manifest(locations):
    """Hash actual retained artifacts; mutable analysis summaries are excluded."""
    result = {}
    fixed = {
        "plan.json",
        "provenance.json",
        "tag.json",
        "pypi.json",
        "LICENSE",
        "manifest.json",
        "declaration.json",
        "runtime.json",
        "summary.json",
        "stages-index.json",
        "records.json",
    }
    for stage, location in sorted(locations.items()):
        root = Path(location)
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Qualification artifacts cannot alias through symlinks")
            if not path.is_file():
                continue
            if (
                path.suffix in {".gz", ".npz", ".npy", ".mjb", ".xml"}
                or path.name in fixed
                or path.name.startswith(("reset-", "initial-reset-"))
                and path.suffix == ".json"
            ):
                result[f"{stage}/{path.relative_to(root)}"] = dict(
                    bytes=path.stat().st_size, sha256=sha256(path)
                )
    return result


def issue_qualification(locations, *, source_root=ROOT):
    """Reconstruct all62/motor15/reset30 and current actual configuration."""
    expected = {"passive", "powered", "refinement", "motor", "reset", "initial_reset", "engine_provenance"}
    if not isinstance(locations, dict) or set(locations) != expected:
        raise ValueError("Complete distinct actual qualification locations required")
    if len({str(Path(p).resolve()) for p in locations.values()}) != len(expected):
        raise ValueError("Qualification stages cannot alias")
    from .payload_solver_audits import verify_motor_audit, verify_reset_audit
    from .payload_solver_initial_reset import verify_initial_resets

    manifest = make_manifest()
    source_root = Path(source_root)
    binding = current_solver_binding(source_root=source_root)
    try:
        passive_plan = json.loads((Path(locations["passive"]) / "plan.json").read_text())
        if passive_plan["original_asset_sha256"] != ORIGINAL_ASSET_SHA256:
            raise ValueError("Actually qualified original asset differs")
        from .payload_solver_engine import verify_engine_provenance

        engine_provenance = verify_engine_provenance(locations["engine_provenance"])
        actual_engine = binding["engine_source_and_build"]
        if engine_provenance["complete_engine"] != actual_engine:
            raise ValueError("Full actual engine differs from pre-capture provenance")
        recorded_engine = passive_plan["engine_source_and_build"]
        legacy_projection = {p: h for p, h in actual_engine.items() if p.endswith((".py", ".so", ".h"))}
        if recorded_engine not in (actual_engine, legacy_projection):
            raise ValueError("Actual captured engine projection differs from full provenance")
        if passive_plan["lock_sha256"] != binding["reference_lock_sha256"]:
            raise ValueError("Qualified reference lock differs")
        passive = require_passive_barrier(
            manifest,
            {c["id"]: Path(locations["passive"]) / c["id"] for c in passive_cases(manifest)},
            source_root=source_root,
        )
        powered = {
            c["id"]: verify_powered_probe(
                Path(locations["powered"]) / c["id"], manifest, c, source_root=source_root
            )
            for c in manifest["cases"]
            if c["kind"] == "powered"
        }
        refinement = verify_refinement_pairs(locations["refinement"], manifest, source_root=source_root)
        motor = verify_motor_audit(locations["motor"], source_root=source_root)
        resets = verify_reset_audit(locations["reset"], source_root=source_root)
        initial = verify_initial_resets(locations["initial_reset"], source_root=source_root)
        raw = _raw_manifest(locations)
        import hashlib

        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return dict(
            schema=QUALIFICATION_SCHEMA,
            status="INDEPENDENTLY_QUALIFIED",
            physical_training_admitted=True,
            actual_binding=binding,
            raw_manifest_sha256=digest,
            engine_provenance=engine_provenance,
            raw_manifest=raw,
            qualification=dict(
                passive32=passive,
                powered18=powered,
                refinement6pairs12traces=refinement,
                motor15=motor,
                reset30=resets,
                initial_reset30=initial,
            ),
            scope="Only the actual bound N1 RK4 configuration; no policy or operational acceptance claim",
        )
    except (OSError, KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete actual full qualification evidence") from exc


def require_qualification(receipt, actual_binding, locations):
    """A status string cannot admit training; replay raw at every consumption."""
    if not isinstance(receipt, dict) or receipt.get("schema") != QUALIFICATION_SCHEMA:
        raise ValueError("Complete all62 + motor15 + actual reset30 raw qualification required")
    if receipt.get("actual_binding") != actual_binding:
        raise ValueError("Consumer binding differs from qualification")
    recomputed = issue_qualification(locations)
    if (
        receipt.get("raw_manifest_sha256") != recomputed["raw_manifest_sha256"]
        or receipt.get("raw_manifest") != recomputed["raw_manifest"]
        or actual_binding != recomputed["actual_binding"]
    ):
        raise ValueError("Actual raw/source/build differs from qualification at consumption")
    return recomputed
