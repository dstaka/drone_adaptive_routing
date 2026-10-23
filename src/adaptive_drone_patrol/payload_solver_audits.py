"""Actual same-backend motor15 and selective geometry-bank reset30 evidence.

These routines never admit learning by themselves. Each archive retains native
model binaries, actual getters, source bindings and every observed motor stage.
"""

import gzip
import json
from pathlib import Path

import numpy as np

from .jobs import write_json
from .payload_contract import Payload, rotation
from .payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG, HOVER_RPM
from .payload_solver_motor import original_fixture
from .payload_solver_work import reconstruct_interval_work
from .util import sha256

AUDIT_SOURCES = (
    "src/adaptive_drone_patrol/payload_solver_mujoco.py",
    "src/adaptive_drone_patrol/payload_solver_motor.py",
    "src/adaptive_drone_patrol/payload_solver_audits.py",
    "src/adaptive_drone_patrol/payload_ball_asset.py",
    "src/adaptive_drone_patrol/payload_contract.py",
    "src/adaptive_drone_patrol/payload_solver_work.py",
)


def _bind(root):
    return {p: sha256(Path(root) / p) for p in AUDIT_SOURCES}


def _save_model(fixture, path):
    fixture.mj.mj_saveModel(fixture.model, str(path), None)
    return dict(path=Path(path).name, bytes=Path(path).stat().st_size, sha256=sha256(path))


def record_motor_audit(output, original):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    source = _bind(Path.cwd())
    plan = dict(
        schema="solver-motor15-plan-v1",
        source_binding=source,
        original_sha256=sha256(original),
        fixtures=["original", "unloaded", "loaded"],
        rotors=[None, 0, 1, 2, 3],
        hover_duration_s=1.0,
        single_rotor_duration_s=0.05,
        substeps=16,
        control_dt_s=0.01,
        input="Hover RPM scaled by total/original mass, or original HOVER_RPM on one rotor and zero others.",
        rejection="Any source/property/wrench disagreement, nonfinite, missing native/raw/stage data or unloaded mismatch.",
        physical_training_admitted=False,
    )
    write_json(root / "plan.json", plan)
    records = []
    for mode in plan["fixtures"]:
        for rotor in plan["rotors"]:
            if _bind(Path.cwd()) != source:
                raise ValueError("Frozen motor source changed")
            name = mode + "-" + ("hover" if rotor is None else f"rotor{rotor}")
            run = root / name
            run.mkdir()
            if mode == "original":
                f = original_fixture(original, run / "fixture", substeps=16)
            else:
                f = MuJoCoFixture(
                    original,
                    run / "fixture",
                    Payload(FREE_MASS_KG, 0.2, 0.045),
                    loaded=mode == "loaded",
                    substeps=16,
                )
            runtime = f.runtime()
            write_json(run / "runtime.json", runtime)
            model = _save_model(f, run / "actual-model.mjb")
            duration = 1.0 if rotor is None else 0.05
            total = FREE_MASS_KG + (f.payload.mass + 0.0001 if mode == "loaded" else 0)
            rpm = (
                np.full(4, HOVER_RPM * np.sqrt(total / FREE_MASS_KG))
                if rotor is None
                else np.array([HOVER_RPM if i == rotor else 0.0 for i in range(4)])
            )
            declaration = dict(
                mode=mode,
                rotor=rotor,
                duration_s=duration,
                rpm=rpm.tolist(),
                source_binding=source,
                native_model=model,
            )
            write_json(run / "declaration.json", declaration)
            with (
                gzip.open(run / "raw.jsonl.gz", "wt") as raw,
                gzip.open(run / "stages.jsonl.gz", "wt") as stages,
            ):
                for step in range(round(duration / 0.01) + 1):
                    row = f.raw_state()
                    row["step"] = step
                    raw.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
                    if (
                        not np.isfinite(row["qpos"]).all()
                        or not np.isfinite(row["qvel"]).all()
                        or any(row["warning_counts"])
                    ):
                        raise ValueError("Motor raw excursion")
                    if step == round(duration / 0.01):
                        break
                    observed = f.step_rpm(rpm, observe_stages=True)
                    stages.write(
                        json.dumps(dict(step=step, stages=observed), separators=(",", ":"), allow_nan=True)
                        + "\n"
                    )
            receipt = dict(
                id=name,
                raw_sha256=sha256(run / "raw.jsonl.gz"),
                stages_sha256=sha256(run / "stages.jsonl.gz"),
                runtime_sha256=sha256(run / "runtime.json"),
                declaration_sha256=sha256(run / "declaration.json"),
            )
            write_json(run / "receipt.json", receipt)
            records.append(receipt)
    write_json(root / "records.json", records)
    return records


def _read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(x) for x in stream]


def verify_motor_audit(output, *, source_root=None):
    root = Path(output)
    source_root = Path.cwd() if source_root is None else Path(source_root)
    try:
        plan = json.loads((root / "plan.json").read_text())
        source = _bind(source_root)
        if plan["schema"] != "solver-motor15-plan-v1" or plan["source_binding"] != source:
            raise ValueError("Motor source differs")
        expected = {
            mode + "-" + ("hover" if rotor is None else f"rotor{rotor}")
            for mode in ["original", "unloaded", "loaded"]
            for rotor in [None, 0, 1, 2, 3]
        }
        records = json.loads((root / "records.json").read_text())
        if len(records) != 15 or {r["id"] for r in records} != expected:
            raise ValueError("Complete distinct motor15 required")
        all_rows = {}
        import mujoco as mj

        for receipt in records:
            run = root / receipt["id"]
            declaration = json.loads((run / "declaration.json").read_text())
            for name, key in [
                ("raw.jsonl.gz", "raw_sha256"),
                ("stages.jsonl.gz", "stages_sha256"),
                ("runtime.json", "runtime_sha256"),
                ("declaration.json", "declaration_sha256"),
            ]:
                if sha256(run / name) != receipt[key]:
                    raise ValueError("Motor raw bytes differ")
            if declaration["source_binding"] != source:
                raise ValueError("Motor source differs")
            native = declaration["native_model"]
            if sha256(run / native["path"]) != native["sha256"]:
                raise ValueError("Native actual model differs")
            model = mj.MjModel.from_binary_path(str(run / native["path"]))
            mode = declaration["mode"]
            expected_mode, expected_excitation = receipt["id"].split("-", 1)
            expected_rotor = None if expected_excitation == "hover" else int(expected_excitation[-1])
            expected_total = FREE_MASS_KG + (FREE_MASS_KG * 0.2 + 0.0001 if expected_mode == "loaded" else 0)
            expected_rpm = (
                np.full(4, HOVER_RPM * np.sqrt(expected_total / FREE_MASS_KG))
                if expected_rotor is None
                else np.array([HOVER_RPM if i == expected_rotor else 0.0 for i in range(4)])
            )
            if (
                declaration["mode"] != expected_mode
                or declaration["rotor"] != expected_rotor
                or declaration["duration_s"] != (1.0 if expected_rotor is None else 0.05)
                or not np.array_equal(declaration["rpm"], expected_rpm)
            ):
                raise ValueError("Original predeclared motor case/input differs")
            ids = [
                mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
                for n in ["base_link", *[f"prop{i}_link" for i in range(4)]]
            ]
            if not np.isclose(sum(model.body_mass[ids]), FREE_MASS_KG, atol=0, rtol=1e-12):
                raise ValueError("Original motor mass differs")
            for index in ids[1:]:
                if not np.isclose(model.body_mass[index], 1.1920928955078125e-7, atol=0, rtol=1e-12):
                    raise ValueError("Original rotor mass differs")
            expected_damping = np.zeros(model.nv)
            expected_damping[3:6] = (expected_total - FREE_MASS_KG + 0.027) * 1e-5
            if not np.allclose(model.dof_damping, expected_damping, rtol=1e-12, atol=1e-15):
                raise ValueError("Inherited authored-mass motor damping differs")
            rows = _read_gzip(run / "raw.jsonl.gz")
            stage_groups = _read_gzip(run / "stages.jsonl.gz")
            count = 100 if declaration["rotor"] is None else 5
            if len(rows) != count + 1 or len(stage_groups) != count:
                raise ValueError("Incomplete motor trace")
            for i, row in enumerate(rows):
                if row["step"] != i or not np.isclose(row["time_s"], i * 0.01, atol=1e-8, rtol=0):
                    raise ValueError("Motor sample continuity")
                if not np.isfinite(row["qpos"]).all() or not np.isfinite(row["qvel"]).all() or row["contact"]:
                    raise ValueError("Motor raw failure")
            total = FREE_MASS_KG + (FREE_MASS_KG * 0.2 + 0.0001 if mode == "loaded" else 0)
            for i, group in enumerate(stage_groups):
                if group["step"] != i or len(group["stages"]) != 64:
                    raise ValueError("Missing actual motor stages")
                padded = []
                for stage in group["stages"]:
                    if stage["rpm"] != declaration["rpm"]:
                        raise ValueError("Applied motor input differs")
                    stage = dict(stage)
                    if mode != "loaded":
                        # Zero padding embeds an unloaded root into the independent
                        # rotor-work map; no physical state or raw is modified.
                        stage.update(
                            qpos=stage["qpos"] + [1.0, 0.0, 0.0, 0.0] * 2,
                            qvel=stage["qvel"] + [0.0] * 6,
                            motor_generalized_force=stage["motor_generalized_force"] + [0.0] * 6,
                        )
                    padded.append(stage)
                reconstruct_interval_work(padded, substeps=16, start_time_s=i * 0.01, total_mass_kg=total)
            all_rows[receipt["id"]] = rows
        for rotor in [None, 0, 1, 2, 3]:
            suffix = "hover" if rotor is None else f"rotor{rotor}"
            a, b = all_rows["original-" + suffix], all_rows["unloaded-" + suffix]
            for ra, rb in zip(a, b):
                for field in ["xyz", "quaternion_wxyz", "velocity", "angular_velocity"]:
                    if not np.allclose(ra["drone_" + field], rb["drone_" + field], atol=1e-8, rtol=1e-5):
                        raise ValueError("Same-backend unloaded motor mismatch")
        return dict(passed=True, motor_cases=15, physical_training_admitted=False)
    except (OSError, KeyError, TypeError, IndexError) as exc:
        raise ValueError("Complete actual motor15 evidence required") from exc


def record_reset_audit(output, original):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    source = _bind(Path.cwd())
    rng = np.random.default_rng(71027)
    draws = [
        dict(
            side_m=[0.03, 0.045, 0.06][i % 3],
            mass_ratio=float(rng.uniform(0.1, 0.3)),
            swing_deg=float(i % 11),
            direction_rad=float(rng.uniform(-np.pi, np.pi)),
        )
        for i in range(30)
    ]
    plan = dict(
        schema="solver-reset30-plan-v1",
        source_binding=source,
        draws=draws,
        substeps=16,
        before="Advance actual selected scene with nonzero rotor input before each reset.",
        reset="Apply actual mass/inertia, reset actual native coordinates, velocities, RPM and all force buffers.",
        scope="N=1 active logical episode in a three-size physical scene bank; no N>1 logical neighbor claim.",
        verification="All actual selected getters/native binaries and byte-equal unselected scene states/properties.",
        physical_training_admitted=False,
    )
    write_json(root / "plan.json", plan)
    fixtures = {
        side: MuJoCoFixture(
            original, root / f"fixture-{side:.3f}", Payload(FREE_MASS_KG, 0.2, side), substeps=16
        )
        for side in [0.03, 0.045, 0.06]
    }
    records = []
    for i, draw in enumerate(draws):
        if _bind(Path.cwd()) != source:
            raise ValueError("Frozen reset source changed")
        selected = fixtures[draw["side_m"]]
        selected.step_rpm(np.full(4, HOVER_RPM * 1.1))
        before = {str(side): dict(raw=f.raw_state(), runtime=f.runtime()) for side, f in fixtures.items()}
        selected.apply_payload(Payload(FREE_MASS_KG, draw["mass_ratio"], draw["side_m"]))
        selected.reset(swing_deg=draw["swing_deg"], direction_rad=draw["direction_rad"])
        after = {str(side): dict(raw=f.raw_state(), runtime=f.runtime()) for side, f in fixtures.items()}
        native = _save_model(selected, root / f"reset-{i:02d}.mjb")
        record = dict(index=i, draw=draw, before=before, after=after, native_model=native)
        write_json(root / f"reset-{i:02d}.json", record)
        records.append(dict(index=i, path=f"reset-{i:02d}.json", sha256=sha256(root / f"reset-{i:02d}.json")))
    write_json(root / "records.json", records)
    return records


def verify_reset_audit(output, *, source_root=None):
    root = Path(output)
    source_root = Path.cwd() if source_root is None else Path(source_root)
    try:
        plan = json.loads((root / "plan.json").read_text())
        if plan["schema"] != "solver-reset30-plan-v1" or plan["source_binding"] != _bind(source_root):
            raise ValueError("Reset source differs")
        records = json.loads((root / "records.json").read_text())
        if len(records) != 30 or [r["index"] for r in records] != list(range(30)):
            raise ValueError("Thirty real resets required")
        import mujoco as mj

        for receipt in records:
            if sha256(root / receipt["path"]) != receipt["sha256"]:
                raise ValueError("Reset raw bytes differ")
            row = json.loads((root / receipt["path"]).read_text())
            draw = plan["draws"][row["index"]]
            key = str(draw["side_m"])
            if row["draw"] != draw:
                raise ValueError("Reset episode draw differs")
            before, after = row["before"][key], row["after"][key]
            if not np.any(np.asarray(before["raw"]["actual_applied_generalized_force"]) != 0):
                raise ValueError("Reset was not preceded by actual held force")
            for other in row["before"]:
                if other != key and row["before"][other] != row["after"][other]:
                    raise ValueError("Unselected physical scene changed")
            raw = after["raw"]
            runtime = after["runtime"]
            mass = FREE_MASS_KG * draw["mass_ratio"]
            if (
                raw["time_s"] != 0
                or np.any(raw["qvel"])
                or np.any(raw["rpm"])
                or np.any(raw["actual_applied_generalized_force"])
                or np.any(raw["actual_xfrc_applied"])
            ):
                raise ValueError("Actual reset input/state not cleared")
            if not np.isclose(runtime["package_mass_kg"], mass, atol=0, rtol=1e-12):
                raise ValueError("Actual reset mass differs")
            native = row["native_model"]
            if sha256(root / native["path"]) != native["sha256"]:
                raise ValueError("Actual native model bytes differ")
            model = mj.MjModel.from_binary_path(str(root / native["path"]))
            idx = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "payload")
            if not np.isclose(model.body_mass[idx], mass, atol=0, rtol=1e-12) or not np.allclose(
                model.body_inertia[idx], mass * draw["side_m"] ** 2 / 6, atol=0, rtol=1e-12
            ):
                raise ValueError("Native reset mass/inertia differs")
            expected_damping = np.zeros(model.nv)
            expected_damping[3:6] = (0.027 + 0.0001 + mass) * 1e-5
            if not np.allclose(model.dof_damping, expected_damping, rtol=1e-12, atol=1e-15):
                raise ValueError("Inherited authored-mass reset damping differs")
            geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "payload_collision")
            if not np.allclose(model.geom_size[geom], draw["side_m"] / 2, atol=1e-12, rtol=0):
                raise ValueError("Actual reset cube size differs")
            q = raw["rod_quaternion_wxyz"]
            half = np.radians(draw["swing_deg"]) / 2
            expected = [
                np.cos(half),
                np.sin(draw["direction_rad"]) * np.sin(half),
                -np.cos(draw["direction_rad"]) * np.sin(half),
                0.0,
            ]
            if not np.allclose(q, expected, atol=1e-10, rtol=0) or not np.allclose(
                raw["package_quaternion_wxyz"], expected, atol=1e-10, rtol=0
            ):
                raise ValueError("Actual reset pose differs")
            a = np.asarray(raw["drone_xyz"]) + rotation(raw["drone_quaternion_wxyz"]) @ [0, 0, -0.02]
            b = np.asarray(raw["package_xyz"]) + rotation(raw["package_quaternion_wxyz"]) @ [
                0,
                0,
                draw["side_m"] / 2,
            ]
            if not np.isclose(np.linalg.norm(b - a), 0.3, atol=1e-10, rtol=0):
                raise ValueError("Actual reset constraint differs")
        return dict(passed=True, actual_resets=30, logical_batch_size=1, physical_training_admitted=False)
    except (OSError, KeyError, TypeError, IndexError) as exc:
        raise ValueError("Complete actual reset30 evidence required") from exc
