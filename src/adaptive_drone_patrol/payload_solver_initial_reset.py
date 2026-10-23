"""Actual common-runtime reset admission for independent drone/package poses."""

import json
from pathlib import Path
import numpy as np

from .jobs import write_json
from .payload_cases import initial_state
from .payload_contract import Payload
from .payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG
from .payload_solver_runtime import SolverRuntime
from .util import sha256

SOURCES = (
    "src/adaptive_drone_patrol/payload_solver_initial_reset.py",
    "src/adaptive_drone_patrol/payload_solver_runtime.py",
    "src/adaptive_drone_patrol/payload_solver_mujoco.py",
    "src/adaptive_drone_patrol/payload_ball_bank.py",
    "src/adaptive_drone_patrol/payload_retry_contract.py",
    "src/adaptive_drone_patrol/payload_contract.py",
)


def record_initial_resets(output, original):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(77013)
    draws = []
    for i in range(30):
        side = [0.03, 0.045, 0.06][i % 3]
        draws.append(
            dict(side_m=side, mass_ratio=float(rng.uniform(0.1, 0.3)), initial=initial_state(rng, side))
        )
    plan = dict(
        schema="solver-initial-reset30-v1",
        source_binding={p: sha256(p) for p in SOURCES},
        draws=draws,
        batch_size=1,
        scope="One active logical episode and three independent physical size scenes.",
        physical_training_admitted=False,
    )
    write_json(root / "plan.json", plan)
    fixtures = {
        side: MuJoCoFixture(
            original, root / f"fixture-{side:.3f}", Payload(FREE_MASS_KG, 0.2, side), substeps=16
        )
        for side in [0.03, 0.045, 0.06]
    }
    cores = {side: SolverRuntime(f, root / f"runtime-{side:.3f}") for side, f in fixtures.items()}
    for core in cores.values():
        core.reset_case()
    records = []
    try:
        for i, draw in enumerate(draws):
            side = draw["side_m"]
            f = fixtures[side]
            core = cores[side]
            core.step([0.1] * 4)
            before = {str(s): dict(raw=x.raw_state(), runtime=x.runtime()) for s, x in fixtures.items()}
            f.apply_payload(Payload(FREE_MASS_KG, draw["mass_ratio"], side))
            core.payload = f.payload
            core.reset_case(initial=draw["initial"], target=[0, 0, 1])
            after = {str(s): dict(raw=x.raw_state(), runtime=x.runtime()) for s, x in fixtures.items()}
            native = root / f"initial-reset-{i:02d}.mjb"
            f.mj.mj_saveModel(f.model, str(native), None)
            record = dict(
                index=i,
                draw=draw,
                before=before,
                after=after,
                actions=core.last_actions.tolist(),
                observation=core.observation.tolist(),
                native_model=dict(path=native.name, sha256=sha256(native)),
            )
            path = root / f"initial-reset-{i:02d}.json"
            write_json(path, record)
            records.append(dict(index=i, path=path.name, sha256=sha256(path)))
    finally:
        for core in cores.values():
            core.close()
    write_json(root / "records.json", records)
    return records


def verify_initial_resets(output, *, source_root=None):
    root = Path(output)
    source_root = Path.cwd() if source_root is None else Path(source_root)
    try:
        plan = json.loads((root / "plan.json").read_text())
        records = json.loads((root / "records.json").read_text())
        if (
            plan["schema"] != "solver-initial-reset30-v1"
            or plan["batch_size"] != 1
            or plan["source_binding"] != {p: sha256(source_root / p) for p in SOURCES}
            or len(records) != 30
            or [r["index"] for r in records] != list(range(30))
        ):
            raise ValueError("Actual initial-reset source/count differs")
        import mujoco as mj
        from .payload_retry_contract import actor_observation

        for receipt in records:
            if sha256(root / receipt["path"]) != receipt["sha256"]:
                raise ValueError("Initial-reset raw differs")
            row = json.loads((root / receipt["path"]).read_text())
            draw = plan["draws"][row["index"]]
            if row["draw"] != draw:
                raise ValueError("Initial-reset draw differs")
            key = str(draw["side_m"])
            before, after = row["before"][key], row["after"][key]
            if not np.any(before["raw"]["actual_applied_generalized_force"]):
                raise ValueError("Reset lacked a real preceding input")
            for other in row["before"]:
                if other != key and row["before"][other] != row["after"][other]:
                    raise ValueError("Unselected size scene changed")
            raw = after["raw"]
            for body in ["drone", "package"]:
                for field in ["xyz", "quaternion_wxyz", "velocity", "angular_velocity"]:
                    if not np.allclose(
                        raw[body + "_" + field], draw["initial"][body + "_" + field], atol=1e-9, rtol=0
                    ):
                        raise ValueError("Actual requested world initial state differs")
            if (
                raw["time_s"] != 0
                or np.any(raw["qvel"])
                or np.any(raw["actual_applied_generalized_force"])
                or np.any(raw["actual_xfrc_applied"])
                or np.any(raw["rpm"])
                or np.any(row["actions"])
            ):
                raise ValueError("Actual initial reset input/history not cleared")
            native = row["native_model"]
            if sha256(root / native["path"]) != native["sha256"]:
                raise ValueError("Native initial-reset bytes differ")
            model = mj.MjModel.from_binary_path(str(root / native["path"]))
            body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "payload")
            mass = FREE_MASS_KG * draw["mass_ratio"]
            if not np.isclose(model.body_mass[body], mass, atol=0, rtol=1e-12) or not np.allclose(
                model.body_inertia[body], mass * draw["side_m"] ** 2 / 6, atol=0, rtol=1e-12
            ):
                raise ValueError("Initial reset actual mass/inertia differs")
            expected_damping = np.zeros(model.nv)
            expected_damping[3:6] = (0.027 + 0.0001 + mass) * 1e-5
            if not np.allclose(model.dof_damping, expected_damping, rtol=1e-12, atol=1e-15):
                raise ValueError("Initial reset inherited authored-mass damping differs")
            sample = dict(raw)
            from .payload_contract import rotation

            for name, label, offset in [
                ("a", "drone", [0, 0, -0.02]),
                ("b", "package", [0, 0, draw["side_m"] / 2]),
            ]:
                arm = rotation(raw[label + "_quaternion_wxyz"]) @ offset
                sample["attachment_" + name + "_world"] = (np.array(raw[label + "_xyz"]) + arm).tolist()
                sample["attachment_" + name + "_velocity"] = (
                    np.array(raw[label + "_velocity"]) + np.cross(raw[label + "_angular_velocity"], arm)
                ).tolist()
            expected = actor_observation(sample, [0, 0, 1], [0.0] * 4)
            if len(row["observation"]) != 29 or not np.array_equal(row["observation"], expected):
                raise ValueError("Actual reset observation contract differs")
        return dict(passed=True, actual_initial_resets=30, batch_size=1, physical_training_admitted=False)
    except (OSError, KeyError, TypeError, IndexError) as exc:
        raise ValueError("Actual common-runtime initial reset30 evidence required") from exc
