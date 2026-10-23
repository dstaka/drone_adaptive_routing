"""Common actual-state boundary for RK4 learning, missions and capture.

This low-level diagnostic core does not authorize an optimizer. Production
consumers must independently reconstruct the physical admission receipt first.
"""

from collections import deque
import copy
import gzip
import json
from pathlib import Path

import numpy as np

from .payload_contract import SharedSettling, payload_metrics, rotation, settled
from .payload_retry_contract import actor_observation
from .payload_solver_mujoco import HOVER_RPM


class SolverRuntime:
    """Explicit episode resets; continuous state and history across retargets."""

    dt = 0.01

    def __init__(self, fixture, output):
        if fixture.fixed or not fixture.loaded or fixture.dt != self.dt:
            raise ValueError("Actual free loaded fixture required")
        self.fixture = self.bank = fixture
        self.payload = fixture.payload
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.trace = gzip.open(self.output / "raw.jsonl.gz", "wt")
        self.recent = deque(maxlen=128)
        self.episode_id = 0
        self.step_count = 0
        self.target = np.array([0.0, 0.0, 1.0])
        self.target_id = None
        self.last_actions = np.zeros(4)
        self.observation = None
        self.current_sample = None
        self.last_row = None
        self.quarantined = False
        self.dwell = SharedSettling(self.dt)

    def close(self):
        self.trace.close()

    def _require_active(self):
        if self.quarantined:
            raise RuntimeError("Solver runtime is quarantined")

    def _capture(self, kind, **extra):
        raw = self.fixture.raw_state()
        row = dict(
            kind=kind,
            episode_id=self.episode_id,
            step=self.step_count,
            target=self.target.tolist(),
            target_id=self.target_id,
            actions=self.last_actions.tolist(),
            actual=raw,
            **extra,
        )
        self.trace.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
        self.trace.flush()
        self.recent.append(copy.deepcopy(row))
        return raw

    def _abort(self, exc):
        self.quarantined = True
        try:
            actual = self.fixture.raw_state()
        except Exception as getter_error:
            actual = dict(
                capture_error=repr(getter_error),
                qpos=self.fixture.data.qpos.tolist(),
                qvel=self.fixture.data.qvel.tolist(),
            )
        with (self.output / "abort.json").open("w") as stream:
            json.dump(
                dict(
                    error=repr(exc),
                    actual=actual,
                    preceding=list(self.recent),
                    step_completion="UNKNOWN",
                    last_confirmed_step=self.step_count,
                ),
                stream,
                allow_nan=True,
            )
        self.trace.flush()
        raise RuntimeError("Solver runtime quarantined; actual raw retained") from exc

    @staticmethod
    def _guard(raw):
        for body in ("drone", "rod", "package"):
            for field, size in [("xyz", 3), ("quaternion_wxyz", 4), ("velocity", 3), ("angular_velocity", 3)]:
                v = np.asarray(raw[body + "_" + field])
                if v.shape != (size,) or not np.isfinite(v).all():
                    raise ValueError("INVALID_ACTUAL_STATE")
                if field == "quaternion_wxyz":
                    rotation(v)
                if field == "velocity" and np.linalg.norm(v) > 20:
                    raise ValueError("FINITE_SPEED_EXCURSION")
                if field == "angular_velocity" and np.linalg.norm(v) > 200:
                    raise ValueError("FINITE_SPIN_EXCURSION")
        if any(raw["warning_counts"]):
            raise ValueError("BACKEND_WARNING_OR_INTERNAL_RESET")
        if not np.isfinite(raw["qpos"]).all() or not np.isfinite(raw["qvel"]).all():
            raise ValueError("INVALID_GENERALIZED_STATE")

    def _derive(self):
        sample = self.fixture.sample()
        # Exact support of the oriented original cylinder, not body-center z.
        r = rotation(sample["drone_quaternion_wxyz"])
        p = np.asarray(sample["drone_xyz"])
        extent = np.array(
            [0.06 * np.sqrt(max(0.0, 1 - r[i, 2] ** 2)) + 0.0125 * abs(r[i, 2]) for i in range(3)]
        )
        sample.update(
            vehicle_collision_aabbs={"vehicle_collision": [(p - extent).tolist(), (p + extent).tolist()]},
            contacts=bool(sample["contact"]),
            done=False,
            reset=False,
        )
        self.current_sample = sample
        self.observation = actor_observation(sample, self.target, self.last_actions)
        return sample

    def reset_case(
        self,
        *,
        swing_deg=0.0,
        direction_rad=0.0,
        position=(0.0, 0.0, 1.0),
        target=None,
        target_id=None,
        initial=None,
    ):
        self._require_active()
        try:
            if initial is None:
                self.fixture.reset(swing_deg=swing_deg, direction_rad=direction_rad, position=position)
            else:
                from .payload_ball_bank import ball_joint_configuration

                upper, lower = ball_joint_configuration(initial, self.payload.side_m)
                position = initial["drone_xyz"]
                self.fixture.reset(position=position, quaternion=initial["drone_quaternion_wxyz"])
                for name, values in [("suspension_ball", upper), ("payload_ball", lower)]:
                    joint = self.fixture.mj.mj_name2id(
                        self.fixture.model, self.fixture.mj.mjtObj.mjOBJ_JOINT, name
                    )
                    start = self.fixture.model.jnt_qposadr[joint]
                    self.fixture.data.qpos[start : start + 4] = values
                self.fixture.mj.mj_forward(self.fixture.model, self.fixture.data)
            self.episode_id += 1
            self.step_count = 0
            self.last_actions = np.zeros(4)
            self.target = np.asarray(position if target is None else target, dtype=float)
            self.target_id = target_id
            self.dwell = SharedSettling(self.dt)
            raw = self._capture("RESET")
            self._guard(raw)
            sample = self._derive()
            metrics = payload_metrics(sample, self.target, self.payload)
            metrics["drone_floor_m"] = sample["vehicle_collision_aabbs"]["vehicle_collision"][0][2]
            self.dwell.update(0.0, bool(settled(metrics) and not sample["contacts"]), target_id)
            self.last_row = None
            return self.observation.copy()
        except Exception as exc:
            self._abort(exc)

    def set_target(self, target, target_id=None):
        self._require_active()
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Finite target required")
        self.target = target.copy()
        self.target_id = target_id
        self.dwell = SharedSettling(self.dt)
        self.dwell.update(self.step_count * self.dt, False, target_id)
        self._capture("RETARGET")
        self._derive()
        return self.observation.copy()

    def step(self, actions):
        self._require_active()
        try:
            values = np.asarray(actions, dtype=float)
            if values.shape != (4,) or not np.isfinite(values).all():
                raise ValueError("Finite motor actions required")
            self.last_actions = np.clip(values, -1, 1)
            rpm = (1 + 0.8 * self.last_actions) * HOVER_RPM
            before = self.observation.copy()
            self.fixture.step_rpm(rpm)
            self.step_count += 1
            raw = self._capture("STEP", actor_observation_before_step=before.tolist())
            self._guard(raw)
            if not np.isclose(raw["time_s"], self.step_count * self.dt, atol=1e-7, rtol=0):
                raise ValueError("INTERNAL_RESET_OR_TIME_DISCONTINUITY")
            sample = self._derive()
            metrics = payload_metrics(sample, self.target, self.payload)
            metrics["drone_floor_m"] = sample["vehicle_collision_aabbs"]["vehicle_collision"][0][2]
            unsafe = bool(
                sample["contacts"]
                or metrics["drone_floor_m"] <= 0.1
                or metrics["package_floor_m"] <= 0.1
                or abs(metrics["drone_roll_deg"]) > 60
                or abs(metrics["drone_pitch_deg"]) > 60
                or metrics["constraint_residual_m"] > 2e-5
            )
            complete = self.dwell.update(
                self.step_count * self.dt, bool(settled(metrics) and not unsafe), self.target_id
            )
            self.last_row = copy.deepcopy(sample)
            self.last_row.update(
                step=self.step_count,
                time_s=self.step_count * self.dt,
                effective_target=self.target.tolist(),
                actions=self.last_actions.tolist(),
                metrics=metrics,
                unsafe=unsafe,
                complete=complete,
                dwell_s=self.dwell.dwell_s,
                actor_observation=self.observation.tolist(),
            )
            return copy.deepcopy(self.last_row)
        except Exception as exc:
            self._abort(exc)
