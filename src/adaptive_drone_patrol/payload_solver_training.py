"""Actual N1 suspended PPO environment using the common qualified runtime.

No optimizer or physical admission is created here. The training entry must
independently reconstruct all62/motor/reset/raw bindings before construction.
"""

import gzip
import json
import math
from pathlib import Path

import numpy as np

from .jobs import write_json
from .payload_cases import initial_state
from .payload_contract import Payload, PayloadSampler, settled, training_outside_envelope
from .payload_retry_contract import OBSERVATION_CONTRACT
from .payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG
from .payload_solver_runtime import SolverRuntime


REWARD_CONTRACT = dict(
    id="loaded-shared-settling-dense-position-v5",
    running_dt_s=0.01,
    weights=dict(
        progress_squared_distance=2.0,
        hover=0.5,
        position_error_squared=-2.0,
        vehicle_speed_squared=-0.03,
        altitude_error_squared=-0.5,
        vehicle_angular_speed_squared=-0.1,
        swing_radians_squared=-2.0,
        swing_rate_radians_squared=-0.1,
        package_speed_squared=-0.5,
        package_spin_radians_squared=-0.1,
        simultaneous_shared_valid=2.0,
    ),
    hover_exponents=dict(position_denominator_m2=0.04, speed_denominator_m2_s2=0.09),
    package_spin_cost_cap_rad_s=1.0,
    terminal_failure=-200.0,
    terminal_integration="Discrete failure penalty; running reward multiplied by physical dt",
)
CURRICULUM_CONTRACT = dict(
    stages=[
        "hover",
        "short_xyz",
        "medium_xyz",
        "full_xyz",
        "corner_reversal_ascent_descent",
        "dispersed_ten_targets",
    ],
    stage_allocation="Six equal valid-transition stages, retain stage of current actual target at boundaries",
    required_exposure=dict(
        min_valid_transitions_per_stage=10000,
        min_continuous_episode_steps_per_stage=500,
        min_full_stage_continuous_ten_target_completions=1,
    ),
    target_timeout_steps=800,
    episode_timeout_steps=12000,
    episode_complete="Ten actual shared completions, excluding target timeouts",
    retarget="Preserve physical bodies/joints/velocities/applied action history; restart dwell only",
    distribution="Independent full mass uniform.10-.30 and equiprobable real three sizes in every randomized episode",
)


def validate_learning_steps(steps, n_envs, rollout_steps):
    if any(type(v) is not int or v <= 0 for v in (steps, n_envs, rollout_steps)) or steps % (
        n_envs * rollout_steps
    ):
        raise ValueError("S must contain complete equal N*T rollouts; no method-specific rounding")
    return steps // (n_envs * rollout_steps)


def sample_ladder(n_envs, rollout_steps):
    rungs = [300000, 1000000, 3000000]
    for s in rungs:
        validate_learning_steps(s, n_envs, rollout_steps)
    return rungs


def learning_ppo_config(parent, *, gae_lambda=0.95, discount=None):
    """Freeze a common PPO recipe with explicit development credit parameters."""
    import copy

    if (isinstance(gae_lambda, bool) or not isinstance(gae_lambda, (float, int))
            or not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1):
        raise ValueError("GAE lambda must be finite within0..1")
    result = copy.deepcopy(parent)
    result.update(learning_rate=3e-5, schedule="fixed", entropy_coef=0.0,
                  max_grad_norm=0.3, lam=float(gae_lambda))
    if discount is not None:
        if (isinstance(discount,bool) or not isinstance(discount,(float,int))
                or not math.isfinite(discount) or not 0 <= discount < 1):
            raise ValueError("Discount must be finite within0..1 exclusive")
        result["gamma"] = float(discount)
    return result


def reward_components(before, current, target, angular_velocity, metrics, failed, shared_valid):
    before, current, target = np.asarray(before), np.asarray(current), np.asarray(target)
    return dict(
        progress=2 * (np.linalg.norm(before - target) ** 2 - np.linalg.norm(current - target) ** 2),
        hover=0.5
        * math.exp(-(metrics["drone_error_m"] ** 2) / 0.04)
        * math.exp(-(metrics["drone_speed_m_s"] ** 2) / 0.09),
        position=-2.0 * float(np.linalg.norm(current - target)) ** 2,
        velocity=-0.03 * metrics["drone_speed_m_s"] ** 2,
        altitude=-0.5 * (current[2] - target[2]) ** 2,
        angular=-0.1 * float(np.linalg.norm(angular_velocity)) ** 2,
        swing=-2.0 * math.radians(metrics["swing_angle_deg"]) ** 2,
        swing_rate=-0.1 * math.radians(metrics["swing_rate_deg_s"]) ** 2,
        package_velocity=-0.5 * metrics["package_speed_m_s"] ** 2,
        package_spin=-0.1 * min(math.radians(metrics["package_spin_deg_s"]) ** 2, 1.0),
        shared_settling=2.0 * bool(shared_valid),
        crash=-200.0 * bool(failed),
    )


def transition_reward(terms):
    """Integrate running costs; a physical terminal penalty is a discrete event."""
    return 0.01 * sum(v for key, v in terms.items() if key != "crash") + terms["crash"]


class SolverTrainingEnv:
    """One active logical environment, three actual cube geometry scenes."""

    def __init__(self, output, original, *, randomized, seed, curriculum_steps=300000, pilot=False):
        if type(randomized) is not bool or type(seed) is not int or seed < 0:
            raise ValueError("Explicit learning method and independent seed required")
        import torch

        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.num_envs, self.num_actions, self.device = 1, 4, "cpu"
        self.cfg = dict(
            profile="cf2x-mujoco-rk4-f64-s16-v2",
            num_actions=4,
            observation_size=29,
            dt=0.01,
            observation_contract=OBSERVATION_CONTRACT,
            randomized_payload=randomized,
            source="Actual common runtime, current immediate clipped motor action; no hidden load actor fields",
        )
        self.max_episode_length = 12000
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self.fixtures = {
            side: MuJoCoFixture(
                original, self.output / f"scene-{side:.3f}", Payload(FREE_MASS_KG, 0.2, side), substeps=16
            )
            for side in [0.03, 0.045, 0.06]
        }
        self.cores = {
            side: SolverRuntime(f, self.output / f"runtime-{side:.3f}") for side, f in self.fixtures.items()
        }
        self.sampler = PayloadSampler(FREE_MASS_KG, seed, "training", randomized=randomized)
        self.pose_rng = np.random.default_rng(np.random.SeedSequence([seed, 710, 101]))
        self.command_rng = np.random.default_rng(np.random.SeedSequence([seed, 710, 102]))
        self.curriculum_steps, self.pilot = curriculum_steps, pilot
        self.total_steps = self.generated_steps = self.invalid_steps = self.episode_count = self.failures = (
            self.completed_stops
        ) = 0
        self.reward_totals = {}
        self.stage_counts = [0] * 6
        self.stage_experience = []
        self.trace = gzip.open(self.output / "training.jsonl.gz", "wt")
        self.terminal = gzip.open(self.output / "terminal.jsonl.gz", "wt")
        self.reset_trace = gzip.open(self.output / "reset-properties.jsonl.gz", "wt")
        import yaml

        layout = yaml.safe_load(Path("configs/demo_scenarios_dispersed/normal.yaml").read_text())
        self.destinations = np.asarray(layout["coords"][1:], dtype=float)
        self._reset()

    @property
    def stage(self):
        return 0 if self.pilot else min(5, self.total_steps * 6 // self.curriculum_steps)

    def _target(self):
        stage = self.stage
        if self.pilot:
            return np.array([[0.0, 0.0, 1.0], [0.15, 0.0, 1.0], [0.0, 0.0, 1.0]][self.command_count % 3])
        if stage == 0:
            return np.array([0.0, 0.0, 1.0])
        if stage == 4:
            return np.array(
                [[-2.7, -1.8, 1.0], [2.7, 1.8, 3.0], [2.7, -1.8, 1.0], [-2.7, 1.8, 3.0]][
                    self.command_count % 4
                ]
            )
        if stage == 5:
            return self.destinations[self.target_order[self.command_count % 10]].copy()
        high = np.array([(0, 0, 1), (0.5, 0.5, 1.5), (1.5, 1.2, 2.0), (2.7, 1.8, 3.0)][stage])
        return self.command_rng.uniform([-high[0], -high[1], 1.0], high)

    def _observation(self):
        import torch
        from tensordict import TensorDict

        return TensorDict(
            {"policy": torch.as_tensor(self.core.observation, dtype=torch.float32)[None]}, batch_size=[1]
        )

    def _reset(self):
        self.payload = self.sampler.reset([0])[0]
        self.core = self.cores[self.payload.side_m]
        fixture = self.fixtures[self.payload.side_m]
        fixture.apply_payload(self.payload)
        self.core.payload = self.payload
        self.initial = initial_state(self.pose_rng, self.payload.side_m)
        self.episode_count += 1
        self.episode_steps = self.command_steps = self.command_count = self.episode_completed_stops = 0
        self.target_order = self.command_rng.permutation(10)
        self.target = self._target()
        self.target_stage = self.stage
        self.episode_stage = self.stage
        self.core.reset_case(initial=self.initial, target=self.target, target_id="0")
        runtime = fixture.runtime()
        if not np.isclose(runtime["package_mass_kg"], self.payload.mass, atol=0, rtol=1e-12):
            raise ValueError("Actual training reset mass differs")
        idx = [x["name"] for x in runtime["links"]].index("payload")
        if not np.allclose(
            runtime["inertia_kg_m2"][idx], np.diag(self.payload.inertia), atol=1e-15, rtol=1e-12
        ):
            raise ValueError("Actual training reset inertia differs")
        self.reset_trace.write(
            json.dumps(
                dict(
                    episode=self.episode_count,
                    draw_index=self.sampler.draw_indices[0],
                    payload=vars(self.payload),
                    initial=self.initial,
                    actual=fixture.raw_state(),
                    runtime=runtime,
                ),
                separators=(",", ":"),
            )
            + "\n"
        )
        self.reset_trace.flush()
        self.episode_length_buf[:] = 0
        self.obs = self._observation()

    def get_observations(self):
        return self.obs

    def close(self):
        for core in self.cores.values():
            core.close()
        self.trace.close()
        self.terminal.close()
        self.reset_trace.close()

    def counters(self):
        return dict(
            generated=self.generated_steps,
            valid=self.total_steps,
            invalid=self.invalid_steps,
            episodes=self.episode_count,
            failed_episodes=self.failures,
            completed_training_stops=self.completed_stops,
            stage_valid_transitions=self.stage_counts,
            stage_episode_experience=self.stage_experience,
        )

    def step(self, actions):
        import torch

        raw_action = actions[0].detach().cpu().numpy()
        before = np.asarray(self.core.current_sample["drone_xyz"])
        self.generated_steps += 1
        self.trace.write(
            json.dumps(
                dict(
                    kind="ACTION_INPUT",
                    generated_step=self.generated_steps,
                    episode=self.episode_count,
                    raw_action=raw_action.tolist(),
                    actor_observation=self.core.observation.tolist(),
                ),
                separators=(",", ":"),
                allow_nan=True,
            )
            + "\n"
        )
        try:
            row = self.core.step(raw_action)
        except BaseException:
            self.invalid_steps += 1
            self.trace.flush()
            write_json(self.output / "aborted-counters.json", self.counters())
            raise
        self.total_steps += 1
        self.episode_steps += 1
        self.command_steps += 1
        stage = self.target_stage
        self.stage_counts[stage] += 1
        self.episode_length_buf[:] = self.episode_steps
        metrics = row["metrics"]
        failed = bool(row["unsafe"] or training_outside_envelope(row, self.target))
        terms = reward_components(
            before,
            row["drone_xyz"],
            self.target,
            row["drone_angular_velocity"],
            metrics,
            failed,
            bool(settled(metrics) and not failed),
        )
        reward = transition_reward(terms)
        if not math.isfinite(reward) or abs(reward) > 1e6:
            raise ValueError("Invalid reward; physical sample retained before abort")
        for key, value in terms.items():
            self.reward_totals[key] = self.reward_totals.get(key, 0) + (
                1.0 if key == "crash" else 0.01
            ) * float(value)
        complete = bool(row["complete"] and not failed)
        self.completed_stops += int(complete)
        self.episode_completed_stops += int(complete)
        timeout = self.episode_steps >= self.max_episode_length
        goal_complete = bool(complete and self.episode_completed_stops >= 10)
        done = bool(failed or timeout or goal_complete)
        self.trace.write(
            json.dumps(
                dict(
                    kind="VALID_TRANSITION",
                    valid_step=self.total_steps,
                    episode=self.episode_count,
                    episode_step=self.episode_steps,
                    stage=stage,
                    target=self.target.tolist(),
                    reward=reward,
                    reward_components=terms,
                    unsafe=failed,
                    done=done,
                    shared_complete=complete,
                    actual=row,
                ),
                separators=(",", ":"),
            )
            + "\n"
        )
        self.trace.flush()
        self.obs = self._observation()
        extras = {"time_outs": torch.tensor([timeout and not failed and not goal_complete], dtype=torch.bool)}
        if done:
            self.terminal.write(
                json.dumps(
                    dict(
                        valid_step=self.total_steps,
                        episode=self.episode_count,
                        episode_steps=self.episode_steps,
                        stage=stage,
                        command_count=self.command_count,
                        target=self.target.tolist(),
                        payload=vars(self.payload),
                        sample=row,
                        failed=failed,
                        timeout=timeout,
                    ),
                    separators=(",", ":"),
                )
                + "\n"
            )
            self.terminal.flush()
            self.stage_experience.append(
                dict(
                    episode=self.episode_count,
                    start_stage=self.episode_stage,
                    end_stage=stage,
                    continuous_steps=self.episode_steps,
                    completed_target_count=self.episode_completed_stops,
                    attempted_target_count=self.command_count + 1,
                    failed=failed,
                )
            )
            self.failures += int(failed)
            extras["episode"] = {
                "failure": float(failed),
                "completed_stops": float(self.episode_completed_stops),
            }
            self._reset()
        elif complete or self.command_steps >= 800:
            self.command_count += 1
            self.command_steps = 0
            self.target = self._target()
            self.target_stage = self.stage
            self.core.set_target(self.target, target_id=str(self.command_count))
            self.obs = self._observation()
        return (
            self.obs,
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.bool),
            extras,
        )


def require_learning_dependencies(lock):
    """Bind exact installed learning closure and actual RSL/Torch source/build."""
    import importlib.metadata as metadata
    import re
    from .util import sha256

    pins = [
        line.split("==", 1)
        for line in Path(lock).read_text().splitlines()
        if line and not line.startswith(("#", "--"))
    ]
    if not pins or any(len(pair) != 2 for pair in pins):
        raise ValueError("Exact complete learning dependency pins required")
    expected = {re.sub("[-_.]+", "-", name).lower(): version for name, version in pins}
    installed = {
        re.sub("[-_.]+", "-", dist.metadata["Name"]).lower(): dist.version
        for dist in metadata.distributions()
    }
    if any(installed.get(name) != version for name, version in expected.items()):
        raise ValueError("Actual learning dependency differs from frozen lock")
    rsl = metadata.distribution("rsl-rl-lib")
    torch = metadata.distribution("torch")
    source = {
        str(p): sha256(rsl.locate_file(p))
        for p in rsl.files
        if str(p).endswith(".py") and rsl.locate_file(p).is_file()
    }
    build = {
        str(p): sha256(torch.locate_file(p))
        for p in torch.files
        if str(p).startswith(("torch/_C.", "torch/lib/"))
        and str(p).endswith(".so")
        or str(p) == "torch/version.py"
    }
    torch_source = {
        str(p): sha256(torch.locate_file(p))
        for p in torch.files
        if str(p).startswith("torch/") and str(p).endswith(".py")
        and torch.locate_file(p).is_file()
    }
    if not source or not build or "torch/optim/adam.py" not in torch_source:
        raise ValueError("Actual learning source/build missing")
    return dict(
        lock_sha256=sha256(lock), installed_pins=expected, rsl_source_sha256=source,
        torch_build_sha256=build, torch_source_sha256=torch_source
    )
