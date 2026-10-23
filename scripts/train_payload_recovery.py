#!/usr/bin/env python
"""Matched RK4 PPO learning; full raw admission precedes every optimizer.

Development checkpoints can never initialize either final method. Every final
run requires one common frozen plan and starts from the verified model4296
transfer with a fresh optimizer. No cumulative GPU budget helper is imported.
"""

import argparse
import copy
import json
from pathlib import Path
import pickle
import time

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_checkpoint import validate_parent_config, actor_signature
from adaptive_drone_patrol.payload_retry_checkpoint import transfer_state, initialization_comparison
from adaptive_drone_patrol.payload_retry_contract import OBSERVATION_CONTRACT
from adaptive_drone_patrol.payload_solver_qualification import current_solver_binding, require_qualification
from adaptive_drone_patrol.payload_solver_training import (
    validate_learning_steps,
    require_learning_dependencies,
    REWARD_CONTRACT,
    CURRICULUM_CONTRACT,
    learning_ppo_config,
)
from adaptive_drone_patrol.util import sha256

PARENT_SHA = "26658a31560f71c36fd9e9605b15d8e4f7b9cae8174c1465fd12dcd0d10b3431"
CONFIG_SHA = "35c6b0be8db90e4e799d453d447a484275317f85098a22cbdf04c19c84056e55"
SOURCES = (
    "scripts/train_payload_recovery.py",
    "src/adaptive_drone_patrol/payload_solver_training.py",
    "src/adaptive_drone_patrol/payload_solver_runtime.py",
    "src/adaptive_drone_patrol/payload_solver_mujoco.py",
    "src/adaptive_drone_patrol/payload_retry_contract.py",
    "src/adaptive_drone_patrol/payload_retry_checkpoint.py",
    "src/adaptive_drone_patrol/payload_solver_qualification.py",
    "src/adaptive_drone_patrol/payload_solver_work_check.py",
    "src/adaptive_drone_patrol/payload_solver_exposure.py",
    "src/adaptive_drone_patrol/payload_solver_training_plan.py",
    "src/adaptive_drone_patrol/payload_solver_validation.py",
    "src/adaptive_drone_patrol/payload_solver_trace.py",
    "src/adaptive_drone_patrol/payload_solver_policy.py",
    "src/adaptive_drone_patrol/payload_solver_models.py",
    "src/adaptive_drone_patrol/payload_cases.py",
    "src/adaptive_drone_patrol/payload_contract.py",
    "configs/demo_scenarios_dispersed/normal.yaml",
    "requirements/payload-solver-flight-linux.lock.txt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["qualification", "locations", "parent", "config", "original", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--method", choices=["randomized", "fixed"], required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--curriculum-steps", type=int, default=300000)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve existing training attempts")
    iterations = validate_learning_steps(args.steps, 1, 100)
    checkpoints = sorted(set([int(x) for x in args.checkpoints.split(",") if x] + [args.steps]))
    if any(s > args.steps or s <= 0 or s % 100 for s in checkpoints):
        raise ValueError("Common checkpoint S must be a complete100-step rollout")
    if args.pilot and not args.development:
        raise ValueError("Pilots are development only")
    seed = args.seed if args.seed is not None else (71 if args.development else 7)
    if not args.development and (seed != 7 or args.plan is None):
        raise ValueError("Final seed7 and common frozen training plan required")
    # Missing qualification fails before importing Torch, constructing an engine,
    # deserializing model pickle, creating output, or creating any optimizer.
    qualification = json.loads(args.qualification.read_text())
    locations = json.loads(args.locations.read_text())
    binding = current_solver_binding()
    verified = require_qualification(qualification, binding, locations)
    if sha256(args.parent) != PARENT_SHA or sha256(args.config) != CONFIG_SHA:
        raise ValueError("Only the actual adopted model4296 and configuration can initialize recovery")
    if sha256(args.original) != binding["original_asset_sha256"]:
        raise ValueError("Actual original vehicle asset differs")
    with args.config.open("rb") as stream:
        configs = pickle.load(stream)
    validate_parent_config(configs)
    train = copy.deepcopy(configs[4])
    train.update(num_steps_per_env=100, save_interval=10**12, run_name=args.method)
    train["algorithm"] = learning_ppo_config(train["algorithm"], gae_lambda=args.gae_lambda,
                                            discount=args.discount)
    source = {p: sha256(p) for p in SOURCES}
    common = dict(
        seed=seed,
        N=1,
        T=100,
        S=args.steps,
        K=iterations,
        architecture={k: train[k] for k in ["actor", "critic", "obs_groups"]},
        optimizer=train["algorithm"],
        curriculum_steps=args.curriculum_steps,
        reward=REWARD_CONTRACT,
        curriculum=CURRICULUM_CONTRACT,
        pilot=args.pilot,
        observation_contract=OBSERVATION_CONTRACT,
        source_binding=source,
        parent_sha256=PARENT_SHA,
        parent_config_sha256=CONFIG_SHA,
        actual_binding=binding,
        learning_dependencies=require_learning_dependencies(
            "requirements/payload-solver-flight-linux.lock.txt"
        ),
        development_evaluation=dict(
            pilot_seed=83011,
            development_seeds=[83021, 83022],
            rng_key=[720],
            loads=[
                dict(side_m=0.03, mass_ratio=0.13),
                dict(side_m=0.045, mass_ratio=0.2),
                dict(side_m=0.06, mass_ratio=0.27),
            ],
            declaration_sha256=sha256("configs/payload_validation.yaml"),
            pilot_targets=[[0.0, 0.0, 1.0], [0.15, 0.0, 1.0], [0.0, 0.0, 1.0]],
            development_targets="Unchanged ten targets from the hashed declaration; independent development seeds only",
            ladder=[300000, 1000000, 3000000],
            selection_rule="Compare both methods at every common rung; choose the smallest common S with randomized development6/6 complete actual ten-target sequences and required actual exposure for both methods. Preserve fixed comparison as observed. If none qualify, diagnose and predeclare a common next rung or new development recipe before running; no held-out tuning.",
        ),
        final_checkpoint_rule="Use only the common declared final S; no method-specific sample extension",
    )
    if not args.development:
        frozen = json.loads(args.plan.read_text())
        from adaptive_drone_patrol.payload_solver_training_plan import require_final_plan

        require_final_plan(frozen)
        if frozen["common"] != common:
            raise ValueError("Final learning differs from common frozen plan")
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from adaptive_drone_patrol.payload_solver_training import SolverTrainingEnv

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    initial = transfer_state(parent)
    comparison = initialization_comparison(parent, initial)
    if not args.development:
        if (
            frozen["initial_actor_tensor_sha256"] != comparison["initial_actor_tensor_sha256"]
            or frozen["initial_critic_tensor_sha256"] != comparison["initial_critic_tensor_sha256"]
        ):
            raise ValueError("Initial tensors differ from the common frozen final plan")
    initial["infos"].pop("physics_signature", None)
    initial["infos"].update(actual_solver_binding=binding, training_split="initialization-only")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(initial, args.out / "initialization.pt")
    manifest = dict(
        schema="payload-solver-learning-v1",
        status="DECLARED_BEFORE_OPTIMIZER",
        method=args.method,
        split="development" if args.development else "final",
        common=common,
        checkpoints=checkpoints,
        fresh_optimizer=True,
        initial_iteration=0,
        initialization_comparison=comparison,
        initialization_sha256=sha256(args.out / "initialization.pt"),
        qualification_sha256=sha256(args.qualification),
        qualified_raw_manifest_sha256=verified["raw_manifest_sha256"],
        heldout_opened=False,
        sampling="Full uniform independent mass.10-.30 and equiprobable three sizes in every randomized episode; nominal fixed otherwise",
        total_transition_identity="S=N*T*K; generated/valid/invalid/consumed and optimizer gradient steps counted separately",
    )
    write_json(args.out / "manifest.json", manifest)
    env = SolverTrainingEnv(
        args.out / "environment",
        args.original,
        randomized=args.method == "randomized",
        seed=seed,
        curriculum_steps=args.curriculum_steps,
        pilot=args.pilot,
    )
    runner = OnPolicyRunner(env, train, str(args.out / "rsl"), device="cpu")
    runner.load(
        str(args.out / "initialization.pt"),
        load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False),
    )
    actual = actor_signature(runner.get_inference_policy(device="cpu").state_dict())
    actual_critic = actor_signature(runner.alg.critic.state_dict())
    if (
        actual != comparison["initial_actor_tensor_sha256"]
        or actual_critic != comparison["initial_critic_tensor_sha256"]
        or runner.current_learning_iteration != 0
        or runner.alg.optimizer.state_dict()["state"]
    ):
        raise ValueError("Actual parent transfer or fresh optimizer differs")
    counts = dict(rollout_updates=0, optimizer_gradient_steps=0, consumed_transitions=0)
    original_optimizer_step = runner.alg.optimizer.step

    def optimizer_step(*a, **kw):
        result = original_optimizer_step(*a, **kw)
        counts["optimizer_gradient_steps"] += 1
        return result

    runner.alg.optimizer.step = optimizer_step
    original_update = runner.alg.update

    def update():
        physical_before = env.core.fixture.raw_state()
        history_before = env.core.last_actions.copy()
        result = original_update()
        counts["rollout_updates"] += 1
        counts["consumed_transitions"] += 100
        if (
            env.core.fixture.raw_state() != physical_before
            or not (env.core.last_actions == history_before).all()
        ):
            raise ValueError("Optimizer/rollout boundary changed actual physical state or history")
        if env.total_steps in checkpoints:
            if any(sha256(p) != h for p, h in source.items()):
                raise ValueError("Frozen learning source changed")
            if require_learning_dependencies("requirements/payload-solver-flight-linux.lock.txt") != common["learning_dependencies"]:
                raise ValueError("Actual learning Python source/build changed during optimization")
            path = args.out / f"checkpoint_s{env.total_steps}.pt"
            if path.exists():
                raise FileExistsError("Checkpoint samples must be unique")
            infos = dict(
                observation_contract=OBSERVATION_CONTRACT,
                actual_solver_binding=binding,
                training_split=manifest["split"],
                method=args.method,
                valid_environment_transitions=env.total_steps,
                completed_rollout_updates=counts["rollout_updates"],
                common_training_plan=common,
                initial_actor_tensor_sha256=comparison["initial_actor_tensor_sha256"],
                initial_critic_tensor_sha256=comparison["initial_critic_tensor_sha256"],
            )
            runner.save(str(path), infos=infos)
            write_json(
                path.with_suffix(".json"),
                dict(
                    checkpoint_sha256=sha256(path),
                    valid_transitions=env.total_steps,
                    counters=env.counters(),
                    optimizer=counts.copy(),
                    rollout_boundaries_preserve_state_history=True,
                    status="DEVELOPMENT_NOT_ADOPTED" if args.development else "FINAL_TRAINED_NOT_VALIDATED",
                ),
            )
        return result

    runner.alg.update = update
    write_json(
        args.out / "started.json",
        dict(
            status="QUALIFIED_OPTIMIZER_STARTED",
            initial_actor_tensor_sha256=actual,
            fresh_optimizer=True,
            common=common,
        ),
    )
    try:
        learning_started = time.monotonic()
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
        if (
            env.total_steps != args.steps
            or counts["consumed_transitions"] != args.steps
            or env.invalid_steps
            or env.generated_steps != env.total_steps
        ):
            raise ValueError("Actual sample accounting differs from common S")
        write_json(
            args.out / "completed.json",
            dict(
                status="TRAINED_NOT_VALIDATED",
                counters=env.counters(),
                optimizer=counts,
                reward_components=env.reward_totals,
                effective_valid_transitions_per_second=env.total_steps
                / (time.monotonic() - learning_started),
                heldout_opened=False,
                final_checkpoint=f"checkpoint_s{args.steps}.pt",
            ),
        )
    except BaseException as exc:
        write_json(
            args.out / "failure.json",
            dict(
                status="FAILED_ATTEMPT_PRESERVED", error=repr(exc), counters=env.counters(), optimizer=counts
            ),
        )
        raise
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
