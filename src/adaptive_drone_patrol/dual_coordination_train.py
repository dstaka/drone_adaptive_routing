"""Finite qualified three-seed PPO training with complete boundary recovery.

No production worlds start without a current file-bound learning capability,
the selected measured native backend/worker count and exact frozen inputs.
Each seed consumes4096x256 joint transitions across64/176/16 updates. The only
selectable artifact is its final terminal checkpoint. Atomic current recovery
contains partial rollout, all native workers/auditors/samplers, optimizer and
RNG state; a failed optimizer transaction restarts from its saved full rollout.
Training retains current recovery, curves/case assignments and bounded failure
receipts. Full replay retention is disabled only for admitted training worlds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import copy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
import multiprocessing as mp
import random
import time
import traceback
import uuid

import numpy as np
import torch

from .dual_coordination_benchmark import ProcessWorlds
from .dual_coordination_policy import (
    JointActorCritic,
    JointPPO,
    PPOBatch,
    PolicyConfig,
    TrainingBudget,
    generalized_advantage,
    action_permutation,
    permute_observation,
)
from .dual_coordination_schema import SCENARIOS, validate_case


from .util import sha256


class ProductionUnavailable(RuntimeError):
    pass


def production_plan(backend):
    if backend not in ("native-cpu", "native-cuda", "warp-cuda"):
        raise ValueError("Declared qualified execution backend required")
    return [
        dict(
            seed=seed,
            total_transitions=1048576,
            total_updates=256,
            selectable_checkpoint="terminal_only",
            stages=[
                dict(
                    stage=stage,
                    transitions=count,
                    updates=count // 4096,
                    physics_backend="native-mujoco" if stage == 3 or backend != "warp-cuda" else "warp",
                    policy_device="cuda" if backend.endswith("cuda") else "cpu",
                )
                for stage, count in enumerate((262144, 720896, 65536), 1)
            ],
        )
        for seed in (42, 43, 44)
    ]


# These primitives also support small nonphysical software fixtures in tests;
# only start_production can create a production campaign receipt.


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class CaseSampler:
    """Uniform geometry sampling; stage2/3 use balanced shuffled scenario bags."""

    def __init__(self, cases, *, seed, worker, stage):
        if (
            seed not in (42, 43, 44)
            or stage not in (1, 2, 3)
            or type(worker) is not int
            or worker < 0
            or not cases
        ):
            raise ValueError("Finite production sampler identity required")
        for index, row in enumerate(cases):
            if set(row) != set(SCENARIOS):
                raise ValueError("All three prospective scenario variants required")
            for scenario, case in row.items():
                validate_case(case)
                if case["split"] != 10 or case["index"] != index or case["scenario"] != scenario:
                    raise ValueError("Ordered split10 training cases required")
        self.cases = cases
        self.identity = dict(seed=seed, worker=worker, stage=stage, manifest_sha256=_digest(cases))
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, worker, stage, 10272711]))
        self.bag = []
        self.draws = 0

    def next(self):
        if not self.bag:
            self.bag = ["normal"] if self.identity["stage"] == 1 else list(self.rng.permutation(SCENARIOS))
        scenario = self.bag.pop()
        self.draws += 1
        return copy.deepcopy(self.cases[int(self.rng.integers(len(self.cases)))][scenario])

    def to_dict(self):
        return dict(
            schema="dual-training-sampler-v1",
            identity=copy.deepcopy(self.identity),
            rng=copy.deepcopy(self.rng.bit_generator.state),
            bag=list(self.bag),
            draws=self.draws,
        )

    def restore(self, state):
        if state.get("schema") != "dual-training-sampler-v1" or state.get("identity") != self.identity:
            raise ValueError("Sampler identity differs from recovery state")
        bag = state["bag"]
        allowed = {"normal"} if self.identity["stage"] == 1 else set(SCENARIOS)
        if (
            not isinstance(bag, list)
            or len(bag) > len(allowed)
            or len(set(bag)) != len(bag)
            or not set(bag) <= allowed
            or type(state["draws"]) is not int
            or state["draws"] < 0
        ):
            raise ValueError("Malformed scenario bag/counters")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.bag = list(bag)
        self.draws = state["draws"]


class EpisodeAugmentation:
    """Fixed episode relabeling, with a stream separate from case sampling."""

    def __init__(self, *, seed, worker, stage):
        self.identity = dict(seed=seed, worker=worker, stage=stage)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, worker, stage, 10272713]))
        self.goal_order = list(range(10))
        self.swap_drones = False
        self._maps()

    def _maps(self):
        self.canonical_to_augmented = action_permutation(
            self.goal_order, swap_drones=self.swap_drones
        ).tolist()
        self.augmented_to_canonical = np.argsort(self.canonical_to_augmented).tolist()

    def new_episode(self):
        self.goal_order = self.rng.permutation(10).tolist()
        self.swap_drones = bool(self.rng.integers(2))
        self._maps()

    def canonical_action(self, action):
        if type(action) is not int or not 0 <= action < 242:
            raise ValueError("Augmented categorical action required")
        return self.augmented_to_canonical[action]

    def observe(self, observation, mask):
        changed, changed_mask = permute_observation(
            {k: torch.as_tensor(v).unsqueeze(0) for k, v in observation.items()},
            torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0),
            self.goal_order,
            swap_drones=self.swap_drones,
        )
        return {k: v[0].numpy().copy() for k, v in changed.items()}, changed_mask[0].numpy().copy()

    def to_dict(self):
        return dict(
            schema="dual-episode-augmentation-v1",
            identity=copy.deepcopy(self.identity),
            rng=copy.deepcopy(self.rng.bit_generator.state),
            goal_order=list(self.goal_order),
            swap_drones=self.swap_drones,
            canonical_to_augmented=list(self.canonical_to_augmented),
            augmented_to_canonical=list(self.augmented_to_canonical),
        )

    def restore(self, state):
        if (
            state.get("schema") != "dual-episode-augmentation-v1"
            or state.get("identity") != self.identity
            or type(state.get("swap_drones")) is not bool
        ):
            raise ValueError("Complete matching episode augmentation required")
        mapping = action_permutation(state["goal_order"], swap_drones=state["swap_drones"]).tolist()
        inverse = np.argsort(mapping).tolist()
        if mapping != state.get("canonical_to_augmented") or inverse != state.get("augmented_to_canonical"):
            raise ValueError("Augmentation action mapping is inconsistent")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.goal_order = list(state["goal_order"])
        self.swap_drones = state["swap_drones"]
        self._maps()


def _rng_state():
    return dict(
        python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().numpy().copy()
    )


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))


def _training_worker(connection, factory, seed, stage, worker, snapshot):
    env = None
    try:
        torch.set_num_threads(1)
        worker_seed = int(np.random.SeedSequence([seed, stage, worker, 10272712]).generate_state(1)[0])
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        sampler = CaseSampler(factory.cases, seed=seed, worker=worker, stage=stage)
        augmentation = EpisodeAugmentation(seed=seed, worker=worker, stage=stage)
        episode = 0
        assignments = []
        pending_terminal = None
        if snapshot is None:
            case = sampler.next()
            assignments.append(dict(episode=episode, case_id=case["case_id"], scenario=case["scenario"]))
            env = factory(worker, episode, case)
            if env.terminated:
                raise ValueError("Training world failed before first joint decision")
            augmentation.new_episode()
            observation, mask = augmentation.observe(env.observation(), env.legal_mask())
        else:
            if snapshot.get("schema") != "dual-training-worker-v2" or snapshot.get("worker") != worker:
                raise ValueError("Complete matching worker recovery required")
            sampler.restore(snapshot["sampler"])
            augmentation.restore(snapshot["augmentation"])
            episode = snapshot["episode"]
            assignments = copy.deepcopy(snapshot["assignments"])
            env = factory(worker, episode, snapshot["environment"]["case"])
            before = snapshot["environment"].get("mission", {}).get("state", {}).get("reward_components", {})
            env.restore(snapshot["environment"])
            observation, mask = copy.deepcopy(snapshot["observation"]), np.array(snapshot["mask"], dtype=bool)
            pending_terminal = copy.deepcopy(snapshot.get("pending_terminal"))
            if env.terminated and pending_terminal is None:
                if not hasattr(env, "mission"):
                    raise ValueError("Unexpected restored terminal worker without terminal reward protocol")
                reward = sum(env.mission.reward_components.values()) - sum(before.values())
                pending_terminal = dict(reward=reward, terminal_reason=env.mission.terminal_reason)
            elif pending_terminal is None:
                actual, actual_mask = augmentation.observe(env.observation(), env.legal_mask())
                if any(
                    not np.array_equal(actual[k], observation[k]) for k in observation
                ) or not np.array_equal(actual_mask, mask):
                    raise ValueError("Restored worker observation/mask differs from checkpoint")
            _restore_rng(snapshot["rng"])
        connection.send(dict(ok=True, observation=observation, mask=mask))
        while True:
            command, payload = connection.recv()
            if command == "close":
                env.close()
                env = None
                connection.send(dict(ok=True, command="close", final_flush_complete=True))
                break
            if command == "snapshot":
                state = dict(
                    schema="dual-training-worker-v2",
                    worker=worker,
                    episode=episode,
                    environment=env.to_dict(),
                    sampler=sampler.to_dict(),
                    augmentation=augmentation.to_dict(),
                    assignments=copy.deepcopy(assignments),
                    observation=copy.deepcopy(observation),
                    mask=np.array(mask, copy=True),
                    rng=_rng_state(),
                    pending_terminal=copy.deepcopy(pending_terminal),
                )
                connection.send(dict(ok=True, command="snapshot", state=state))
                continue
            if command != "step":
                raise ValueError("Unknown training worker command")
            case_id = env.case["case_id"]
            if type(payload) is not int or not 0 <= payload < 242 or not mask[payload]:
                raise ValueError("Sampled augmented action violates its preserved mask")
            canonical_action = augmentation.canonical_action(payload)
            if pending_terminal is not None:
                reward = pending_terminal["reward"]
                terminal = True
                info = dict(
                    elapsed_simulated_s=0.0,
                    terminal_reason=pending_terminal["terminal_reason"],
                    resume_wall_timeout=True,
                )
                pending_terminal = None
            else:
                observation, reward, terminal, info = env.step(canonical_action)
            elapsed = info["elapsed_simulated_s"]
            if (
                type(terminal) is not bool
                or not math.isfinite(reward)
                or not math.isfinite(elapsed)
                or not 0 <= elapsed <= 0.200000001
            ):
                raise ValueError("Finite reward, explicit terminal and actual elapsed0..0.20 required")
            if terminal:
                env.close()
                env = None
                episode += 1
                case = sampler.next()
                assignments.append(dict(episode=episode, case_id=case["case_id"], scenario=case["scenario"]))
                env = factory(worker, episode, case)
                if env.terminated:
                    raise ValueError("Training reset failed before first joint decision")
                observation = env.observation()
                augmentation.new_episode()
            observation, mask = augmentation.observe(observation, env.legal_mask())
            connection.send(
                dict(
                    ok=True,
                    observation=observation,
                    mask=mask,
                    reward=float(reward),
                    terminal=terminal,
                    elapsed_simulated_s=float(elapsed),
                    terminal_reason=info.get("terminal_reason"),
                    case_id=case_id,
                    canonical_action=canonical_action,
                    resume_wall_timeout=info.get("resume_wall_timeout", False),
                )
            )
    except BaseException:
        try:
            connection.send(dict(ok=False, error=traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                try:
                    connection.send(
                        dict(ok=False, error="Final training flush failed: " + traceback.format_exc())
                    )
                except (OSError, EOFError):
                    pass
        connection.close()


class TrainingWorlds(ProcessWorlds):
    """Isolated native worlds with acknowledged complete-state snapshots."""

    def __init__(
        self,
        factory,
        *,
        seed,
        stage,
        workers,
        snapshots=None,
        timeout_seconds=650.0,
        close_timeout_seconds=60.0,
        heartbeat=None,
    ):
        if (
            type(workers) is not int
            or workers < 1
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Positive worker count and finite timeout required")
        if snapshots is not None and len(snapshots) != workers:
            raise ValueError("Exactly one recovery snapshot per worker required")
        if not math.isfinite(close_timeout_seconds) or close_timeout_seconds <= 0:
            raise ValueError("Positive finite close timeout required")
        self.workers, self.timeout_seconds, self.heartbeat = workers, timeout_seconds, heartbeat
        self.close_timeout_seconds, self.close_wall_seconds = close_timeout_seconds, None
        self.connections, self.processes, self.observations, self.masks = [], [], [], []
        self.worker_peak_rss_bytes = [None] * workers
        self.closed = False
        ctx = mp.get_context("spawn")
        try:
            for worker in range(workers):
                parent, child = ctx.Pipe()
                process = ctx.Process(
                    target=_training_worker,
                    args=(
                        child,
                        factory,
                        seed,
                        stage,
                        worker,
                        None if snapshots is None else snapshots[worker],
                    ),
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for worker in range(workers):
                result = self._receive(worker)
                self.observations.append(result["observation"])
                self.masks.append(result["mask"])
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f"Training initialization cleanup failed: {cleanup}")
            raise

    def snapshot(self):
        for connection in self.connections:
            connection.send(("snapshot", None))
        states = []
        for worker in range(self.workers):
            result = self._receive(worker)
            if result.get("command") != "snapshot":
                raise ValueError("Explicit worker snapshot acknowledgement required")
            states.append(result["state"])
        return states


class PartialRollout:
    """Bounded 4096-row chronological records, including actual terminal duration."""

    def __init__(self, workers):
        if type(workers) is not int or workers < 1:
            raise ValueError("Positive worker count required")
        self.workers = workers
        self.rows = []

    @property
    def count(self):
        return len(self.rows)

    def append(self, *, worker, observation, mask, action, log_prob, value, reward, terminal, elapsed):
        mask = np.asarray(mask)
        if self.count >= 4096 or type(worker) is not int or not 0 <= worker < self.workers:
            raise ValueError("Bounded rollout and matching worker required")
        if (
            mask.shape != (242,)
            or mask.dtype != bool
            or type(action) is not int
            or not 0 <= action < 242
            or not mask[action]
        ):
            raise ValueError("Legal categorical rollout action required")
        if (
            type(terminal) is not bool
            or not all(math.isfinite(x) for x in (log_prob, value, reward, elapsed))
            or not 0 <= elapsed <= 0.200000001
        ):
            raise ValueError("Finite rollout scalars and explicit terminal required")
        if set(observation) != {"goals", "drones", "global_features"} or not all(
            np.isfinite(x).all() for x in observation.values()
        ):
            raise ValueError("Complete finite rollout observation required")
        self.rows.append(
            dict(
                worker=worker,
                observation={k: np.array(v, copy=True) for k, v in observation.items()},
                mask=mask.copy(),
                action=action,
                log_prob=float(log_prob),
                value=float(value),
                reward=float(reward),
                terminal=terminal,
                elapsed=float(elapsed),
            )
        )

    def to_dict(self):
        return dict(
            schema="dual-partial-rollout-v1",
            workers=self.workers,
            count=self.count,
            rows=copy.deepcopy(self.rows),
        )

    @classmethod
    def from_dict(cls, state):
        if state.get("schema") != "dual-partial-rollout-v1" or state.get("count") != len(state["rows"]):
            raise ValueError("Complete rollout buffer/counter required")
        value = cls(state["workers"])
        for row in state["rows"]:
            value.append(**row)
        return value

    def batch(self, policy, observations):
        if not self.rows or len(observations) != self.workers:
            raise ValueError("Nonempty rollout and one bootstrap observation per worker required")
        device = next(policy.parameters()).device

        def tensor(data, dtype=torch.float32):
            return torch.as_tensor(data, dtype=dtype, device=device)

        with torch.no_grad():
            _, last = policy({k: tensor(np.stack([o[k] for o in observations])) for k in observations[0]})
        rewards = tensor([r["reward"] for r in self.rows])
        values = tensor([r["value"] for r in self.rows])
        terminals = tensor([r["terminal"] for r in self.rows], torch.bool)
        advantages = torch.empty_like(values)
        returns = torch.empty_like(values)
        for worker in range(self.workers):
            ids = [i for i, r in enumerate(self.rows) if r["worker"] == worker]
            if ids:
                a, v = generalized_advantage(
                    rewards[ids, None], values[ids, None], terminals[ids, None], last[worker : worker + 1]
                )
                advantages[ids], returns[ids] = a[:, 0], v[:, 0]
        return PPOBatch(
            {k: tensor(np.stack([r["observation"][k] for r in self.rows])) for k in observations[0]},
            tensor(np.stack([r["mask"] for r in self.rows]), torch.bool),
            tensor([r["action"] for r in self.rows], torch.long),
            tensor([r["log_prob"] for r in self.rows]),
            advantages,
            returns,
        )


def collect_training_round(policy, worlds, rollout):
    if rollout.workers != worlds.workers or rollout.count >= 4096:
        raise ValueError("Open matching partial rollout required")
    ids = list(range(min(worlds.workers, 4096 - rollout.count)))
    observations = [copy.deepcopy(worlds.observations[i]) for i in ids]
    masks = [np.array(worlds.masks[i], copy=True) for i in ids]
    device = next(policy.parameters()).device
    policy.eval()
    with torch.no_grad():
        actions, logps, values = policy.act(
            {
                k: torch.as_tensor(np.stack([o[k] for o in observations]), device=device)
                for k in observations[0]
            },
            torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device),
        )
    outcomes = worlds.step(ids, actions.cpu().tolist())
    for local, (worker, outcome) in enumerate(zip(ids, outcomes)):
        rollout.append(
            worker=worker,
            observation=observations[local],
            mask=masks[local],
            action=int(actions[local]),
            log_prob=float(logps[local]),
            value=float(values[local]),
            reward=outcome["reward"],
            terminal=outcome["terminal"],
            elapsed=outcome["elapsed_simulated_s"],
        )
    return outcomes


def _stage_counts(transitions):
    return [
        min(transitions, 262144),
        max(0, min(transitions - 262144, 720896)),
        max(0, min(transitions - 983040, 65536)),
    ]


def save_recovery(path, trainer, worlds, rollout, *, binding, stage, curves, sampling_history=None):
    if rollout.count and stage != trainer.budget.stage:
        raise ValueError("Partial rollout must belong to the current stage")
    state = dict(
        schema="dual-training-recovery-v1",
        binding=copy.deepcopy(binding),
        stage=stage,
        workers=worlds.snapshot(),
        rollout=rollout.to_dict(),
        curves=copy.deepcopy(curves),
        sampling_history=copy.deepcopy(sampling_history or []),
        collected_transitions=trainer.budget.transitions + rollout.count,
        update_transaction="complete-update-boundary",
    )
    trainer.save(
        path,
        environment_state=state,
        backend_state=dict(
            stage_transition_counts=_stage_counts(trainer.budget.transitions),
            stage3_backend="native-mujoco",
            selected_backend=binding["backend"],
            learning_lock_sha256=binding["learning_lock_sha256"],
        ),
    )


def load_recovery(path, *, binding, device="cpu"):
    # Check identity before deserializing optimizer into a runnable trainer or
    # changing process RNG. These are trusted local torch checkpoints only.
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("environment_state")
    if (
        not isinstance(state, dict)
        or state.get("schema") != "dual-training-recovery-v1"
        or state.get("binding") != binding
    ):
        raise ValueError("Training recovery binding differs")
    buffer = PartialRollout.from_dict(state["rollout"])
    if (
        len(state["workers"]) != binding["workers"]
        or buffer.workers != binding["workers"]
        or raw["budget"]["seed"] != binding["seed"]
    ):
        raise ValueError("Worker count or production seed differs")
    transitions = raw["budget"]["transitions"]
    budget = TrainingBudget(**raw["budget"])
    allowed_stages = {budget.stage}
    if transitions in (262144, 983040) and not buffer.count:
        allowed_stages.add(budget.stage - 1)
    if state["stage"] not in allowed_stages:
        raise ValueError("Recovery stage disagrees with counters")
    if len(state["curves"]) != budget.updates or any(
        row.get("update") != index
        or row.get("transitions") != index * 4096
        or row.get("stage") != TrainingBudget(budget.seed, (index - 1) * 4096, index - 1).stage
        or row.get("ppo", {}).get("minibatches") != 40
        for index, row in enumerate(state["curves"], 1)
    ):
        raise ValueError("Recovery curves disagree with complete update counters")
    if len(state.get("sampling_history", [])) != state["stage"] - 1:
        raise ValueError("Complete previous-stage sampling history required")
    if raw.get("versions") != dict(torch=torch.__version__, numpy=np.__version__):
        raise ValueError("Exact checkpoint Torch/NumPy versions required")
    if "policy_config" in binding and raw.get("policy_config") != binding["policy_config"]:
        raise ValueError("Policy architecture differs from production binding")
    if (
        state["collected_transitions"] != transitions + buffer.count
        or state["collected_transitions"] > 1048576
    ):
        raise ValueError("Recovery transition counters disagree")
    expected = dict(
        stage_transition_counts=_stage_counts(transitions),
        stage3_backend="native-mujoco",
        selected_backend=binding["backend"],
        learning_lock_sha256=binding["learning_lock_sha256"],
    )
    if raw.get("backend_state") != expected or raw.get("physical_resume_available") is not True:
        raise ValueError("Complete bound backend recovery required")
    trainer, _ = JointPPO.load(path, device=device)
    return trainer, state


def append_training_episode_receipt(path, record):
    """Append one exact JSON gzip member, fsync, and verify its complete readback.

    Shared with development storage measurements. This helper never admits an
    environment or deletes files; cleanup remains exclusively in the admitted
    training wrapper after successful persistence and ownership checks.
    """
    import gzip
    import os

    if record.get("schema") != "dual-training-episode-v1":
        raise ValueError("Training episode receipt schema required")
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    member = gzip.compress(payload, compresslevel=3, mtime=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        start = stream.tell()
        stream.write(member)
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(start)
        if gzip.decompress(stream.read(len(member))) != payload:
            raise OSError("Training receipt verification failed")
    return len(member)


class _StoredTrainingEpisode:
    """Bounded derived storage for freshly owned, admitted training worlds only.

    Complete canonical proposals travel in recovery snapshots and one compressed
    receipt per closed episode. Receipt members are fsynced and read back before
    any newly derived world directory is removed. Existing attempts are never
    pruned, and a failed close/receipt leaves the world available for diagnosis.
    """

    def __init__(self, env, *, container, token, archive, provenance):
        if env.mode != "training" or not env.production_admitted or env.retain_replay:
            raise ValueError("Storage cleanup requires admitted training without permanent replay")
        self._env = env
        self._container, self._archive = Path(container), Path(archive)
        self._token = token
        self._provenance = copy.deepcopy(provenance)
        self._actions = []
        self._initial = self._native_state()
        self._restored_from = None
        self._closed = False
        self._execution_error = None
        self._close_failed = None
        self._check_ownership()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _check_ownership(self):
        marker = self._container / "ownership.json"
        expected = dict(schema="dual-training-owned-episode-v1", token=self._token)
        if (
            self._container.is_symlink()
            or not marker.is_file()
            or json.loads(marker.read_text()) != expected
            or Path(self._env.output) != self._container / "world"
            or Path(self._env.output).is_symlink()
            or self._archive.resolve().is_relative_to(self._container.resolve())
        ):
            raise ValueError("Missing or mismatched training episode ownership")

    def _native_state(self):
        sample = self._env.fixture.sample()
        return {key: copy.deepcopy(sample[key]) for key in ("world_time", "qpos", "qvel")}

    def step(self, action):
        self._actions.append(dict(tick=self._env.mission.tick, action=int(action)))
        try:
            return self._env.step(action)
        except Exception:
            self._execution_error = traceback.format_exc()
            raise

    def to_dict(self):
        if self._execution_error is not None:
            raise ValueError("Cannot checkpoint after a failed native step; use prior durable recovery")
        state = self._env.to_dict()
        state["training_storage"] = dict(
            schema="dual-training-storage-v1",
            actions=copy.deepcopy(self._actions),
            initial_native=copy.deepcopy(self._initial),
            provenance=copy.deepcopy(self._provenance),
            restored_from=self._restored_from,
        )
        return state

    def restore(self, state):
        from .dual_coordination_campaign import _digest

        storage = state.get("training_storage")
        if (
            not isinstance(storage, dict)
            or storage.get("schema") != "dual-training-storage-v1"
            or storage.get("provenance") != self._provenance
        ):
            raise ValueError("Complete matching training storage recovery required")
        actions = storage["actions"]
        tick = state["mission"]["state"]["tick"]
        previous = -1
        for row in actions:
            if (
                set(row) != {"tick", "action"}
                or type(row["tick"]) is not int
                or row["tick"] <= previous
                or row["tick"] > tick
                or row["tick"] % 20
                or type(row["action"]) is not int
                or not 0 <= row["action"] < 242
            ):
                raise ValueError("Invalid canonical proposal history in recovery")
            previous = row["tick"]
        physical = copy.deepcopy(state)
        del physical["training_storage"]
        self._env.restore(physical)
        self._actions = copy.deepcopy(actions)
        self._initial = copy.deepcopy(storage["initial_native"])
        self._restored_from = _digest(state)

    def close(self):
        import shutil

        if self._closed:
            return
        if self._close_failed is not None:
            raise RuntimeError(
                "Previous training close failed; derived files remain preserved"
            ) from self._close_failed
        self._check_ownership()
        error = None
        try:
            self._env.close()
        except Exception as caught:
            error = caught
            self._close_failed = caught
        try:
            record = dict(
                schema="dual-training-episode-v1",
                case=copy.deepcopy(self._env.case),
                provenance=self._provenance,
                physical_binding=copy.deepcopy(self._env.binding),
                actions=self._actions,
                initial_native=self._initial,
                final_native=self._native_state(),
                mission=self._env.mission.to_dict(),
                audit=self._env.online_auditor.report(),
                restored_from=self._restored_from,
                episode_token=self._token,
                execution_error=self._execution_error,
                close_error=None if error is None else f"{type(error).__name__}: {error}",
                derived_directory_retired=False,
                retirement_policy="only_after_successful_close_and_verified_durable_receipt",
            )
            append_training_episode_receipt(self._archive,record)
        except Exception as receipt_error:
            self._close_failed = receipt_error
            if error is not None:
                error.add_note(f"Preserving world; receipt persistence also failed: {receipt_error}")
                raise error
            raise
        if error is not None:
            raise error
        try:
            self._check_ownership()
            shutil.rmtree(self._container)
        except Exception as cleanup_error:
            self._close_failed = cleanup_error
            raise
        self._closed = True


@dataclass
class NativeTrainingFactory:
    original: str
    design: dict
    cases: list
    output: str
    learning_lock: str
    generation: str
    _admission: object = field(default=None, init=False, repr=False)

    def __call__(self, worker, episode, case):
        from .dual_coordination_campaign import VerifiedLearningLock
        from .dual_coordination_env import DualCoordinationEnv

        if self._admission is None:
            self._admission = VerifiedLearningLock.load(self.learning_lock)
        admission = self._admission
        admission.require_current()
        admission.verify_case(case, "training")
        from .dual_coordination_progress import atomic_json

        container = Path(self.output) / f"worker-{worker}" / f"episode-{episode}-{self.generation}"
        container.mkdir(parents=True, exist_ok=False)
        token = uuid.uuid4().hex
        atomic_json(container / "ownership.json", dict(schema="dual-training-owned-episode-v1", token=token))
        env = DualCoordinationEnv(
            self.original,
            container / "world",
            case,
            self.design,
            mode="training",
            admission=admission,
            retain_replay=False,
        )
        from .dual_coordination_campaign import _digest

        provenance = dict(
            learning_lock_sha256=admission.digest,
            **{
                role + "_sha256": _digest(admission.binding.files[role])
                for role in ("source", "controller", "assets")
            },
        )
        return _StoredTrainingEpisode(
            env,
            container=container,
            token=token,
            archive=container.parent / f"episodes-{self.generation}.jsonl.gz",
            provenance=provenance,
        )


def _load_training_cases(lock):
    records = lock.binding.files["declaration"]
    declarations = [r for r in records if Path(r["path"]).name == "declaration.json"]
    if len(declarations) != 1:
        raise ValueError("Exactly one bound declaration required")
    path = Path(declarations[0]["path"])
    declaration = json.loads((lock.binding.root / path).read_text())
    relative = str(path.parent / "training.jsonl")
    selected = [r for r in records if r["path"] == relative]
    if len(selected) != 1 or declaration.get("files", {}).get("training.jsonl") != {
        k: selected[0][k] for k in ("bytes", "sha256")
    }:
        raise ValueError("Exact training manifest must be bound to the learning lock")
    cases = [json.loads(line) for line in (lock.binding.root / relative).read_text().splitlines()]
    if len(cases) != 10000:
        raise ValueError("Exactly10000 three-variant training rows required")
    CaseSampler(cases, seed=42, worker=0, stage=1)  # Structural validation without drawing cases.
    return cases, selected[0]["sha256"]


def _require_bound_file(lock, role, path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Missing production {role} input")
    identity = (path.stat().st_size, sha256(path))
    if not any(identity == (record["bytes"], record["sha256"]) for record in lock.binding.files[role]):
        raise ValueError(f"Production {role} input differs from learning lock")
    return path


def _run_seed(
    factory, *, output, binding, resume=False, checkpoint_rounds=64, heartbeat=None, after_update=None
):
    """Finite update transaction loop; tests may stop through after_update.

    There is no reduced production budget parameter. A failed update resumes
    from the pre-update checkpoint and repeats the same five RNG-bound epochs.
    """
    from .dual_coordination_progress import atomic_json
    import os

    if type(checkpoint_rounds) is not int or checkpoint_rounds < 1:
        raise ValueError("Positive decision-round checkpoint interval required")
    output = Path(output)
    current = output / "current-recovery.pt"
    terminal = output / "terminal.pt"
    device = "cuda" if binding["backend"] == "native-cuda" else "cpu"
    if resume:
        if not output.is_dir() or not current.is_file():
            raise ValueError("Resume requires existing complete recovery; never restart an incomplete seed")
        trainer, state = load_recovery(current, binding=binding, device=device)
        curves = state["curves"]
        rollout = PartialRollout.from_dict(state["rollout"])
        stage = state["stage"]
        snapshots = state["workers"]
        history = state.get("sampling_history", [])
    else:
        output.mkdir(parents=True, exist_ok=False)
        seed = binding["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        trainer = JointPPO(JointActorCritic().to(device), seed=seed)
        curves = []
        rollout = PartialRollout(binding["workers"])
        stage = 1
        snapshots = None
        history = []
    if terminal.exists():
        if not trainer.budget.selectable or rollout.count or sha256(terminal) != sha256(current):
            raise ValueError("Terminal checkpoint differs from exact complete recovery")
        return dict(
            seed=binding["seed"],
            transitions=1048576,
            updates=256,
            terminal=str(terminal),
            sha256=sha256(terminal),
        )
    worlds = None
    rounds = 0

    def save():
        save_recovery(
            current,
            trainer,
            worlds,
            rollout,
            binding=binding,
            stage=stage,
            curves=curves,
            sampling_history=history,
        )
        atomic_json(output / "curves.json", dict(seed=binding["seed"], selectable=False, updates=curves))

    try:
        while not trainer.budget.selectable:
            wanted = trainer.budget.stage
            if stage != wanted:
                if rollout.count:
                    raise ValueError("Stage boundary has unconsumed rollout")
                previous = worlds.snapshot() if worlds is not None else snapshots
                history.append(
                    dict(
                        stage=stage,
                        workers=[
                            dict(worker=s["worker"], assignments=s["assignments"], sampler=s["sampler"])
                            for s in previous
                        ],
                    )
                )
                if worlds is not None:
                    worlds.close()
                worlds = None
                snapshots = None
                stage = wanted
            if worlds is None:
                stage_factory = (
                    replace(factory, output=str(Path(factory.output) / f"stage-{stage}"))
                    if isinstance(factory, NativeTrainingFactory)
                    else factory
                )
                worlds = TrainingWorlds(
                    stage_factory,
                    seed=binding["seed"],
                    stage=stage,
                    workers=binding["workers"],
                    snapshots=snapshots,
                    heartbeat=heartbeat,
                )
                snapshots = None
                save()
            while rollout.count < 4096:
                collect_training_round(trainer.policy, worlds, rollout)
                rounds += 1
                if rounds % checkpoint_rounds == 0:
                    save()
                if heartbeat is not None:
                    heartbeat()
            save()  # Last complete rollout + pre-update Adam/RNG transaction.
            start = time.monotonic()
            metrics = trainer.update(
                rollout.batch(trainer.policy, worlds.observations), backend="native-mujoco"
            )
            curve = dict(
                update=trainer.budget.updates,
                stage=stage,
                transitions=trainer.budget.transitions,
                rollout_reward=sum(r["reward"] for r in rollout.rows),
                terminal_episodes=sum(r["terminal"] for r in rollout.rows),
                simulated_seconds=sum(r["elapsed"] for r in rollout.rows),
                ppo=metrics,
                update_wall_seconds=time.monotonic() - start,
            )
            curves.append(curve)
            rollout = PartialRollout(binding["workers"])
            save()
            if after_update is not None:
                after_update(copy.deepcopy(curve))
        if worlds is None and snapshots is not None:
            stage_factory = (
                replace(factory, output=str(Path(factory.output) / f"stage-{stage}"))
                if isinstance(factory, NativeTrainingFactory)
                else factory
            )
            worlds = TrainingWorlds(
                stage_factory,
                seed=binding["seed"],
                stage=stage,
                workers=binding["workers"],
                snapshots=snapshots,
                heartbeat=heartbeat,
            )
        if worlds is not None:
            worlds.close()
            worlds = None
        # No terminal selectable artifact is published until every worker's final
        # flush has acknowledged success. Existing terminal bytes are immutable.
        os.link(current, terminal)
        result = dict(
            seed=binding["seed"],
            transitions=1048576,
            updates=256,
            terminal=str(terminal),
            sha256=sha256(terminal),
            stage_transition_counts=[262144, 720896, 65536],
            stage3_backend="native-mujoco",
        )
        atomic_json(output / "terminal.json", result)
        return result
    except BaseException as error:
        if worlds is not None:
            try:
                worlds.close()
            except BaseException as cleanup:
                error.add_note(f"Training worker cleanup failed: {cleanup}")
            worlds = None
        try:
            atomic_json(
                output / f"failure-{time.time_ns()}.json",
                dict(
                    error=repr(error),
                    notes=getattr(error, "__notes__", []),
                    recovery_exists=current.exists(),
                    saved_recovery_sha256=sha256(current) if current.exists() else None,
                    meaning="Resume the saved complete boundary; never select or restart this incomplete seed",
                ),
            )
        except BaseException as diagnostic_error:
            error.add_note(f"Failure receipt could not be saved: {diagnostic_error}")
        raise


def _qualified_adapter_available():
    import inspect
    from .dual_coordination_env import DualCoordinationEnv

    parameters = inspect.signature(DualCoordinationEnv).parameters
    return {"admission", "retain_replay"} <= set(parameters)


def _verified_learning_lock(path):
    from .dual_coordination_campaign import VerifiedLearningLock

    lock = VerifiedLearningLock.load(path)
    lock.require_current()
    return lock


def production_status(learning_lock):
    result = dict(
        schema="dual-production-admission-v1",
        ready=False,
        production_transitions_started=0,
        learning_lock_verified=False,
        resume_supported=True,
    )
    try:
        lock = _verified_learning_lock(learning_lock)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return dict(result, reason="learning_lock_unavailable", detail=str(error))
    result.update(
        learning_lock_verified=True,
        learning_lock_sha256=lock.digest,
        plan=production_plan(lock.selected_backend),
    )
    if lock.selected_backend == "warp-cuda":
        return dict(result, reason="qualified_warp_adapter_unavailable")
    if lock.selected_backend == "native-cuda" and not torch.cuda.is_available():
        return dict(result, reason="qualified_cuda_device_unavailable")
    if not _qualified_adapter_available():
        return dict(result, reason="qualified_production_adapter_unavailable")
    return dict(
        result,
        ready=True,
        reason="qualified_lock_and_adapter_available",
        requires="Bound original/controller/manifest and measured worker configuration checked at start",
    )


def start_production(
    learning_lock,
    *,
    original=None,
    design_path=None,
    output=None,
    workers=None,
    resume=False,
    journal_root=None,
):
    from .dual_coordination_progress import RunJournal, atomic_json

    status = production_status(learning_lock)
    if not status["ready"]:
        raise ProductionUnavailable(json.dumps(status, sort_keys=True))
    if original is None or design_path is None or output is None or type(workers) is not int or workers < 1:
        raise ProductionUnavailable(
            "Explicit bound original, controller design, output and measured worker count required"
        )
    lock = _verified_learning_lock(learning_lock)
    if lock.receipt["benchmark"].get("selected_workers") != workers:
        raise ValueError("Worker count must equal the measured frozen benchmark selection")
    original = _require_bound_file(lock, "assets", original)
    design_path = _require_bound_file(lock, "controller", design_path)
    for name in (
        "dual_coordination_train.py",
        "dual_coordination_policy.py",
        "dual_coordination_benchmark.py",
    ):
        _require_bound_file(lock, "source", Path(__file__).with_name(name))
    cases, manifest_digest = _load_training_cases(lock)
    design = json.loads(design_path.read_text())
    from .payload_hybrid_controller import Controller

    if not isinstance(design, dict) or not {"gain", "hover_action"} <= set(design):
        raise ValueError("Frozen finite controller gain and hover action required")
    Controller(design["gain"], design["hover_action"])
    output = Path(output).resolve()
    declaration = dict(
        schema="dual-training-run-v1",
        learning_lock_sha256=lock.digest,
        backend=lock.selected_backend,
        workers=workers,
        manifest_sha256=manifest_digest,
        original_sha256=sha256(original),
        controller_sha256=sha256(design_path),
        seeds=[42, 43, 44],
        plan=production_plan(lock.selected_backend),
        output=str(output),
        checkpoint_rounds=64,
    )
    if resume:
        if (
            not (output / "campaign.json").is_file()
            or json.loads((output / "campaign.json").read_text()) != declaration
        ):
            raise ValueError("Exact existing campaign declaration required; no incomplete seed restart")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "campaign.json", declaration)
    journal = RunJournal(output if journal_root is None else journal_root)
    attempt = journal.start(
        "finite-production-training", "Exact three-seed finite production campaign", category="training"
    )
    last_heartbeat = time.monotonic()

    def heartbeat():
        nonlocal last_heartbeat
        if time.monotonic() - last_heartbeat >= 30:
            journal.heartbeat(
                attempt, reason="Finite training worker/optimizer active; durable recovery retained"
            )
            last_heartbeat = time.monotonic()

    torch.set_num_threads(1)
    results = []
    try:
        for seed in (42, 43, 44):
            lock.require_current()
            root = output / f"seed-{seed}"
            binding = dict(
                learning_lock_sha256=lock.digest,
                workers=workers,
                backend=lock.selected_backend,
                seed=seed,
                manifest_sha256=manifest_digest,
                original_sha256=declaration["original_sha256"],
                controller_sha256=declaration["controller_sha256"],
                policy_config=asdict(PolicyConfig()),
            )
            factory = NativeTrainingFactory(
                str(original), design, cases, str(root / "worlds"), str(lock.path), uuid.uuid4().hex
            )
            results.append(
                _run_seed(factory, output=root, binding=binding, resume=root.exists(), heartbeat=heartbeat)
            )
        result = dict(
            schema="dual-training-complete-v1",
            learning_lock_sha256=lock.digest,
            seeds=results,
            production_transitions=3145728,
            selectable="terminal-only",
            validation_performed=False,
        )
        atomic_json(output / "training-complete.json", result)
        journal.finish(
            attempt,
            status="passed",
            reason="All three terminal budgets complete; validation remains separate",
        )
        return result
    except BaseException as error:
        try:
            journal.finish(attempt, status="failed", reason=repr(error))
        except BaseException as diagnostic_error:
            error.add_note(f"Training journal failure could not be saved: {diagnostic_error}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-lock", required=True)
    parser.add_argument(
        "--run", action="store_true", help="Run the full qualified three-seed finite campaign"
    )
    parser.add_argument("--original")
    parser.add_argument("--design")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--journal-root")
    args = parser.parse_args(argv)
    if not args.run:
        result = production_status(args.learning_lock)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ready"] else 2
    result = start_production(
        args.learning_lock,
        original=args.original,
        design_path=args.design,
        output=args.output,
        workers=args.workers,
        resume=args.resume,
        journal_root=args.journal_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
