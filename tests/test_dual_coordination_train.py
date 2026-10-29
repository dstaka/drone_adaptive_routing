"""Administrative software admission tests; fixture receipts are not physics results."""

import importlib
import json

import pytest


def module():
    try:
        return importlib.import_module("adaptive_drone_patrol.dual_coordination_train")
    except ModuleNotFoundError:
        pytest.fail("Finite production admission guard is not implemented")


def test_finite_plan_has_all_three_native_tails_and_exact_total():
    m = module()
    plan = m.production_plan("native-cuda")
    assert [r["seed"] for r in plan] == [42, 43, 44]
    assert sum(r["total_transitions"] for r in plan) == 3145728
    for row in plan:
        assert [s["transitions"] for s in row["stages"]] == [262144, 720896, 65536]
        assert [s["updates"] for s in row["stages"]] == [64, 176, 16]
        assert row["stages"][2]["physics_backend"] == "native-mujoco"
        assert row["selectable_checkpoint"] == "terminal_only"
    with pytest.raises(ValueError):
        m.production_plan("surrogate")


def test_missing_incomplete_and_mutated_learning_locks_cannot_start(tmp_path):
    m = module()
    absent = m.production_status(tmp_path / "missing.json")
    assert absent["ready"] is False and absent["reason"] == "learning_lock_unavailable"
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"schema": "dual-learning-lock-v1"}))
    assert m.production_status(path)["ready"] is False
    with pytest.raises(m.ProductionUnavailable):
        m.start_production(path)


def test_administratively_valid_lock_still_cannot_bypass_missing_qualified_adapter(tmp_path, monkeypatch):
    m = module()
    from test_dual_coordination_campaign import experiment_fixture
    from adaptive_drone_patrol.dual_coordination_campaign import create_learning_lock

    roles, qualification, benchmark, _ = experiment_fixture(tmp_path)
    lock = tmp_path / "lock.json"
    create_learning_lock(lock, root=tmp_path, roles=roles, qualification=qualification, benchmark=benchmark)
    monkeypatch.setattr(m, "_qualified_adapter_available", lambda: False)
    result = m.production_status(lock)
    assert result["learning_lock_verified"] is True
    assert result["ready"] is False and result["reason"] == "qualified_production_adapter_unavailable"
    assert result["resume_supported"] is True
    with pytest.raises(m.ProductionUnavailable):
        m.start_production(lock)
    (tmp_path / "controller.json").write_text('{"mutated":true}')
    assert m.production_status(lock)["learning_lock_verified"] is False


def training_fixture_cases(count=3):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case

    scenarios = ("normal", "priority", "availability-delay")
    return [{scenario: generate_case(10, i, scenario) for scenario in scenarios} for i in range(count)]


def test_sampling_curriculum_is_balanced_and_completely_resumable():
    m = module()
    cases = training_fixture_cases()
    sampler = m.CaseSampler(cases, seed=42, worker=1, stage=1)
    assert {sampler.next()["scenario"] for _ in range(12)} == {"normal"}
    sampler = m.CaseSampler(cases, seed=42, worker=1, stage=2)
    for _ in range(4):
        assert {sampler.next()["scenario"] for _ in range(3)} == {"normal", "priority", "availability-delay"}
    state = sampler.to_dict()
    expected = [sampler.next()["case_id"] for _ in range(11)]
    restored = m.CaseSampler(cases, seed=42, worker=1, stage=2)
    restored.restore(state)
    assert [restored.next()["case_id"] for _ in range(11)] == expected
    with pytest.raises(ValueError):
        m.CaseSampler(cases, seed=43, worker=1, stage=2).restore(state)


def test_partial_rollout_gae_cuts_terminal_and_roundtrips():
    import numpy as np
    import torch
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, PolicyConfig

    m = module()
    policy = JointActorCritic(PolicyConfig(hidden=16))
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
        policy.critic.bias.fill_(10)
    observation = dict(
        goals=np.zeros((10, 11), np.float32),
        drones=np.zeros((2, 66), np.float32),
        global_features=np.zeros(11, np.float32),
    )
    mask = np.zeros(242, bool)
    mask[240] = True
    rollout = m.PartialRollout(2)
    for worker, terminal in ((0, False), (1, True), (0, True)):
        rollout.append(
            worker=worker,
            observation=observation,
            mask=mask,
            action=240,
            log_prob=0.0,
            value=10.0,
            reward=1.0,
            terminal=terminal,
            elapsed=0.2,
        )
    restored = m.PartialRollout.from_dict(rollout.to_dict())
    batch = restored.batch(policy, [observation, observation])
    assert batch.returns.tolist() == pytest.approx([2.44855, 1.0, 1.0])
    assert restored.count == 3
    with pytest.raises(ValueError):
        restored.append(
            worker=0,
            observation=observation,
            mask=mask,
            action=0,
            log_prob=0.0,
            value=1.0,
            reward=1.0,
            terminal=False,
            elapsed=0.2,
        )


def only_actions(worlds):
    import numpy as np

    return [int(np.flatnonzero(mask)[0]) for mask in worlds.masks]


class RecoverableToyEnv:
    """Nonphysical software fixture, never a qualification source."""

    def __init__(self, case):
        import numpy as np

        self.case = case
        self.rng = np.random.default_rng(123)
        self.ticks = 0
        self.terminated = False
        self.closed = False

    def observation(self):
        import numpy as np

        return dict(
            goals=np.full((10, 11), self.ticks, np.float32),
            drones=np.zeros((2, 66), np.float32),
            global_features=np.zeros(11, np.float32),
        )

    def legal_mask(self):
        import numpy as np

        mask = np.zeros(242, bool)
        mask[240] = True
        return mask

    def step(self, action):
        assert action == 240
        self.ticks += 1
        self.terminated = self.ticks == 3
        return (
            self.observation(),
            float(self.rng.random()),
            self.terminated,
            dict(elapsed_simulated_s=0.2, terminal_reason="toy" if self.terminated else None),
        )

    def to_dict(self):
        return dict(
            case=self.case, ticks=self.ticks, rng=self.rng.bit_generator.state, terminated=self.terminated
        )

    def restore(self, state):
        self.ticks = state["ticks"]
        self.rng.bit_generator.state = state["rng"]
        self.terminated = state["terminated"]

    def close(self):
        self.closed = True


class ToyTrainingFactory:
    def __init__(self, cases):
        self.cases = cases

    def __call__(self, worker, episode, case):
        return RecoverableToyEnv(case)


def test_process_worker_state_restores_next_rewards_and_sampling(tmp_path):
    m = module()
    cases = training_fixture_cases(2)
    factory = ToyTrainingFactory(cases)
    with m.TrainingWorlds(factory, seed=42, stage=2, workers=2) as worlds:
        worlds.step([0, 1], only_actions(worlds))
        snapshot = worlds.snapshot()
        expected = [worlds.step([0, 1], only_actions(worlds)) for _ in range(4)]
    with m.TrainingWorlds(factory, seed=42, stage=2, workers=2, snapshots=snapshot) as restored:
        actual = [restored.step([0, 1], only_actions(restored)) for _ in range(4)]
    assert [[row["reward"] for row in round_] for round_ in actual] == [
        [row["reward"] for row in round_] for round_ in expected
    ]
    assert [[row["case_id"] for row in round_] for round_ in actual] == [
        [row["case_id"] for row in round_] for round_ in expected
    ]


def test_complete_recovery_keeps_partial_buffer_optimizer_and_rng(tmp_path):
    import random
    import numpy as np
    import torch
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, JointPPO, PolicyConfig

    m = module()
    torch.set_num_threads(1)
    ppo = JointPPO(JointActorCritic(PolicyConfig(hidden=16)), seed=42)
    factory = ToyTrainingFactory(training_fixture_cases(2))
    binding = {"learning_lock_sha256": "test-only", "workers": 1, "backend": "native-cpu", "seed": 42}
    with m.TrainingWorlds(factory, seed=42, stage=1, workers=1) as worlds:
        rollout = m.PartialRollout(1)
        m.collect_training_round(ppo.policy, worlds, rollout)
        m.save_recovery(tmp_path / "current.pt", ppo, worlds, rollout, binding=binding, stage=1, curves=[])
        expected = (torch.rand(3), np.random.random(3), random.random())
    loaded, data = m.load_recovery(tmp_path / "current.pt", binding=binding)
    assert data["rollout"]["count"] == 1 and loaded.budget.transitions == 0
    assert torch.equal(torch.rand(3), expected[0])
    assert np.array_equal(np.random.random(3), expected[1]) and random.random() == expected[2]
    assert data["workers"][0]["environment"]["ticks"] == 1
    with pytest.raises(ValueError, match="binding"):
        m.load_recovery(tmp_path / "current.pt", binding={**binding, "workers": 2})


def test_real_ppo_update_is_atomic_and_resume_keeps_exact_budget(tmp_path, monkeypatch):
    import torch
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, PolicyConfig

    m = module()
    torch.set_num_threads(1)
    monkeypatch.setattr(
        m,
        "JointActorCritic",
        lambda: JointActorCritic(PolicyConfig(hidden=16)),
    )
    factory = ToyTrainingFactory(training_fixture_cases(2))
    binding = {"learning_lock_sha256": "test-only", "workers": 1, "backend": "native-cpu", "seed": 42}

    def stop(curve):
        raise InterruptedError("test stop after atomic update")

    with pytest.raises(InterruptedError):
        m._run_seed(
            factory, output=tmp_path / "seed42", binding=binding, after_update=stop, checkpoint_rounds=1024
        )
    ppo, state = m.load_recovery(tmp_path / "seed42" / "current-recovery.pt", binding=binding)
    assert ppo.budget.transitions == 4096 and ppo.budget.updates == 1 and not ppo.budget.selectable
    assert state["rollout"]["count"] == 0 and len(state["curves"]) == 1
    assert state["curves"][0]["ppo"]["minibatches"] == 40
    assert not (tmp_path / "seed42" / "terminal.pt").exists()
    with pytest.raises(InterruptedError):
        m._run_seed(
            factory,
            output=tmp_path / "seed42",
            binding=binding,
            resume=True,
            after_update=stop,
            checkpoint_rounds=1024,
        )
    ppo, state = m.load_recovery(tmp_path / "seed42" / "current-recovery.pt", binding=binding)
    assert ppo.budget.transitions == 8192 and ppo.budget.updates == 2
    assert [row["update"] for row in state["curves"]] == [1, 2]


def test_training_rejects_warp_and_unbound_inputs_before_output(tmp_path):
    m = module()
    with pytest.raises((m.ProductionUnavailable, ValueError)):
        m.start_production(
            tmp_path / "absent.json",
            original="missing.xml",
            design_path="missing.json",
            output=tmp_path / "campaign",
            workers=1,
        )
    assert not (tmp_path / "campaign").exists()
    from test_dual_coordination_campaign import experiment_fixture
    from adaptive_drone_patrol.dual_coordination_campaign import create_learning_lock

    root = tmp_path / "warp"
    roles, qualification, benchmark, _ = experiment_fixture(root)
    benchmark.update(selected_backend="warp-cuda", warp_fidelity36_passed=True, selected_workers=1)
    (root / "benchmark.json").write_text(json.dumps(benchmark))
    lock = root / "lock.json"
    create_learning_lock(lock, root=root, roles=roles, qualification=qualification, benchmark=benchmark)
    assert m.production_status(lock)["reason"] == "qualified_warp_adapter_unavailable"
    with pytest.raises(m.ProductionUnavailable, match="warp"):
        m.start_production(
            lock,
            original="missing.xml",
            design_path="missing.json",
            output=tmp_path / "warp-campaign",
            workers=1,
        )
    assert not (tmp_path / "warp-campaign").exists()


def test_invalid_recovery_stage_does_not_mutate_process_rng(tmp_path):
    import numpy as np
    import torch
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, JointPPO, PolicyConfig

    m = module()
    trainer = JointPPO(JointActorCritic(PolicyConfig(hidden=16)), seed=42)
    binding = {"learning_lock_sha256": "test-only", "workers": 1, "backend": "native-cpu", "seed": 42}
    factory = ToyTrainingFactory(training_fixture_cases(1))
    with m.TrainingWorlds(factory, seed=42, stage=1, workers=1) as worlds:
        buffer = m.PartialRollout(1)
        m.collect_training_round(trainer.policy, worlds, buffer)
        m.save_recovery(tmp_path / "valid.pt", trainer, worlds, buffer, binding=binding, stage=1, curves=[])
    state = torch.load(tmp_path / "valid.pt", weights_only=False)
    state["environment_state"]["stage"] = 2
    torch.save(state, tmp_path / "invalid.pt")
    torch.manual_seed(999)
    np.random.seed(999)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="stage"):
        m.load_recovery(tmp_path / "invalid.pt", binding=binding)
    assert torch.equal(torch.get_rng_state(), before)


def test_primary_failure_survives_failure_receipt_write_error(tmp_path, monkeypatch):
    import torch
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, PolicyConfig
    from adaptive_drone_patrol import dual_coordination_progress as progress

    m = module()
    torch.set_num_threads(1)
    monkeypatch.setattr(
        m,
        "JointActorCritic",
        lambda: JointActorCritic(PolicyConfig(hidden=16)),
    )

    def broken_world(*args, **kwargs):
        raise ValueError("primary world initialization failed")

    def broken_write(*args, **kwargs):
        raise OSError("failure receipt disk full")

    monkeypatch.setattr(m, "TrainingWorlds", broken_world)
    monkeypatch.setattr(progress, "atomic_json", broken_write)
    with pytest.raises(ValueError, match="primary world") as caught:
        m._run_seed(
            ToyTrainingFactory(training_fixture_cases(1)),
            output=tmp_path / "seed",
            binding={"learning_lock_sha256": "test-only", "workers": 1, "backend": "native-cpu", "seed": 42},
        )
    assert any("disk full" in note for note in caught.value.__notes__)


def test_orchestration_runs_exact_stage_counts_and_closes_before_terminal(tmp_path, monkeypatch):
    """Control-flow fixture only: no optimizer/physics/qualification claims."""
    from types import SimpleNamespace
    from adaptive_drone_patrol.dual_coordination_policy import TrainingBudget

    m = module()
    stages = []
    closed = []

    class Trainer:
        def __init__(self, policy, seed):
            self.policy = policy
            self.budget = TrainingBudget(seed)

        def update(self, batch, backend):
            stages.append(self.budget.stage)
            self.budget.record_update(4096, backend=backend)
            return {"minibatches": 40}

    class Worlds:
        def __init__(self, factory, seed, stage, workers, snapshots, heartbeat):
            self.stage = stage
            self.workers = workers
            self.observations = [{}]

        def snapshot(self):
            return [{"worker": 0, "assignments": [], "sampler": {}}]

        def close(self):
            assert not (tmp_path / "seed" / "terminal.pt").exists()
            closed.append(self.stage)

    class Rollout:
        def __init__(self, workers):
            self.count = 0
            self.workers = workers
            self.rows = []

        def batch(self, policy, observations):
            return None

    def collect(policy, worlds, buffer):
        buffer.count = 4096
        buffer.rows = [{"reward": 1.0, "terminal": False, "elapsed": 0.2}] * 4096

    def save(path, trainer, worlds, rollout, **kwargs):
        path.write_bytes(b"software-only-control-flow-fixture")

    monkeypatch.setattr(m, "JointActorCritic", lambda: SimpleNamespace(to=lambda device: object()))
    monkeypatch.setattr(m, "JointPPO", Trainer)
    monkeypatch.setattr(m, "TrainingWorlds", Worlds)
    monkeypatch.setattr(m, "PartialRollout", Rollout)
    monkeypatch.setattr(m, "collect_training_round", collect)
    monkeypatch.setattr(m, "save_recovery", save)
    result = m._run_seed(
        object(), output=tmp_path / "seed", binding={"seed": 42, "workers": 1, "backend": "native-cpu"}
    )
    assert stages == [1] * 64 + [2] * 176 + [3] * 16
    assert closed == [1, 2, 3] and result["transitions"] == 1048576 and result["updates"] == 256


class DowntimeToyEnv(RecoverableToyEnv):
    def __init__(self, case):
        from types import SimpleNamespace

        super().__init__(case)
        self.mission = SimpleNamespace(
            reward_components={"failure": 0.0, "elapsed": -12.0}, terminal_reason=None
        )

    def to_dict(self):
        return {
            **super().to_dict(),
            "mission": {
                "schema": "dual-mission-state-v1",
                "state": {"reward_components": dict(self.mission.reward_components)},
            },
        }

    def restore(self, state):
        super().restore(state)
        self.terminated = True
        self.mission.reward_components["failure"] = -1000.0
        self.mission.terminal_reason = "infrastructure:wall_timeout"


class DowntimeToyFactory(ToyTrainingFactory):
    def __call__(self, worker, episode, case):
        return DowntimeToyEnv(case)


def test_restored_wall_timeout_is_one_zero_elapsed_terminal_transition():
    m = module()
    factory = DowntimeToyFactory(training_fixture_cases(1))
    with m.TrainingWorlds(factory, seed=42, stage=1, workers=1) as worlds:
        snapshot = worlds.snapshot()
    with m.TrainingWorlds(factory, seed=42, stage=1, workers=1, snapshots=snapshot) as worlds:
        timeout = worlds.step([0], only_actions(worlds))[0]
        next_step = worlds.step([0], only_actions(worlds))[0]
    assert timeout["reward"] == -1000.0 and timeout["terminal"] is True
    assert timeout["elapsed_simulated_s"] == 0.0 and timeout["resume_wall_timeout"] is True
    assert next_step["terminal"] is False and next_step["elapsed_simulated_s"] == 0.2


def test_episode_augmentation_preserves_action_mask_and_restores_independent_rng():
    import numpy as np
    from adaptive_drone_patrol.dual_coordination_policy import action_permutation
    from adaptive_drone_patrol.dual_coordination_schema import encode_action

    m = module()
    aug = m.EpisodeAugmentation(seed=42, worker=0, stage=2)
    aug.new_episode()
    obs = dict(
        goals=np.arange(110, dtype=np.float32).reshape(10, 11),
        drones=np.arange(132, dtype=np.float32).reshape(2, 66),
        global_features=np.arange(11, dtype=np.float32),
    )
    mask = np.zeros(242, bool)
    old_action = encode_action(0, 10, 1)
    mask[old_action] = True
    changed, new_mask = aug.observe(obs, mask)
    state = aug.to_dict()
    mapping = action_permutation(state["goal_order"], swap_drones=state["swap_drones"]).tolist()
    transformed = mapping[old_action]
    assert np.flatnonzero(new_mask).tolist() == [transformed]
    assert aug.canonical_action(transformed) == old_action
    assert np.array_equal(changed["goals"][:, 0], obs["goals"][state["goal_order"], 0])
    restored = m.EpisodeAugmentation(seed=42, worker=0, stage=2)
    restored.restore(state)
    for _ in range(4):
        aug.new_episode()
        restored.new_episode()
        assert aug.to_dict() == restored.to_dict()
    bad = dict(state, augmented_to_canonical=list(range(242)))
    with pytest.raises(ValueError):
        restored.restore(bad)


class StorageEnv:
    """Nonphysical world used only to check admitted-storage bookkeeping."""

    mode = "training"
    production_admitted = True
    retain_replay = False

    def __init__(self, output, *, fail_close=False):
        from types import SimpleNamespace

        self.output = output
        output.mkdir()
        (output / "duplicated.mesh").write_bytes(b"derived mesh copy")
        self.case = {"case_id": "synthetic"}
        self.binding = {"model_sha256": "model", "profile_sha256": "profile"}
        self.mission = SimpleNamespace(
            tick=0,
            terminal_reason=None,
            to_dict=lambda: {
                "state": {"tick": self.mission.tick, "terminal_reason": self.mission.terminal_reason}
            },
        )
        self.fixture = SimpleNamespace(
            sample=lambda: {"qpos": [0.0], "qvel": [0.0], "world_time": self.mission.tick * 0.01}
        )
        self.online_auditor = SimpleNamespace(report=lambda: {"synthetic": True})
        self.fail_close = fail_close

    def step(self, action):
        self.mission.tick += 20
        return {}, 0.0, False, {"elapsed_simulated_s": 0.2}

    def close(self):
        if self.fail_close:
            raise OSError("recovery flush failed")

    def to_dict(self):
        return {"case": self.case, "mission": self.mission.to_dict()}

    def restore(self, state):
        self.mission.tick = state["mission"]["state"]["tick"]


def storage_episode(tmp_path, name="episode", *, fail_close=False):
    m = module()
    container = tmp_path / name
    container.mkdir()
    (container / "ownership.json").write_text(
        json.dumps({"schema": "dual-training-owned-episode-v1", "token": name})
    )
    env = StorageEnv(container / "world", fail_close=fail_close)
    return m._StoredTrainingEpisode(
        env,
        container=container,
        token=name,
        archive=tmp_path / "episodes.jsonl.gz",
        provenance={"learning_lock_sha256": "synthetic", "source": [], "controller": [], "assets": []},
    )


def test_training_storage_preserves_exact_actions_and_result_before_owned_cleanup(tmp_path):
    import gzip

    episode = storage_episode(tmp_path)
    episode.step(123)
    episode.step(240)
    episode.close()
    assert not (tmp_path / "episode").exists()
    rows = [
        json.loads(line)
        for line in gzip.decompress((tmp_path / "episodes.jsonl.gz").read_bytes()).splitlines()
    ]
    assert rows[0]["actions"] == [{"tick": 0, "action": 123}, {"tick": 20, "action": 240}]
    assert (
        rows[0]["mission"]["state"]["tick"] == 40
        and rows[0]["provenance"]["learning_lock_sha256"] == "synthetic"
    )
    assert rows[0]["initial_native"]["qpos"] == [0.0] and rows[0]["final_native"]["qvel"] == [0.0]
    episode.close()  # Idempotent close does not append another receipt.
    assert len(gzip.decompress((tmp_path / "episodes.jsonl.gz").read_bytes()).splitlines()) == 1


def test_training_storage_restore_carries_precheckpoint_actions_into_new_owned_attempt(tmp_path):
    import gzip

    first = storage_episode(tmp_path, "first")
    first.step(123)
    snapshot = first.to_dict()
    second = storage_episode(tmp_path, "second")
    second.restore(snapshot)
    second.step(240)
    second.close()
    rows = [
        json.loads(line)
        for line in gzip.decompress((tmp_path / "episodes.jsonl.gz").read_bytes()).splitlines()
    ]
    assert rows[0]["actions"] == [{"tick": 0, "action": 123}, {"tick": 20, "action": 240}]
    assert rows[0]["restored_from"]
    assert (tmp_path / "first" / "world" / "duplicated.mesh").exists()  # Never delete predecessor attempt.


def test_training_storage_flush_failure_preserves_world_and_failure_receipt(tmp_path):
    import gzip

    episode = storage_episode(tmp_path, fail_close=True)
    with pytest.raises(OSError, match="recovery flush"):
        episode.close()
    assert (tmp_path / "episode" / "world" / "duplicated.mesh").exists()
    rows = [
        json.loads(line)
        for line in gzip.decompress((tmp_path / "episodes.jsonl.gz").read_bytes()).splitlines()
    ]
    assert rows[0]["close_error"] and rows[0]["derived_directory_retired"] is False


def test_training_storage_cannot_cleanup_unowned_or_development_world(tmp_path):
    episode = storage_episode(tmp_path)
    (tmp_path / "episode" / "ownership.json").unlink()
    with pytest.raises(ValueError, match="ownership"):
        episode.close()
    assert (tmp_path / "episode" / "world" / "duplicated.mesh").exists()
    env = StorageEnv(tmp_path / "development")
    env.mode = "development"
    with pytest.raises(ValueError, match="admitted training"):
        module()._StoredTrainingEpisode(
            env, container=tmp_path, token="no", archive=tmp_path / "a.gz", provenance={}
        )


def test_training_storage_preserves_primary_close_error_when_diagnostics_also_fail(tmp_path):
    episode = storage_episode(tmp_path, fail_close=True)

    def failed_sample():
        raise RuntimeError("diagnostics failed")

    episode.fixture.sample = failed_sample
    with pytest.raises(OSError, match="recovery flush") as caught:
        episode.close()
    assert any("diagnostics failed" in note for note in caught.value.__notes__)
    assert (tmp_path / "episode" / "world").exists()


def test_native_training_factory_enables_owned_storage_only_after_admission(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import adaptive_drone_patrol.dual_coordination_env as env_module
    import adaptive_drone_patrol.dual_coordination_campaign as campaign

    m = module()
    checks = []
    capability = SimpleNamespace(
        digest="synthetic",
        require_current=lambda: checks.append("current"),
        verify_case=lambda case, mode: checks.append(mode),
        binding=SimpleNamespace(files={role: [] for role in ("source", "controller", "assets")}),
    )
    monkeypatch.setattr(campaign.VerifiedLearningLock, "load", lambda path: capability)

    def construct(original, output, case, design, **kwargs):
        assert kwargs == {"mode": "training", "admission": capability, "retain_replay": False}
        return StorageEnv(output)

    monkeypatch.setattr(env_module, "DualCoordinationEnv", construct)
    factory = m.NativeTrainingFactory("original", {}, [], str(tmp_path / "worlds"), "lock", "0" * 32)
    episode = factory(0, 0, {"case_id": "synthetic"})
    assert isinstance(episode, m._StoredTrainingEpisode) and checks == ["current", "training"]
    container = episode._container
    with pytest.raises(FileExistsError):
        factory(0, 0, {"case_id": "synthetic"})
    episode.step(240)
    episode.close()
    assert not container.exists()
    assert list((tmp_path / "worlds" / "worker-0").glob("episodes-*.jsonl.gz"))


def test_training_storage_retains_step_failure_and_refuses_partial_physics_snapshot(tmp_path):
    import gzip

    episode = storage_episode(tmp_path)

    def fail(action):
        raise RuntimeError("native mid-step failure")

    episode._env.step = fail
    with pytest.raises(RuntimeError, match="mid-step"):
        episode.step(240)
    with pytest.raises(ValueError, match="failed native step"):
        episode.to_dict()
    episode.close()
    receipt = json.loads(gzip.decompress((tmp_path / "episodes.jsonl.gz").read_bytes()))
    assert "native mid-step failure" in receipt["execution_error"]
    assert receipt["actions"] == [{"tick": 0, "action": 240}]
