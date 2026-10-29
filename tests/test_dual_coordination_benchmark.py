"""Software-only vector/rollout fixtures; never native qualification evidence."""

from dataclasses import dataclass
import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def module():
    try:
        return importlib.import_module("adaptive_drone_patrol.dual_coordination_benchmark")
    except ModuleNotFoundError:
        pytest.fail("Development benchmark runner is not implemented")


class ToyWorld:
    """Deliberately nonphysical short episodes for IPC and terminal GAE tests."""

    def __init__(self, worker, episode):
        self.worker, self.episode, self.ticks = worker, episode, 0
        self.terminated = False

    def observation(self):
        return dict(
            goals=np.full((10, 2), self.ticks, dtype=np.float32),
            drones=np.zeros((2, 2), dtype=np.float32),
            global_features=np.zeros(2, dtype=np.float32),
        )

    def legal_mask(self):
        result = np.zeros(242, dtype=bool)
        result[240] = True
        return result

    def step(self, action):
        assert action == 240
        self.ticks += 1
        self.terminated = self.ticks == 2
        return (
            self.observation(),
            1.0,
            self.terminated,
            dict(elapsed_simulated_s=0.2, terminal_reason="toy_end" if self.terminated else None),
        )

    def close(self):
        pass


@dataclass
class ToyFactory:
    def __call__(self, worker, episode):
        return ToyWorld(worker, episode)


def test_worker_candidates_respect_actual_cpu_quota():
    m = module()
    assert m.worker_candidates(7.68) == (1, 2, 4, 7)
    assert m.worker_candidates(2.5) == (1, 2)
    with pytest.raises(ValueError):
        m.worker_candidates(float("nan"))


def test_disposable_pilot_budget_never_becomes_production_or_selectable():
    m = module()
    b = m.DisposablePilotBudget()
    with pytest.raises(ValueError):
        b.record_update(4095, backend="native-mujoco")
    for _ in range(4):
        b.record_update(4096, backend="native-mujoco")
    assert b.transitions == 16384 and b.updates == 4 and b.selectable is False
    assert b.production_transitions == 0 and b.seed not in (42, 43, 44)
    with pytest.raises(ValueError, match="pilot"):
        b.record_update(4096, backend="native-mujoco")


def test_spawn_world_rollout_exact_remainder_and_terminal_no_bootstrap():
    m = module()
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, PolicyConfig

    p = JointActorCritic(PolicyConfig(goal_dim=2, drone_dim=2, global_dim=2, hidden=16))
    with torch.no_grad():
        for parameter in p.parameters():
            parameter.zero_()
        p.critic.bias.fill_(10.0)
    with m.ProcessWorlds(ToyFactory(), workers=3) as worlds:
        batch, receipt = m.collect_rollout(p, worlds, transitions=7)
    assert batch.actions.tolist() == [240] * 7
    assert receipt["joint_transitions"] == 7
    assert receipt["world_transition_counts"] == [3, 2, 2]
    assert receipt["completed_episodes"] == 3
    assert receipt["elapsed_simulated_seconds"] == pytest.approx(1.4)
    # Initial round indices0..2 then terminal indices3..5: terminal return=1,
    # regardless of critic value10 and reset observations.
    assert batch.returns[3:6].tolist() == [1.0, 1.0, 1.0]
    assert batch.returns[6].item() == pytest.approx(10.99)
    assert receipt["scope"] == "DEVELOPMENT_ONLY"


def test_benchmark_requires_full_three_repetitions_for_receipt():
    m = module()
    rows = [dict(joint_transitions=4096, epochs=5, total_wall_seconds=t) for t in (4.0, 3.0, 5.0)]
    result = m.benchmark_summary(rows, workers=2, device="cpu")
    assert result["median_repetition_seconds"] == 4.0
    assert result["measured_transitions"] == 12288
    assert result["production_transitions"] == 0
    assert result["schema"] != "dual-benchmark-v1"
    with pytest.raises(ValueError):
        m.benchmark_summary(rows[:2], workers=2, device="cpu")
    rows[1]["joint_transitions"] = 4095
    with pytest.raises(ValueError):
        m.benchmark_summary(rows, workers=2, device="cpu")


class FailingCloseWorld(ToyWorld):
    def close(self):
        raise OSError("replay final flush failed")


@dataclass
class FailingCloseFactory:
    def __call__(self, worker, episode):
        return FailingCloseWorld(worker, episode)


class SlowCloseWorld(ToyWorld):
    def close(self):
        import time

        time.sleep(0.15)


@dataclass
class SlowCloseFactory:
    def __call__(self, worker, episode):
        return SlowCloseWorld(worker, episode)


def test_worker_final_flush_failure_cannot_be_reported_successful():
    m = module()
    with pytest.raises(RuntimeError, match="replay final flush failed"):
        with m.ProcessWorlds(FailingCloseFactory(), workers=1):
            pass
    with pytest.raises(ValueError, match="primary failure") as caught:
        with m.ProcessWorlds(FailingCloseFactory(), workers=1):
            raise ValueError("primary failure")
    assert any("replay final flush failed" in note for note in caught.value.__notes__)


def test_slow_flush_is_waited_and_timeout_is_explicit_failure():
    m = module()
    with m.ProcessWorlds(SlowCloseFactory(), workers=1, close_timeout_seconds=2.0) as worlds:
        pass
    assert worlds.close_wall_seconds >= 0.15
    with pytest.raises(RuntimeError, match="close|flush"):
        with m.ProcessWorlds(SlowCloseFactory(), workers=1, close_timeout_seconds=0.02):
            pass


def test_close_rejects_unrelated_success_response_instead_of_flush_ack():
    m = module()

    class Connection:
        def send(self, message):
            pass

        def poll(self, timeout):
            return True

        def recv(self):
            return {"ok": True, "observation": {}}

        def close(self):
            pass

    class Process:
        exitcode = 0
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            self.alive = False

    worlds = m.ProcessWorlds.__new__(m.ProcessWorlds)
    worlds.closed = False
    worlds.close_timeout_seconds = 1.0
    worlds.connections, worlds.processes = [Connection()], [Process()]
    with pytest.raises(RuntimeError, match="acknowledgement"):
        worlds.close()


def test_factory_compiled_capture_profile_reaches_development_environment(monkeypatch, tmp_path):
    import adaptive_drone_patrol.dual_coordination_env as env_module

    # Native construction requires unavailable local MuJoCo; capture the real
    # factory boundary, including the declaration that selects development.
    monkeypatch.setattr(env_module, "DualCoordinationEnv", lambda *args, **kwargs: (args, kwargs))
    factory = module().NativeDevelopmentFactory(
        "original", str(tmp_path), ({"split": 15},), {}, native_audit_library="candidate.so"
    )
    args, options = factory(0, 0)
    assert options["native_audit_library"] == "candidate.so"
    assert options["replay_stride_ticks"] == 20
    assert options["mode"] == "development" and options["record_raw"] is False


def test_input_identity_detects_asset_library_and_controller_changes(tmp_path):
    m = module()
    original = tmp_path / "cf2x.urdf"
    original.write_text(
        '<robot><link><visual><geometry><mesh filename="part.obj"/></geometry></visual></link></robot>'
    )
    (tmp_path / "part.obj").write_text("v 0 0 0\n")
    library = tmp_path / "audit.so"
    library.write_bytes(b"library")
    library.with_suffix(".json").write_text("{}")
    factory = m.NativeDevelopmentFactory(
        str(original), str(tmp_path / "out"), ({"split": 15},), {"gain": 1}, native_audit_library=str(library)
    )
    first = m.capture_input_identity(factory)
    assert first["audit_backend"] == "compiled-native"
    assert len(first["assets"]) == 2 and first["source"]
    (tmp_path / "part.obj").write_text("v 1 0 0\n")
    assert first != m.capture_input_identity(factory)
    library.write_bytes(b"changed")
    assert first["audit_library"] != m.capture_input_identity(factory)["audit_library"]
    factory.design["gain"] = 2
    assert first["controller_sha256"] != m.capture_input_identity(factory)["controller_sha256"]


def test_machine_inventory_records_measured_quota_memory_and_storage(tmp_path):
    inventory = module().machine_inventory(tmp_path)
    assert inventory["cpu"]["effective_quota"] >= 1
    assert inventory["memory"]["physical_bytes"] > 0
    assert inventory["disk"]["free_bytes"] > 0
    assert inventory["disk"]["total_bytes"] >= inventory["disk"]["free_bytes"]
    assert inventory["dependencies"]["torch"] == torch.__version__
    assert inventory["machine_id"] and inventory["observed_at"]


def test_worker_candidates_never_silently_add_unmeasured_eight_worker_configuration():
    assert module().worker_candidates(64) == (1, 2, 4, 7)


def test_reducer_missing_full_runs_is_diagnostic_and_cannot_admit_campaign(tmp_path):
    result = module().reduce_benchmarks([], output=tmp_path / "report.json")
    assert result["schema"] == "dual-benchmark-diagnostic-v1"
    assert result["selected_backend"] is None
    assert result["storage_feasible"] is not True
    assert result["missing_evidence"]
    assert result["gpu_physics"]["qualified"] is False


def reducer_fixture(tmp_path, *, gpu_factor=0.95, replays=0):
    """Synthetic file-bound software evidence; never an actual performance run."""
    import json
    from adaptive_drone_patrol.dual_coordination_campaign import BoundFiles, EXPERIMENT_ROLES, NATIVE_CHECKS
    from adaptive_drone_patrol.util import sha256

    m = module()

    def put(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    roles = {}
    for role in EXPERIMENT_ROLES:
        path = tmp_path / f"{role}.txt"
        path.write_text(role)
        roles[role] = [path.name]
    (tmp_path / "controller.txt").write_text('{"gain":1}')
    (tmp_path / "declaration.txt").write_text('[{"split":15}]')
    bound = BoundFiles.capture(tmp_path, roles)
    binding = put("binding.json", bound.to_dict())
    profile = put(
        "profile.json",
        dict(
            schema="dual-execution-profile-v1",
            replays_per_mission=replays,
            training_updates=768,
            validation_missions=1500,
            validation_diagnostic_missions=600,
            heldout_missions=1500,
            physical_missions=127,
            checkpoint_rounds=64,
            media_items=2,
            recovery_attempts=1,
        ),
    )
    qualification = put(
        "qualification.json",
        dict(
            schema="dual-qualification-v1",
            passed=True,
            checks={key: True for key in NATIVE_CHECKS},
            safety_violations=0,
            profile="dual-payload-v1",
            version="3.15.0",
            experiment_inputs=bound.files,
            audit_backend="python-native",
            execution_profile_sha256=sha256(profile),
            compact_online_evidence_verified=True,
        ),
    )
    machine = dict(
        machine_id="synthetic-machine",
        observed_at="2026-10-28T00:00:00+00:00",
        cpu=dict(effective_quota=7),
        gpu_devices=[dict(name="synthetic-test-device")],
        disk=dict(free_bytes=10**12, total_bytes=2 * 10**12),
        memory=dict(physical_bytes=10**10),
        dependencies={},
    )
    identity = dict(
        audit_backend="python-native",
        source=bound.files["source"],
        assets=bound.files["assets"],
        controller_sha256=m._json_hash({"gain": 1}),
        cases_sha256=m._json_hash([{"split": 15}]),
        audit_library=None,
    )
    configurations = []
    for device in ("cpu", "cuda"):
        for workers in (1, 2, 4, 7):
            seconds = 70 / workers * (gpu_factor if device == "cuda" else 1)
            report = m.benchmark_summary(
                [
                    dict(
                        joint_transitions=4096,
                        epochs=5,
                        total_wall_seconds=seconds,
                        phase_timed_transitions=4096,
                        world_transition_counts=[
                            4096 // workers + int(i < 4096 % workers) for i in range(workers)
                        ],
                        world_phase_timed_counts=[
                            4096 // workers + int(i < 4096 % workers) for i in range(workers)
                        ],
                        summed_worker_execution_seconds=4.5,
                        summed_worker_phase_seconds={
                            key: (4 if key == "total_step" else 1) for key in m.PHASE_TIMINGS
                        },
                    )
                ]
                * 3,
                workers=workers,
                device=device,
            )
            checkpoint = tmp_path / f"{device}-{workers}-current.pt"
            checkpoint.write_bytes(b"x" * 1000)
            terminal = tmp_path / f"{device}-{workers}-terminal.pt"
            import os

            os.link(checkpoint, terminal)
            checkpoints = [
                dict(
                    path=str(checkpoint),
                    sha256=sha256(checkpoint),
                    bytes=1000,
                    wall_seconds=0.1,
                    prior_bytes=1000 if i else 0,
                    atomic_peak_bytes=2000 if i else 1000,
                    readback_verified=True,
                    full_rollout_transitions=4096,
                    retained=i == 2,
                )
                for i in range(3)
            ]
            report.update(
                input_identity=identity,
                experiment_inputs=bound.files,
                machine_before=machine,
                machine_after=machine,
                final_flush_wall_seconds=0.01,
                total_configuration_wall_seconds=seconds * 4,
                checkpoint_measurements=checkpoints,
                checkpoint_storage_probe=dict(
                    schema="dual-checkpoint-storage-probe-v1",
                    overwrite_verified=True,
                    terminal_hardlink_verified=True,
                    unique_current_payloads=1,
                    full_rollout_transitions=4096,
                    max_atomic_bytes=2000,
                    terminal=m._file_record(terminal),
                ),
                peak_storage_bytes=10000,
                completed=True,
                initialization_wall_seconds=0.2,
            )
            configurations.append(put(f"{device}-{workers}.json", report))
    envelope_path = tmp_path / "synthetic-unit-storage-envelope.pt"
    envelope_path.write_bytes(b"x" * 2000)
    envelope = dict(
        path=envelope_path.name,
        sha256=sha256(envelope_path),
        bytes=2000,
        scope="DECLARED_SHAPE_SIZE_PROBE",
        full_rollout_transitions=4096,
        max_curve_rows=256,
        max_assignment_rows=1048576 + 21,
        atomic_write_seconds=0.00001,
    )
    samples = []
    for kind in (
        "training_archive",
        "validation_mission",
        "heldout_mission",
        "physical_mission",
        "qualification",
        "media",
        "recovery",
    ):
        for index in range(3):
            path = put(
                f"{kind}-{index}.json",
                dict(
                    schema="dual-benchmark-cost-sample-v1",
                    kind=kind,
                    units=4096 if kind == "training_archive" else 1,
                    wall_seconds=0.001 if kind == "training_archive" else 2,
                    checkpoint_envelope=envelope if kind == "training_archive" else None,
                    retained_bytes=100,
                    peak_storage_bytes=1000,
                    actual_execution=True,
                    measurement_cohort="development",
                    experiment_inputs=bound.files,
                    machine_id="synthetic-machine",
                    execution_profile_sha256=sha256(profile),
                ),
            )
            samples.append(dict(path=path.name, sha256=sha256(path), bytes=path.stat().st_size))
    costs = put("costs.json", dict(schema="dual-benchmark-costs-v1", samples=samples))
    return dict(
        configuration_paths=configurations,
        experiment_binding=binding,
        qualification_path=qualification,
        execution_profile_path=profile,
        cost_measurements_path=costs,
        output=tmp_path / "reduced.json",
    )


def test_reducer_selects_cpu_within_ten_percent_and_accounts_whole_campaign(tmp_path):
    result = module().reduce_benchmarks(**reducer_fixture(tmp_path))
    assert result["schema"] == "dual-benchmark-v1"
    assert result["selected_backend"] == "native-cpu" and result["selected_workers"] == 7
    # 768 updates*10 + 3727 missions*2 + qual2 + media4 + recovery2 +
    # 8577 checkpoint writes*.1 + 3 seed initializations/flushes*.21 + all pilots1033.5 + archive .768.
    assert result["whole_campaign_projection_seconds"] == pytest.approx(17034.68377)
    assert result["cost_components"]["validation_mission"]["units"] == 2100
    assert result["execution_profile"]["validation_missions"] == 1500
    assert result["execution_profile"]["validation_diagnostic_missions"] == 600
    assert result["storage_feasible"] is True
    assert result["execution_profile"]["replays_per_mission"] == 0
    assert result["gpu_physics"]["qualified"] is False


def test_reducer_selects_cuda_only_with_more_than_ten_percent_whole_campaign_gain(tmp_path):
    result = module().reduce_benchmarks(**reducer_fixture(tmp_path, gpu_factor=0.5))
    assert result["selected_backend"] == "native-cuda" and result["selected_workers"] == 7
    assert result["whole_campaign_speedup"] > 1.1


@pytest.mark.parametrize(
    "damage",
    [
        "partial_run",
        "asset_mismatch",
        "missing_cuda",
        "profile_unqualified",
        "cost_tampered",
        "replay_unmeasured",
        "disk_shortage",
    ],
)
def test_reducer_fails_closed_on_missing_or_mismatched_evidence(tmp_path, damage):
    import json

    args = reducer_fixture(tmp_path, replays=1 if damage == "replay_unmeasured" else 0)

    def edit(path, change):
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    if damage == "partial_run":
        edit(args["configuration_paths"][0], lambda x: x["measurements"].pop())
    elif damage == "asset_mismatch":
        edit(
            args["configuration_paths"][0], lambda x: x["input_identity"]["assets"][0].update(sha256="f" * 64)
        )
    elif damage == "missing_cuda":
        args["configuration_paths"] = args["configuration_paths"][:4]
    elif damage == "profile_unqualified":
        edit(args["qualification_path"], lambda x: x.update(execution_profile_sha256="f" * 64))
    elif damage == "cost_tampered":
        (tmp_path / "media-0.json").write_text("{}")
    elif damage == "disk_shortage":
        for path in args["configuration_paths"]:
            edit(path, lambda x: x["machine_after"]["disk"].update(free_bytes=1))
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-diagnostic-v1"
    assert result["selected_backend"] is None
    assert result["missing_evidence"]


class CheckpointWorld(ToyWorld):
    def to_dict(self):
        return dict(worker=self.worker, episode=self.episode, ticks=self.ticks, flushed=True)


@dataclass
class CheckpointFactory:
    def __call__(self, worker, episode):
        return CheckpointWorld(worker, episode)


def test_worker_checkpoint_captures_actual_current_state_and_rng():
    m = module()
    with m.ProcessWorlds(CheckpointFactory(), workers=1) as worlds:
        worlds.step([0], [240])
        snapshots = worlds.checkpoint()
    assert snapshots[0]["state"] == dict(worker=0, episode=0, ticks=1, flushed=True)
    assert snapshots[0]["python_rng"] and snapshots[0]["numpy_rng"]
    assert isinstance(snapshots[0]["torch_rng"], bytes) and len(snapshots[0]["torch_rng"]) == len(
        torch.get_rng_state()
    )
    assert worlds.worker_peak_rss_bytes[0] > 0


def test_reduce_cli_produces_diagnostic_without_starting_experiments(tmp_path):
    result = module().main(["reduce", "--output", str(tmp_path / "diagnostic.json")])
    assert result["schema"] == "dual-benchmark-diagnostic-v1"


def test_run_rejects_requested_quota_above_actual_before_spawning(tmp_path):
    m = module()
    original = tmp_path / "model.urdf"
    original.write_text("<robot/>")
    factory = m.NativeDevelopmentFactory(str(original), str(tmp_path / "worlds"), ({"split": 15},), {})
    quota = m.machine_inventory(tmp_path)["cpu"]["effective_quota"]
    with pytest.raises(ValueError, match="measured"):
        m.run_configuration(
            factory,
            workers=1,
            device="cpu",
            output=tmp_path / "out",
            journal_root=tmp_path / "journal",
            cpu_quota=quota + 1,
        )
    assert (tmp_path / "out/failure.json").is_file()
    assert not (tmp_path / "worlds").exists()


def test_reducer_rejects_checkpoint_sample_bytes_without_actual_checkpoint(tmp_path):
    import json

    args = reducer_fixture(tmp_path)
    path = args["configuration_paths"][0]
    value = json.loads(path.read_text())
    value["checkpoint_measurements"][-1]["sha256"] = "f" * 64
    path.write_text(json.dumps(value))
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-diagnostic-v1"


@pytest.mark.parametrize("damage", ["dependencies", "future_cohort"])
def test_reducer_rejects_dependency_drift_and_nonprospective_cost_samples(tmp_path, damage):
    import json
    from adaptive_drone_patrol.util import sha256

    args = reducer_fixture(tmp_path)
    if damage == "dependencies":
        path = args["configuration_paths"][1]
        value = json.loads(path.read_text())
        value["machine_after"]["dependencies"] = {"torch": "changed"}
        path.write_text(json.dumps(value))
    else:
        path = tmp_path / "heldout_mission-0.json"
        value = json.loads(path.read_text())
        value["measurement_cohort"] = "heldout"
        path.write_text(json.dumps(value))
        costs = json.loads(args["cost_measurements_path"].read_text())
        for row in costs["samples"]:
            if row["path"] == path.name:
                row.update(sha256=sha256(path), bytes=path.stat().st_size)
        args["cost_measurements_path"].write_text(json.dumps(costs))
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-diagnostic-v1"


def test_reducer_rejects_missing_worker_phase_timing_coverage(tmp_path):
    import json

    args = reducer_fixture(tmp_path)
    path = args["configuration_paths"][0]
    value = json.loads(path.read_text())
    value["measurements"][0].pop("phase_timed_transitions", None)
    path.write_text(json.dumps(value))
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-diagnostic-v1"


class TimedToyWorld(ToyWorld):
    def step(self, action):
        import time

        time.sleep(0.011)  # actual outer duration must contain the illustrative ten-ms phases
        observation, reward, terminal, info = super().step(action)
        info["timings"] = dict(
            controller_reference_supervisor=0.001,
            physics_audit=0.002,
            mask_observation=0.003,
            record_io=0.004,
            total_step=0.01,
        )
        return observation, reward, terminal, info


@dataclass
class TimedToyFactory:
    def __call__(self, worker, episode):
        return TimedToyWorld(worker, episode)


def test_rollout_preserves_phase_coverage_without_adding_overlapping_worker_sums():
    m = module()
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, PolicyConfig

    policy = JointActorCritic(PolicyConfig(goal_dim=2, drone_dim=2, global_dim=2, hidden=16))
    with m.ProcessWorlds(TimedToyFactory(), workers=1) as worlds:
        _, receipt = m.collect_rollout(policy, worlds, transitions=2)
    assert receipt["phase_timed_transitions"] == 2
    assert receipt["summed_worker_phase_seconds"] == pytest.approx(
        dict(
            controller_reference_supervisor=0.002,
            physics_audit=0.004,
            mask_observation=0.006,
            record_io=0.008,
            total_step=0.02,
        )
    )


@pytest.mark.parametrize("damage", ["float_count", "missing_worker", "sum_mismatch", "negative", "nonfinite"])
def test_reducer_rejects_incomplete_or_inconsistent_phase_measurements(tmp_path, damage):
    import json

    args = reducer_fixture(tmp_path)
    path = args["configuration_paths"][0]
    data = json.loads(path.read_text())
    row = data["measurements"][0]
    if damage == "float_count":
        row["phase_timed_transitions"] = 4096.0
    elif damage == "missing_worker":
        row["world_phase_timed_counts"] = [0]
    elif damage == "sum_mismatch":
        row["summed_worker_phase_seconds"]["total_step"] = 0.1
    elif damage == "negative":
        row["summed_worker_phase_seconds"]["record_io"] = -1
    else:
        row["summed_worker_phase_seconds"]["total_step"] = float("nan")
    path.write_text(json.dumps(data))
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-diagnostic-v1"


def test_strict_rollout_rejects_first_missing_phase_and_keeps_recovery_rows():
    m = module()
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, PolicyConfig

    p = JointActorCritic(PolicyConfig(goal_dim=2, drone_dim=2, global_dim=2, hidden=16))
    with m.ProcessWorlds(ToyFactory(), workers=1) as worlds:
        with pytest.raises(ValueError, match="phase"):
            m.collect_rollout(p, worlds, transitions=2, require_phase_timings=True)
    rows = []
    with m.ProcessWorlds(TimedToyFactory(), workers=1) as worlds:
        _, receipt = m.collect_rollout(
            p, worlds, transitions=2, require_phase_timings=True, recovery_rows=rows
        )
    assert receipt["world_phase_timed_counts"] == [2]
    assert len(rows) == 2 and rows[1]["terminal"] is True
    assert rows[0]["worker"] == 0 and rows[0]["action"] == 240


def test_checkpoint_write_schedule_separates_write_count_from_retained_files():
    m = module()
    assert m.training_checkpoint_writes(workers=7) == 8577
    assert m.training_checkpoint_writes(workers=1) == 50697


def test_storage_does_not_claim_all_writes_retained_and_missing_archive_measurement_is_pending(tmp_path):
    import json

    args = reducer_fixture(tmp_path)
    costs = json.loads(args["cost_measurements_path"].read_text())
    costs["samples"] = [r for r in costs["samples"] if not r["path"].startswith("training_archive")]
    args["cost_measurements_path"].write_text(json.dumps(costs))
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-diagnostic-v1"
    assert result["storage_feasible"] is None
    assert any("training" in message for message in result["missing_evidence"])


def test_checkpoint_probe_overwrites_current_and_verifies_full_rollout(tmp_path):
    m = module()
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, JointPPO, PolicyConfig

    trainer = JointPPO(
        JointActorCritic(PolicyConfig(goal_dim=2, drone_dim=2, global_dim=2, hidden=16)), seed=42
    )
    trainer.budget = m.DisposablePilotBudget()
    sum(p.sum() for p in trainer.policy.parameters()).backward()
    trainer.optimizer.step()
    rows = [
        dict(
            worker=0,
            observation={"x": np.array([i], dtype=np.float32)},
            mask=np.ones(242, dtype=bool),
            action=240,
            log_prob=0.0,
            value=0.0,
            reward=0.0,
            terminal=False,
            elapsed=0.2,
        )
        for i in range(4096)
    ]

    class Worlds:
        workers = 1

        def checkpoint(self):
            return [dict(state={"qpos": [1.0, 2.0]}, torch_rng=b"unit-fixture")]

    path = tmp_path / "current-size-probe.pt"
    samples = [m.measure_checkpoint(trainer, Worlds(), rows, path) for _ in range(3)]
    assert len(list(tmp_path.glob("*.pt"))) == 1
    assert samples[0]["prior_bytes"] == 0 and samples[1]["prior_bytes"] == samples[0]["bytes"]
    assert samples[1]["atomic_peak_bytes"] == samples[1]["prior_bytes"] + samples[1]["bytes"]
    assert all(s["readback_verified"] and s["full_rollout_transitions"] == 4096 for s in samples)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert not saved["physical_resume_available"] and not saved["selectable"]
    assert len(saved["environment_state"]["partial_rollout"]["rows"]) == 4096
    assert saved["optimizer"]["state"]
    assert saved["environment_state"]["workers"][0]["state"]["qpos"] == [1.0, 2.0]


@pytest.mark.parametrize("damage", ["artifact", "shape"])
def test_reducer_requires_measured_checkpoint_history_envelope(tmp_path, damage):
    import json
    from adaptive_drone_patrol.util import sha256

    args = reducer_fixture(tmp_path)
    if damage == "artifact":
        (tmp_path / "synthetic-unit-storage-envelope.pt").write_bytes(b"tampered")
    else:
        path = tmp_path / "training_archive-0.json"
        data = json.loads(path.read_text())
        data["checkpoint_envelope"]["max_assignment_rows"] = 1
        path.write_text(json.dumps(data))
        costs = json.loads(args["cost_measurements_path"].read_text())
        for row in costs["samples"]:
            if row["path"] == path.name:
                row.update(sha256=sha256(path), bytes=path.stat().st_size)
        args["cost_measurements_path"].write_text(json.dumps(costs))
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-diagnostic-v1"
    assert result["storage_feasible"] is None


@pytest.mark.parametrize("count", [None, 0, 599, 601, 600.0])
def test_reducer_requires_exact_validation_diagnostic_count(tmp_path, count):
    import json
    from adaptive_drone_patrol.util import sha256

    args = reducer_fixture(tmp_path)
    profile = json.loads(args["execution_profile_path"].read_text())
    if count is None:
        profile.pop("validation_diagnostic_missions")
    else:
        profile["validation_diagnostic_missions"] = count
    args["execution_profile_path"].write_text(json.dumps(profile))
    qualified = json.loads(args["qualification_path"].read_text())
    qualified["execution_profile_sha256"] = sha256(args["execution_profile_path"])
    args["qualification_path"].write_text(json.dumps(qualified))
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-diagnostic-v1"
    assert any("validation_diagnostic_missions" in reason for reason in result["missing_evidence"])


def cpu_only_restriction(args):
    import json
    from adaptive_drone_patrol.util import sha256

    path = args["output"].parent / "cpu-only-direction.json"
    path.write_text(
        json.dumps(
            dict(
                schema="dual-backend-restriction-v1",
                decision="cpu-only",
                source="user",
                authorization="Proceed with remaining training using CPU only",
                experiment_binding_sha256=sha256(args["experiment_binding"]),
            )
        )
    )
    args["backend_restriction_path"] = path
    args["configuration_paths"] = [
        p for p in args["configuration_paths"] if json.loads(p.read_text())["policy_device"] == "cpu"
    ]
    return path


def test_explicit_cpu_only_direction_keeps_gpu_inventory_and_admits_all_cpu_candidates(tmp_path):
    import json

    args = reducer_fixture(tmp_path)
    restriction = cpu_only_restriction(args)
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-v1", result["missing_evidence"]
    assert result["selected_backend"] == "native-cpu" and result["selected_workers"] == 7
    assert result["backend_restriction"]["decision"] == "cpu-only"
    assert (
        result["gpu_selection_note"]
        == "CPU-only continuation requested by user; GPU physics speedup unmeasured"
    )
    assert all(
        json.loads(p.read_text())["machine_before"]["gpu_devices"] for p in args["configuration_paths"]
    )
    assert any(x["path"] == str(restriction.resolve()) for x in result["evidence"])


@pytest.mark.parametrize(
    "damage", ["binding", "source", "decision", "authorization", "missing_worker", "gpu_candidate"]
)
def test_cpu_only_direction_cannot_bypass_binding_or_complete_cpu_measurements(tmp_path, damage):
    import json

    args = reducer_fixture(tmp_path)
    gpu = args["configuration_paths"][-1]
    restriction = cpu_only_restriction(args)
    if damage == "missing_worker":
        args["configuration_paths"] = args["configuration_paths"][:-1]
    elif damage == "gpu_candidate":
        args["configuration_paths"].append(gpu)
    else:
        d = json.loads(restriction.read_text())
        key = {"binding": "experiment_binding_sha256"}.get(damage, damage)
        d[key] = {"source": "model", "decision": "gpu", "authorization": ""}.get(damage, "0" * 64)
        restriction.write_text(json.dumps(d))
    result = module().reduce_benchmarks(**args)
    assert result["schema"] == "dual-benchmark-diagnostic-v1"
    assert result["selected_backend"] is None and result["missing_evidence"]
