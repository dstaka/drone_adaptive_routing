"""Synthetic evaluator contract tests; none are physical qualification evidence."""

import importlib
import json
from types import SimpleNamespace
import numpy as np
import pytest


def module():
    try:
        return importlib.import_module("adaptive_drone_patrol.dual_coordination_evaluate")
    except ModuleNotFoundError:
        pytest.fail("Validation evaluator is not implemented")


def mission_state():
    return dict(
        tick=1200,
        terminal_reason="success",
        completed_counts=[5, 5],
        reward_components={},
        team_completion_tick=1200,
        event_receipts=[{"event": {"kind": "availability-delay"}, "observed_tick": 600}],
        action_receipts=[{"tick": 0, "owners": [0] + [None] * 9}, {"tick": 620, "owners": [1] + [None] * 9}],
        goals=[{"status": "DONE", "owner": 1, "completed_tick": 1000}]
        + [{"status": "DONE", "owner": 0, "completed_tick": 1100} for _ in range(9)],
    )


def test_useful_transfer_requires_post_observation_change_and_new_owner_service():
    m = module()
    state = mission_state()
    assert m.useful_transfer(state)
    state["action_receipts"][1]["tick"] = 580
    assert not m.useful_transfer(state)
    state = mission_state()
    state["goals"][0]["owner"] = 0
    assert not m.useful_transfer(state)
    state = mission_state()
    state["goals"][0]["completed_tick"] = 600
    assert not m.useful_transfer(state)


def report(tick=20):
    return dict(
        control_steps=tick,
        rk_states=tick * 64,
        native_poses=1 + tick * 16,
        reconstructed_intervals=tick * 16,
        body_reconstructed_intervals=tick * 16,
        errors=[],
        proof_gaps=[],
        violation_count=0,
        violations=[],
        reconstruction_failures=[],
    )


def test_audit_gap_or_missing_internal_interval_cannot_be_promoted():
    m = module()
    result = {"identity": {"test": "synthetic"}, "terminal_tick": 20, "outcome": {"test": "synthetic"}}
    good = m.build_audit_receipt(result, report(), auditor_hash="synthetic")
    assert good["complete"] and good["coverage"]["rk_states"] == 1280
    bad = report()
    bad["proof_gaps"] = ["unproved continuous interval"]
    assert not m.build_audit_receipt(result, bad, auditor_hash="synthetic")["complete"]
    bad = report()
    bad["native_poses"] -= 1
    receipt = m.build_audit_receipt(result, bad, auditor_hash="synthetic")
    assert not receipt["complete"] and receipt["coverage"]["native_substep_intervals"] == 319


def test_replay_mismatch_never_becomes_passed():
    m = module()
    capture = dict(
        actions=[{"tick": 0, "action": 240, "mask": [False] * 240 + [True, False]}],
        terminal_tick=20,
        mission={"terminal_reason": "failure"},
        outcome={"success": False},
        final_backend={"qpos": [1.0], "qvel": [0.0]},
    )
    assert m.compare_replays(capture, capture)["passed"]
    changed = {**capture, "actions": [{"tick": 0, "action": 241, "mask": [False] * 240 + [True, False]}]}
    assert not m.compare_replays(capture, changed)["passed"]
    changed = {**capture, "final_backend": {"qpos": [1.001], "qvel": [0.0]}}
    assert not m.compare_replays(capture, changed)["metrics_within_tolerance"]


def test_failed_mission_score_and_intervention_denominator_are_explicit():
    m = module()
    state = mission_state()
    state["terminal_reason"] = "team_timeout"
    state["tick"] = 30000
    case = {"case_id": "test", "scenario": "availability-delay"}
    outcome = m.mission_outcome(
        case,
        state,
        decision_intervals=7,
        intervened_intervals=2,
        held_seconds=0.4,
        safety_violations=0,
        invalid_actions=0,
    )
    assert outcome["success"] is False and outcome["failure_reason"] == "team_timeout"
    assert m.time_score(outcome) == 600.0
    assert outcome["decision_intervals"] == 7 and outcome["intervened_intervals"] == 2


def test_missing_learning_lock_rejected_before_any_output(tmp_path):
    m = module()
    with pytest.raises((ValueError, FileNotFoundError)):
        m.run_validation(
            learning_lock=tmp_path / "missing",
            terminals={42: "a", 43: "b", 44: "c"},
            original="missing",
            design_path="missing",
            output=tmp_path / "new",
        )
    assert not (tmp_path / "new").exists()


def test_proposal_is_durable_even_when_step_raises(tmp_path):
    m = module()

    class Env:
        terminated = False
        mission = SimpleNamespace(tick=0)

        def legal_mask(self):
            return np.array([False] * 240 + [True, False])

        def observation(self):
            return {}

        def step(self, action):
            raise OSError("native capture failed")

    path = tmp_path / "actions.jsonl"
    with pytest.raises(OSError, match="native capture"):
        m.propose_and_step(Env(), lambda env: 240, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["action"] == 240 and rows[0]["tick"] == 0 and rows[0]["mask"][240]


class SyntheticEnv:
    """Finite nonphysical environment for persistence/error-flow tests only."""

    def __init__(self, directory, *, close_error=False):
        self.output = directory
        directory.mkdir(parents=True)
        self.terminated = False
        self.close_error = close_error
        self.state = mission_state()
        self.state.update(tick=0, terminal_reason=None)
        self.mission = SimpleNamespace(tick=0, to_dict=lambda: {"state": self.state})
        self.fixture = SimpleNamespace(sample=lambda: {"qpos": [1.0], "qvel": [0.0], "drones": []})
        self.online_auditor = SimpleNamespace(report=lambda: report(self.mission.tick))
        self.replay_segments = []
        self.ledger = SimpleNamespace(failures=[])

    def legal_mask(self):
        return np.array([False] * 240 + [True, False])

    def step(self, action):
        self.mission.tick = 20
        self.state.update(tick=20, terminal_reason="success")
        self.terminated = True
        return {}, 0.0, True, {"supervisor_held_drone_seconds": 0.2, "supervisor_reason": "hold"}

    def close(self):
        if self.close_error:
            raise OSError("final replay flush")


def test_capture_counts_actual_intervention_and_preserves_close_failure(tmp_path):
    m = module()
    case = {"case_id": "synthetic", "scenario": "normal"}
    capture = m.capture_mission(SyntheticEnv(tmp_path / "a"), lambda env: 240, case, tmp_path / "a")
    assert capture["outcome"]["decision_intervals"] == 1
    assert capture["outcome"]["intervened_intervals"] == 1
    assert capture["outcome"]["supervisor_held_drone_seconds"] == 0.2
    assert capture["infrastructure_errors"] == []
    failed = m.capture_mission(
        SyntheticEnv(tmp_path / "b", close_error=True), lambda env: 240, case, tmp_path / "b"
    )
    assert failed["outcome"]["success"] is False
    assert "final replay flush" in failed["infrastructure_errors"][0]
    assert json.loads((tmp_path / "b" / "capture.json").read_text()) == failed


def test_exact_action_replay_checks_masks_and_decision_ticks():
    m = module()
    env = SimpleNamespace(mission=SimpleNamespace(tick=0), legal_mask=lambda: np.array([True, False]))
    replay = m.ActionReplay([{"tick": 0, "action": 0, "mask": [True, False]}])
    assert replay(env) == 0 and replay.exhausted
    with pytest.raises(ValueError, match="extra"):
        replay(env)
    replay = m.ActionReplay([{"tick": 20, "action": 0, "mask": [True, False]}])
    with pytest.raises(ValueError, match="tick"):
        replay(env)


def test_dynamic_predictor_uses_observed_references_and_measured_envelope():
    m = module()
    from adaptive_drone_patrol.dual_coordination_reference import ContinuousReference
    from adaptive_drone_patrol.dual_coordination_baselines import ConflictRequest

    refs = (ContinuousReference([0, 0, 2]), ContinuousReference([4, 0, 2]))
    rows = [
        {"drone_xyz": [0, 0, 2], "drone_velocity": [0, 0, 0]},
        {"drone_xyz": [4, 0, 2], "drone_velocity": [0, 0, 0]},
    ]
    calls = []

    def predict(references, now, horizon, radii):
        calls.append((references, now, horizon, radii))
        return True, 0.1

    env = SimpleNamespace(
        fixture=SimpleNamespace(sample=lambda: {"drones": rows}),
        mission=SimpleNamespace(targets=[10, 10]),
        supervisor=SimpleNamespace(assembly_radius=0.4, tracking_inflation=0.15, predict=predict),
    )
    request = ConflictRequest((0, 10), refs, ((0, 0, 2), (4, 0, 2)), ((1, 0, 2),) * 10, 0.0, (0.0, 2.0))
    result = m.causal_conflict_predictor(env)(request)
    assert result.pair_conflict and result.unavailable_hover_conflict == (True, False)
    assert calls[0][3] == [0.55, 0.55]
    # A held same-target reference must predict its resumed intention, not hover.
    env.mission.targets[0] = 0
    m.causal_conflict_predictor(env)(request)
    assert calls[-1][0][0].target == (1.0, 0.0, 2.0)


def test_bundle_verifies_raw_checksums_and_never_accepts_missing_reset(tmp_path):
    m = module()
    from adaptive_drone_patrol.dual_coordination_raw import RawWriter
    from adaptive_drone_patrol.util import sha256

    root = tmp_path / "capture"
    root.mkdir()
    path = root / "replay.draw"
    writer = RawWriter(path)
    writer.append({"kind": "RESET", "tick": 0, "actual": {"schema": "dual-raw-v1", "world_time": 0.0}})
    writer.append({"kind": "STEP", "tick": 1, "actual": {"schema": "dual-raw-v1", "world_time": 0.01}})
    writer.close()
    for name in ("reset-inputs.json", "capture.json"):
        (root / name).write_text("{}")
    (root / "proposed-actions.jsonl").write_text("")
    capture = {"replay_segments": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}]}
    bundle = tmp_path / "bundle.zip"
    receipt = m.bundle_captures(bundle, [(root, capture)], root=tmp_path)
    assert receipt["reset_records"] == [1] and receipt["record"]["sha256"] == sha256(bundle)
    path.write_bytes(path.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        m.bundle_captures(tmp_path / "tampered.zip", [(root, capture)], root=tmp_path)


def test_missing_third_terminal_rejected_before_output_even_with_valid_capability(tmp_path, monkeypatch):
    m = module()
    called = []
    fake = SimpleNamespace(require_current=lambda: called.append("verified"))
    monkeypatch.setattr(m.VerifiedLearningLock, "load", lambda path: fake)
    with pytest.raises(ValueError, match="42.*43.*44"):
        m.run_validation(
            learning_lock="synthetic",
            terminals={42: "a", 43: "b"},
            original="a",
            design_path="b",
            output=tmp_path / "out",
        )
    assert called == ["verified"] and not (tmp_path / "out").exists()


def test_partial_campaign_never_selects_and_records_expected_denominator(tmp_path):
    m = module()
    declaration = {"case_sha256": {str(i): str(i) for i in range(300)}}
    summary = m.aggregate_validation(tmp_path, declaration, root=tmp_path)
    assert summary["status"] == "partial" and summary["expected_missions"] == 1500
    assert summary["retained_missions"] == 0 and not (tmp_path / "selection.json").exists()


@pytest.mark.parametrize("version", [1, 2])
def test_full_validation_aggregation_binds_all_receipts_and_rejects_tamper(tmp_path, version):
    m = module()
    from adaptive_drone_patrol.util import sha256

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    bundle = evidence / "synthetic.zip"
    bundle.write_bytes(b"SYNTHETIC-NOT-PHYSICAL")
    record = {"path": "evidence/synthetic.zip", "bytes": bundle.stat().st_size, "sha256": sha256(bundle)}
    declaration = {
        "case_sha256": {str(i): str(i) for i in range(300)},
        "checkpoint_sha256": {str(s): str(s) for s in (42, 43, 44)},
        "baseline_sha256": "baseline",
        "experiment_inputs_sha256": "inputs",
        "auditor_sha256": "audit",
        "configuration_sha256": "config",
    }
    if version == 2:
        declaration["development_replay_reference"] = {"path": "dev.json", "bytes": 1, "sha256": "synthetic"}
    missions = []
    for method in m.METHODS:
        for i in range(300):
            outcome = dict(
                case_id=str(i),
                scenario=("normal", "priority", "availability-delay")[i // 100],
                success=True,
                team_time=100.0 if method.startswith("N") else 120.0,
                safety_violations=0,
                invalid_actions=0,
                duplicate_service=0,
                decision_intervals=600,
                intervened_intervals=0,
                supervisor_held_drone_seconds=0.0,
                useful_post_event_transfer=False,
                failure_reason=None,
            )
            identity = dict(
                method=method,
                case_id=str(i),
                case_sha256=str(i),
                model_sha256=method[1:] if method.startswith("N") else "baseline",
                experiment_inputs_sha256="inputs",
            )
            result = dict(
                schema="dual-validation-result-v1",
                identity=identity,
                outcome=outcome,
                terminal_tick=int(outcome["team_time"] * 100),
            )
            audit = m.build_audit_receipt(result, report(result["terminal_tick"]), auditor_hash="audit")
            inventory = dict.fromkeys(("reset", "actions", "compact_replay"), record)
            replay = dict(
                schema="dual-validation-replay-v1",
                identity=identity,
                result_sha256=m._digest(result),
                audit_sha256=m._digest(audit),
                inventory=inventory,
                inventory_sha256=m._digest(inventory),
                configuration_sha256="config",
                passed=True,
                decisions_match=True,
                terminal_match=True,
                metrics_within_tolerance=True,
                control_steps=result["terminal_tick"],
                reset_records=1,
                outcome=outcome,
            )
            if version == 2:
                for key in ("passed", "decisions_match", "terminal_match", "metrics_within_tolerance"):
                    del replay[key]
                replay.update(
                    schema="dual-validation-replay-v2",
                    execution_mode="retained-input-stream-validation",
                    native_reexecution_performed=False,
                    development_replay_reference=declaration["development_replay_reference"],
                    retained_inputs_verified=True,
                    compact_boundary_consistency=True,
                    errors=[],
                    action_records=outcome["decision_intervals"],
                    retained_state_records=1201,
                    reset_case_sha256=str(i),
                )
            missions.append(dict(result=result, audit=audit, replay=replay))
    path = evidence / "synthetic.json"
    path.write_text(json.dumps({"schema": "dual-validation-evidence-v1", "missions": missions}))
    summary = m.aggregate_validation(tmp_path, declaration, root=tmp_path)
    assert summary["status"] == "complete" and summary["retained_missions"] == 1500
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["schema"] == "dual-selection-bound-v1" and selection["selected_seed"] == 42
    assert len(selection["candidates"]["44"]) == 300
    bundle.write_bytes(b"TAMPERED")
    rejected = m.aggregate_validation(tmp_path, declaration, root=tmp_path)
    assert rejected["status"] == "partial" and rejected["errors"]
    assert not (tmp_path / "selection.json").exists()


def test_terminal_recovery_from_another_learning_lock_is_rejected_without_rng_change(tmp_path):
    import torch
    import random

    m = module()
    checkpoint = tmp_path / "terminal.pt"
    torch.save(
        {
            "environment_state": {
                "schema": "dual-training-recovery-v1",
                "binding": {"learning_lock_sha256": "old"},
            }
        },
        checkpoint,
    )
    original = tmp_path / "original"
    original.write_text("synthetic")
    controller = tmp_path / "controller"
    controller.write_text("synthetic")
    lock = SimpleNamespace(
        digest="new",
        selected_backend="native-cpu",
        receipt={"benchmark": {"selected_workers": 1, "selected_backend": "native-cpu"}},
        binding=SimpleNamespace(
            files={
                "declaration": [{"path": "training.jsonl", "sha256": "manifest"}],
                "assets": [
                    {"path": "cf2x.urdf", "sha256": module().file_record(original, tmp_path)["sha256"]}
                ],
                "controller": [
                    {"path": "controller", "sha256": module().file_record(controller, tmp_path)["sha256"]}
                ],
            }
        ),
    )
    before = torch.get_rng_state().clone()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    with pytest.raises(ValueError, match="binding differs"):
        m.verify_terminal_recovery(checkpoint, lock, seed=42, original=original, design_path=controller)
    assert torch.equal(before, torch.get_rng_state()) and random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])


@pytest.mark.parametrize("reexecute", [False, True])
def test_guarded_single_slot_runs_declared_captures_and_emits_bound_inventory(
    tmp_path, monkeypatch, reexecute
):
    m = module()
    import adaptive_drone_patrol.dual_coordination_env as env_module
    from adaptive_drone_patrol.dual_coordination_raw import RawWriter
    from adaptive_drone_patrol.util import sha256

    inputs = tmp_path / "input"
    inputs.mkdir()
    original = inputs / "original"
    original.write_text("synthetic")
    design = inputs / "design"
    design.write_text("{}")
    terminals = {}
    for seed in (42, 43, 44):
        terminals[seed] = inputs / str(seed)
        terminals[seed].write_text(str(seed))
    case = {"case_id": "synthetic", "scenario": "normal"}
    checks = []
    import adaptive_drone_patrol.dual_coordination_campaign as campaign

    reference = {"path": "development.json", "bytes": 1, "sha256": "synthetic"}
    monkeypatch.setattr(
        campaign,
        "verified_development_replay_reference",
        lambda bound, qualification: reference,
        raising=False,
    )
    lock = SimpleNamespace(
        digest="synthetic",
        receipt={"qualification": {}},
        selected_backend="native-cpu",
        require_current=lambda: checks.append("current"),
        verify_case=lambda c, mode: checks.append(mode),
        binding=SimpleNamespace(root=tmp_path, files={r: [] for r in m.EXPERIMENT_ROLES}),
    )
    monkeypatch.setattr(m.VerifiedLearningLock, "load", lambda path: lock)
    monkeypatch.setattr(
        m,
        "_validate_terminal_checkpoint",
        lambda path, seed: dict(
            sha256=sha256(path), transitions=1048576, updates=256, stage3_transitions=65536
        ),
    )
    monkeypatch.setattr(m, "verify_terminal_recovery", lambda *args, **kwargs: None)
    monkeypatch.setattr(m, "_bound_file", lambda lock, role, path: Path(path))
    monkeypatch.setattr(m, "_validation_cases", lambda bound: {case["case_id"]: case})
    monkeypatch.setattr(m, "_selector", lambda *args: lambda env: 240)
    from pathlib import Path

    admissions = []

    class CaptureEnv(SyntheticEnv):
        def __init__(self, original, output, case, design, **kwargs):
            admissions.append(kwargs)
            super().__init__(output)
            self.writer = RawWriter(output / "replay.draw")
            self.writer.append(
                {
                    "kind": "RESET",
                    "tick": 0,
                    "actual": {"schema": "dual-raw-v1", "world_time": 0.0, "qpos": [1.0], "qvel": [0.0]},
                    "case_id": "synthetic",
                    "replay_stride_ticks": 20,
                }
            )

        def close(self):
            for tick in (1, 20):
                self.writer.append(
                    {
                        "kind": "STEP",
                        "tick": tick,
                        "actual": {
                            "schema": "dual-raw-v1",
                            "world_time": tick * 0.01,
                            "qpos": [1.0],
                            "qvel": [0.0],
                        },
                        "case_id": "synthetic",
                        "replay_stride_ticks": 20,
                        "action": 240,
                        "action_tick": 0,
                        "structural_mask": self.legal_mask().tolist(),
                    }
                )
            self.writer.close()
            path = self.output / "replay.draw"
            self.replay_segments = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))]

    monkeypatch.setattr(env_module, "DualCoordinationEnv", CaptureEnv)
    summary = m.run_validation(
        learning_lock="synthetic",
        terminals=terminals,
        original=original,
        design_path=design,
        output=tmp_path / "out",
        case_id="synthetic",
        method="N42",
        native_reexecute=reexecute,
    )
    assert summary["status"] == "partial" and summary["retained_missions"] == 1
    assert len(admissions) == (2 if reexecute else 1) and all(
        x["mode"] == "validation" and x["replay_stride_ticks"] == 20 and x["retain_replay"]
        for x in admissions
    )
    evidence = json.loads(next((tmp_path / "out" / "evidence").glob("*.json")).read_text())["missions"][0]
    assert evidence["audit"]["complete"]
    assert evidence["replay"]["passed"] if reexecute else evidence["replay"]["retained_inputs_verified"]
    assert evidence["replay"]["inventory"]["actions"]["path"].endswith(".zip")
    assert not (tmp_path / "out" / "selection.json").exists()


def test_compact_replay_detects_intermediate_drift_even_with_identical_terminal(tmp_path):
    m = module()
    from adaptive_drone_patrol.dual_coordination_raw import RawWriter

    captures = []
    for run in (0, 1):
        path = tmp_path / f"{run}.draw"
        with RawWriter(path) as writer:
            for tick in (0, 1, 20):
                writer.append(
                    dict(
                        kind="RESET" if tick == 0 else "STEP",
                        tick=tick,
                        actual=dict(
                            schema="dual-raw-v1",
                            world_time=tick * 0.01,
                            qpos=[0.001 if tick == 1 and run else 0.0],
                            qvel=[0.0],
                        ),
                    )
                )
        captures.append(dict(replay_segments=[{"path": str(path)}]))
    assert m.compare_compact_replay(captures[0], captures[0])
    assert not m.compare_compact_replay(*captures)


@pytest.mark.parametrize(
    "reason,diagnostic",
    [
        ("safety:online_internal_audit", {}),
        ("team_timeout", {"reconstruction_failures": [{"reason": "uncertified swept interval"}]}),
        ("team_timeout", {"reconstructed_path_safe": False}),
        ("supervisor:infeasible_hold", {}),
    ],
)
def test_reconstruction_or_safety_failure_cannot_report_zero_violations(tmp_path, reason, diagnostic):
    m = module()

    class UnsafeEnv(SyntheticEnv):
        def step(self, action):
            result = super().step(action)
            self.state["terminal_reason"] = reason
            return result

    env = UnsafeEnv(tmp_path / "unsafe")
    env.online_auditor = SimpleNamespace(report=lambda: {**report(env.mission.tick), **diagnostic})
    capture = m.capture_mission(
        env, lambda env: 240, {"case_id": "synthetic", "scenario": "normal"}, env.output
    )
    assert capture["outcome"]["safety_violations"] >= 1
    assert not capture["outcome"]["success"]


def test_only_verified_new_owned_slot_can_retire_duplicate_sources(tmp_path):
    m = module()
    import zipfile

    slot = tmp_path / "slot"
    slot.mkdir()
    captures = []
    for name in ("capture", "replay"):
        directory = slot / name
        directory.mkdir()
        (directory / "capture.json").write_text("{}")
        captures.append(directory)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i, directory in enumerate(captures):
            z.write(directory / "capture.json", f"{i}/capture.json")
    record = m.file_record(archive, tmp_path)
    with pytest.raises(ValueError, match="ownership"):
        m.retire_slot_captures(slot, record, root=tmp_path, token="owned")
    assert all(d.exists() for d in captures)
    (slot / "ownership.json").write_text(
        json.dumps({"schema": "dual-evaluator-owned-slot-v1", "token": "owned"})
    )
    (captures[0] / "capture.json").write_text("changed")
    with pytest.raises(ValueError, match="member"):
        m.retire_slot_captures(slot, record, root=tmp_path, token="owned")
    assert all(d.exists() for d in captures)
    (captures[0] / "capture.json").write_text("{}")
    m.retire_slot_captures(slot, record, root=tmp_path, token="owned")
    assert not any(d.exists() for d in captures)
    assert archive.exists() and (slot / "ownership.json").exists()


def retained_capture(tmp_path, *, missing_first=False, wrong_action=False):
    from adaptive_drone_patrol.dual_coordination_raw import RawWriter

    m = module()
    directory = tmp_path / "capture"
    directory.mkdir()
    mask = [False] * 240 + [True, False]
    proposal = dict(tick=0, action=240, mask=mask)
    actual = dict(schema="dual-raw-v1", world_time=0.0, qpos=[0.0], qvel=[0.0])
    (directory / "reset-inputs.json").write_text(
        json.dumps({"case": {"case_id": "synthetic"}, "initial_native": actual})
    )
    (directory / "proposed-actions.jsonl").write_text(json.dumps(proposal) + "\n")
    path = directory / "replay.draw"
    with RawWriter(path) as writer:
        writer.append(dict(kind="RESET", tick=0, actual=actual, case_id="synthetic", replay_stride_ticks=20))
        for tick in [20] if missing_first else [1, 20]:
            writer.append(
                dict(
                    kind="STEP",
                    tick=tick,
                    actual={**actual, "world_time": tick * 0.01},
                    action=241 if wrong_action else 240,
                    action_tick=0,
                    structural_mask=mask,
                    case_id="synthetic",
                    replay_stride_ticks=20,
                )
            )
    capture = dict(
        actions=[proposal],
        terminal_tick=20,
        mission={"tick": 20, "terminal_reason": "success"},
        outcome={"case_id": "synthetic", "success": True, "team_time": 0.2},
        final_backend={"qpos": [0.0], "qvel": [0.0]},
        audit_report=report(20),
        infrastructure_errors=[],
        replay_segments=[m.file_record(path, tmp_path)],
        wall_seconds=1.0,
    )
    capture["replay_segments"][0]["path"] = str(path)
    (directory / "capture.json").write_text(json.dumps(capture))
    return directory, capture


def test_single_flight_stream_validation_checks_every_retained_boundary_and_proposal(tmp_path):
    m = module()
    _, capture = retained_capture(tmp_path)
    evidence = m.validate_retained_inputs(capture)
    assert evidence["retained_inputs_verified"] and evidence["compact_boundary_consistency"]
    assert (
        evidence["reset_records"] == 1 and evidence["action_records"] == 1 and evidence["control_steps"] == 20
    )
    assert evidence["retained_state_records"] == 3
    assert "passed" not in evidence and "native_reexecution_performed" not in evidence


@pytest.mark.parametrize("damage", ["missing_first", "wrong_action", "final_state", "proposal_file"])
def test_single_flight_stream_validation_rejects_missing_or_inconsistent_evidence(tmp_path, damage):
    m = module()
    directory, capture = retained_capture(
        tmp_path, missing_first=damage == "missing_first", wrong_action=damage == "wrong_action"
    )
    if damage == "final_state":
        capture["final_backend"]["qpos"] = [1.0]
    if damage == "proposal_file":
        (directory / "proposed-actions.jsonl").write_text("")
    evidence = m.validate_retained_inputs(capture)
    assert not evidence["retained_inputs_verified"] and evidence["errors"]


def test_v2_receipt_explicitly_does_not_claim_native_reexecution(tmp_path):
    m = module()
    directory, capture = retained_capture(tmp_path)
    cap = SimpleNamespace(binding=SimpleNamespace(files={"audit": [], "configuration": []}))
    bundle = m.bundle_captures(tmp_path / "bundle.zip", [(directory, capture)], root=tmp_path)
    reference = {"path": "development.json", "bytes": 1, "sha256": "synthetic"}
    receipts = m.build_single_capture_receipts(
        cap,
        {"case_id": "synthetic", "case_sha256": m._digest({"case_id": "synthetic"})},
        capture,
        bundle,
        development_replay_reference=reference,
    )
    replay = receipts["replay"]
    assert replay["schema"] == "dual-validation-replay-v2" and replay["native_reexecution_performed"] is False
    assert replay["development_replay_reference"] == reference and replay["retained_inputs_verified"]
    assert not ({"passed", "decisions_match", "terminal_match", "metrics_within_tolerance"} & replay.keys())


def test_unselected_infeasible_candidate_is_conservative_conflict_with_bounded_diagnostics(monkeypatch):
    m = module()
    from adaptive_drone_patrol.dual_coordination_reference import ContinuousReference, ReferenceInfeasible
    from adaptive_drone_patrol.dual_coordination_baselines import ConflictRequest

    refs = (ContinuousReference([0, 0, 2]), ContinuousReference([4, 0, 2]))
    env = SimpleNamespace(
        fixture=SimpleNamespace(
            sample=lambda: {
                "drones": [
                    {"drone_xyz": [0, 0, 2], "drone_velocity": [0, 0, 0]},
                    {"drone_xyz": [4, 0, 2], "drone_velocity": [0, 0, 0]},
                ]
            }
        ),
        mission=SimpleNamespace(targets=[10, 10]),
        supervisor=SimpleNamespace(assembly_radius=0.4, tracking_inflation=0.15),
    )

    def fail(*args, **kwargs):
        raise ReferenceInfeasible("synthetic uncertified C2 candidate")

    monkeypatch.setattr(ContinuousReference, "retarget", fail)
    request = ConflictRequest((0, 10), refs, ((0, 0, 2), (4, 0, 2)), ((1, 0, 2),) * 10, 0.0, (0.0, 2.0))
    predictor = m.causal_conflict_predictor(env)
    for _ in range(30):
        result = predictor(request)
        assert result.pair_conflict and result.unavailable_hover_conflict == (True, False)
    diagnostics = env._baseline_prediction_diagnostics
    assert diagnostics["uncertified_candidates"] == 30 and len(diagnostics["samples"]) == 16
    assert diagnostics["samples"][0]["targets"] == [0, 10]
    assert "uncertified C2" in diagnostics["samples"][0]["reason"]


def test_capture_discloses_baseline_uncertified_candidate_count(tmp_path):
    m = module()
    env = SyntheticEnv(tmp_path / "capture")
    env._baseline_prediction_diagnostics = {
        "uncertified_candidates": 123,
        "samples": [{"reason": "synthetic"}],
    }
    capture = m.capture_mission(
        env, lambda env: 240, {"case_id": "synthetic", "scenario": "normal"}, env.output
    )
    assert capture["baseline_prediction_diagnostics"] == env._baseline_prediction_diagnostics
