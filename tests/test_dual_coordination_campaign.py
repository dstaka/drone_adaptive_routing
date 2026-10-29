"""Frozen file identities and stage access must fail closed before exposure."""

import json
import copy
import pytest


def test_file_snapshot_rejects_escape_symlink_and_mutation(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import BoundFiles

    root = tmp_path / "root"
    root.mkdir()
    p = root / "config.json"
    p.write_text("{}")
    receipt = BoundFiles.capture(root, {"configuration": ["config.json"]})
    receipt.verify()
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        receipt.verify()
    with pytest.raises(ValueError):
        BoundFiles.capture(root, {"configuration": ["../outside"]})
    link = root / "link"
    link.symlink_to(p)
    with pytest.raises(ValueError):
        BoundFiles.capture(root, {"configuration": ["link"]})


def test_production_lock_refuses_partial_or_unqualified_receipts(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import create_learning_lock

    with pytest.raises(ValueError):
        create_learning_lock(tmp_path / "lock.json", root=tmp_path, roles={}, qualification={}, benchmark={})
    assert not (tmp_path / "lock.json").exists()


def test_joint_freeze_is_not_constructible_from_boolean_or_empty_receipt(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import VerifiedJointFreeze

    for value in (True, {}, {"schema":
        "dual-freeze-v1"}):
        with pytest.raises((ValueError, TypeError)):
            VerifiedJointFreeze(value)
    p = tmp_path / "freeze.json"
    p.write_text(json.dumps({"schema": "dual-freeze-v1"}))
    with pytest.raises(ValueError):
        VerifiedJointFreeze.load(p)


def test_physical_manifest_cannot_be_generated_without_joint_freeze(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import generate_heldout

    with pytest.raises(ValueError, match="freeze"):
        generate_heldout(
            tmp_path / "heldout", None, layout_path="configs/demo_scenarios_payload_elevated/layout.json"
        )
    assert not (tmp_path / "heldout").exists()


def test_loading_freeze_revalidates_actual_selection_and_learning_chain(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import (
        BoundFiles,
        JOINT_ROLES,
        NATIVE_CHECKS,
        VerifiedJointFreeze,
    )

    (tmp_path / "empty.json").write_text("{}")
    binding = BoundFiles.capture(tmp_path, {role: ["empty.json"] for role in JOINT_ROLES})
    fake = dict(
        schema="dual-freeze-v1",
        binding=binding.to_dict(),
        selected_seed=42,
        selection_verified=True,
        qualification=dict(
            schema="dual-qualification-v1",
            checks={k: True for k in NATIVE_CHECKS},
            passed=True,
            safety_violations=0,
            profile="dual-payload-v1",
            version="3.15.0",
        ),
        benchmark=dict(
            schema="dual-benchmark-v1",
            repetitions=3,
            transitions_per_repetition=4096,
            epochs_per_update=5,
            storage_feasible=True,
            selected_backend="native-cpu",
            whole_campaign_projection_seconds=1,
        ),
    )
    (tmp_path / "fake.json").write_text(json.dumps(fake))
    with pytest.raises(ValueError):
        VerifiedJointFreeze.load(tmp_path / "fake.json")


def experiment_fixture(root):
    """Synthetic administrative inputs; no physical qualification is claimed."""
    from adaptive_drone_patrol.dual_coordination_campaign import LEARNING_ROLES, NATIVE_CHECKS, BoundFiles
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    from adaptive_drone_patrol.util import sha256

    root.mkdir(exist_ok=True)
    roles = {}
    for role in LEARNING_ROLES - {"qualification", "benchmark", "declaration"}:
        (root / f"{role}.json").write_text("{}")
        roles[role] = [f"{role}.json"]
    cases = [generate_case(20, i, ("normal", "priority", "availability-delay")[i // 100]) for i in range(300)]
    path = root / "validation.jsonl"
    path.write_text("".join(json.dumps(case) + "\n" for case in cases))
    record = dict(bytes=path.stat().st_size, sha256=sha256(path))
    (root / "declaration.json").write_text(
        json.dumps(
            dict(
                schema="dual-campaign-declaration-v1",
                training_cases=10000,
                validation_cases=300,
                files={"validation.jsonl": record},
            )
        )
    )
    roles["declaration"] = ["declaration.json", "validation.jsonl"]
    inputs = BoundFiles.capture(root, roles).files
    qualification = dict(
        schema="dual-qualification-v1",
        checks=dict.fromkeys(NATIVE_CHECKS, True),
        passed=True,
        safety_violations=0,
        profile="dual-payload-v1",
        version="3.15.0",
        experiment_inputs=inputs,
    )
    benchmark = dict(
        schema="dual-benchmark-v1",
        repetitions=3,
        transitions_per_repetition=4096,
        epochs_per_update=5,
        storage_feasible=True,
        selected_backend="native-cpu",
        whole_campaign_projection_seconds=100.0,
        experiment_inputs=inputs,
    )
    for role, value in [("qualification", qualification), ("benchmark", benchmark)]:
        (root / f"{role}.json").write_text(json.dumps(value))
        roles[role] = [f"{role}.json"]
    return roles, qualification, benchmark, cases


def test_learning_lock_binds_experiment_input_hashes(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import create_learning_lock

    roles, qualification, benchmark, _ = experiment_fixture(tmp_path)
    (tmp_path / "controller.json").write_text('{"changed_after_qualification":true}')
    with pytest.raises(ValueError, match="experiment"):
        create_learning_lock(
            tmp_path / "lock.json",
            root=tmp_path,
            roles=roles,
            qualification=qualification,
            benchmark=benchmark,
        )
    assert not (tmp_path / "lock.json").exists()


@pytest.mark.parametrize("role", ["qualification", "benchmark"])
def test_learning_lock_rejects_receipt_from_other_experiment(tmp_path, role):
    from adaptive_drone_patrol.dual_coordination_campaign import create_learning_lock

    roles, qualification, benchmark, _ = experiment_fixture(tmp_path)
    value = qualification if role == "qualification" else benchmark
    # Change this receipt alone, retaining self-consistent receipt file bytes.
    value["experiment_inputs"] = copy.deepcopy(value["experiment_inputs"])
    value["experiment_inputs"]["source"][0]["sha256"] = "0" * 64
    (tmp_path / f"{role}.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="experiment"):
        create_learning_lock(
            tmp_path / "lock.json",
            root=tmp_path,
            roles=roles,
            qualification=qualification,
            benchmark=benchmark,
        )


def test_declaration_requires_actual_complete_validation_cases(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import create_learning_lock, BoundFiles

    roles, qualification, benchmark, _ = experiment_fixture(tmp_path)
    # Even mutually consistent input hashes cannot turn a short cohort into300.
    validation = tmp_path / "validation.jsonl"
    validation.write_text("\n".join(validation.read_text().splitlines()[:-1]) + "\n")
    from adaptive_drone_patrol.util import sha256

    declared = json.loads((tmp_path / "declaration.json").read_text())
    declared["files"]["validation.jsonl"] = dict(bytes=validation.stat().st_size, sha256=sha256(validation))
    (tmp_path / "declaration.json").write_text(json.dumps(declared))
    inputs = BoundFiles.capture(
        tmp_path, {k: v for k, v in roles.items() if k not in ("qualification", "benchmark")}
    ).files
    for name, value in [("qualification", qualification), ("benchmark", benchmark)]:
        value["experiment_inputs"] = inputs
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="300"):
        create_learning_lock(
            tmp_path / "lock.json",
            root=tmp_path,
            roles=roles,
            qualification=qualification,
            benchmark=benchmark,
        )


@pytest.fixture(scope="module")
def complete_checkpoint():
    """Loadable synthetic NN/Adam state; counters are fixture data, not training evidence."""
    torch = pytest.importorskip("torch")
    from dataclasses import asdict
    import numpy as np
    import random
    from adaptive_drone_patrol.dual_coordination_policy import JointActorCritic, JointPPO, TrainingBudget

    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        trainer = JointPPO(JointActorCritic(), seed=42)
        for parameter in trainer.policy.parameters():
            parameter.grad = torch.zeros_like(parameter)
        trainer.optimizer.step()
        for state in trainer.optimizer.state.values():
            state["step"].fill_(10240)
        trainer.budget = TrainingBudget(42, 1048576, 256)
        return dict(
            schema="dual-policy-v1",
            policy_config=asdict(trainer.policy.config),
            ppo_config=asdict(trainer.config),
            policy=trainer.policy.state_dict(),
            optimizer=trainer.optimizer.state_dict(),
            budget=asdict(trainer.budget),
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            cuda_rng=None,
            environment_state={"synthetic_recovery_fixture": True},
            backend_state={
                "stage_transition_counts": [262144, 720896, 65536],
                "stage3_backend": "native-mujoco",
            },
            physical_resume_available=True,
            selectable=True,
            versions={"torch": torch.__version__, "numpy": np.__version__},
            parameter_count=sum(parameter.numel() for parameter in trainer.policy.parameters()),
        )


@pytest.mark.parametrize(
    "damage",
    [
        "metadata_only",
        "missing_weight",
        "nan_weight",
        "normalization",
        "optimizer_empty",
        "adam_shape",
        "adam_step",
        "torch_rng",
        "numpy_rng",
        "python_rng",
        "counter",
        "none",
    ],
)
def test_terminal_checkpoint_requires_loadable_model_adam_rng_and_counters(
    tmp_path, complete_checkpoint, damage
):
    import adaptive_drone_patrol.dual_coordination_campaign as campaign

    torch = pytest.importorskip("torch")
    validator = getattr(campaign, "_validate_terminal_checkpoint", None)
    assert callable(validator), "Terminal checkpoint structural validator is missing"
    state = copy.deepcopy(complete_checkpoint)
    first = next(iter(state["optimizer"]["state"]))
    if damage == "metadata_only":
        state = {k: state[k] for k in ("schema", "budget", "selectable", "backend_state")}
    elif damage == "missing_weight":
        del state["policy"]["actor.weight"]
    elif damage == "nan_weight":
        state["policy"]["actor.weight"][0, 0] = float("nan")
    elif damage == "normalization":
        state["policy"]["goals_scale"][0] = 0
    elif damage == "optimizer_empty":
        state["optimizer"]["state"] = {}
    elif damage == "adam_shape":
        state["optimizer"]["state"][first]["exp_avg"] = torch.zeros(1)
    elif damage == "adam_step":
        state["optimizer"]["state"][first]["step"].fill_(1)
    elif damage == "torch_rng":
        state["torch_rng"] = torch.zeros(1, dtype=torch.uint8)
    elif damage == "numpy_rng":
        state["numpy_rng"] = ("invalid",)
    elif damage == "python_rng":
        state["python_rng"] = ("invalid",)
    elif damage == "counter":
        state["budget"]["transitions"] -= 4096
    path = tmp_path / "terminal.pt"
    torch.save(state, path)
    before = torch.get_rng_state().clone()
    if damage == "none":
        assert validator(path, 42)["transitions"] == 1048576
    else:
        with pytest.raises(ValueError):
            validator(path, 42)
    assert torch.equal(before, torch.get_rng_state())


def validation_fixture(root, manifests=None):
    """Fake administrative receipts only; never physical acceptance evidence."""
    from adaptive_drone_patrol.dual_coordination_campaign import BoundFiles, EXPERIMENT_ROLES, _digest

    roles, _, _, cases = experiment_fixture(root)
    inputs = BoundFiles.capture(root, roles)
    experiment_hash = _digest({k: inputs.files[k] for k in EXPERIMENT_ROLES})
    manifests = manifests or {seed: dict(sha256=str(seed) * 32) for seed in (42, 43, 44)}
    selection = dict(candidates={}, static=[], dynamic=[])
    records, payloads = [], []
    for method in ("N42", "N43", "N44", "B-static", "B-dynamic"):
        outcomes = []
        for case in cases:
            outcome = dict(
                case_id=case["case_id"],
                scenario=case["scenario"],
                success=True,
                team_time=1.0,
                safety_violations=0,
                invalid_actions=0,
                duplicate_service=0,
                decision_intervals=1,
                intervened_intervals=0,
                supervisor_held_drone_seconds=0.0,
                useful_post_event_transfer=False,
                failure_reason=None,
            )
            outcomes.append(outcome)
            identity = dict(
                case_id=case["case_id"],
                case_sha256=_digest(case),
                method=method,
                model_sha256=manifests[int(method[1:])]["sha256"]
                if method.startswith("N")
                else _digest(inputs.files["baseline"]),
                experiment_inputs_sha256=experiment_hash,
            )
            result = dict(
                schema="dual-validation-result-v1", identity=identity, outcome=outcome, terminal_tick=100
            )
            audit = dict(
                schema="dual-validation-audit-v1",
                identity=identity,
                result_sha256=_digest(result),
                auditor_sha256=_digest(inputs.files["audit"]),
                outcome=outcome,
                complete=True,
                errors=[],
                proof_gaps=[],
                coverage=dict(
                    control_steps=100, rk_states=6400, native_substep_intervals=1600, swept_intervals=1600
                ),
            )
            # Invented compact bytes deliberately labelled synthetic, with a bound inventory.
            payload = f"{method}-{case['index']}.json"
            (root / payload).write_text(json.dumps({"synthetic_test_only": True, "identity": identity}))
            payloads.append(payload)
            entry = BoundFiles.capture(root, {"payload": [payload]}).files["payload"][0]
            inventory = {kind: entry for kind in ("reset", "actions", "compact_replay")}
            replay = dict(
                schema="dual-validation-replay-v1",
                identity=identity,
                result_sha256=_digest(result),
                audit_sha256=_digest(audit),
                inventory=inventory,
                inventory_sha256=_digest(inventory),
                configuration_sha256=_digest(inputs.files["configuration"]),
                passed=True,
                decisions_match=True,
                terminal_match=True,
                metrics_within_tolerance=True,
                control_steps=100,
                reset_records=1,
                outcome=outcome,
            )
            records.append(dict(result=result, audit=audit, replay=replay))
        if method.startswith("N"):
            selection["candidates"][method[1:]] = outcomes
        else:
            selection["static" if method == "B-static" else "dynamic"] = outcomes
    (root / "evidence.json").write_text(
        json.dumps(dict(schema="dual-validation-evidence-v1", missions=records))
    )
    roles["validation_evidence"] = ["evidence.json"] + payloads
    selection["validation_evidence"] = ["evidence.json"]
    return roles, selection, {c["case_id"]: c for c in cases}, manifests


@pytest.mark.parametrize(
    "damage", ["missing", "case", "case_cohort", "model", "outcome", "audit", "replay", "payload", "none"]
)
def test_validation_requires_bound_case_result_audit_and_compact_replay(tmp_path, damage):
    import adaptive_drone_patrol.dual_coordination_campaign as campaign

    validator = getattr(campaign, "_verify_validation_evidence", None)
    assert callable(validator), "Bound validation evidence validator is missing"
    roles, selection, cases, manifests = validation_fixture(tmp_path)
    document = json.loads((tmp_path / "evidence.json").read_text())
    first = document["missions"][0]
    if damage == "missing":
        document["missions"].pop()
    elif damage == "case":
        first["result"]["identity"]["case_sha256"] = "0" * 64
    elif damage == "case_cohort":
        cases.pop(next(iter(cases)))
    elif damage == "model":
        first["result"]["identity"]["model_sha256"] = "0" * 64
    elif damage == "outcome":
        selection["candidates"]["42"][0]["team_time"] = 2.0
    elif damage == "audit":
        first["audit"]["coverage"]["rk_states"] -= 1
    elif damage == "replay":
        first["replay"]["decisions_match"] = False
    (tmp_path / "evidence.json").write_text(json.dumps(document))
    bound = campaign.BoundFiles.capture(tmp_path, roles)
    if damage == "payload":
        (tmp_path / first["replay"]["inventory"]["reset"]["path"]).write_text("tampered")
    if damage == "none":
        validator(bound, selection, cases, manifests)
    else:
        with pytest.raises(ValueError):
            validator(bound, selection, cases, manifests)


def test_verified_learning_lock_checks_current_inputs_and_receipt(tmp_path):
    import adaptive_drone_patrol.dual_coordination_campaign as campaign

    lock_type = getattr(campaign, "VerifiedLearningLock", None)
    assert lock_type is not None, "Verified learning-lock capability is missing"
    roles, qualification, benchmark, _ = experiment_fixture(tmp_path)
    path = tmp_path / "lock.json"
    campaign.create_learning_lock(
        path, root=tmp_path, roles=roles, qualification=qualification, benchmark=benchmark
    )
    lock = lock_type.load(path)
    assert lock.selected_backend == "native-cpu"
    assert lock.require_current() is lock
    with pytest.raises(ValueError):
        lock_type(lock.receipt)
    (tmp_path / "controller.json").write_text('{"changed":true}')
    with pytest.raises(ValueError):
        lock.require_current()
    with pytest.raises(ValueError):
        lock_type.load(path)
    (tmp_path / "controller.json").write_text("{}")
    lock.require_current()
    lock.binding.files["controller"] = []
    with pytest.raises(ValueError):
        lock.require_current()


def test_complete_synthetic_joint_chain_loads_and_rechecks_retained_files(tmp_path, complete_checkpoint,monkeypatch):
    """Exercise the admission plumbing only, without opening/generating heldout."""
    import torch
    import adaptive_drone_patrol.dual_coordination_campaign as campaign
    from adaptive_drone_patrol.util import sha256

    # This fixture has no real physics assets/training. Bind its explicit
    # synthetic execution identity while separately testing real input identity
    # derivation below; do not generate/open actual held-out cases here.
    def synthetic_binding(bound,learning_lock_sha,benchmark,seed):
        return dict(learning_lock_sha256='synthetic-only',backend='native-cpu',seed=seed)
    monkeypatch.setattr(campaign,'expected_terminal_binding',synthetic_binding)

    manifests = {}
    for seed in (42, 43, 44):
        state = copy.deepcopy(complete_checkpoint)
        state["budget"]["seed"] = seed
        state['environment_state']=dict(schema='dual-training-recovery-v1',binding=synthetic_binding(None,None,None,seed))
        state['backend_state'].update(learning_lock_sha256='synthetic-only',selected_backend='native-cpu')
        path = tmp_path / f"terminal{seed}.pt"
        torch.save(state, path)
        manifests[seed] = dict(sha256=sha256(path))
    roles, selection, _, _ = validation_fixture(tmp_path, manifests)
    qualification = json.loads((tmp_path / "qualification.json").read_text())
    benchmark = json.loads((tmp_path / "benchmark.json").read_text())
    lock_path = tmp_path / "lock.json"
    campaign.create_learning_lock(
        lock_path,
        root=tmp_path,
        roles={k: v for k, v in roles.items() if k in campaign.LEARNING_ROLES},
        qualification=qualification,
        benchmark=benchmark,
    )
    # Baseline times must make the three synthetic NN candidates numerically eligible.
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    for method in ("static", "dynamic"):
        for row in selection[method]:
            row["team_time"] = 2.0
    for mission in evidence["missions"]:
        if mission["result"]["identity"]["method"].startswith("B-"):
            result, audit, replay = (mission[k] for k in ("result", "audit", "replay"))
            result["outcome"]["team_time"] = 2.0
            result["terminal_tick"] = 200
            audit["outcome"]["team_time"] = replay["outcome"]["team_time"] = 2.0
            audit["coverage"] = dict(
                control_steps=200, rk_states=12800, native_substep_intervals=3200, swept_intervals=3200
            )
            audit["result_sha256"] = replay["result_sha256"] = campaign._digest(result)
            replay["audit_sha256"] = campaign._digest(audit)
            replay["control_steps"] = 200
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    selection.update(
        schema="dual-selection-bound-v1",
        selected_seed=42,
        checkpoint_sha256={str(k): v["sha256"] for k, v in manifests.items()},
    )
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    (tmp_path / "camera.json").write_text("{}")
    roles.update(camera=["camera.json"], selection=["selection.json"], learning_lock=["lock.json"])
    roles.update({f"terminal{s}": [f"terminal{s}.pt"] for s in (42, 43, 44)})
    freeze = campaign.create_joint_freeze(
        tmp_path / "freeze.json", learning_lock=lock_path, root=tmp_path, roles=roles, selection=selection
    )
    assert freeze.receipt["selected_seed"] == 42
    freeze.require_current()
    assert not (tmp_path / "heldout").exists()
    payload = next(p for p in roles["validation_evidence"] if p != "evidence.json")
    (tmp_path / payload).write_text("tampered")
    with pytest.raises(ValueError):
        freeze.require_current()
    with pytest.raises(ValueError):
        campaign.VerifiedJointFreeze.load(tmp_path / "freeze.json")


def training_manifest_fixture(root, count=10000):
    """Synthetic declared cases, used solely to exercise exact admission."""
    from adaptive_drone_patrol.dual_coordination_campaign import BoundFiles, EXPERIMENT_ROLES
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    from adaptive_drone_patrol.util import sha256

    roles, qualification, benchmark, validation = experiment_fixture(root)
    scenarios = ("normal", "priority", "availability-delay")
    template = {s: generate_case(10, 0, s) for s in scenarios}
    path = root / "training.jsonl"
    first, last = None, None
    with path.open("w") as stream:
        for index in range(count):
            row = copy.deepcopy(template)
            for scenario, case in row.items():
                case.update(index=index, case_id=f"10-{index:05d}-{scenario}")
            stream.write(json.dumps(row) + "\n")
            if index == 0:
                first = row
            last = row
    roles["declaration"].append("training.jsonl")
    declaration = json.loads((root / "declaration.json").read_text())
    declaration["files"]["training.jsonl"] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    (root / "declaration.json").write_text(json.dumps(declaration))
    inputs = BoundFiles.capture(root, {k: v for k, v in roles.items() if k in EXPERIMENT_ROLES}).files
    for name, receipt in [("qualification", qualification), ("benchmark", benchmark)]:
        receipt["experiment_inputs"] = inputs
        (root / f"{name}.json").write_text(json.dumps(receipt))
    return roles, qualification, benchmark, validation, first, last


def test_case_admission_matches_manifest_and_caches_immutable_hashes(tmp_path, monkeypatch):
    import adaptive_drone_patrol.dual_coordination_campaign as campaign

    verifier = getattr(campaign.VerifiedLearningLock, "verify_case", None)
    assert callable(verifier), "Exact case admission API is missing"
    roles, qualification, benchmark, validation, first, last = training_manifest_fixture(tmp_path)
    campaign.create_learning_lock(
        tmp_path / "lock.json", root=tmp_path, roles=roles, qualification=qualification, benchmark=benchmark
    )
    lock = campaign.VerifiedLearningLock.load(tmp_path / "lock.json")
    calls = []
    real_parse = campaign._training_case_hashes

    def counted(binding):
        calls.append(True)
        return real_parse(binding)

    monkeypatch.setattr(campaign, "_training_case_hashes", counted)
    for row in (first, last):
        for case in row.values():
            assert lock.verify_case(case, "training") == case
    assert calls == [True]
    assert lock.verify_case(validation[0], "validation") == validation[0]
    changed = copy.deepcopy(first["normal"])
    changed["priorities"][0] = 1 if changed["priorities"][0] != 1 else 2
    changed_validation = copy.deepcopy(validation[0])
    changed_validation["loads"][0]["mass_ratio"] = 0.15
    for case, mode in (
        (changed, "training"),
        (first["normal"], "validation"),
        (validation[0], "training"),
        (validation[0], "heldout"),
        (changed_validation, "validation"),
    ):
        with pytest.raises(ValueError):
            lock.verify_case(case, mode)
    with pytest.raises(TypeError):
        lock._case_digests["training"][changed["case_id"]] = campaign._digest(changed)
    with pytest.raises(AttributeError):
        lock._case_digests = {}
    with pytest.raises(AttributeError):
        del lock._case_digests
    # Verification returns a separate object, never a mutable cached case.
    returned = lock.verify_case(first["normal"], "training")
    returned["priorities"][0] = 999
    lock.verify_case(first["normal"], "training")
    # Cache avoids reparsing; file tampering is still rejected by boundary checks.
    (tmp_path / "training.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        lock.require_current()


def test_training_case_admission_rejects_short_bound_manifest(tmp_path):
    import adaptive_drone_patrol.dual_coordination_campaign as campaign

    verifier = getattr(campaign.VerifiedLearningLock, "verify_case", None)
    assert callable(verifier), "Exact case admission API is missing"
    roles, qualification, benchmark, _, first, _ = training_manifest_fixture(tmp_path, count=1)
    campaign.create_learning_lock(
        tmp_path / "lock.json", root=tmp_path, roles=roles, qualification=qualification, benchmark=benchmark
    )
    lock = campaign.VerifiedLearningLock.load(tmp_path / "lock.json")
    with pytest.raises(ValueError, match="10000"):
        lock.verify_case(first["normal"], "training")


@pytest.mark.parametrize("damage", ["index", "split", "scenario", "missing_variant", "duplicate_id", "hash"])
def test_training_manifest_semantics_are_checked_before_cache_creation(tmp_path, damage):
    import adaptive_drone_patrol.dual_coordination_campaign as campaign
    from adaptive_drone_patrol.util import sha256

    roles, qualification, benchmark, _, first, _ = training_manifest_fixture(tmp_path, count=1)
    row = copy.deepcopy(first)
    if damage == "index":
        row["normal"]["index"] = 9999
    elif damage == "split":
        row["normal"]["split"] = 20
    elif damage == "scenario":
        row["priority"] = copy.deepcopy(row["normal"])
    elif damage == "missing_variant":
        del row["priority"]
    elif damage == "duplicate_id":
        row["priority"]["case_id"] = row["normal"]["case_id"]
    (tmp_path / "training.jsonl").write_text(json.dumps(row) + "\n")
    declaration = json.loads((tmp_path / "declaration.json").read_text())
    path = tmp_path / "training.jsonl"
    declaration["files"]["training.jsonl"] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    if damage == "hash":
        declaration["files"]["training.jsonl"]["sha256"] = "0" * 64
    (tmp_path / "declaration.json").write_text(json.dumps(declaration))
    binding = campaign.BoundFiles.capture(tmp_path, roles)
    with pytest.raises(ValueError):
        campaign._training_case_hashes(binding)


def test_heldout_runtime_capability_refuses_unfrozen_files_before_reading_cases(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import VerifiedHeldout
    for freeze in (None,True,{}):
        with pytest.raises(ValueError,match='freeze'):
            VerifiedHeldout.load(tmp_path/'missing',freeze=freeze)
    with pytest.raises(ValueError,match='Load'):
        VerifiedHeldout({})


def test_heldout_runtime_checks_recipe_bytes_modes_and_immutable_cache(tmp_path,monkeypatch):
    """Synthetic repeated geometry exercises plumbing; no real held-out RNG is used."""
    from types import SimpleNamespace
    import adaptive_drone_patrol.dual_coordination_campaign as campaign
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    from adaptive_drone_patrol.util import sha256
    (tmp_path/'layout.json').write_text('{}')
    class SyntheticFreeze:
        digest='a'*64
        receipt={'created_at':'2026-01-01T00:00:00+00:00','benchmark':{'selected_backend':'native-cpu'}}
        binding=SimpleNamespace(root=tmp_path,files={'layout':[{'path':'layout.json'}]})
        def require_current(self):
            return self
    monkeypatch.setattr(campaign,'VerifiedJointFreeze',SyntheticFreeze)
    base=generate_case(15,0,'normal')
    def rows(split,n):
        result=[]
        for i in range(n):
            case=copy.deepcopy(base)
            case.update(case_id=f'synthetic-{split}-{i}',split=split,index=i)
            result.append(case)
        return result
    routing=rows(30,300)
    physical=[{'group':'mixed','case':case} for case in rows(40,127)]
    monkeypatch.setattr(campaign,'_heldout_case_lists',lambda *args,**kwargs:(routing,physical))
    for name,cases in [('routing300.json',routing),('physical127.json',physical)]:
        (tmp_path/name).write_text(json.dumps(cases))
    receipt=dict(schema='dual-heldout-manifest-v1',freeze_sha256='a'*64,
                 opened_at='2026-01-01T00:00:01+00:00',routing_cases=300,physical_cases=127,
                 files={name:dict(bytes=(tmp_path/name).stat().st_size,sha256=sha256(tmp_path/name))
                        for name in ('routing300.json','physical127.json')})
    (tmp_path/'manifest.json').write_text(json.dumps(receipt))
    capability=campaign.VerifiedHeldout.load(tmp_path,freeze=SyntheticFreeze())
    case=capability.verify_case(routing[0],'routing-heldout')
    case['goals'][0][0]+=1
    assert capability.verify_case(routing[0],'routing-heldout')==routing[0]
    with pytest.raises(ValueError,match='differs'):
        capability.verify_case(case,'routing-heldout')
    with pytest.raises(ValueError,match='split'):
        capability.verify_case(routing[0],'physical-heldout')
    with pytest.raises(AttributeError):
        capability._case_digests={}
    (tmp_path/'routing300.json').write_text('[]')
    with pytest.raises(ValueError,match='changed'):
        capability.require_current()
    with pytest.raises(ValueError,match='recipe'):
        campaign.VerifiedHeldout.load(tmp_path,freeze=SyntheticFreeze())


def test_expected_terminal_binding_uses_exact_frozen_inputs(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import BoundFiles,expected_terminal_binding
    for name in ('cf2x.urdf','controller.json','training.jsonl'):
        (tmp_path/name).write_text(name)
    bound=BoundFiles.capture(tmp_path,dict(assets=['cf2x.urdf'],controller=['controller.json'],declaration=['training.jsonl']))
    actual=expected_terminal_binding(bound,'a'*64,dict(selected_workers=7,selected_backend='native-cuda'),42)
    assert actual['original_sha256']==bound.files['assets'][0]['sha256']
    assert actual['controller_sha256']==bound.files['controller'][0]['sha256']
    assert actual['manifest_sha256']==bound.files['declaration'][0]['sha256']
    assert actual['workers']==7 and actual['seed']==42 and actual['learning_lock_sha256']=='a'*64
    with pytest.raises(ValueError,match='worker'):
        expected_terminal_binding(bound,'a'*64,dict(selected_backend='native-cpu'),42)


def test_terminal_checkpoint_cannot_come_from_another_learning_lock(tmp_path,complete_checkpoint):
    import torch
    from adaptive_drone_patrol.dual_coordination_campaign import _validate_terminal_checkpoint
    path=tmp_path/'terminal.pt'
    state=copy.deepcopy(complete_checkpoint)
    state['environment_state']=dict(schema='dual-training-recovery-v1',binding=dict(learning_lock_sha256='old'))
    torch.save(state,path)
    with pytest.raises(ValueError,match='binding'):
        _validate_terminal_checkpoint(path,42,expected_binding=dict(learning_lock_sha256='current'))


def test_single_capture_reference_requires_measured_replay_and_frozen_source(tmp_path):
    from adaptive_drone_patrol.dual_coordination_campaign import BoundFiles,EXPERIMENT_ROLES,_digest,verified_development_replay_reference
    from adaptive_drone_patrol.util import sha256
    from adaptive_drone_patrol.dual_coordination_env import OBSERVATION_SCALES
    roles={}
    for role in EXPERIMENT_ROLES:
        (tmp_path/(role+'.json')).write_text('{}')
        roles[role]=[role+'.json']
    names=('dual_coordination_env.py','dual_coordination_mission.py','dual_coordination_audit.py',
           'dual_coordination_reference.py','dual_coordination_safety.py','dual_payload_mujoco.py','payload_hybrid_controller.py')
    for name in names:
        (tmp_path/name).write_text(name)
    roles['source']=list(names)
    profile=dict(schema='dual-evidence-profile-v1',replays_per_mission=0,replay_stride_ticks=20,online_internal_audit=True)
    (tmp_path/'dual_coordination_evidence.json').write_text(json.dumps(profile))
    roles['configuration'].append('dual_coordination_evidence.json')
    bound=BoundFiles.capture(tmp_path,roles)
    evidence_hash=sha256(tmp_path/'dual_coordination_evidence.json')
    reference=dict(schema='dual-development-replay-qualification-v1',passed=True,native_reexecution_performed=True,
                   decisions_match=True,terminal_match=True,metrics_within_tolerance=True,decision_count=129,
                   checked_states=259,maximum_state_absolute_error=0.,
                   experiment_inputs_sha256=_digest(bound.files),source_sha256=_digest(bound.files['source']),
                   audit_sha256=_digest(bound.files['audit']),configuration_sha256=_digest(bound.files['configuration']),
                   profile_sha256=_digest(dict(scales=OBSERVATION_SCALES,source_sha256={n:sha256(tmp_path/n) for n in names},profile='dual-payload-v1')),
                   audit_library_sha256=None,evidence_profile_sha256=evidence_hash)
    path=tmp_path/'development-replay.json'
    path.write_text(json.dumps(reference))
    record=dict(path=path.name,bytes=path.stat().st_size,sha256=sha256(path))
    qualification=dict(compact_online_evidence_verified=True,evidence_profile_sha256=evidence_hash,
                       checks={'deterministic_replay':True},
                       development_replay_reference=record,audit_backend='python-native')
    assert verified_development_replay_reference(bound,qualification)==record
    reference['native_reexecution_performed']=False
    path.write_text(json.dumps(reference))
    qualification['development_replay_reference']=dict(path=path.name,bytes=path.stat().st_size,sha256=sha256(path))
    with pytest.raises(ValueError,match='measured'):
        verified_development_replay_reference(bound,qualification)


def single_capture_validation_fixture(root):
    """Synthetic serialized streams exercise authentication, never native physics."""
    from types import SimpleNamespace
    import adaptive_drone_patrol.dual_coordination_campaign as c
    import adaptive_drone_patrol.dual_coordination_evaluate as e
    from adaptive_drone_patrol.dual_coordination_env import OBSERVATION_SCALES
    from adaptive_drone_patrol.dual_coordination_raw import RawWriter
    from adaptive_drone_patrol.util import sha256

    roles,selection,cases,manifests = validation_fixture(root)
    names=('dual_coordination_env.py','dual_coordination_mission.py','dual_coordination_audit.py',
           'dual_coordination_reference.py','dual_coordination_safety.py','dual_payload_mujoco.py','payload_hybrid_controller.py')
    for name in names:
        (root/name).write_text('Synthetic source identity: '+name)
    roles['source']=list(names)
    (root/'dual_coordination_evidence.json').write_text(json.dumps(dict(
        schema='dual-evidence-profile-v1',replays_per_mission=0,replay_stride_ticks=20,online_internal_audit=True)))
    roles['configuration'].append('dual_coordination_evidence.json')
    bound=c.BoundFiles.capture(root,roles)
    experiment=c._digest({r:bound.files[r] for r in c.EXPERIMENT_ROLES})
    profile=c._digest(dict(scales=OBSERVATION_SCALES,source_sha256={n:sha256(root/n) for n in names},profile='dual-payload-v1'))
    reference=dict(schema='dual-development-replay-qualification-v1',passed=True,native_reexecution_performed=True,
        decisions_match=True,terminal_match=True,metrics_within_tolerance=True,decision_count=5,checked_states=11,
        maximum_state_absolute_error=0.,experiment_inputs_sha256=experiment,
        source_sha256=c._digest(bound.files['source']),audit_sha256=c._digest(bound.files['audit']),
        configuration_sha256=c._digest(bound.files['configuration']),profile_sha256=profile,
        audit_library_sha256=None,evidence_profile_sha256=sha256(root/'dual_coordination_evidence.json'))
    (root/'development-replay.json').write_text(json.dumps(reference))
    reference_record=e.file_record(root/'development-replay.json',root)
    qualification=json.loads((root/'qualification.json').read_text())
    qualification.update(compact_online_evidence_verified=True,development_replay_reference=reference_record,
                         evidence_profile_sha256=reference['evidence_profile_sha256'])
    (root/'qualification.json').write_text(json.dumps(qualification))
    document=json.loads((root/'evidence.json').read_text())
    for mission in document['missions']:
        for name in ('result','audit','replay'):
            mission[name]['identity']['experiment_inputs_sha256']=experiment
        mission['audit']['result_sha256']=c._digest(mission['result'])
        mission['replay'].update(result_sha256=c._digest(mission['result']),audit_sha256=c._digest(mission['audit']),
                                 configuration_sha256=c._digest(bound.files['configuration']))
    first=document['missions'][0]
    case=cases[first['result']['identity']['case_id']]
    outcome=selection['candidates']['42'][0]
    outcome['decision_intervals']=5
    directory=root/'capture'
    directory.mkdir()
    actual=dict(schema='dual-raw-v1',world_time=0.,qpos=[0.],qvel=[0.],synthetic_test_only=True)
    mask=[False]*240+[True,False]
    actions=[dict(tick=tick,action=240,mask=mask) for tick in range(0,100,20)]
    (directory/'reset-inputs.json').write_text(json.dumps(dict(case=case,initial_native=actual)))
    (directory/'proposed-actions.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in actions))
    path=directory/'replay.draw'
    with RawWriter(path) as writer:
        writer.append(dict(kind='RESET',tick=0,actual=actual,case_id=case['case_id'],replay_stride_ticks=20))
        for proposal in actions:
            for tick in (proposal['tick']+1,proposal['tick']+20):
                writer.append(dict(kind='STEP',tick=tick,actual=dict(actual,world_time=tick*.01),
                    action=240,action_tick=proposal['tick'],structural_mask=mask,
                    case_id=case['case_id'],replay_stride_ticks=20))
    segment=e.file_record(path,root)
    segment['path']=str(path)
    capture=dict(schema='dual-validation-capture-v1',actions=actions,terminal_tick=100,
        mission=dict(tick=100,terminal_reason='success',synthetic_test_only=True),outcome=outcome,
        final_backend=dict(qpos=[0.],qvel=[0.]),infrastructure_errors=[],replay_segments=[segment],wall_seconds=1.,
        audit_report=dict(control_steps=100,rk_states=6400,native_poses=1601,reconstructed_intervals=1600,
                          body_reconstructed_intervals=1600,proof_gaps=[],errors=[]))
    (directory/'capture.json').write_text(json.dumps(capture))
    bundle=e.bundle_captures(root/'capture.zip',[(directory,capture)],root=root)
    document['missions'][0]=e.build_single_capture_receipts(SimpleNamespace(binding=bound),first['result']['identity'],
        capture,bundle,development_replay_reference=reference_record)
    old_payload=first['replay']['inventory']['reset']['path']
    roles['validation_evidence'].remove(old_payload)
    roles['validation_evidence'].append('capture.zip')
    (root/'evidence.json').write_text(json.dumps(document))
    return roles,selection,cases,manifests


@pytest.mark.parametrize('damage',[
    'none','reexecution','forged_v1_flag','unqualified_reference','different_reference','reference_tamper',
    'action_count','state_count','reset_case','retained_unverified','stream_mismatch','missing_member',
    'duplicate_member','unsafe_member','capture_result','capture_audit','inventory_split',
])
def test_v2_validation_requires_bound_qualified_reference_and_actual_retained_stream(tmp_path,damage):
    import zipfile
    import warnings
    import adaptive_drone_patrol.dual_coordination_campaign as c
    from adaptive_drone_patrol.dual_coordination_evaluate import file_record

    roles,selection,cases,manifests=single_capture_validation_fixture(tmp_path)
    document=json.loads((tmp_path/'evidence.json').read_text())
    mission=document['missions'][0]
    replay=mission['replay']
    if damage=='reexecution':
        replay['native_reexecution_performed']=True
    elif damage=='forged_v1_flag':
        replay['passed']=False
    elif damage=='different_reference':
        replay['development_replay_reference']['sha256']='0'*64
    elif damage=='action_count':
        replay['action_records']=4
    elif damage=='state_count':
        replay['retained_state_records']=10
    elif damage=='reset_case':
        replay['reset_case_sha256']='0'*64
    elif damage=='retained_unverified':
        replay['retained_inputs_verified']=False
    elif damage=='unqualified_reference':
        path=tmp_path/'qualification.json'
        value=json.loads(path.read_text())
        value['compact_online_evidence_verified']=False
        path.write_text(json.dumps(value))
    elif damage=='reference_tamper':
        (tmp_path/'development-replay.json').write_text('{}')
    elif damage=='inventory_split':
        replay['inventory']['actions']=document['missions'][1]['replay']['inventory']['actions']
        replay['inventory_sha256']=c._digest(replay['inventory'])
    elif damage in ('stream_mismatch','missing_member','duplicate_member','unsafe_member','capture_result','capture_audit'):
        path=tmp_path/'capture.zip'
        with zipfile.ZipFile(path) as archive:
            members=[(i.filename,archive.read(i)) for i in archive.infolist()]
        if damage=='missing_member':
            members=[r for r in members if r[0]!='0/proposed-actions.jsonl']
        elif damage=='duplicate_member':
            members.append(members[0])
        elif damage=='unsafe_member':
            members.append(('../escape',b'unsafe'))
        elif damage=='stream_mismatch':
            members=[(name,b'{}\n' if name=='0/proposed-actions.jsonl' else value) for name,value in members]
        else:
            changed=[]
            for name,value in members:
                if name=='0/capture.json':
                    capture=json.loads(value)
                    if damage=='capture_result':
                        capture['final_backend']['qpos']=[1.]
                    else:
                        capture['audit_report']['rk_states']=0
                    value=json.dumps(capture).encode()
                changed.append((name,value))
            members=changed
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(path,'w') as archive:
                for name,value in members:
                    archive.writestr(name,value)
        record=file_record(path,tmp_path)
        replay['inventory']=dict.fromkeys(('reset','actions','compact_replay'),record)
        replay['inventory_sha256']=c._digest(replay['inventory'])
    (tmp_path/'evidence.json').write_text(json.dumps(document))
    bound=c.BoundFiles.capture(tmp_path,roles)
    if damage=='none':
        c._verify_validation_evidence(bound,selection,cases,manifests)
    else:
        with pytest.raises(ValueError):
            c._verify_validation_evidence(bound,selection,cases,manifests)
