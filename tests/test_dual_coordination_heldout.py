"""Software-only heldout orchestration tests; no heldout cases are generated."""

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    return importlib.import_module("adaptive_drone_patrol.dual_coordination_heldout")


def test_requires_typed_verified_capability_before_output_or_case_access(tmp_path):
    m = module()
    for invalid in (None, True, {"selected_seed": 42}):
        with pytest.raises(ValueError, match="VerifiedHeldout"):
            m.run_heldout(
                capability=invalid,
                cohort="routing",
                original="missing",
                design_path="missing",
                output=tmp_path / "new",
            )
    assert not (tmp_path / "new").exists()


def test_fixed_priority_keeps_canonical_single_active_priority_and_counts_eligible_only(tmp_path):
    m = module()
    from adaptive_drone_patrol.dual_coordination_schema import encode_action, WAIT

    chosen = [encode_action(WAIT, 3, 2), encode_action(2, 3, 2), encode_action(2, 3, 1)]
    calls = []

    def policy(env):
        calls.append(env.mission.tick)
        return chosen[len(calls) - 1]

    selector = m.AblationSelector(policy, "N-fixed-priority", tmp_path / "actions.jsonl")
    mask = np.zeros(242, dtype=bool)
    mask[chosen[0]] = True
    env = SimpleNamespace(mission=SimpleNamespace(tick=0), legal_mask=lambda: mask)
    assert selector(env) == chosen[0]
    mask[:] = False
    mask[chosen[1]] = mask[chosen[2]] = True
    env.mission.tick = 20
    assert selector(env) == chosen[2]
    env.mission.tick = 40
    assert selector(env) == chosen[2]
    assert calls == [0, 20, 40]
    assert selector.counters == dict(forward_passes=3, priority_opportunities=2, priority_overwrites=1)
    assert len((tmp_path / "actions.jsonl").read_text().splitlines()) == 3


def test_invalid_network_proposal_is_not_canonicalized_or_repaired(tmp_path):
    m = module()
    mask = np.zeros(242, dtype=bool)
    mask[0] = True
    env = SimpleNamespace(mission=SimpleNamespace(tick=0), legal_mask=lambda: mask)
    selector = m.AblationSelector(lambda env: 1, "N-fixed-priority", tmp_path / "actions.jsonl")
    with pytest.raises(ValueError, match="illegal"):
        selector(env)
    assert json.loads((tmp_path / "actions.jsonl").read_text())["proposed_action"] == 1


def test_no_transfer_calls_policy_on_current_world_each_decision(tmp_path):
    m = module()
    seen = []

    def policy(env):
        seen.append(env.mission.tick)
        return int(env.mission.tick // 20)

    env = SimpleNamespace(mission=SimpleNamespace(tick=0), legal_mask=lambda: np.ones(242, dtype=bool))
    selector = m.AblationSelector(policy, "N-no-transfer", tmp_path / "actions.jsonl")
    assert selector(env) == 0
    env.mission.tick = 20
    assert selector(env) == 1
    assert seen == [0, 20] and selector.counters["priority_overwrites"] == 0


def test_claimed_slot_is_never_rerun_even_after_incomplete_attempt(tmp_path):
    m = module()
    slot = tmp_path / "slot"
    assert m.claim_slot(slot, {"case_id": "synthetic"}) is not None
    assert m.claim_slot(slot, {"case_id": "synthetic"}) is None
    assert (slot / "ownership.json").is_file()
    with pytest.raises(ValueError, match="identity"):
        m.claim_slot(slot, {"case_id": "changed"})


def test_unverified_failure_is_conservative_and_scored600():
    m = module()
    case = {"case_id": "synthetic", "scenario": "normal"}
    outcome = m.unverified_outcome(case, "missing audit")
    assert not outcome["success"] and outcome["safety_violations"] == 1
    assert m.time_score(outcome) == 600.0
    assert outcome["failure_reason"] == "missing audit"


def test_typed_identity_rejects_wrong_seed_ablation_or_physical_method():
    m = module()
    fields = dict(
        cohort="routing",
        method="N-no-transfer",
        case_id="synthetic",
        case_sha256="a" * 64,
        model_sha256="b" * 64,
        model_kind="checkpoint",
        selected_seed=42,
        freeze_sha256="c" * 64,
        heldout_manifest_sha256="d" * 64,
        experiment_inputs_sha256="e" * 64,
        no_transfer=True,
    )
    assert m.HeldoutIdentity(**fields).to_dict() == fields
    for change in (
        {"selected_seed": 45},
        {"cohort": "physical"},
        {"no_transfer": False},
        {"model_kind": "baseline"},
    ):
        with pytest.raises(ValueError):
            m.HeldoutIdentity(**dict(fields, **change))


def test_routing_numeric_report_keeps_all_unverified_slots_and_crossing_unmet():
    m = module()
    rows = [
        m.unverified_outcome(
            {"case_id": f"synthetic-{i}", "scenario": ("normal", "priority", "availability-delay")[i // 100]},
            "never run",
        )
        for i in range(300)
    ]
    result = m.numeric_reports({method: rows for method in m.ROUTING_METHODS}, "routing")
    assert result["comparison"]["denominator"] == 300
    assert result["comparison"]["mean_score"] == 600.0
    assert result["comparison"]["safety_violations"] == 300
    assert result["bootstrap"]["resamples"] == 10000
    assert result["bootstrap"]["seed"] == 10272799
    assert result["checks"]["crossing_yield"] is False
    assert result["passed"] is False


def test_verified_ablation_log_counters_and_tampering(tmp_path):
    m = module()
    from adaptive_drone_patrol.dual_coordination_schema import encode_action

    action = encode_action(2, 3, 2)
    mask = np.zeros(242, dtype=bool)
    mask[action] = mask[encode_action(2, 3, 1)] = True
    path = tmp_path / "nn.jsonl"
    choose = m.AblationSelector(lambda env: action, "N-fixed-priority", path)
    env = SimpleNamespace(mission=SimpleNamespace(tick=0), legal_mask=lambda: mask)
    executed = choose(env)
    import zipfile

    archive = tmp_path / "capture.zip"
    with zipfile.ZipFile(archive, "x") as stream:
        stream.writestr(
            "0/proposed-actions.jsonl", json.dumps(dict(tick=0, action=executed, mask=mask.tolist())) + "\n"
        )
    document = dict(
        identity=dict(method="N-fixed-priority", no_transfer=False),
        ablation=dict(counters=choose.counters, proposals=m.file_record(path, tmp_path)),
        receipts=dict(
            result=dict(outcome=dict(decision_intervals=1), mission=dict(no_transfer=False)),
            replay=dict(inventory=dict(actions=m.file_record(archive, tmp_path))),
        ),
    )
    assert m.verify_ablation(document, root=tmp_path) == choose.counters
    row = json.loads(path.read_text())
    row["executed_action"] = action
    path.write_text(json.dumps(row) + "\n")
    document["ablation"]["proposals"] = m.file_record(path, tmp_path)
    with pytest.raises(ValueError, match="Unauthorized"):
        m.verify_ablation(document, root=tmp_path)


def single_capture_receipts(tmp_path):
    """Synthetic receipt contract only: these files are not flight evidence."""
    m = module()
    case = dict(case_id="synthetic-contract", scenario="normal")
    identity = dict(case_sha256=m._digest(case), no_transfer=False)
    outcome = dict(
        m.unverified_outcome(case, "synthetic safe timeout"), safety_violations=0, decision_intervals=1
    )
    result = dict(
        schema="dual-heldout-result-v1",
        identity=identity,
        terminal_tick=20,
        outcome=outcome,
        time_score=600.0,
        mission=dict(no_transfer=False),
    )
    audit = dict(
        schema="dual-heldout-audit-v1",
        identity=identity,
        result_sha256=m._digest(result),
        outcome=outcome,
        auditor_sha256="auditor",
        complete=True,
        errors=[],
        proof_gaps=[],
        coverage=dict(control_steps=20, rk_states=1280, native_substep_intervals=320, swept_intervals=320),
        report=dict(body_reconstructed_intervals=320, safe=True),
    )
    artifact = tmp_path / "synthetic-bundle"
    artifact.write_text("synthetic contract fixture")
    reference = tmp_path / "synthetic-development-reference.json"
    reference.write_text('{"synthetic":true}')
    record = m.file_record(artifact, tmp_path)
    dev = m.file_record(reference, tmp_path)
    inventory = dict.fromkeys(("reset", "actions", "compact_replay"), record)
    replay = dict(
        schema="dual-heldout-replay-v2",
        identity=identity,
        result_sha256=m._digest(result),
        audit_sha256=m._digest(audit),
        outcome=outcome,
        configuration_sha256="configuration",
        inventory=inventory,
        inventory_sha256=m._digest(inventory),
        control_steps=20,
        reset_records=1,
        execution_mode="retained-input-stream-validation",
        native_reexecution_performed=False,
        development_replay_reference=dev,
        retained_inputs_verified=True,
        compact_boundary_consistency=True,
        errors=[],
        reset_case_sha256=m._digest(case),
        action_records=1,
        retained_state_records=3,
    )
    return dict(result=result, audit=audit, replay=replay), identity, case, dev


def test_single_capture_receipt_accepts_safe_failure_without_claiming_native_reexecution(tmp_path):
    m = module()
    receipts, identity, case, dev = single_capture_receipts(tmp_path)
    outcome = m.verify_receipts(
        receipts,
        identity,
        case,
        root=tmp_path,
        auditor_hash="auditor",
        configuration_hash="configuration",
        development_replay_reference=dev,
    )
    assert outcome["success"] is False and outcome["safety_violations"] == 0
    assert m.time_score(outcome) == 600.0


@pytest.mark.parametrize(
    "change",
    [
        dict(native_reexecution_performed=True),
        dict(passed=True),
        dict(decisions_match=True),
        dict(terminal_match=True),
        dict(metrics_within_tolerance=True),
        dict(audit={}),
        dict(action_records=2),
        dict(retained_inputs_verified=False),
        dict(errors=["missing reset"]),
        dict(reset_case_sha256="wrong"),
        dict(retained_state_records=0),
    ],
)
def test_single_capture_receipt_rejects_missing_or_falsely_claimed_evidence(tmp_path, change):
    m = module()
    receipts, identity, case, dev = single_capture_receipts(tmp_path)
    receipts["replay"].update(change)
    with pytest.raises(ValueError):
        m.verify_receipts(
            receipts,
            identity,
            case,
            root=tmp_path,
            auditor_hash="auditor",
            configuration_hash="configuration",
            development_replay_reference=dev,
        )


def test_single_capture_requires_exact_still_present_frozen_development_reference(tmp_path):
    m = module()
    receipts, identity, case, dev = single_capture_receipts(tmp_path)
    kwargs = dict(root=tmp_path, auditor_hash="auditor", configuration_hash="configuration")
    with pytest.raises(ValueError, match="development"):
        m.verify_receipts(receipts, identity, case, **kwargs)
    (tmp_path / dev["path"]).write_text("changed")
    with pytest.raises(ValueError, match="development"):
        m.verify_receipts(receipts, identity, case, development_replay_reference=dev, **kwargs)


@pytest.mark.parametrize(
    "change",
    [
        dict(decision_intervals=-1),
        dict(success="false"),
        dict(safety_violations=-1),
        dict(supervisor_held_drone_seconds=-1.0),
        dict(intervened_intervals=2),
        dict(useful_post_event_transfer=True),
        dict(failure_reason=None),
        dict(extra="not allowed"),
    ],
)
def test_malformed_outcomes_fail_before_aggregate_denominator_reduction(tmp_path, change):
    m = module()
    receipts, identity, case, dev = single_capture_receipts(tmp_path)
    receipts["result"]["outcome"].update(change)
    receipts["audit"]["result_sha256"] = m._digest(receipts["result"])
    receipts["replay"]["result_sha256"] = m._digest(receipts["result"])
    receipts["replay"]["audit_sha256"] = m._digest(receipts["audit"])
    with pytest.raises(ValueError, match="outcome"):
        m.verify_receipts(
            receipts,
            identity,
            case,
            root=tmp_path,
            auditor_hash="auditor",
            configuration_hash="configuration",
            development_replay_reference=dev,
        )


def test_physical_numeric_report_retains_every_unverified_group_slot():
    m = module()
    rows = []
    for group, counts in [("basic", (3, 3, 3)), ("endpoints", (6, 6, 6)), ("mixed", (34, 33, 33))]:
        for scenario, count in zip(("normal", "priority", "availability-delay"), counts):
            for index in range(count):
                row = m.unverified_outcome(
                    dict(case_id=f"synthetic-{group}-{scenario}-{index}", scenario=scenario), "unstarted"
                )
                row.update(group=group, independently_audited=False)
                rows.append(row)
    report = m.numeric_reports({"N-full": rows}, "physical")
    assert report["denominator"] == 127 and report["mean_score"] == 600
    assert report["safety_violations"] == 127 and report["passed"] is False
    assert report["groups"]["basic"]["denominator"] == 9
    assert report["groups"]["endpoints"]["denominator"] == 18
    assert report["groups"]["mixed"]["denominator"] == 100
