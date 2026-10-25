import copy
from collections import Counter

import pytest


def test_development_allocation_is_fifteen_disjoint_full_missions_without_formal_draw(monkeypatch):
    import adaptive_drone_patrol.payload_cases as original

    monkeypatch.setattr(original, "build_cases", lambda: pytest.fail("formal draw before freeze"))
    from adaptive_drone_patrol.payload_hybrid_final import robustness_cases, validation_cases

    cases = robustness_cases()
    assert len(cases) == 15
    assert Counter(c["allocation"] for c in cases) == dict(nominal=3, interior=6, endpoint=6)
    assert Counter(c["scenario"] for c in cases) == dict(normal=5, emergency=5, cost_shift=5)
    assert len({c["execution_seed"] for c in cases}) == 15
    assert all(c["execution_seed"] > 82627011 for c in cases)
    interior = [c for c in cases if c["allocation"] == "interior"]
    assert {c["side_m"] for c in interior} == {0.03, 0.045, 0.06}
    assert all(0.1 < c["mass_ratio"] < 0.3 for c in interior)
    assert len({c["mass_ratio"] for c in interior}) == 6
    endpoint = [c for c in cases if c["allocation"] == "endpoint"]
    assert {(c["side_m"], c["mass_ratio"]) for c in endpoint} == {
        (s, m) for s in (0.03, 0.045, 0.06) for m in (0.1, 0.3)
    }
    assert all(c["initial"]["swing_angle_deg"] == 10 for c in endpoint)
    validation = validation_cases()
    assert len(validation) == 12
    assert Counter(c["execution_seed"] for c in validation) == {11: 3, 12: 3, 13: 3, 14: 3}
    assert all(len(c["targets"]) == 10 for c in validation)
    assert robustness_cases() == cases


def sample(step=0):
    s = dict(
        step=step,
        time_s=step * 0.01,
        actions=[0.0] * 4,
        effective_target=[0.0, 0.0, 1.0],
        contact=False,
        contacts=False,
        done=False,
        reset=False,
    )
    for body, xyz in [("drone", [0.0, 0.0, 1.0]), ("package", [0.0, 0.0, 0.6575])]:
        s.update(
            {
                body + "_xyz": xyz,
                body + "_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                body + "_velocity": [0.0] * 3,
                body + "_angular_velocity": [0.0] * 3,
            }
        )
    return s


def sequence(steps=101, *, start=0):
    rows = [dict(kind="INITIAL", step=0, sample=sample())]
    rows.append(dict(kind="RETARGET", step=0, target_index="0", target=[0.0, 0.0, 1.0], sample=sample()))
    rows.extend(
        dict(
            kind="STEP",
            step=i,
            target_index="0",
            target=[0.0, 0.0, 1.0],
            sample=sample(i),
            reset_elapsed_wall_s=i * 0.02,
        )
        for i in range(1, steps + 1)
    )
    return rows


def test_sequence_reconstruction_requires_complete_intervals_and_finite_flush():
    from adaptive_drone_patrol.payload_hybrid_final_audit import reconstruct_sequence

    case = dict(side_m=0.045, targets=[[0.0, 0.0, 1.0]])
    checked = reconstruct_sequence(sequence(), case, flush_elapsed_wall_s=3.0)
    assert checked["success"] and checked["completed_targets"] == 1
    assert not reconstruct_sequence(sequence(100), case, flush_elapsed_wall_s=3.0)["success"]
    assert not reconstruct_sequence(sequence(), case, flush_elapsed_wall_s=600.0)["success"]
    for late in (float("nan"), -1, True):
        with pytest.raises(ValueError):
            reconstruct_sequence(sequence(), case, flush_elapsed_wall_s=late)


@pytest.mark.parametrize("mutation", ["clock", "skip", "unsafe", "retarget", "wall", "hidden"])
def test_sequence_mutations_reject(mutation):
    from adaptive_drone_patrol.payload_hybrid_final_audit import reconstruct_sequence

    rows = sequence()
    if mutation == "clock":
        rows[3]["sample"]["time_s"] = 0.5
    if mutation == "skip":
        rows[3]["step"] = 7
    if mutation == "unsafe":
        rows[3]["sample"]["contacts"] = rows[3]["sample"]["contact"] = True
    if mutation == "retarget":
        rows[1]["sample"]["drone_xyz"][0] = 0.1
    if mutation == "wall":
        rows[3]["reset_elapsed_wall_s"] = float("nan")
    if mutation == "hidden":
        rows[3]["sample"]["reset"] = True
    with pytest.raises(ValueError):
        reconstruct_sequence(rows, dict(side_m=0.045, targets=[[0.0, 0.0, 1.0]]), flush_elapsed_wall_s=3.0)


def test_no_extra_intervals_after_completion_or_stationary_command_relabel():
    from adaptive_drone_patrol.payload_hybrid_final_audit import reconstruct_sequence

    with pytest.raises(ValueError):
        reconstruct_sequence(
            sequence(102), dict(side_m=0.045, targets=[[0.0, 0.0, 1.0]]), flush_elapsed_wall_s=3.0
        )
    rows = sequence()
    rows.append(dict(kind="RETARGET", step=101, target_index="1", target=[0.0, 0.0, 1.0], sample=sample(101)))
    # Inherited prescribed duplicate target is a new settling leg, never a physical reset.
    check = reconstruct_sequence(
        rows, dict(side_m=0.045, targets=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]), flush_elapsed_wall_s=3.0
    )
    assert not check["success"] and check["completed_targets"] == 1


def test_final_capabilities_reject_early_receipts_and_arbitrary_construction():
    from adaptive_drone_patrol.payload_hybrid_final import FinalFreeze, formal_cases

    with pytest.raises(ValueError):
        FinalFreeze({}, None)
    with pytest.raises(ValueError):
        formal_cases(dict(success=True, physical_acceptance=True))


def test_statistics_fixed_denominators_failures_wilson_and_subgroups():
    from adaptive_drone_patrol.payload_hybrid_final_audit import summarize_outcomes

    cases = []
    for suite, n in [("basic", 9), ("endpoints", 18), ("heldout", 100)]:
        for i in range(n):
            cases.append(
                dict(
                    case_id=f"{suite}-{i}",
                    suite=suite,
                    scenario=("normal", "emergency", "cost_shift")[i % 3],
                    side_m=(0.03, 0.045, 0.06)[i % 3],
                    mass_ratio=0.1 + 0.2 * (i % 10) / 9,
                )
            )
    rows = [
        dict(case_id=c["case_id"], success=(i < 122), strict_shadow=[dict(passed=False)], routing_gaps=[])
        for i, c in enumerate(cases)
    ]
    result = summarize_outcomes(cases, rows)
    assert result["accepted"] and result["suites"]["heldout"]["successes"] == 95
    assert result["suites"]["heldout"]["wilson95"][0] < 0.95
    rows[-6]["success"] = False
    assert not summarize_outcomes(cases, rows)["accepted"]
    with pytest.raises(ValueError):
        summarize_outcomes(cases, rows[:-1])
    with pytest.raises(ValueError):
        summarize_outcomes(cases, rows[:-1] + [rows[0]])
    rows[0]["success"] = 1
    with pytest.raises(ValueError):
        summarize_outcomes(cases, rows)


def test_required_duration_gate_uses_independent_auditor_and_binds_design_reference(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from adaptive_drone_patrol.payload_hybrid_final_audit import audit_duration_prerequisite
    import adaptive_drone_patrol.payload_hybrid_duration_audit as duration

    design = {"recipe": "C-local-lqr-v1"}
    (tmp_path / "design.json").write_text(json.dumps(design))
    declaration = {"design": design}
    (tmp_path / "inputs.json").write_text(json.dumps(declaration))
    context = SimpleNamespace(
        paths={"design": str(tmp_path / "design.json")}, reference=SimpleNamespace(binding={"physics": "old"})
    )
    proof = dict(
        schema="hybrid-independent-continuous-duration-group-v1",
        passed=True,
        distinct_cases=7,
        intervals_per_case=33000,
        duration_s=330.0,
        full_native_physics_replay=True,
        reference_proof={"actual_binding": {"physics": "old"}},
    )
    monkeypatch.setattr(duration, "audit_group", lambda root, source_root: copy.deepcopy(proof))
    assert audit_duration_prerequisite(context, tmp_path)["summary"]["distinct_cases"] == 7
    for key, bad in [
        ("passed", False),
        ("full_native_physics_replay", False),
        ("distinct_cases", 6),
        ("duration_s", 329.99),
    ]:
        changed = copy.deepcopy(proof)
        changed[key] = bad
        monkeypatch.setattr(duration, "audit_group", lambda root, source_root: changed)
        with pytest.raises(ValueError):
            audit_duration_prerequisite(context, tmp_path)
    monkeypatch.setattr(duration, "audit_group", lambda root, source_root: copy.deepcopy(proof))
    declaration["design"] = {"recipe": "different"}
    (tmp_path / "inputs.json").write_text(json.dumps(declaration))
    with pytest.raises(ValueError):
        audit_duration_prerequisite(context, tmp_path)


def test_freeze_current_verification_does_not_accept_mutated_prerequisite_bytes(tmp_path):
    from adaptive_drone_patrol.payload_hybrid_final_audit import artifact_hashes, require_artifact_hashes

    (tmp_path / "raw.json").write_text('{"kind":"RESET"}')
    saved = artifact_hashes(tmp_path)
    require_artifact_hashes(saved)
    (tmp_path / "raw.json").write_text('{"kind":"RESET","hidden":1}')
    with pytest.raises(ValueError):
        require_artifact_hashes(saved)


def test_final_source_closure_binds_route_checker_sequence_and_duration_without_draw(monkeypatch):
    import adaptive_drone_patrol.payload_cases as original

    monkeypatch.setattr(original, "build_cases", lambda: pytest.fail("draw during source binding"))
    from adaptive_drone_patrol.payload_hybrid_final import _source_closure

    sources = _source_closure()
    for name in (
        "payload_hybrid_flight",
        "payload_hybrid_controller",
        "payload_hybrid_runtime",
        "payload_hybrid_contract",
        "payload_solver_mujoco",
        "payload_solver_runtime",
        "payload_solver_acceptance",
        "payload_hybrid_native",
        "payload_hybrid_duration",
        "payload_hybrid_duration_audit",
        "routing_agent",
        "objective",
        "scenario",
        "payload_hybrid_final",
        "payload_hybrid_final_audit",
    ):
        assert f"src/adaptive_drone_patrol/{name}.py" in sources
    assert "scripts/run_payload_hybrid_final.py" in sources
    assert all(
        f"configs/demo_scenarios_dispersed/{s}.yaml" in sources for s in ("normal", "emergency", "cost_shift")
    )


def test_formal_declaration_cannot_be_consumed_without_typed_freeze(tmp_path):
    from adaptive_drone_patrol.payload_hybrid_final import require_formal_declaration

    with pytest.raises(ValueError):
        require_formal_declaration(tmp_path, {"accepted": True})


def test_flush_audit_binds_closed_raw_and_control_and_preserves_late_result(tmp_path):
    import json
    from adaptive_drone_patrol.payload_hybrid_final_audit import audit_flush_receipt
    from adaptive_drone_patrol.util import sha256

    (tmp_path / "flight/runtime").mkdir(parents=True)
    raw = tmp_path / "flight/runtime/raw.jsonl.gz"
    raw.write_bytes(b"raw")
    control = tmp_path / "flight/control.jsonl.gz"
    control.write_bytes(b"control")
    receipt = dict(
        schema="payload-hybrid-final-flush-v1",
        flushed=True,
        elapsed_wall_s=600.0,
        error=None,
        raw_sha256=sha256(raw),
        control_sha256=sha256(control),
    )
    (tmp_path / "final-flush.json").write_text(json.dumps(receipt))
    assert audit_flush_receipt(tmp_path) == 600.0
    control.write_bytes(b"changed")
    with pytest.raises(ValueError):
        audit_flush_receipt(tmp_path)
    receipt["control_sha256"] = sha256(control)
    for bad in (float("nan"), True, -1):
        receipt["elapsed_wall_s"] = bad
        (tmp_path / "final-flush.json").write_text(json.dumps(receipt))
        with pytest.raises(ValueError):
            audit_flush_receipt(tmp_path)


def test_development_case_requires_preexisting_external_campaign_declaration(tmp_path):
    from adaptive_drone_patrol.payload_hybrid_final import require_development_attempt, robustness_cases

    case = robustness_cases()[0]
    with pytest.raises(ValueError):
        require_development_attempt(tmp_path / "robustness" / case["case_id"], None, case)


def test_unreadable_evidence_cannot_consume_the_five_allowed_physical_failures():
    from adaptive_drone_patrol.payload_hybrid_final_audit import evidence_gate

    rows = [dict(case_id=f"case-{i}", success=i < 122, evidence_complete=True) for i in range(127)]
    assert evidence_gate(rows)["complete"]
    rows[-1]["evidence_complete"] = False
    result = evidence_gate(rows)
    assert not result["complete"] and result["incomplete_case_ids"] == ["case-126"]
    rows[-1]["evidence_complete"] = True
    rows[-1]["success"] = False
    assert evidence_gate(rows)["complete"]
    rows[-1]["evidence_complete"] = 1
    assert not evidence_gate(rows)["complete"]


def failed_cpu_mission(tmp_path, mode):
    from test_payload_recovery_mission import CpuFlight, Planner, scenario
    from adaptive_drone_patrol.payload_recovery_mission import execute_payload_mission

    if mode == "unsafe":
        flight = CpuFlight(override=lambda f: f.current.update(contacts=True))

        def clock():
            return 0.0
    elif mode == "post-wall":
        flight = CpuFlight()
        ticks = iter([0.0, 0.0, 0.0, 600.0])

        def clock():
            return next(ticks, 600.0)
    else:
        flight = CpuFlight()
        ticks = iter([0.0, 0.0, 600.0])

        def clock():
            return next(ticks, 600.0)

    root = tmp_path / mode
    execute_payload_mission(
        flight,
        Planner(),
        scenario(),
        root,
        dict(evidence_kind="cpu-fixture", routing_model_hash="a" * 64),
        clock=clock,
        revision2=True,
    )
    return root


@pytest.mark.parametrize("mode,termination", [("unsafe", "SAFETY_FAILURE"), ("post-wall", "WALL_TIMEOUT")])
def test_genuine_terminal_raw_without_predicate_is_independently_audited(tmp_path, mode, termination):
    from test_payload_recovery_mission import Planner
    from adaptive_drone_patrol.payload_hybrid_final_audit import verify_final_mission

    root = failed_cpu_mission(tmp_path, mode)
    result = verify_final_mission(root, Planner())
    assert not result["success"] and result["completed_count"] == 0
    assert result["independent_termination_code"] == termination
    assert result["terminal_raw_without_predicate_verified"]
    assert result["steps"] == 1 and result["native_proof"] is None  # CPU protocol fixture only.


def test_missing_ordinary_predicate_is_not_a_terminal_safety_or_wall_failure(tmp_path):
    import json
    from test_payload_recovery_mission import Planner
    from adaptive_drone_patrol.payload_hybrid_final_audit import verify_final_mission
    from adaptive_drone_patrol.util import sha256

    root = failed_cpu_mission(tmp_path, "unsafe")
    rows = [json.loads(s) for s in (root / "samples.jsonl").read_text().splitlines()]
    rows[-1]["sample"]["contacts"] = False
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = json.loads((root / "mission_summary.json").read_text())
    summary["telemetry_sha256"] = sha256(root / "samples.jsonl")
    (root / "mission_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        verify_final_mission(root, Planner())


def test_final_zero_step_command_is_reconstructed_against_original_routing(tmp_path):
    import json
    from test_payload_recovery_mission import Planner
    from adaptive_drone_patrol.payload_hybrid_final_audit import verify_final_mission
    from adaptive_drone_patrol.util import sha256

    root = failed_cpu_mission(tmp_path, "pre-wall")
    result = verify_final_mission(root, Planner())
    assert not result["success"] and result["steps"] == 0
    decisions = [json.loads(s) for s in (root / "decisions.jsonl").read_text().splitlines()]
    decisions[0]["selected_action"] = 1
    (root / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decisions))
    summary = json.loads((root / "mission_summary.json").read_text())
    summary["decisions_sha256"] = sha256(root / "decisions.jsonl")
    (root / "mission_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        verify_final_mission(root, Planner())


def test_terminal_offending_raw_must_match_active_original_route_command(tmp_path):
    import json
    from test_payload_recovery_mission import Planner
    from adaptive_drone_patrol.payload_hybrid_final_audit import verify_final_mission
    from adaptive_drone_patrol.util import sha256

    root = failed_cpu_mission(tmp_path, "unsafe")
    rows = [json.loads(s) for s in (root / "samples.jsonl").read_text().splitlines()]
    rows[-1]["target"] = [9.0, 9.0, 9.0]
    rows[-1]["target_id"] = "fabricated"
    rows[-1]["sample"]["effective_target"] = [9.0, 9.0, 9.0]
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = json.loads((root / "mission_summary.json").read_text())
    summary["telemetry_sha256"] = sha256(root / "samples.jsonl")
    (root / "mission_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        verify_final_mission(root, Planner())


def test_sequence_cannot_continue_after_wall_deadline_even_as_allowed_failure():
    from adaptive_drone_patrol.payload_hybrid_final_audit import reconstruct_sequence

    rows = sequence()
    for row in rows[2:]:
        row["reset_elapsed_wall_s"] = 600.0 + row["step"] * 0.01
    with pytest.raises(ValueError):
        reconstruct_sequence(rows, dict(side_m=0.045, targets=[[0.0, 0.0, 1.0]]), flush_elapsed_wall_s=700.0)


def test_prescribed_duplicate_holds_require_new_full_physical_interval_dwell():
    from adaptive_drone_patrol.payload_hybrid_final_audit import reconstruct_sequence

    rows = sequence()
    rows.append(dict(kind="RETARGET", step=101, target_index="1", target=[0.0, 0.0, 1.0], sample=sample(101)))
    for i in range(102, 203):
        rows.append(
            dict(
                kind="STEP",
                step=i,
                target_index="1",
                target=[0.0, 0.0, 1.0],
                sample=sample(i),
                reset_elapsed_wall_s=i * 0.02,
            )
        )
    result = reconstruct_sequence(
        rows, dict(side_m=0.045, targets=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]), flush_elapsed_wall_s=5.0
    )
    assert result["success"]
    assert result["commanded_target_legs"] == 2 and result["distinct_coordinate_transfers"] == 0
    assert result["initial_or_duplicate_holds"] == 2
    assert all(
        leg["valid_intervals"] == 100 and leg["end_step"] - leg["start_step"] == 101 for leg in result["legs"]
    )


@pytest.mark.parametrize("kind", ["duplicate-start", "duplicate-leg", "early-terminal"])
def test_failed_transcript_rejects_duplicate_or_postterminal_events(tmp_path, kind):
    import json
    from test_payload_recovery_mission import Planner
    from adaptive_drone_patrol.payload_hybrid_final_audit import verify_final_mission
    from adaptive_drone_patrol.util import sha256

    root = failed_cpu_mission(tmp_path, "pre-wall")
    events = [json.loads(s) for s in (root / "events.jsonl").read_text().splitlines()]
    source = events[0] if kind == "duplicate-start" else events[2] if kind == "duplicate-leg" else events[-1]
    events.insert(3, copy.deepcopy(source))
    for i, e in enumerate(events):
        e["event_seq"] = i + 1
    (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    summary = json.loads((root / "mission_summary.json").read_text())
    summary["events_sha256"] = sha256(root / "events.jsonl")
    (root / "mission_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        verify_final_mission(root, Planner())


def test_verified_artifact_cache_detects_byte_mutation_even_with_restored_mtime(tmp_path):
    import os
    from adaptive_drone_patrol.payload_hybrid_final_audit import VerifiedArtifactSet, artifact_hashes

    path = tmp_path / "native.gz"
    path.write_bytes(b"first")
    cache = VerifiedArtifactSet(artifact_hashes(tmp_path))
    before = path.stat()
    cache.require_current()
    path.write_bytes(b"other")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError):
        cache.require_current()


def reader_bindings():
    from adaptive_drone_patrol.payload_hybrid_final import PINNED_V1_READER_SOURCES

    old = dict(
        schema="payload-hybrid-final-binding-v1",
        source_sha256=dict(PINNED_V1_READER_SOURCES),
        design_sha256="gain-original",
        design_value_sha256="gain-value",
        actual_reference={"mass": 0.027, "engine": "original"},
        dependencies={"numpy": "same"},
        video_profile_sha256="original-profile",
        motor_inputs="original-four",
        routing_sha256="original44",
    )
    old["source_sha256"]["src/adaptive_drone_patrol/payload_hybrid_controller.py"] = "controller-original"
    current = copy.deepcopy(old)
    for p in PINNED_V1_READER_SOURCES:
        current["source_sha256"][p] = "new-reader-" + p
    return old, current


def test_reader_rebind_changes_only_exact_pinned_reader_sources():
    from adaptive_drone_patrol.payload_hybrid_final import validate_reader_binding_delta

    old, current = reader_bindings()
    assert len(validate_reader_binding_delta(old, current)) == 3
    bad = copy.deepcopy(old)
    bad["source_sha256"][next(iter(bad["source_sha256"]))] = "not-pinned-v1"
    with pytest.raises(ValueError):
        validate_reader_binding_delta(bad, current)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("design_sha256", "different-gain"),
        ("design_value_sha256", "different-gain-values"),
        ("actual_reference", {"mass": 0.0, "engine": "different-model"}),
        ("video_profile_sha256", "different-profile"),
        ("motor_inputs", "different-motor"),
        ("dependencies", {"numpy": "other"}),
        ("routing_sha256", "other-routing"),
    ],
)
def test_reader_rebind_never_exempts_physics_gain_model_profile_motor_dependencies(field, bad):
    from adaptive_drone_patrol.payload_hybrid_final import validate_reader_binding_delta

    old, current = reader_bindings()
    current[field] = bad
    with pytest.raises(ValueError):
        validate_reader_binding_delta(old, current)


def test_reader_rebind_rejects_other_source_and_changed_source_inventory():
    from adaptive_drone_patrol.payload_hybrid_final import validate_reader_binding_delta

    old, current = reader_bindings()
    current["source_sha256"]["src/adaptive_drone_patrol/payload_hybrid_controller.py"] = (
        "different-controller"
    )
    with pytest.raises(ValueError):
        validate_reader_binding_delta(old, current)
    old, current = reader_bindings()
    current["source_sha256"]["unexpected.py"] = "new"
    with pytest.raises(ValueError):
        validate_reader_binding_delta(old, current)


def test_recorded_binding_compatibility_is_never_formal_and_never_allows_dev_execution(monkeypatch):
    from adaptive_drone_patrol.payload_hybrid_final import (
        FinalContext,
        prepare_development,
        run_final_case,
        robustness_cases,
    )

    old, current = reader_bindings()
    context = object.__new__(FinalContext)
    context._binding = current
    context._reader_rebinding = {"old_binding": old, "current_binding": current}
    monkeypatch.setattr(context, "require_current", lambda: None)
    assert context.require_recorded_binding(current, "basic") == current
    with pytest.raises(ValueError):
        context.require_recorded_binding(old, "basic")
    with pytest.raises(ValueError):
        prepare_development("/unused/new-development", context)
    with pytest.raises(ValueError):
        run_final_case(context, robustness_cases()[0], "/unused/case")


def test_preserved_paired_diagnostic_allocation_is_c0_not_c1():
    from adaptive_drone_patrol.payload_hybrid_final_audit import expected_diagnostic_keys

    keys = expected_diagnostic_keys()
    assert len(keys) == 70
    assert all((f"C0-{i:02d}", 16, False) in keys and (f"C0-{i:02d}", 32, True) in keys for i in range(10))
    assert not any(
        case.startswith("C1") and (substeps != 16 or feedback is not True)
        for case, substeps, feedback in keys
    )


def test_reader_receipt_checks_actual_archive_members_and_preserved_declaration(tmp_path, monkeypatch):
    import hashlib
    import io
    import json
    import tarfile
    import adaptive_drone_patrol.payload_hybrid_final as final
    from adaptive_drone_patrol.util import sha256

    old, current = reader_bindings()
    original = {p: ("original-source:" + p).encode() for p in final.PINNED_V1_READER_SOURCES}
    pins = {p: hashlib.sha256(v).hexdigest() for p, v in original.items()}
    monkeypatch.setattr(final, "PINNED_V1_READER_SOURCES", pins)
    old["source_sha256"].update(pins)
    package = tmp_path / "original.tar"
    with tarfile.open(package, "w") as archive:
        for p, data in original.items():
            item = tarfile.TarInfo(p)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps({"binding": old}))
    record = dict(
        schema="payload-hybrid-final-reader-rebinding-v1",
        scope=final.READER_REBIND_SCOPE,
        old_binding=old,
        current_binding=current,
        reader_source_changes=final.validate_reader_binding_delta(old, current),
        pinned_v1_reader_sources=pins,
        old_source_package=str(package),
        old_source_package_sha256=sha256(package),
        old_source_package_files=final.reader_source_package_files(package),
        old_development_declaration=str(declaration),
        old_development_declaration_sha256=sha256(declaration),
        formal_compatibility=False,
    )
    assert final.validate_reader_rebinding(record, current) == record
    declaration.write_text(json.dumps({"binding": current}))
    with pytest.raises(ValueError):
        final.validate_reader_rebinding(record, current)
    declaration.write_text(json.dumps({"binding": old}))
    with tarfile.open(package, "w") as archive:
        for p, data in original.items():
            data += b"mutation"
            item = tarfile.TarInfo(p)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    record["old_source_package_sha256"] = sha256(package)
    with pytest.raises(ValueError):
        final.validate_reader_rebinding(record, current)


def test_reader_delta_is_type_exact_in_unchanged_physics_fields():
    from adaptive_drone_patrol.payload_hybrid_final import validate_reader_binding_delta

    old, current = reader_bindings()
    old["actual_reference"]["batch_size"] = 1
    current["actual_reference"]["batch_size"] = True
    with pytest.raises(ValueError):
        validate_reader_binding_delta(old, current)


def diagnostic_manifest_fixture(tmp_path, cases):
    import json
    from adaptive_drone_patrol.util import sha256

    root = tmp_path / "C-v1-C0"
    root.mkdir()
    path = tmp_path / "development-manifest-v1.json"
    path.write_text(json.dumps({"cases": cases}))
    (root / "inputs.json").write_text(json.dumps({"manifest_sha256": sha256(path), "gate": "C0"}))
    return root, path


def test_original_diagnostic_manifest_keeps_ulp_initial_draw_exact(tmp_path):
    import numpy as np
    from adaptive_drone_patrol.payload_hybrid_manifest import development_cases
    from adaptive_drone_patrol.payload_hybrid_final_audit import frozen_diagnostic_cases

    cases = development_cases()
    original = cases[1]["initial"]["package_xyz"][1]
    cases[1]["initial"]["package_xyz"][1] = float(np.nextafter(original, np.inf))
    root, path = diagnostic_manifest_fixture(tmp_path, cases)
    accepted, manifest = frozen_diagnostic_cases(root)
    assert manifest == path
    assert accepted["C0-01"]["initial"]["package_xyz"][1] == cases[1]["initial"]["package_xyz"][1]
    assert accepted["C0-01"]["initial"]["package_xyz"][1] != original


@pytest.mark.parametrize("mutation", ["target", "allocation", "mass", "initial"])
def test_original_diagnostic_manifest_rejects_material_changed_inputs(tmp_path, mutation):
    from adaptive_drone_patrol.payload_hybrid_manifest import development_cases
    from adaptive_drone_patrol.payload_hybrid_final_audit import frozen_diagnostic_cases

    cases = development_cases()
    if mutation == "target":
        cases[0]["targets"][0][0] = 0.0001
    if mutation == "allocation":
        cases[0]["seed"] += 1
    if mutation == "mass":
        cases[0]["mass_ratio"] += 1e-10
    if mutation == "initial":
        cases[0]["initial"]["package_xyz"][1] += 1e-10
    root, _ = diagnostic_manifest_fixture(tmp_path, cases)
    with pytest.raises(ValueError):
        frozen_diagnostic_cases(root)


def test_original_diagnostic_manifest_hash_must_match_preserved_producer_input(tmp_path):
    from adaptive_drone_patrol.payload_hybrid_manifest import development_cases
    from adaptive_drone_patrol.payload_hybrid_final_audit import frozen_diagnostic_cases

    root, path = diagnostic_manifest_fixture(tmp_path, development_cases())
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        frozen_diagnostic_cases(root)


@pytest.mark.parametrize("field", ["target", "initial"])
def test_raw_case_still_must_equal_original_manifest_exactly_after_redraw_guard(tmp_path, field):
    import json
    import numpy as np
    from adaptive_drone_patrol.payload_hybrid_manifest import development_cases
    from adaptive_drone_patrol.payload_hybrid_final_audit import frozen_diagnostic_cases
    from adaptive_drone_patrol.payload_hybrid_audit import audit_case

    root, _ = diagnostic_manifest_fixture(tmp_path, development_cases())
    frozen, _ = frozen_diagnostic_cases(root)
    expected = frozen["C0-01"]
    actual = copy.deepcopy(expected)
    if field == "target":
        actual["targets"][0][0] = float(np.nextafter(0.0, np.inf))
    else:
        actual["initial"]["package_xyz"][1] = float(np.nextafter(actual["initial"]["package_xyz"][1], np.inf))
    trial = root / "C0-01"
    trial.mkdir()
    (trial / "case.json").write_text(json.dumps(actual))
    with pytest.raises(ValueError, match="Case/command stream differs"):
        audit_case(trial, expected_case=expected)
