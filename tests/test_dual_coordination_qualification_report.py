"""Administrative contract tests use synthetic files; never physical receipts."""

import importlib
import json

import pytest


def module():
    return importlib.import_module("adaptive_drone_patrol.dual_coordination_qualification_report")


def test_empty_evidence_is_diagnostic_with_all_missing_checks(tmp_path):
    m = module()
    result = m.build_qualification_report(
        tmp_path, experiment_roles={}, declaration_path=None, case_directories=[], supplemental_paths=[]
    )
    assert result["schema"] == "dual-qualification-diagnostic-v1"
    assert not result["passed"] and not result["admission_granted"]
    assert set(result["missing_checks"]) == m.NATIVE_CHECKS
    assert result["accepted_case_count"] == 0


def test_boolean_supplement_cannot_claim_physical_experiments(tmp_path):
    m = module()
    path = tmp_path / "fake.json"
    path.write_text(json.dumps(dict(check="address_actuation", passed=True)))
    with pytest.raises(ValueError):
        m.inspect_supplement(tmp_path, "fake.json", {})


def test_measurement_binding_rejects_tamper_and_symlink(tmp_path):
    m = module()
    p = tmp_path / "measurements.json"
    p.write_text("{}")
    binding = m.file_record(tmp_path, "measurements.json")
    p.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed"):
        m.read_bound_json(tmp_path, binding)
    link = tmp_path / "linked.json"
    link.symlink_to(p)
    with pytest.raises(ValueError):
        m.file_record(tmp_path, "linked.json")


def test_pilot_or_boolean_case_is_not_complete_evidence(tmp_path):
    m = module()
    case = m.qualification_cases()[0]
    path = tmp_path / "case"
    path.mkdir()
    (path / "result.json").write_text(json.dumps(dict(case=case, passed=True, pilot=True)))
    with pytest.raises(ValueError, match="pilot|complete"):
        m.inspect_case(tmp_path, "case", case, {})


def test_numeric_monitor_rejects_missing_or_excess_stage_counts():
    m = module()
    row = dict(
        rk_states=64,
        native_states=16,
        maximum_constraint_residual_m=0.0,
        maximum_signed_energy_work_residual_j=0.0,
        motor_work_j=0.0,
        passive_work_j=0.0,
        failures=[],
    )
    m.check_numerics(row, ticks=1, substeps=16)
    for key in ["rk_states", "native_states"]:
        invalid = dict(row, **{key: row[key] - 1})
        with pytest.raises(ValueError, match="coverage"):
            m.check_numerics(invalid, ticks=1, substeps=16)
    with pytest.raises(ValueError):
        m.check_numerics(dict(row, maximum_constraint_residual_m=float("nan")), ticks=1, substeps=16)


def test_all_eighty_one_ids_are_required_not_just_group_totals():
    m = module()
    assert len(m.qualification_cases()) == 81
    rows = m.qualification_cases()
    with pytest.raises(ValueError, match="duplicate"):
        m.index_cases([dict(case=rows[0]), dict(case=rows[0])])


def online_row(ticks=1):
    return dict(
        audited_points=ticks * 80 + 2,
        rk_states=ticks * 64,
        native_poses=ticks * 16 + 1,
        control_steps=ticks,
        reconstructed_intervals=ticks * 16,
        body_reconstructed_intervals=ticks * 16,
        errors=[],
        violations=[],
        reconstruction_failures=[],
        proof_gaps=[],
        safe=True,
        retained_points_safe=True,
        reconstructed_path_safe=True,
        continuous_safety_qualified=True,
        minimum_clearance_m=0.5,
        reconstructed_clearance_lower_bound_m=0.5,
        reconstructed_constraint_upper_bound_m=1e-6,
        reconstructed_tilt_upper_bound_deg=2.0,
        floor_minima_m=[0.5, 0.5],
        reconstructed_self_clearance_lower_bound_m=0.01,
        evidence_mode="online-live",
    )


def test_compiled_reset_count_requires_exact_declared_mode_and_frozen_library():
    m = module()
    python = online_row()
    m.check_online(python, 1)
    compiled = dict(
        python,
        audited_points=81,
        evidence_mode="compiled-native-with-python-near-interval-fallback",
        compiled_library_sha256="a" * 64,
    )
    m.check_online(compiled, 1, compiled_library_sha256="a" * 64)
    for changed in [
        dict(compiled, evidence_mode="online-live"),
        dict(compiled, compiled_library_sha256="b" * 64),
        dict(compiled, audited_points=82),
        dict(compiled, rk_states=63),
        dict(compiled, body_reconstructed_intervals=15),
    ]:
        with pytest.raises(ValueError):
            m.check_online(changed, 1, compiled_library_sha256="a" * 64)
    with pytest.raises(ValueError, match="library"):
        m.check_online(compiled, 1)


def test_v2_reference_requires_actual_replay_and_explicit_compiled_parity(tmp_path):
    m = module()
    result = m.build_qualification_report(
        tmp_path,
        experiment_roles={},
        declaration_path=None,
        case_directories=[],
        supplemental_paths=[],
        audit_backend="compiled-native",
        audit_library_path="missing.so",
        development_reference_path="new/reference.json",
    )
    assert result["schema"] == "dual-qualification-diagnostic-v1"
    assert result["compact_online_evidence_verified"] is False
    assert result["development_replay_reference"] is None
    assert any("compiled" in error.lower() or "library" in error.lower() for error in result["errors"])
    assert not (tmp_path / "new/reference.json").exists()


def reference_contract_fixture(tmp_path):
    """Synthetic schema interoperability only; never an admissible physical campaign."""
    m = module()
    roles = {}
    for role in m.EXPERIMENT_ROLES:
        path = tmp_path / (role + ".json")
        path.write_text("{}")
        roles[role] = [path.name]
    names = (
        "dual_coordination_env.py",
        "dual_coordination_mission.py",
        "dual_coordination_audit.py",
        "dual_coordination_reference.py",
        "dual_coordination_safety.py",
        "dual_payload_mujoco.py",
        "payload_hybrid_controller.py",
    )
    roles["source"] = []
    for name in names:
        (tmp_path / name).write_text("# synthetic contract fixture only\n")
        roles["source"].append(name)
    evidence = tmp_path / "dual_coordination_evidence.json"
    evidence.write_text(
        json.dumps(
            dict(
                schema="dual-evidence-profile-v1",
                replays_per_mission=0,
                replay_stride_ticks=20,
                online_internal_audit=True,
            )
        )
    )
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps(dict(schema="dual-execution-profile-v1", replays_per_mission=0)))
    roles["configuration"] = [evidence.name, execution.name]
    bound = m.BoundFiles.capture(tmp_path, roles)
    measured = dict(
        passed=True,
        native_reexecution_performed=True,
        decisions_match=True,
        terminal_match=True,
        metrics_within_tolerance=True,
        decision_count=3,
        checked_states=7,
        maximum_state_absolute_error=0.0,
        audit_library_sha256=None,
        profile_sha256=m._expected_profile(bound.files),
    )
    supplements = [dict(details="deterministic-replay", measured_summary=measured, files=[])]
    return bound, supplements


def test_generated_reference_matches_campaign_reader_and_is_outside_experiment_hash(tmp_path):
    m = module()
    from adaptive_drone_patrol.dual_coordination_campaign import verified_development_replay_reference

    bound, supplements = reference_contract_fixture(tmp_path)
    fields = m._v2_reference(
        tmp_path,
        bound.files,
        supplements,
        audit_backend="python-native",
        audit_library_path=None,
        development_reference_path="qualification/replay-reference.json",
        execution_profile_path="execution.json",
    )
    qualification = dict(fields, checks=dict(deterministic_replay=True))
    assert (
        verified_development_replay_reference(bound, qualification) == fields["development_replay_reference"]
    )
    assert fields["development_replay_reference"]["path"] not in {
        r["path"] for rows in bound.files.values() for r in rows
    }
    (tmp_path / "qualification/replay-reference.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        verified_development_replay_reference(bound, qualification)


def test_compiled_reference_never_falls_back_to_python_trials_only(tmp_path):
    m = module()
    bound, supplements = reference_contract_fixture(tmp_path)
    path = tmp_path / "candidate.so"
    path.write_bytes(b"synthetic library identity only")
    record = m.file_record(tmp_path, path.name)
    bound.files["audit"].append(record)
    supplements[0]["measured_summary"]["audit_library_sha256"] = record["sha256"]
    with pytest.raises(ValueError, match="parity/checkpoint/copyback"):
        m._v2_reference(
            tmp_path,
            bound.files,
            supplements,
            audit_backend="compiled-native",
            audit_library_path=path.name,
            development_reference_path="qualification/replay-reference.json",
            execution_profile_path="execution.json",
        )
    assert not (tmp_path / "qualification/replay-reference.json").exists()


def test_declared_passive_half_step_counts_survive_complete_report_reduction(tmp_path):
    """Synthetic full-shape evidence detects the real seven-case reducer bug."""
    import gzip

    m = module()
    expected = next(c for c in m.qualification_cases() if c["group"] == "passive" and c["substeps"] == 32)
    p = tmp_path / "case"
    (p / "fixture").mkdir(parents=True)
    (p / "fixture/anchored-probe.xml").write_text("synthetic model identity only")
    (tmp_path / "controller.json").write_text("{}")
    (tmp_path / "source.py").write_text("# synthetic binding only")
    source = m.file_record(tmp_path, "source.py")
    bound = dict(source=[source], assets=[source], controller=[m.file_record(tmp_path, "controller.json")])
    model = m.file_record(tmp_path, "case/fixture/anchored-probe.xml")

    def numbers(ticks):
        return dict(
            rk_states=ticks * 32 * 4,
            native_states=ticks * 32,
            failures=[],
            maximum_constraint_residual_m=0.0,
            maximum_signed_energy_work_residual_j=0.0,
            motor_work_j=0.0,
            passive_work_j=0.0,
        )

    trace = p / "compact-replay.jsonl.gz"
    with gzip.open(trace, "wt") as stream:
        stream.write(json.dumps(dict(kind="RESET", actual=dict(world_time=0.0))) + "\n")
        for tick in range(1, 3001):
            stream.write(
                json.dumps(
                    dict(kind="STEP", tick=tick, actual=dict(world_time=tick * 0.01), numerical=numbers(tick))
                )
                + "\n"
            )
    numerical = numbers(3000)
    numerical["passive"] = [dict(maximum_absolute_energy_fraction=0.01)] * 2
    documents = {
        "result.json": dict(
            schema="dual-qualification-case-v1",
            case=expected,
            pilot=False,
            passed=True,
            completed_declared_duration=True,
            failure=None,
            completed_control_steps=3000,
            numerical=numerical,
            final_references=[],
            compact_bytes=trace.stat().st_size,
            compact_sha256=m.sha256(trace),
        ),
        "inputs.json": dict(
            case=expected,
            pilot=False,
            requested_duration_s=30.0,
            source_hashes={"source.py": source["sha256"]},
            original_sha256=source["sha256"],
            design={},
        ),
        "properties.json": dict(
            runtime=dict(
                version="3.15.0",
                substeps=32,
                timestep_s=0.01 / 32,
                precision="float64",
                integrator="RK4",
                profile="dual-anchored-passive-probe-v1",
            )
        ),
        "initial-state.json": dict(base=dict(model_sha256=model["sha256"])),
    }
    for name, value in documents.items():
        (p / name).write_text(json.dumps(value))
    assert m.inspect_case(tmp_path, "case", expected, bound)["ticks"] == 3000
    # A declared half-step case with an ordinary16-step runtime must fail.
    documents["properties.json"]["runtime"].update(substeps=16, timestep_s=0.01 / 16)
    (p / "properties.json").write_text(json.dumps(documents["properties.json"]))
    with pytest.raises(ValueError, match="stepping profile"):
        m.inspect_case(tmp_path, "case", expected, bound)
