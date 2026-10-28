"""Derive native admission from file-bound measurements, never pass switches.

Missing/failed evidence produces a diagnostic schema which campaign admission
rejects. JSON hashes establish provenance/integrity, not an adversarial signature
or an assertion that files alone mathematically prove real continuous physics.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path


from .dual_coordination_campaign import BoundFiles, EXPERIMENT_ROLES, NATIVE_CHECKS, _path, _digest
from .dual_coordination_qualify import qualification_cases
from .util import sha256


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, *, lower=0.0, upper=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper


def _json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON evidence: " + value)

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def file_record(root, relative):
    root = Path(root).resolve()
    path = _path(root, relative)
    return dict(path=relative, bytes=path.stat().st_size, sha256=sha256(path))


def read_bound_json(root, record):
    _require(set(record) == {"path", "bytes", "sha256"}, "Malformed measured file binding")
    _require(file_record(root, record["path"]) == record, "Measured evidence changed")
    return _json(_path(Path(root).resolve(), record["path"]))


def check_numerics(row, *, ticks, substeps):
    _require(
        row.get("rk_states") == ticks * substeps * 4 and row.get("native_states") == ticks * substeps,
        "Incomplete actual numerical coverage",
    )
    _require(row.get("failures") == [], "Numerical failure details present or missing")
    for key, limit in [
        ("maximum_constraint_residual_m", 2e-5),
        ("maximum_signed_energy_work_residual_j", 0.02),
    ]:
        _require(_finite(row.get(key), upper=limit), "Invalid numerical bound: " + key)
    for key in ("motor_work_j", "passive_work_j"):
        _require(_finite(row.get(key), lower=-math.inf), "Missing finite work measurement")


def check_online(row, ticks, *, compiled_library_sha256=None):
    compiled = row.get("evidence_mode") == "compiled-native-with-python-near-interval-fallback"
    if compiled:
        _require(
            isinstance(compiled_library_sha256, str)
            and len(compiled_library_sha256) == 64
            and row.get("compiled_library_sha256") == compiled_library_sha256,
            "Explicit bound compiled library required for native point count",
        )
    else:
        _require(
            compiled_library_sha256 is None and row.get("compiled_library_sha256") is None,
            "Compiled library cannot use undeclared audit evidence mode",
        )
    _require(
        row.get("audited_points") == ticks * 80 + (1 if compiled else 2),
        "Missing native/RK point diagnostics",
    )
    _require(
        row.get("rk_states") == ticks * 64
        and row.get("native_poses") == ticks * 16 + 1
        and row.get("control_steps") == ticks,
        "Incomplete online native/RK coverage",
    )
    _require(
        row.get("reconstructed_intervals") == ticks * 16
        and row.get("body_reconstructed_intervals") == ticks * 16,
        "Missing reconstructed interval coverage",
    )
    for key in ("errors", "violations", "reconstruction_failures", "proof_gaps"):
        _require(row.get(key) == [], "Online audit failures or missing coverage: " + key)
    for key in ("safe", "retained_points_safe", "reconstructed_path_safe", "continuous_safety_qualified"):
        _require(row.get(key) is True, "Online safety not established: " + key)
    _require(_finite(row.get("minimum_clearance_m"), lower=0.2), "Assembly clearance failed")
    _require(
        _finite(row.get("reconstructed_clearance_lower_bound_m"), lower=0.2), "Interval clearance failed"
    )
    _require(
        _finite(row.get("reconstructed_constraint_upper_bound_m"), upper=2e-5),
        "Interval attachment bound failed",
    )
    _require(_finite(row.get("reconstructed_tilt_upper_bound_deg"), upper=60.0), "Interval tilt failed")

    floors = row.get("floor_minima_m", [])
    _require(
        len(floors) == 2 and all(_finite(x, lower=0.1) and x > 0.1 for x in floors), "Floor clearance failed"
    )
    _require(_finite(row.get("reconstructed_self_clearance_lower_bound_m")), "Self collision interval failed")


def _source_binding(inputs, experiment_inputs):
    records = experiment_inputs.get("source", [])
    hashes = {r["sha256"] for r in records}
    source = inputs.get("source_hashes", {})
    _require(source and set(source.values()) <= hashes, "Case source hashes not in frozen experiment source")
    _require(
        inputs.get("original_sha256") in {r["sha256"] for r in experiment_inputs.get("assets", [])},
        "Original asset not frozen with experiment",
    )


def inspect_case(root, directory, expected, experiment_inputs):
    root = Path(root).resolve()
    prefix = Path(directory)
    result = _json(_path(root, str(prefix / "result.json")))
    _require(
        result.get("pilot") is False
        and result.get("passed") is True
        and result.get("completed_declared_duration") is True,
        "Full complete nonpilot case required",
    )
    _require(
        result.get("schema") == "dual-qualification-case-v1" and result.get("case") == expected,
        "Case differs from prospective exact declaration",
    )
    _require(result.get("failure") is None, "Case failure cannot qualify")
    inputs = _json(_path(root, str(prefix / "inputs.json")))
    _require(
        inputs.get("case") == expected
        and inputs.get("pilot") is False
        and inputs.get("requested_duration_s") == expected["duration_s"],
        "Incomplete case input binding",
    )
    _source_binding(inputs, experiment_inputs)
    controllers = [read_bound_json(root, r) for r in experiment_inputs.get("controller", [])]
    _require(inputs.get("design") in controllers, "Controller differs from frozen experiment")
    ticks = round(expected["duration_s"] * 100)
    _require(result.get("completed_control_steps") == ticks, "Case control duration incomplete")
    properties = _json(_path(root, str(prefix / "properties.json")))
    runtime = properties["runtime"]
    _require(runtime.get("mujoco_version", runtime.get("version")) == "3.15.0", "Wrong native version")
    _require(
        runtime.get("substeps") == 16 and runtime.get("timestep_s") == 0.01 / 16,
        "Wrong native stepping profile",
    )
    _require(
        runtime.get("precision") == "float64" and runtime.get("integrator") == "RK4",
        "Wrong numerical profile",
    )
    initial = _json(_path(root, str(prefix / "initial-state.json")))
    model_name = "fixture/anchored-probe.xml" if expected["group"] == "passive" else "fixture/model.xml"
    model = file_record(root, str(prefix / model_name))
    _require(initial["base"]["model_sha256"] == model["sha256"], "Initial state model identity differs")
    check_numerics(result["numerical"], ticks=ticks, substeps=16)
    if expected["group"] in ("duration", "mixed"):
        check_online(result["online_audit"], ticks)
    if expected["group"] != "passive":
        _require(
            type(result.get("final_settling_ticks")) is int and result["final_settling_ticks"] >= 100,
            "Final continuous settling not established",
        )
    else:
        passive = result["numerical"].get("passive", [])
        _require(
            len(passive) == 2
            and all(_finite(r.get("maximum_absolute_energy_fraction"), upper=0.05 + 1e-12) for r in passive),
            "Passive energy drift/loss failed",
        )
        if expected["kind"] == "anchor":
            periods = result.get("periods", [])
            _require(
                len(periods) == 2
                and all(
                    _finite(r.get("predicted_s"), lower=1e-12)
                    and _finite(r.get("measured_s"), lower=1e-12)
                    and abs(r["measured_s"] - r["predicted_s"]) / r["predicted_s"] <= 0.1
                    for r in periods
                ),
                "Inherited passive period threshold failed",
            )
    if expected["group"] == "refinement":
        half = properties["half_runtime"]
        _require(
            half.get("version") == "3.15.0"
            and half.get("precision") == "float64"
            and half.get("integrator") == "RK4"
            and half.get("substeps") == 32
            and half.get("timestep_s") == 0.01 / 32,
            "Wrong paired half-dt native profile",
        )
        half_model = file_record(root, str(prefix / "half-dt-fixture/model.xml"))
        _require(initial["half"]["model_sha256"] == half_model["sha256"], "Half-dt initial model differs")
        check_numerics(result["half_numerical"], ticks=ticks, substeps=32)
        comparison = result["refinement"]
        _require(
            _finite(comparison.get("maximum_position_difference_m"), upper=0.02)
            and _finite(comparison.get("maximum_angle_difference_deg"), upper=2.0),
            "Half-dt refinement failed",
        )
    trace = _path(root, str(prefix / "compact-replay.jsonl.gz"))
    _require(
        trace.stat().st_size == result.get("compact_bytes") and sha256(trace) == result.get("compact_sha256"),
        "Compact case replay changed or missing",
    )
    from .dual_coordination_qualify import _settled
    from .dual_coordination_reference import ContinuousReference
    from .payload_contract import Payload
    from .payload_solver_mujoco import FREE_MASS_KG

    targets = [ContinuousReference.from_dict(r).target for r in result.get("final_references", [])]
    payloads = [
        Payload(FREE_MASS_KG, ratio, side) for ratio, side in zip(expected["ratios"], expected["sides"])
    ]
    dwell = 0
    with gzip.open(trace, "rt") as stream:
        reset = json.loads(next(stream))
        _require(reset.get("kind") == "RESET" and reset["actual"]["world_time"] == 0.0, "Missing reset state")
        previous_valid = bool(targets) and _settled(reset["actual"]["drones"], payloads, targets)
        count = 0
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            _require(
                row.get("kind") == "STEP"
                and row.get("tick") == count
                and abs(row["actual"]["world_time"] - count * 0.01) < 5e-8,
                "Missing or reordered replay controls",
            )
            check_numerics(row["numerical"], ticks=count, substeps=16)
            if expected["group"] != "passive":
                _require(len(targets) == 2, "Missing final reference targets")
                valid = _settled(row["actual"]["drones"], payloads, targets)
                dwell = dwell + 1 if previous_valid and valid else 0
                previous_valid = valid
        _require(count == ticks, "Truncated case replay")
        if expected["group"] != "passive":
            _require(dwell >= 100, "Replay does not establish one continuous second of final settling")
    files = [
        file_record(root, str(prefix / name))
        for name in (
            "result.json",
            "inputs.json",
            "properties.json",
            "initial-state.json",
            "compact-replay.jsonl.gz",
        )
    ]
    files.append(model)
    if expected["group"] == "refinement":
        files.append(half_model)
    return dict(case=expected, files=files, ticks=ticks)


def index_cases(rows):
    result = {}
    for row in rows:
        case_id = row["case"]["case_id"]
        _require(case_id not in result, "duplicate qualification case ID")
        result[case_id] = row
    return result


def inspect_supplement(root, relative, experiment_inputs):
    """Validate a bound measurement set using its experiment-specific reducer."""
    value = _json(_path(Path(root).resolve(), relative))
    _require(value.get("schema") == "dual-native-supplement-v1", "Actual measurement supplement required")
    _require(
        value.get("experiment_inputs") == experiment_inputs and experiment_inputs,
        "Supplement experiment inputs differ",
    )
    measurements = read_bound_json(root, value["measurements"])
    from .dual_coordination_qualification_supplement import (
        reduce_measurements,
        replay_measurement_summary,
        validate_compiled_parity,
    )

    summary = None
    if measurements.get("kind") == "deterministic-replay":
        summary = replay_measurement_summary(measurements, root=Path(root).resolve())
        checks = {"deterministic_replay"}
    elif measurements.get("kind") == "compiled-adapter-parity":
        summary = validate_compiled_parity(measurements, root=Path(root).resolve())
        checks = set()
        _source_binding(measurements, experiment_inputs)
        _require(
            measurements["design"] in [read_bound_json(root, r) for r in experiment_inputs["controller"]],
            "Compiled parity controller differs from frozen experiment",
        )
        build = read_bound_json(root, measurements["build_receipt"])
        _require(
            build["source"]["sha256"]
            in {r["sha256"] for role in ("source", "audit") for r in experiment_inputs[role]},
            "Compiled C++ source is not frozen",
        )
        _require(
            measurements["compiled_profile"]["profile_sha256"] == _expected_profile(experiment_inputs),
            "Compiled parity executing source differs from frozen profile",
        )
        for name in ("dual_coordination_compiled_audit.py", "dual_coordination_native_audit.py"):
            records = [r for r in experiment_inputs["source"] if Path(r["path"]).name == name]
            _require(
                len(records) == 1 and measurements["source_hashes"].get(name) == records[0]["sha256"],
                "Compiled adapter executing source is not frozen: " + name,
            )
    else:
        checks = reduce_measurements(measurements, root=Path(root).resolve())
    if summary is not None and summary.get("audit_library_sha256", summary.get("library_sha256")) is not None:
        record = measurements.get("audit_library", measurements.get("library"))
        _require(record in experiment_inputs["audit"], "Measured compiled library is not frozen audit input")
    if measurements.get("kind") == "deterministic-replay":
        metadata = _json(
            _path(Path(root).resolve(), str(Path(measurements["replay_directory"]) / "environment.json"))
        )
        names = (
            "dual_coordination_env.py",
            "dual_coordination_mission.py",
            "dual_coordination_audit.py",
            "dual_coordination_reference.py",
            "dual_coordination_safety.py",
            "dual_payload_mujoco.py",
            "payload_hybrid_controller.py",
        )
        hashes = {}
        for name in names:
            rows = [r for r in experiment_inputs["source"] if Path(r["path"]).name == name]
            _require(len(rows) == 1, "Exactly one frozen replay executable source required: " + name)
            hashes[name] = rows[0]["sha256"]
        expected_profile = _digest(
            dict(scales=metadata["observation_scales"], source_sha256=hashes, profile="dual-payload-v1")
        )
        _require(
            metadata["binding"]["profile_sha256"] == expected_profile,
            "Verified replay executable profile differs from frozen experiment",
        )
        _require(
            expected_profile == _expected_profile(experiment_inputs),
            "Replay observation/source profile differs",
        )
    return dict(
        checks=checks,
        files=[file_record(root, relative), value["measurements"]],
        details=measurements.get("kind"),
        measured_summary=summary,
    )


def _expected_profile(binding):
    from .dual_coordination_env import OBSERVATION_SCALES

    names = (
        "dual_coordination_env.py",
        "dual_coordination_mission.py",
        "dual_coordination_audit.py",
        "dual_coordination_reference.py",
        "dual_coordination_safety.py",
        "dual_payload_mujoco.py",
        "payload_hybrid_controller.py",
    )
    hashes = {}
    for name in names:
        rows = [r for r in binding["source"] if Path(r["path"]).name == name]
        _require(len(rows) == 1, "Exactly one frozen replay executable required: " + name)
        hashes[name] = rows[0]["sha256"]
    return _digest(dict(scales=OBSERVATION_SCALES, source_sha256=hashes, profile="dual-payload-v1"))


def _v2_reference(
    root,
    binding,
    supplements,
    *,
    audit_backend,
    audit_library_path,
    development_reference_path,
    execution_profile_path,
):
    """Write only measured replay facts, outside experiment input identities."""
    from .dual_coordination_progress import atomic_json

    _require(audit_backend in ("python-native", "compiled-native"), "Declared audit backend required")
    library_hash = None
    if audit_backend == "compiled-native":
        _require(audit_library_path is not None, "Explicit compiled library input required")
        library = file_record(root, audit_library_path)
        _require(library in binding.get("audit", []), "Compiled library is not a frozen audit input")
        library_hash = library["sha256"]
        parity = [s for s in supplements if s["details"] == "compiled-adapter-parity"]
        _require(
            len(parity) == 1 and parity[0]["measured_summary"]["library_sha256"] == library_hash,
            "Actual same-library compiled parity/checkpoint/copyback measurements required",
        )
    else:
        _require(audit_library_path is None, "Python audit cannot name an unqualified compiled library")
    replay = [s for s in supplements if s["details"] == "deterministic-replay"]
    _require(len(replay) == 1, "Exactly one complete measured development replay required")
    measured = replay[0]["measured_summary"]
    _require(
        measured["audit_library_sha256"] == library_hash, "Development replay audit backend/library differs"
    )
    evidence = [
        r for r in binding["configuration"] if Path(r["path"]).name == "dual_coordination_evidence.json"
    ]
    _require(len(evidence) == 1, "One frozen single-capture evidence profile required")
    profile = read_bound_json(root, evidence[0])
    _require(
        profile.get("schema") == "dual-evidence-profile-v1"
        and type(profile.get("replays_per_mission")) is int
        and profile["replays_per_mission"] == 0
        and type(profile.get("replay_stride_ticks")) is int
        and profile["replay_stride_ticks"] == 20
        and profile.get("online_internal_audit") is True,
        "Exact compact single-capture profile required",
    )
    _require(execution_profile_path is not None, "Frozen execution profile required for benchmark reducer")
    execution_record = file_record(root, execution_profile_path)
    _require(
        execution_record in binding["configuration"], "Execution profile must be a frozen configuration input"
    )
    execution = read_bound_json(root, execution_record)
    _require(
        execution.get("schema") == "dual-execution-profile-v1"
        and execution.get("replays_per_mission") == 0
        and type(execution.get("replays_per_mission")) is int,
        "Execution/replay profiles disagree",
    )
    _require(isinstance(development_reference_path, str), "New measured development reference path required")
    relative = Path(development_reference_path)
    _require(
        not relative.is_absolute() and ".." not in relative.parts and relative.parts,
        "Confined development reference path required",
    )
    _require(
        development_reference_path not in {r["path"] for rows in binding.values() for r in rows},
        "Development reference must not be an experiment input (hash cycle)",
    )
    path = Path(root) / relative
    _require(
        not any(p.is_symlink() for p in (path, *path.parents) if p != Path(root).parent),
        "Development reference must not traverse symlinks",
    )
    value = dict(measured)
    value.update(
        schema="dual-development-replay-qualification-v1",
        experiment_inputs_sha256=_digest({r: binding[r] for r in EXPERIMENT_ROLES}),
        source_sha256=_digest(binding["source"]),
        audit_sha256=_digest(binding["audit"]),
        configuration_sha256=_digest(binding["configuration"]),
        profile_sha256=_expected_profile(binding),
        evidence_profile_sha256=evidence[0]["sha256"],
        supplemental_evidence=replay[0]["files"],
        compiled_adapter_evidence=[
            s["files"] for s in supplements if s["details"] == "compiled-adapter-parity"
        ],
    )
    if path.exists():
        _require(
            _json(_path(root, development_reference_path)) == value,
            "Preserve prior development reference; use a new path",
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)
    return dict(
        development_replay_reference=file_record(root, development_reference_path),
        compact_online_evidence_verified=True,
        evidence_profile_sha256=evidence[0]["sha256"],
        execution_profile_sha256=execution_record["sha256"],
        audit_backend=audit_backend,
        audit_library_sha256=library_hash,
    )


def build_qualification_report(
    root,
    *,
    experiment_roles,
    declaration_path,
    case_directories,
    supplemental_paths,
    audit_backend="python-native",
    audit_library_path=None,
    development_reference_path=None,
    execution_profile_path=None,
):
    root = Path(root).resolve()
    errors, accepted, supplements = [], [], []
    binding = {}
    try:
        _require(set(experiment_roles) == EXPERIMENT_ROLES, "All frozen experiment roles required")
        binding = BoundFiles.capture(root, experiment_roles).files
        declaration = _json(_path(root, declaration_path))
        _require(
            declaration.get("schema") == "dual-native-qualification-declaration-v2"
            and declaration.get("cases") == qualification_cases(),
            "Exact prospective qualification v2 required",
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        errors.append(str(error))
    expected = {c["case_id"]: c for c in qualification_cases()}
    for directory in case_directories:
        try:
            data = _json(_path(root, str(Path(directory) / "result.json")))
            case = expected[data["case"]["case_id"]]
            accepted.append(inspect_case(root, directory, case, binding))
        except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
            errors.append(f"{directory}: {error}")
    try:
        indexed = index_cases(accepted)
    except ValueError as error:
        errors.append(str(error))
        indexed = {}
    checks = dict.fromkeys(sorted(NATIVE_CHECKS), False)

    def complete(group, kinds=None):
        needed = [
            c for c in expected.values() if c["group"] == group and (kinds is None or c["kind"] in kinds)
        ]
        return bool(needed) and all(c["case_id"] in indexed for c in needed)

    checks["nominal330"] = "dual-duration330-00" in indexed
    checks["endpoints6"] = all(f"dual-duration330-{i:02d}" in indexed for i in range(1, 7))
    checks["mixed36"] = complete("mixed")
    checks["passive_powered"] = complete("passive") and complete("refinement") and complete("duration")
    for relative in supplemental_paths:
        try:
            row = inspect_supplement(root, relative, binding)
            supplements.append(row)
            for key in row["checks"]:
                _require(key in checks, "Unknown native qualification check")
                checks[key] = True
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(f"{relative}: {error}")
    reference = dict(
        development_replay_reference=None,
        compact_online_evidence_verified=False,
        evidence_profile_sha256=None,
        execution_profile_sha256=None,
        audit_backend=audit_backend,
        audit_library_sha256=None,
    )
    try:
        reference = _v2_reference(
            root,
            binding,
            supplements,
            audit_backend=audit_backend,
            audit_library_path=audit_library_path,
            development_reference_path=development_reference_path,
            execution_profile_path=execution_profile_path,
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        errors.append(f"Compact/backend qualification [{audit_backend}]: " + str(error))
    passed = (
        len(indexed) == 81
        and all(checks.values())
        and not errors
        and reference["compact_online_evidence_verified"]
    )
    return dict(
        schema="dual-qualification-v1" if passed else "dual-qualification-diagnostic-v1",
        passed=passed,
        admission_granted=passed,
        checks=checks,
        missing_checks=[key for key, value in checks.items() if not value],
        accepted_case_count=len(indexed),
        required_case_count=81,
        missing_case_ids=sorted(set(expected) - set(indexed)),
        errors=errors,
        safety_violations=0 if passed else None,
        profile="dual-payload-v1",
        version="3.15.0",
        experiment_inputs=binding,
        case_evidence=accepted,
        supplemental_evidence=supplements,
        **reference,
        limitations=[
            "Native saved points and declared reconstruction are qualified; true unsaved continuous physics is not mathematically proved."
        ],
    )


def main(argv=None):
    import argparse
    from .dual_coordination_progress import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--request",
        required=True,
        help="JSON kwargs: experiment_roles, declaration_path, case_directories, supplemental_paths",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    _require(not output.exists(), "Refusing to overwrite qualification attempt")
    result = build_qualification_report(args.root, **_json(args.request))
    atomic_json(output, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
