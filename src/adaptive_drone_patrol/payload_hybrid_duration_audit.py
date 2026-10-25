"""Independent complete duration reconstruction and same-input native replay.

Never trusts a saved success flag, settling event or nominal duration. All actual
native states and16x4 RK stages are checked, including the final partial chunk.
No mission, acceptance, learner or new native-batch capability is issued here.
"""

import copy
import gzip
import json
from pathlib import Path
import tempfile

import numpy as np

from .payload_hybrid_audit import independent_metrics, interval_update
from .payload_hybrid_contract import VIDEO_PROFILE, STRICT_PROFILE
from .payload_hybrid_controller import Controller, SENSOR_FIELDS
from .payload_hybrid_native import verify_native_kinematics
from .payload_solver_qualification import physical_errors
from .payload_solver_trace import match_native_sample
from .payload_solver_validation import verify_initial_sample
from .payload_solver_work import pack_stages, WIDTH, total_mechanical_energy
from .payload_solver_work_check import reconstruct_batch_work
from .util import sha256


def check_duration_boundary(step, *, unsafe, failed):
    if (
        type(step) is not int
        or not 0 <= step <= 33000
        or type(unsafe) is not bool
        or type(failed) is not bool
    ):
        raise ValueError("Explicit actual finite duration interval/terminal inputs required")
    return (
        "SAFETY_FAILURE"
        if unsafe
        else "RUNTIME_FAILURE"
        if failed
        else "DURATION_COMPLETE"
        if step == 33000
        else "CONTINUE"
    )


def check_native_equal(expected, actual, *, atol=1e-10):
    if not isinstance(expected, dict) or not isinstance(actual, dict) or set(expected) != set(actual):
        raise ValueError("Complete matching native field inventories required")
    maximum = 0.0
    for key in expected:
        left, right = np.asarray(expected[key]), np.asarray(actual[key])
        if (
            left.shape != right.shape
            or left.dtype.kind not in "biuf"
            or right.dtype.kind not in "biuf"
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
        ):
            raise ValueError("Finite native replay shapes/fields differ: " + key)
        if left.dtype.kind in "biu" or right.dtype.kind in "biu":
            if left.dtype.kind != right.dtype.kind or not np.array_equal(left, right):
                raise ValueError("Native discrete replay field differs: " + key)
        elif not np.allclose(left, right, rtol=0, atol=atol):
            raise ValueError("Native replay field differs: " + key)
        if left.size:
            maximum = max(maximum, float(np.max(np.abs(left.astype(float) - right.astype(float)))))
    return maximum


def _equal(expected, actual, *, atol=1e-10):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError("Independent reconstruction field inventory differs")
        for key in expected:
            _equal(expected[key], actual[key], atol=atol)
    elif isinstance(expected, (list, tuple)) and any(isinstance(value, (dict, str)) for value in expected):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            raise ValueError("Independent reconstruction list differs")
        for a, b in zip(expected, actual):
            _equal(a, b, atol=atol)
    elif isinstance(expected, (str, bool)) or expected is None:
        if type(expected) is not type(actual) or expected != actual:
            raise ValueError("Independent discrete reconstruction differs")
    else:
        a, b = np.asarray(expected, float), np.asarray(actual, float)
        if (
            a.shape != b.shape
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
            or not np.allclose(a, b, rtol=0, atol=atol)
        ):
            raise ValueError("Independent numeric reconstruction differs")


def _read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON numeric evidence: " + value)

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _lines(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite actual evidence")),
            )


def _stages(root, count):
    index = _read(root / "stages-index.json")
    cursor = 0
    for i, chunk in enumerate(index):
        path = root / chunk["path"]
        if (
            chunk["path"] != f"stages-{i:05d}.npz"
            or chunk["first_step"] != cursor
            or type(chunk["intervals"]) is not int
            or not 1 <= chunk["intervals"] <= 128
            or sha256(path) != chunk["sha256"]
        ):
            raise ValueError("Consecutive complete bound stage chunks required")
        with np.load(path, allow_pickle=False) as packed:
            if set(packed.files) != {"stages", "qacc_before_solve"}:
                raise ValueError("Every actual observed stage field required")
            stage, acc = packed["stages"], packed["qacc_before_solve"]
            if (
                stage.dtype != np.float64
                or acc.dtype != np.float64
                or stage.shape != (chunk["intervals"], 64, WIDTH)
                or acc.shape != (chunk["intervals"], 64, 12)
                or not np.isfinite(stage).all()
                or not np.isfinite(acc).all()
            ):
                raise ValueError("Actual float64 full16x4 RK stages required")
            yield from zip(stage, acc)
        cursor += chunk["intervals"]
    if cursor != count:
        raise ValueError("Stage interval denominator differs from actual native trace")


def _unsafe(metrics, contact):
    return bool(
        contact
        or min(metrics["drone_floor_m"], metrics["package_floor_m"]) <= 0.1
        or max(abs(metrics["drone_roll_deg"]), abs(metrics["drone_pitch_deg"])) > 60
        or metrics["constraint_residual_m"] > 2e-5
    )


def audit_case(root, *, source_root, reference=None):
    from .payload_hybrid_duration import (
        DurationReference,
        ObservedFixture,
        duration_declaration,
        validate_duration_case,
        sources,
        SCOPE,
    )
    from .payload_contract import Payload
    from .payload_ball_bank import ball_joint_configuration
    from .payload_solver_mujoco import FREE_MASS_KG, HOVER_RPM

    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    case, design, metadata, result = (
        _read(root / name) for name in ("case.json", "design.json", "metadata.json", "result.json")
    )
    validate_duration_case(case)
    if _read(root / "declaration.json") != duration_declaration():
        raise ValueError("Exact prospectively declared330s duration protocol required")
    proof = _read(root / "reference-proof.json")
    if reference is None:
        reference = DurationReference.from_proof(proof)
    if not isinstance(reference, DurationReference) or reference.proof != proof:
        raise ValueError("Actual unchanged historical reference proof required")
    reference.require_current()
    binding = sources(source_root)
    if (
        metadata["source_binding"] != binding
        or result["source_after"] != binding
        or any(sha256(root / "source" / name) != digest for name, digest in binding.items())
    ):
        raise ValueError("Executed duration source/snapshot changed")
    if (
        metadata["scope"] != SCOPE
        or metadata["learning_admitted"] is not False
        or metadata["reference_binding"] != reference.admission.binding
        or metadata["profile_sha256"] != VIDEO_PROFILE.sha256
        or metadata["strict_shadow_sha256"] != STRICT_PROFILE.sha256
        or metadata["substeps"] != 16
        or metadata["observation_fields"] != list(SENSOR_FIELDS)
        or metadata["nn"] is not None
        or metadata["optimizer"] is not None
    ):
        raise ValueError("Qualified C-only selected reference/profile required")
    if (
        sha256(metadata["original"]) != metadata["original_sha256"]
        or metadata["original_sha256"] != reference.admission.binding["original_asset_sha256"]
        or sha256(metadata["executable"]) != metadata["executable_sha256"]
    ):
        raise ValueError("Actual original asset/executable binding changed")
    if design.get("recipe") != "C-local-lqr-v1" or design.get("design_load") != dict(
        side_m=0.045, mass_ratio=0.2
    ):
        raise ValueError("Frozen nominal C-local-lqr-v1 required")
    for name, key in [
        ("runtime/raw.jsonl.gz", "raw_sha256"),
        ("control.jsonl.gz", "control_sha256"),
        ("stages-index.json", "stages_index_sha256"),
    ]:
        if sha256(root / name) != result[key]:
            raise ValueError("Recorded final raw evidence changed")
    properties = _read(root / "runtime-properties.json")
    if (
        physical_errors(properties, dict(case, substeps=16), fixed=False)
        or sha256(root / "fixture/model.xml") != properties["model_xml_sha256"]
    ):
        raise ValueError("Actual compiled model/settings/properties differ")
    if (
        type(result["steps"]) is not int
        or not 0 <= result["steps"] <= 33000
        or result["native_samples"] != result["steps"] + 1
        or result["recorded_stage_intervals"] != result["steps"]
        or result["final_flush_complete"] is not True
        or result["reset_count"] != 1
        or type(result["success"]) is not bool
    ):
        raise ValueError("Complete single-reset native/stage/flush denominator required")
    raw, controls, stages = (
        iter(_lines(root / "runtime/raw.jsonl.gz")),
        iter(_lines(root / "control.jsonl.gz")),
        iter(_stages(root, result["steps"])),
    )
    initial = next(raw, None)
    if (
        initial is None
        or initial["kind"] != "RESET"
        or initial["episode_id"] != 1
        or initial["step"] != 0
        or initial["target"] != case["target"]
        or initial["target_id"] != "S"
        or np.any(initial["actions"])
        or np.any(initial["actual"]["rpm"])
    ):
        raise ValueError("Exactly one initial physical zero-input RESET required")
    verify_initial_sample(initial["actual"], case["initial"])
    controller = Controller(design["gain"], design["hover_action"], vibration_feedback=True)
    energy0 = total_mechanical_energy(initial["actual"], properties, case["side_m"])
    previous = initial["actual"]
    count = 0
    metrics = independent_metrics(previous, case["target"], case["side_m"])
    live, strict = (
        (VIDEO_PROFILE.valid(metrics, contact=previous["contact"]), 0),
        (STRICT_PROFILE.valid(metrics, contact=previous["contact"]), 0),
    )
    final_verdict = "SAFETY_FAILURE" if _unsafe(metrics, previous["contact"]) else "CONTINUE"
    motor = loss = bound = maximum_native = maximum_stage = maximum_work = maximum_constraint = 0.0
    longest_live = longest_strict = 0.0
    first_live = first_strict = None
    last_wall = result["reset_started_monotonic_s"]
    if not np.isfinite(last_wall) or last_wall < 0:
        raise ValueError("Finite actual reset wall origin required")
    with tempfile.TemporaryDirectory(prefix="duration-independent-replay-") as temporary:
        fresh = ObservedFixture(
            metadata["original"],
            Path(temporary) / "fixture",
            Payload(FREE_MASS_KG, case["mass_ratio"], case["side_m"]),
            substeps=16,
        )
        expected_properties = fresh.runtime()
        for key in properties:
            if key != "model_xml_sha256":
                _equal(expected_properties[key], properties[key])
        fresh.reset(
            position=case["initial"]["drone_xyz"], quaternion=case["initial"]["drone_quaternion_wxyz"]
        )
        upper, lower = ball_joint_configuration(case["initial"], case["side_m"])
        for name, values in [("suspension_ball", upper), ("payload_ball", lower)]:
            joint = fresh.mj.mj_name2id(fresh.model, fresh.mj.mjtObj.mjOBJ_JOINT, name)
            start = fresh.model.jnt_qposadr[joint]
            fresh.data.qpos[start : start + 4] = values
        fresh.mj.mj_forward(fresh.model, fresh.data)
        kinematics_data = fresh.mj.MjData(fresh.model)
        maximum_native = check_native_equal(fresh.raw_state(), previous)
        for record in raw:
            if final_verdict != "CONTINUE":
                raise ValueError("Physical integration continued after first duration terminal")
            count += 1
            if (
                record["kind"] != "STEP"
                or record["episode_id"] != 1
                or record["step"] != count
                or record["target"] != case["target"]
                or record["target_id"] != "S"
            ):
                raise ValueError("Duration changed target/reset or skipped actual intervals")
            control, observed = next(controls, None), next(stages, None)
            if control is None or observed is None:
                raise ValueError("Missing control/all-stage evidence")
            stage, acceleration = observed
            actual = record["actual"]
            match_native_sample(dict(actual, contacts=actual["contact"]), actual, properties)
            verify_native_kinematics(fresh.model, kinematics_data, actual)
            if abs(actual["time_s"] - count * 0.01) > 1e-7:
                raise ValueError("Native continuous clock differs")
            start = np.asarray(initial["actual"]["drone_xyz"])
            delta = np.asarray(case["target"]) - start
            length = np.linalg.norm(delta)
            duration = max(
                3.0, 1.875 * length / 1.2, np.sqrt(5.774 * length / 0.8), (60 * length / 2) ** (1 / 3)
            )
            u = np.clip((count - 1) * 0.01 / duration, 0, 1)
            position = start + (10 * u**3 - 15 * u**4 + 6 * u**5) * delta
            velocity = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration * delta
            action = controller.action(
                {k: copy.deepcopy(previous[k]) for k in SENSOR_FIELDS},
                position,
                tick=count - 1,
                reference_velocity=velocity,
            )
            _equal(controller.last_audit, control["control"])
            _equal(position, control["reference_position"])
            _equal(velocity, control["reference_velocity"])
            _equal(action, record["actions"])
            _equal((1 + 0.8 * action) * HOVER_RPM, actual["rpm"], atol=1e-8)
            fresh.step_rpm(actual["rpm"])
            maximum_native = max(maximum_native, check_native_equal(fresh.raw_state(), actual))
            expected_stage = pack_stages(fresh.stage_rows)
            expected_acc = np.asarray([r["qacc_before_solve"] for r in fresh.stage_rows])
            _equal(expected_stage, stage)
            _equal(expected_acc, acceleration)
            maximum_stage = max(
                maximum_stage,
                float(np.max(abs(expected_stage - stage))),
                float(np.max(abs(expected_acc - acceleration))),
            )
            work = reconstruct_batch_work(
                stage[None],
                substeps=16,
                first_step=count - 1,
                total_mass_kg=sum(r["mass_kg"] for r in properties["links"]),
            )
            if not np.allclose(work["rpm_all_stages"], actual["rpm"], rtol=0, atol=1e-8):
                raise ValueError("Current original RPM not held for every RK stage")
            motor += float(work["motor_work_j"][0])
            loss += float(work["dissipative_work_j"][0])
            bound += float(work["endpoint_absolute_work_bound_j"][0])
            energy = total_mechanical_energy(actual, properties, case["side_m"])
            residual = energy - energy0 - motor - loss
            maximum_work = max(maximum_work, abs(residual))
            metrics = independent_metrics(actual, case["target"], case["side_m"])
            maximum_constraint = max(maximum_constraint, metrics["constraint_residual_m"])
            live = interval_update(live, VIDEO_PROFILE.valid(metrics, contact=actual["contact"]))
            strict = interval_update(strict, STRICT_PROFILE.valid(metrics, contact=actual["contact"]))
            longest_live, longest_strict = (
                max(longest_live, live[1] * 0.01),
                max(longest_strict, strict[1] * 0.01),
            )
            if first_live is None and live[1] >= 100:
                first_live = count * 0.01
            if first_strict is None and strict[1] >= 100:
                first_strict = count * 0.01
            final_verdict = (
                "SAFETY_FAILURE"
                if _unsafe(metrics, actual["contact"])
                else "WORK_FAILURE"
                if abs(residual) > 0.02 or energy - energy0 > 0.02 + 1.1 * bound
                else check_duration_boundary(count, unsafe=False, failed=False)
            )
            _equal(metrics, control["metrics"])
            for key, value in dict(
                kind="STEP",
                step=count,
                time_s=count * 0.01,
                target=case["target"],
                target_id="S",
                dwell_s=live[1] * 0.01,
                strict_dwell_s=strict[1] * 0.01,
                energy_j=energy,
                motor_work_j=motor,
                dissipative_work_j=loss,
                absolute_work_bound_j=bound,
                signed_work_residual_j=residual,
                verdict=final_verdict,
            ).items():
                _equal(value, control[key])
            a, b = control["native_step_started_monotonic_s"], control["native_step_ended_monotonic_s"]
            flushed = result["reset_started_monotonic_s"] + control["reset_elapsed_wall_s"]
            if not np.isfinite([a, b, flushed]).all() or not last_wall <= a <= b <= flushed:
                raise ValueError("Actual native/recording wall clock order differs")
            last_wall = flushed
            previous = actual
    if next(controls, None) is not None or next(stages, None) is not None or count != result["steps"]:
        raise ValueError("Extra/missing actual control/native/stage intervals")
    terminal = _read(root / "terminal-evidence.json")
    elapsed = terminal["evidence_complete_elapsed_wall_s"]
    if (
        not np.isfinite(elapsed)
        or elapsed < last_wall - result["reset_started_monotonic_s"]
        or result["simulated_s"] != count * 0.01
        or terminal["result_sha256"] != sha256(root / "result.json")
        or terminal["scope"] != SCOPE
        or terminal["ipc_claim"] is not False
        or not np.isfinite(terminal["evidence_complete_monotonic_s"])
        or abs(terminal["evidence_complete_monotonic_s"] - result["reset_started_monotonic_s"] - elapsed)
        > 1e-7
        or not np.isfinite(result["final_flush_elapsed_wall_s"])
        or not 0 <= result["final_flush_elapsed_wall_s"] <= elapsed
    ):
        raise ValueError("Actual complete recording flush/performance clock differs")
    succeeded = final_verdict == "DURATION_COMPLETE" and count == 33000
    if result["success"] != succeeded or result["failure"] != (None if succeeded else final_verdict):
        raise ValueError("Saved success/failure is not independently derived duration evidence")
    return dict(
        schema="hybrid-independent-continuous-duration-case-v1",
        case_id=case["case_id"],
        passed=succeeded,
        steps=count,
        samples=count + 1,
        simulated_s=count * 0.01,
        resets=1,
        targets_visited_claim=False,
        failure=None if succeeded else final_verdict,
        full_native_physics_replay=True,
        actual_rk_stages=count * 64,
        maximum_native_replay_residual=maximum_native,
        maximum_stage_replay_residual=maximum_stage,
        maximum_signed_work_residual_j=maximum_work,
        maximum_constraint_residual_m=maximum_constraint,
        first_live_settling_s=first_live,
        first_strict_shadow_s=first_strict,
        longest_live_valid_s=longest_live,
        longest_strict_valid_s=longest_strict,
        actual_wall_s=elapsed,
        simulated_seconds_per_wall_second=count * 0.01 / elapsed if elapsed > 0 else None,
        artifacts={
            str(p.relative_to(root)): sha256(p)
            for p in root.rglob("*")
            if p.is_file() and p.name != "independent-audit.json"
        },
        scope=SCOPE,
        learning_admitted=False,
    )


def audit_group(root, *, source_root):
    from .payload_hybrid_duration import (
        DurationReference,
        duration_cases,
        duration_declaration,
        sources,
        SCOPE,
    )

    root = Path(root)
    if Path(source_root).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("Actual imported duration checker/project source differs")
    declaration = _read(root / "inputs.json")
    expected = duration_declaration()
    if any(declaration.get(key) != value for key, value in expected.items()) or declaration[
        "source_binding"
    ] != sources(source_root):
        raise ValueError("Complete exact predeclared seven-case source-bound duration group required")
    proof = declaration["reference_proof"]
    reference = DurationReference.from_proof(proof)
    if reference.proof != proof:
        raise ValueError("Actual historical reference evidence changed")
    outcomes = []
    for case in duration_cases():
        path = root / case["case_id"]
        if _read(path / "case.json") != case or _read(path / "design.json") != declaration["design"]:
            raise ValueError("Frozen duration input/design differs from its pre-execution declaration")
        outcomes.append(audit_case(path, source_root=source_root, reference=reference))
    return dict(
        schema="hybrid-independent-continuous-duration-group-v1",
        passed=all(r["passed"] for r in outcomes),
        distinct_cases=7,
        intervals_per_case=33000,
        duration_s=330.0,
        full_native_physics_replay=True,
        outcomes=outcomes,
        declaration_sha256=sha256(root / "inputs.json"),
        reference_proof=proof,
        source_binding=sources(source_root),
        scope=SCOPE,
        targets_visited_claim=False,
        learning_admitted=False,
    )
