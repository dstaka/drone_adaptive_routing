"""Independent final C-only raw audits and fixed-denominator statistics.

Pure reconstruction helpers support rejection tests, never physical admission.
Only audit_final_campaign combines current typed freeze with actual native data.
"""

from collections import Counter
import copy
import json
import math
from pathlib import Path

import numpy as np

from .payload_hybrid_audit import independent_metrics, _valid
from .payload_solver_validation import measured_predicates, _continuous
from .payload_solver_acceptance import wilson
from .util import sha256


def _read(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _finite_clock(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative physical wall clock required")
    return value


def audit_flush_receipt(root):
    root = Path(root)
    flush = _read(root / "final-flush.json")
    if flush.get("schema") != "payload-hybrid-final-flush-v1" or flush.get("flushed") is not True:
        raise ValueError("Actual native/control close did not complete")
    elapsed = _finite_clock(flush["elapsed_wall_s"])
    for key, path in [
        ("raw_sha256", "flight/runtime/raw.jsonl.gz"),
        ("control_sha256", "flight/control.jsonl.gz"),
    ]:
        if flush[key] != sha256(root / path):
            raise ValueError("Terminal close raw binding differs")
    return elapsed


def artifact_hashes(root):
    root = Path(root).resolve()
    # Independent summaries are regenerated and are never producer evidence.
    return {
        str(p): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "independent-check.json"
    }


def require_artifact_hashes(hashes):
    if not hashes:
        raise ValueError("Immutable raw evidence required")
    for path, value in hashes.items():
        if not Path(path).is_file() or sha256(path) != value:
            raise ValueError("Frozen prerequisite/raw evidence changed: " + path)


class VerifiedArtifactSet:
    """One exact hash pass, then metadata guards for immutable closed evidence.

    Unlike a persisted PASS flag, this process-local cache starts by checking
    every actual byte. A changed inode/size/mtime/ctime triggers a new exact hash.
    Restoring mtime does not restore ctime. Source/build bytes are checked by
    the separate current binding on every capability use.
    """

    def __init__(self, hashes):
        self.hashes = copy.deepcopy(hashes)
        require_artifact_hashes(self.hashes)
        self.fingerprints = {p: self._fingerprint(p) for p in self.hashes}

    @staticmethod
    def _fingerprint(path):
        s = Path(path).stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def require_current(self):
        for path, value in self.hashes.items():
            current = self._fingerprint(path)
            if current != self.fingerprints[path]:
                require_artifact_hashes({path: value})
                self.fingerprints[path] = current


def reconstruct_sequence(rows, case, *, flush_elapsed_wall_s):
    """All prescribed transitions, including terminal failures, from measured bodies."""
    flush = _finite_clock(flush_elapsed_wall_s)
    if not rows or rows[0]["kind"] != "INITIAL" or rows[0]["step"] != 0:
        raise ValueError("Initial physical boundary missing")
    targets = case["targets"]
    if np.asarray(targets).shape != (len(targets), 3) or not np.isfinite(targets).all():
        raise ValueError("Finite prescribed targets required")
    previous = rows[0]["sample"]
    if previous["step"] != 0 or previous["time_s"] != 0:
        raise ValueError("Initial control clock differs")
    # Full initial pose is also reconciled against the external case by native reader.
    index = -1
    step = 0
    completed = []
    intervals = 0
    previous_valid = False
    strict_previous = False
    strict_count = longest = 0
    strict = []
    begin = 0
    last_wall = 0.0
    unsafe_seen = False
    wall_terminal = False
    legs = []
    for position, row in enumerate(rows[1:], 1):
        if wall_terminal:
            raise ValueError("Continued after physical wall deadline")
        if row["kind"] == "RETARGET":
            if index >= 0 and len(completed) != index + 1:
                raise ValueError("Retarget before100 complete shared intervals")
            index += 1
            if (
                index >= len(targets)
                or row["step"] != step
                or row["target_index"] != str(index)
                or row["target"] != targets[index]
            ):
                raise ValueError("Prescribed target/time/order differs")
            before = {k: v for k, v in previous.items() if k != "effective_target"}
            after = {k: v for k, v in row["sample"].items() if k != "effective_target"}
            if before != after or row["sample"]["effective_target"] != targets[index]:
                raise ValueError("Retarget changed physical state/history")
            if index:
                strict.append(
                    dict(target=str(index - 1), passed=longest >= 100, longest_dwell_s=longest * 0.01)
                )
            intervals = 0
            previous_valid = False
            strict_previous = False
            strict_count = longest = 0
            begin = step
            previous = row["sample"]
            continue
        if row["kind"] != "STEP" or index < 0 or len(completed) > index or unsafe_seen:
            raise ValueError("Unexpected interval after completion/failure or before command")
        step += 1
        s = row["sample"]
        if (
            row["step"] != step
            or row["target_index"] != str(index)
            or row["target"] != targets[index]
            or s["step"] != step
            or s["time_s"] != step * 0.01
            or s["effective_target"] != targets[index]
            or step - begin > 3000
        ):
            raise ValueError("Physical interval/command/leg clock differs")
        wall = _finite_clock(row["reset_elapsed_wall_s"])
        if wall < last_wall or wall > flush:
            raise ValueError("Nonmonotonic clock or interval after terminal flush")
        last_wall = wall
        wall_terminal = wall >= 600
        _continuous(previous, s)
        measured = measured_predicates(s, targets[index], case["side_m"], revision2=True)
        metric = independent_metrics(s, np.asarray(targets[index]), case["side_m"])
        valid = _valid(metric, s["contacts"])
        intervals = intervals + 1 if valid and previous_valid else 0
        previous_valid = valid
        shadow = _valid(metric, s["contacts"], True)
        strict_count = strict_count + 1 if shadow and strict_previous else 0
        strict_previous = shadow
        longest = max(longest, strict_count)
        unsafe_seen = measured["unsafe"]
        if unsafe_seen and position != len(rows) - 1:
            raise ValueError("Continued after physical safety failure")
        if intervals >= 100 and wall < 600 and not unsafe_seen:
            completed.append(index)
            kind = (
                "initial_hold"
                if index == 0
                else "duplicate_hold"
                if targets[index] == targets[index - 1]
                else "coordinate_transfer"
            )
            legs.append(
                dict(
                    target_index=index, kind=kind, start_step=begin, end_step=step, valid_intervals=intervals
                )
            )
        previous = s
    if index >= 0:
        strict.append(dict(target=str(index), passed=longest >= 100, longest_dwell_s=longest * 0.01))
    return dict(
        success=len(completed) == len(targets) and not unsafe_seen and flush < 600,
        completed_targets=len(completed),
        completed=completed,
        steps=step,
        safe=not unsafe_seen,
        commanded_target_legs=len(completed),
        distinct_coordinate_transfers=sum(leg["kind"] == "coordinate_transfer" for leg in legs),
        initial_or_duplicate_holds=sum(leg["kind"] != "coordinate_transfer" for leg in legs),
        legs=legs,
        strict_shadow=strict,
        terminal_flush_wall_s=flush,
        scope="PURE_RECONSTRUCTION_NOT_PHYSICAL_ADMISSION",
    )


def _native_arguments(root, context, case):
    return dict(
        native_path=root / "flight/runtime/raw.jsonl.gz",
        properties_path=root / "flight/runtime-properties.json",
        checkpoint=root / "flight/design.json",
        expected_hash=sha256(root / "flight/design.json"),
        case=case,
        binding=context.reference.binding,
        control_path=root / "flight/control.jsonl.gz",
        admission=context.reference,
    )


def _mission_native_rows(root):
    samples = _rows(root / "mission/samples.jsonl")
    if samples[0]["kind"] != "INITIAL":
        raise ValueError("Mission initial boundary missing")
    rows = [dict(kind="INITIAL", step=0, sample=samples[0]["sample"])]
    previous = samples[0]["sample"]
    decisions = _rows(root / "mission/decisions.jsonl")
    by_step = {d["simulation_step"]: d for d in decisions}
    if len(by_step) != len(decisions):
        raise ValueError("Duplicate decision boundary")
    physical_steps = [r for r in samples[1:] if r["kind"] == "RAW_STEP"]
    # Include a final command even if timeout occurs before its first STEP.
    for i in range(len(physical_steps) + 1):
        if i in by_step:
            d = by_step[i]
            after = copy.deepcopy(previous)
            after["effective_target"] = d["target"]
            rows.append(
                dict(kind="RETARGET", step=i, target_index=d["target_id"], target=d["target"], sample=after)
            )
        if i < len(physical_steps):
            r = physical_steps[i]
            rows.append(
                dict(
                    kind="STEP",
                    step=r["simulation_step"],
                    target_index=r["target_id"],
                    target=r["target"],
                    sample=r["sample"],
                )
            )
            previous = r["sample"]
    return rows


def _route_commands(root, planner):
    """Replay every command, including a last command with no subsequent STEP."""
    from .analytical import snapshot, content_hash
    from .objective import RoutingState
    from .scenario import Scenario
    from .payload_recovery_mission import explain_payload_decision
    from .payload_solver_acceptance import _objective_fields

    scenario = Scenario.from_dict(_read(root / "scenario.json"))
    state = RoutingState(scenario)
    decisions = _rows(root / "decisions.jsonl")
    explanations = _rows(root / "decision_explanations.jsonl")
    events = _rows(root / "events.jsonl")
    metadata = _read(root / "metadata.json")
    samples = _rows(root / "samples.jsonl")
    side = _read(root / "payload.json")["side_m"]
    poses = {0: samples[0]["sample"]["drone_xyz"]}
    poses.update({r["simulation_step"]: r["sample"]["drone_xyz"] for r in samples if r["kind"] == "RAW_STEP"})
    raw = [r for r in samples[1:] if r["kind"] == "RAW_STEP"]
    commands = {d["simulation_step"]: d for d in decisions}
    if len(commands) != len(decisions):
        raise ValueError("Repeated actual command boundary")
    settled = {e["simulation_step"]: e for e in events if e["type"] == "SHARED_SETTLED"}
    command = None
    completed_command = False
    for i, r in enumerate(raw, 1):
        if i - 1 in commands:
            command = commands[i - 1]
            completed_command = False
        if (
            command is None
            or completed_command
            or r["simulation_step"] != i
            or r["target_id"] != command["target_id"]
            or r["target"] != command["target"]
            or r["sample"]["effective_target"] != command["target"]
            or i - command["simulation_step"] > 3000
        ):
            raise ValueError("Every actual STEP must follow the active original routing command/leg clock")
        if i in settled:
            if settled[i]["target_id"] != command["target_id"]:
                raise ValueError("Actual completion target differs")
            completed_command = True
        m = measured_predicates(r["sample"], r["target"], side, revision2=True)
        if m["unsafe"] and i != len(raw):
            raise ValueError("Continued after physical safety failure")
    active = None
    selected = 0
    completed = []
    event_applied = False
    pending_leg = False
    terminal = None
    if not events or events[0]["type"] != "MISSION_STARTED":
        raise ValueError("Exactly one initial mission start required")
    for event_index, e in enumerate(events):
        if terminal is not None:
            if not (
                terminal["type"] == "MISSION_FINISHED"
                and e["type"] == "MISSION_FAILED"
                and e["termination_code"] == "WALL_TIMEOUT"
                and event_index == len(events) - 1
                and e["simulation_step"] == terminal["simulation_step"]
            ):
                raise ValueError("Continued/repeated events after the first terminal event")
        step = e["simulation_step"]
        if step not in poses or e["sim_time_s"] != step * 0.01 or e["physical_pose"] != poses[step]:
            raise ValueError("Original terminal event clock/pose differs")
        if e["type"] == "TARGET_SELECTED":
            if active is not None or state.terminal or selected >= len(decisions):
                raise ValueError("Unexpected terminal routing command")
            d = decisions[selected]
            action = int(planner.predict(state.copy()))
            expected = snapshot(state)
            if (
                d["selected_action"] != action
                or e["selected_action"] != action
                or d["target_id"] != scenario.checkpoint_ids[action]
                or e["target_id"] != d["target_id"]
                or d["target"] != state.scenario.coords[action + 1]
                or d["boundary_snapshot"] != expected
                or d["boundary_hash"] != content_hash(expected)
                or d["simulation_step"] != step
                or d["event_seq"] != e["event_seq"]
                or d["physical_pose"] != poses[step]
                or d["policy_hash"] != metadata["routing_model_hash"]
                or not np.array_equal(d["feature_vector"], state.observation())
                or d["mask"] != state.mask().tolist()
                or _objective_fields(explanations[selected])
                != _objective_fields(explain_payload_decision(d, planner.predict))
            ):
                raise ValueError("Original terminal routing/snapshot/explanation differs")
            active = action
            pending_leg = True
            selected += 1
        elif e["type"] == "LEG_STARTED":
            if (
                active is None
                or not pending_leg
                or e["target_id"] != scenario.checkpoint_ids[active]
                or step != decisions[selected - 1]["simulation_step"]
            ):
                raise ValueError("Terminal leg start differs from selected command")
            pending_leg = False
        elif e["type"] == "SHARED_SETTLED":
            if active is None or e["target_id"] != scenario.checkpoint_ids[active]:
                raise ValueError("Unexpected route completion")
            state.complete(active, state.departure(active))
            completed.append(e["target_id"])
            active = None
        elif e["type"] in ("PRIORITY_UPDATED", "COST_UPDATED"):
            if event_applied or len(completed) != 1 or state.apply_event() != e["update"]:
                raise ValueError("Actual remaining routing event differs")
            event_applied = True
        elif e["type"] == "MISSION_FINISHED":
            if not state.terminal:
                raise ValueError("Mission finish before all real completions")
            terminal = e
        elif e["type"] == "MISSION_FAILED":
            terminal = e
        elif e["type"] == "MISSION_STARTED":
            if event_index != 0 or step != 0:
                raise ValueError("Repeated or late mission start")
        elif e["type"] not in ("MISSION_STARTED", "MISSION_FAILED"):
            raise ValueError("Unknown actual mission event")
    if (
        selected != len(decisions)
        or len(explanations) != len(decisions)
        or pending_leg
        or terminal is None
        or terminal["simulation_step"] != len(raw)
    ):
        raise ValueError("Actual terminal command coverage differs")


def verify_final_mission(output, planner, *, native=None):
    """Validate raw-first failure without changing original producer evidence.

    An audit-only temporary copy validates the nonoffending prefix with the
    unchanged reader. The original offending interval is independently checked
    here and remains covered by the caller's full native/controller proof.
    No generated predicate or completion is substituted into retained evidence.
    """
    import shutil
    import tempfile
    from .payload_solver_acceptance import verify_mission

    root = Path(output)
    samples = _rows(root / "samples.jsonl")
    predicates = _rows(root / "predicates.jsonl")
    raw = [r for r in samples[1:] if r["kind"] == "RAW_STEP"]
    summary = _read(root / "mission_summary.json")
    terminal_verified = False
    classification = summary["termination_code"]
    if len(predicates) == len(raw):
        checked = verify_mission(root, planner, native=native, revision2=True)
    else:
        if (
            summary["status"] != "FAILED"
            or not raw
            or len(predicates) != len(raw) - 1
            or samples[-1] != raw[-1]
        ):
            raise ValueError("Only one actual terminal raw-first failure may lack a predicate")
        bindings = (
            ("samples.jsonl", "telemetry_sha256"),
            ("events.jsonl", "events_sha256"),
            ("decisions.jsonl", "decisions_sha256"),
            ("predicates.jsonl", "predicates_sha256"),
            ("decision_explanations.jsonl", "explanations_sha256"),
            ("scenario.json", "scenario_sha256"),
            ("metadata.json", "metadata_sha256"),
            ("payload.json", "payload_sha256"),
            ("initial_state.json", "initial_state_sha256"),
        )
        if any(sha256(root / name) != summary[key] for name, key in bindings):
            raise ValueError("Original failed mission artifact binding differs")
        last = raw[-1]
        n = len(raw)
        if (
            last["simulation_step"] != n
            or last["sim_time_s"] != n * 0.01
            or last["sample"]["step"] != n
            or last["sample"]["time_s"] != n * 0.01
            or last["sample"]["effective_target"] != last["target"]
        ):
            raise ValueError("Terminal offending raw clock/command differs")
        previous = samples[0]["sample"] if n == 1 else raw[-2]["sample"]
        _continuous(previous, last["sample"])
        side = _read(root / "payload.json")["side_m"]
        m = measured_predicates(last["sample"], last["target"], side, revision2=True)
        wall = _finite_clock(summary["wall_residence_s"])
        if m["unsafe"]:
            if classification not in ("PHYSICAL_SAFETY_FAILURE", "WALL_TIMEOUT"):
                raise ValueError("Terminal unsafe raw does not explain the saved failure")
            classification = "SAFETY_FAILURE"
        elif classification != "WALL_TIMEOUT" or wall < 600:
            raise ValueError("Missing ordinary predicate is not an explained terminal failure")
        terminal_events = _rows(root / "events.jsonl")
        if (
            not terminal_events
            or terminal_events[-1]["type"] != "MISSION_FAILED"
            or terminal_events[-1]["simulation_step"] != n
            or terminal_events[-1]["termination_code"] != summary["termination_code"]
        ):
            raise ValueError("Missing actual offending-interval terminal event")
        with tempfile.TemporaryDirectory(prefix="hybrid-final-failure-prefix-") as temporary:
            prefix = Path(temporary) / "mission"
            shutil.copytree(root, prefix)
            (prefix / "samples.jsonl").write_text(
                "".join(json.dumps(r, allow_nan=False) + "\n" for r in samples[:-1])
            )
            adjusted = copy.deepcopy(terminal_events)
            for e in adjusted:
                if e["simulation_step"] == n:
                    if e["type"] != "MISSION_FAILED":
                        raise ValueError("Unexplained event after an offending raw interval")
                    e.update(
                        simulation_step=n - 1, sim_time_s=(n - 1) * 0.01, physical_pose=previous["drone_xyz"]
                    )
            (prefix / "events.jsonl").write_text(
                "".join(json.dumps(e, allow_nan=False) + "\n" for e in adjusted)
            )
            updated = copy.deepcopy(summary)
            updated.update(
                steps=n - 1,
                sim_time_s=(n - 1) * 0.01,
                telemetry_sha256=sha256(prefix / "samples.jsonl"),
                events_sha256=sha256(prefix / "events.jsonl"),
            )
            (prefix / "mission_summary.json").write_text(json.dumps(updated, allow_nan=False))
            checked = verify_mission(prefix, planner, native=native, revision2=True)
        if checked["success"]:
            raise ValueError("Failed prefix unexpectedly completed the mission")
        checked.update(steps=n, summary_sha256=sha256(root / "mission_summary.json"))
        terminal_verified = True
    _route_commands(root, planner)
    checked.update(
        independent_termination_code=classification,
        terminal_raw_without_predicate_verified=terminal_verified,
        producer_termination_code=summary["termination_code"],
    )
    return checked


def audit_final_case(output, *, context, expected_case, planner=None):
    from .payload_hybrid_final import FinalContext, digest
    from .payload_hybrid_native import verify_hybrid_native

    if not isinstance(context, FinalContext):
        raise ValueError("Typed current final context required")
    context.require_current()
    root = Path(output)
    case = _read(root / "case.json")
    recorded_binding = _read(root / "binding.json")
    if case != expected_case:
        raise ValueError("Actual case/source binding differs from external declaration")
    context.require_recorded_binding(recorded_binding, case["suite"])
    attempt = _read(root / "attempt.json")
    if attempt["attempt"] != 1 or attempt["stage"] != case["suite"]:
        raise ValueError("Exactly one declared physical attempt required")
    if case["suite"] in ("final-development", "final-validation"):
        from .payload_hybrid_final import require_development_attempt

        declaration_hash = require_development_attempt(root, context, case)
    else:
        declaration_hash = digest(_read(root.parent / "declaration.json"))
    if attempt["declaration_sha256"] != declaration_hash:
        raise ValueError("Physical attempt does not bind its prior external declaration")
    if (root / "flight/design.json").is_file() and digest(
        _read(root / "flight/design.json")
    ) != context.binding["design_value_sha256"]:
        raise ValueError("Actual controller design differs from final binding")
    errors = []
    proof = None
    live = None
    flush = _read(root / "final-flush.json")
    try:
        elapsed = audit_flush_receipt(root)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        elapsed = None
        errors.append("flush: " + str(exc))
    if case["suite"] == "final-validation":
        try:
            rows = _rows(root / "sequence/samples.jsonl")
            if elapsed is not None:
                live = reconstruct_sequence(rows, case, flush_elapsed_wall_s=elapsed)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append("validation: " + str(exc))
            rows = None
        if rows is not None:
            try:
                proof = verify_hybrid_native(rows, **_native_arguments(root, context, case))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                errors.append("native: " + str(exc))
    else:
        try:
            from .payload_hybrid_final import ROUTING_SHA256

            if planner is None or planner.hash != ROUTING_SHA256:
                raise ValueError("Original routing planner required")
            metadata = _read(root / "mission/metadata.json")
            if (
                metadata["final_binding_sha256"] != digest(recorded_binding)
                or metadata["final_stage"] != case["suite"]
                or metadata["actual_binding"] != context.reference.binding
                or metadata["routing_model_hash"] != ROUTING_SHA256
                or metadata["controller"] != "C-local-lqr-v1"
                or metadata["flight_nn"] != "NOT_USED"
            ):
                raise ValueError("Actual mission final source/controller binding differs")
            live = verify_final_mission(
                root / "mission", planner, native=_native_arguments(root, context, case)
            )
            proof = live["native_proof"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append("mission: " + str(exc))
        # The inherited reader only validates native evidence on success.
        if proof is None:
            try:
                proof = verify_hybrid_native(
                    _mission_native_rows(root), **_native_arguments(root, context, case)
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                errors.append("native: " + str(exc))
    evidence_complete = bool(live is not None and proof is not None and not errors and elapsed is not None)
    success = bool(
        live
        and live["success"]
        and proof
        and not errors
        and elapsed is not None
        and elapsed < 600
        and not flush.get("error")
    )
    gaps = []
    if live and case["suite"] != "final-validation":
        for explanation in _rows(root / "mission/decision_explanations.jsonl"):
            p = explanation["pose_cost_analysis"]
            chosen = p["actual_selected_action"]
            forced = next(v for v in p["forced_first"] if v["action"] == chosen)
            gaps.append(
                dict(
                    decision_id=explanation.get("decision_id"),
                    actual_pose=p["actual_pose"],
                    selected_action=chosen,
                    policy_suffix_cost=p["policy_suffix"]["cost"],
                    optimal_suffix_cost=p["exact"]["cost"],
                    policy_suffix_gap=p["policy_suffix"]["cost"] - p["exact"]["cost"],
                    selected_first_gap=forced["cost"] - p["exact"]["cost"],
                    scope=p["scope"],
                )
            )
    return dict(
        schema="payload-hybrid-final-case-audit-v1",
        case_id=case["case_id"],
        success=success,
        physical_acceptance=success,
        evidence_complete=evidence_complete,
        native_reconstructed=proof is not None,
        mission_reconstructed=live is not None,
        completed_targets=live.get("completed_targets", live.get("completed_count", 0)) if live else 0,
        validation_target_legs=live.get("commanded_target_legs") if live else None,
        validation_coordinate_transfers=live.get("distinct_coordinate_transfers") if live else None,
        validation_initial_or_duplicate_holds=live.get("initial_or_duplicate_holds") if live else None,
        validation_legs=live.get("legs") if live else None,
        independent_termination_code=(
            "WALL_TIMEOUT"
            if elapsed is not None and elapsed >= 600
            else live.get("independent_termination_code")
            if live
            else None
        ),
        steps=live["steps"] if live else None,
        terminal_flush_wall_s=elapsed,
        audit_errors=errors,
        execution_error=flush.get("error"),
        strict_shadow=proof["strict_shadow"] if proof else [],
        native_proof=proof,
        routing_gaps=gaps,
        artifacts=artifact_hashes(root),
    )


def expected_diagnostic_keys():
    from .payload_hybrid_manifest import development_cases

    cases = {c["case_id"]: c for c in development_cases()}
    return {(k, 16, True) for k in cases} | {
        (k, s, f) for k, c in cases.items() if c["gate"] == "C0" for s, f in ((16, False), (32, True))
    }


def frozen_diagnostic_cases(location):
    """Use the producer's original case bytes after its existing redraw guard.

    The shared validator permits only its established initial-state1e-14
    cross-build roundoff. audit_case still compares the actual saved case and
    native state against these ORIGINAL values, never against a fresh redraw.
    """
    from .payload_hybrid_manifest import validate_cases

    root = Path(location)
    manifest = root.parent / "development-manifest-v1.json"
    if sha256(manifest) != _read(root / "inputs.json")["manifest_sha256"]:
        raise ValueError("Original diagnostic manifest differs from preserved producer input")
    declared = _read(manifest)["cases"]
    validate_cases(declared)
    return {c["case_id"]: c for c in declared}, manifest


def audit_reference_gates(context, diagnostic_roots):
    """Reuse unchanged50 gate traces +20 paired diagnostic traces, re-read all raw."""
    from .payload_hybrid_audit import audit_case
    from .payload_hybrid_manifest import validate_gate_result
    from .payload_solver_qualification import physical_errors

    context.require_current()
    cases = None
    original_manifest_sha256 = None
    expected = expected_diagnostic_keys()
    outcomes = {}
    artifacts = {}
    for location in diagnostic_roots:
        root = Path(location)
        original, manifest = frozen_diagnostic_cases(root)
        actual_manifest_sha256 = sha256(manifest)
        if cases is None:
            cases = original
            original_manifest_sha256 = actual_manifest_sha256
        elif actual_manifest_sha256 != original_manifest_sha256:
            raise ValueError("All C50+20 traces must use the same original frozen manifest")
        artifacts[str(manifest.resolve())] = actual_manifest_sha256
        for casefile in sorted(root.glob("*/case.json")):
            directory = casefile.parent
            case = _read(casefile)
            metadata = _read(directory / "metadata.json")
            key = (case["case_id"], metadata["substeps"], metadata["vibration_feedback"])
            if key not in expected or key in outcomes or type(metadata["vibration_feedback"]) is not bool:
                raise ValueError("Duplicate/unexpected C diagnostic case/settings")
            if (
                metadata["reference_binding"] != context.reference.binding
                or metadata.get("backend") != "mujoco-cpu-float64-RK4"
                or _read(root / "inputs.json")["design_sha256"] != context.binding["design_sha256"]
                or _read(directory / "design.json") != _read(context.paths["design"])
            ):
                raise ValueError("C diagnostic reference/design binding differs")
            props = _read(directory / "runtime-properties.json")
            if (
                physical_errors(
                    props,
                    dict(side_m=case["side_m"], mass_ratio=case["mass_ratio"], substeps=metadata["substeps"]),
                    fixed=False,
                )
                or sha256(directory / "fixture/model.xml") != props["model_xml_sha256"]
            ):
                raise ValueError("C diagnostic actual physics properties differ")
            outcomes[key] = dict(
                audit_case(directory, expected_case=cases[case["case_id"]]), case_id=case["case_id"]
            )
            artifacts.update(artifact_hashes(directory))
        artifacts[str((root / "inputs.json").resolve())] = sha256(root / "inputs.json")
    if set(outcomes) != expected:
        raise ValueError("Exact C50+20 paired diagnostic denominator required")
    for gate in ("C0", "C1", "C2"):
        rows = []
        for k, case in cases.items():
            if case["gate"] != gate:
                continue
            r = outcomes[k, 16, True]
            rows.append(dict(r, success=r["physical_success"]))
        if not validate_gate_result(gate, rows):
            raise ValueError("Independent C gate failed: " + gate)
    return dict(gates50=True, diagnostics20=True, artifacts=artifacts)


def audit_duration_prerequisite(context, root):
    from .payload_hybrid_final import ROOT
    from .payload_hybrid_duration_audit import audit_group

    proof = audit_group(root, source_root=ROOT)
    if (
        proof.get("schema") != "hybrid-independent-continuous-duration-group-v1"
        or proof.get("passed") is not True
        or proof.get("distinct_cases") != 7
        or proof.get("intervals_per_case") != 33000
        or proof.get("duration_s") != 330.0
        or proof.get("full_native_physics_replay") is not True
        or proof["reference_proof"]["actual_binding"] != context.reference.binding
        or _read(Path(root) / "inputs.json")["design"] != _read(context.paths["design"])
    ):
        raise ValueError("Complete independently reconstructed matching C duration7 required")
    return dict(summary=proof, artifacts=artifact_hashes(root))


def audit_prerequisites(context, development, diagnostic_roots, duration_root, planner):
    from .payload_hybrid_final import robustness_cases, validation_cases

    artifacts = {
        str((Path(development) / "declaration.json").resolve()): sha256(
            Path(development) / "declaration.json"
        )
    }
    gates = audit_reference_gates(context, diagnostic_roots)
    artifacts.update(gates["artifacts"])
    duration = audit_duration_prerequisite(context, duration_root)
    artifacts.update(duration["artifacts"])
    summaries = {}
    for stage, cases in [("robustness", robustness_cases()), ("validation", validation_cases())]:
        stage_root = Path(development) / stage
        if {p.name for p in stage_root.iterdir() if p.is_dir()} != {c["case_id"] for c in cases}:
            raise ValueError("Exact declared development directories required; retries forbidden")
        results = []
        for case in cases:
            proof = audit_final_case(
                Path(development) / stage / case["case_id"],
                context=context,
                expected_case=case,
                planner=planner,
            )
            if not proof["success"] or not proof["native_reconstructed"]:
                raise ValueError("Final prerequisite failed: " + case["case_id"])
            artifacts.update(proof["artifacts"])
            results.append(
                dict(
                    case_id=case["case_id"],
                    steps=proof["steps"],
                    completed_targets=proof["completed_targets"],
                    commanded_target_legs=proof["validation_target_legs"],
                    distinct_coordinate_transfers=proof["validation_coordinate_transfers"],
                    initial_or_duplicate_holds=proof["validation_initial_or_duplicate_holds"],
                    legs=proof["validation_legs"],
                )
            )
        summaries[stage] = results
    if sum(r["completed_targets"] for r in summaries["validation"]) != 120:
        raise ValueError("All120 validation transitions required")
    validation_totals = dict(
        commanded_target_legs=sum(r["commanded_target_legs"] for r in summaries["validation"]),
        distinct_coordinate_transfers=sum(
            r["distinct_coordinate_transfers"] for r in summaries["validation"]
        ),
        initial_or_duplicate_holds=sum(r["initial_or_duplicate_holds"] for r in summaries["validation"]),
    )
    if validation_totals != dict(
        commanded_target_legs=120, distinct_coordinate_transfers=96, initial_or_duplicate_holds=24
    ):
        raise ValueError("Inherited120 validation legs must comprise96 transfers and24 holds")
    return dict(
        summary=dict(
            C_gates50=True,
            C_paired_diagnostics20=True,
            duration7=duration["summary"],
            validation_totals=validation_totals,
            **summaries,
        ),
        artifacts=artifacts,
    )


def summarize_outcomes(cases, outcomes):
    """Descriptive pure helper; caller must independently audit every actual mission."""
    if Counter(c["suite"] for c in cases) != dict(basic=9, endpoints=18, heldout=100):
        raise ValueError("Original9/18/100 denominator required")
    expected = {c["case_id"] for c in cases}
    if len(expected) != 127 or len(outcomes) != 127 or {r["case_id"] for r in outcomes} != expected:
        raise ValueError("Exactly127 unique outcomes required, including failures")
    by_id = {r["case_id"]: r for r in outcomes}
    if any(type(r["success"]) is not bool for r in outcomes):
        raise ValueError("Explicit independent Boolean verdicts required")

    def group(selected):
        successes = sum(by_id[c["case_id"]]["success"] for c in selected)
        n = len(selected)
        return dict(
            successes=successes,
            total=n,
            rate=successes / n,
            wilson95=wilson(successes, n),
            failed_case_ids=[c["case_id"] for c in selected if not by_id[c["case_id"]]["success"]],
        )

    suites = {s: group([c for c in cases if c["suite"] == s]) for s in ("basic", "endpoints", "heldout")}
    subgroups = {}
    for field in ("scenario", "side_m"):
        subgroups[field] = {
            str(v): group([c for c in cases if c[field] == v]) for v in sorted({c[field] for c in cases})
        }
    subgroups["mass_ratio_thirds"] = {
        str(i): group([c for c in cases if min(2, int((c["mass_ratio"] - 0.1) / (0.2 / 3))) == i])
        for i in range(3)
    }
    strict = [s for r in outcomes for s in r.get("strict_shadow", [])]
    return dict(
        accepted=suites["basic"]["successes"] == 9
        and suites["endpoints"]["successes"] == 18
        and suites["heldout"]["successes"] >= 95,
        suites=suites,
        overall=group(cases),
        subgroups=subgroups,
        strict_shadow=dict(
            passed=sum(s["passed"] for s in strict),
            attempted=len(strict),
            scope="Observed video-first event stream only; no counterfactual strict mission claim",
        ),
        routing_gaps=[dict(case_id=r["case_id"], **g) for r in outcomes for g in r.get("routing_gaps", [])],
        failures=[r for r in outcomes if not r["success"]],
        physical_admission="NOT_GRANTED_BY_PURE_STATISTICS",
    )


def evidence_gate(outcomes):
    """Missing/corrupt evidence is independent of the allowed empirical failures."""
    incomplete = [r["case_id"] for r in outcomes if r.get("evidence_complete") is not True]
    return dict(
        complete=not incomplete,
        incomplete_case_ids=incomplete,
        scope="Every attempt requires complete independent native/mission/flush reconstruction",
    )


def audit_final_campaign(freeze, output, planner):
    from .payload_hybrid_final import FinalFreeze, formal_cases, digest

    if not isinstance(freeze, FinalFreeze):
        raise ValueError("Typed final freeze required")
    freeze.require_current()
    root = Path(output)
    cases = formal_cases(freeze)
    declaration = _read(root / "declaration.json")
    if declaration != dict(
        schema="payload-hybrid-final-formal-v1",
        freeze_sha256=digest(freeze.record),
        binding=freeze.context.binding,
        cases=cases,
        attempts_per_case=1,
        method="C",
    ):
        raise ValueError("Actual formal declaration differs from final freeze/original draws")
    expected = {c["case_id"] for c in cases}
    actual = {p.name for p in root.iterdir() if p.is_dir()}
    if actual != expected:
        raise ValueError("All127 unique formal case directories required; retries forbidden")
    outcomes = []
    artifacts = {str((root / "declaration.json").resolve()): sha256(root / "declaration.json")}
    for case in cases:
        attempt = _read(root / case["case_id"] / "attempt.json")
        if attempt["freeze_sha256"] != digest(freeze.record):
            raise ValueError("Physical attempt predates/differs from final freeze")
        row = audit_final_case(
            root / case["case_id"], context=freeze.context, expected_case=case, planner=planner
        )
        outcomes.append(row)
        artifacts.update(row["artifacts"])
    report = summarize_outcomes(cases, outcomes)
    completeness = evidence_gate(outcomes)
    report["empirical_thresholds_met"] = report["accepted"]
    report["accepted"] = bool(report["accepted"] and completeness["complete"])
    report["evidence_completeness"] = completeness
    report.update(
        schema="payload-hybrid-final-campaign-audit-v1",
        freeze_sha256=digest(freeze.record),
        formal_declaration_sha256=sha256(root / "declaration.json"),
        outcomes=outcomes,
        artifacts=artifacts,
        branch="controller",
        methods=1,
        learning_admitted=False,
    )
    return report
