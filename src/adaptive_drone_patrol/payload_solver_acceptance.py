"""Independent ten-stop raw replay and fixed paired acceptance denominators."""

from collections import Counter
import copy
import json
import math
from pathlib import Path

import numpy as np

from .analytical import snapshot, content_hash, event_comparison
from .baselines import exact_remaining, remaining_cost
from .objective import RoutingState
from .payload_recovery_mission import explain_payload_decision
from .payload_solver_validation import measured_predicates, _continuous, verify_initial_sample
from .scenario import Scenario
from .util import sha256, load_config


def verify_case_scenario(scenario, case):
    """Bind the complete mission configuration to the frozen physical case."""
    name = case["scenario"]
    if name not in {"normal", "emergency", "cost_shift"}:
        raise ValueError("Unknown frozen case scenario")
    expected = Scenario.from_dict(load_config(f"configs/demo_scenarios_dispersed/{name}.yaml"))
    if scenario.to_dict() != expected.to_dict():
        raise ValueError("Actual mission scenario differs from frozen case scenario")
    return True


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _objective_fields(value):
    """Compare reproducible objective fields; validate solver timing separately."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "elapsed_s":
                if type(item) not in (int, float) or not math.isfinite(item) or item < 0:
                    raise ValueError("Invalid exact-solver duration")
            else:
                result[key] = _objective_fields(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_objective_fields(item) for item in value]
    return value


def verify_mission(output, planner, *, native=None):
    """Derive routing commands/events and all100-interval stops from saved bodies."""
    root = Path(output)
    summary = json.loads((root / "mission_summary.json").read_text())
    for name, key in [
        ("samples.jsonl", "telemetry_sha256"),
        ("events.jsonl", "events_sha256"),
        ("decisions.jsonl", "decisions_sha256"),
        ("predicates.jsonl", "predicates_sha256"),
        ("decision_explanations.jsonl", "explanations_sha256"),
        ("scenario.json", "scenario_sha256"),
        ("metadata.json", "metadata_sha256"),
        ("payload.json", "payload_sha256"),
        ("initial_state.json", "initial_state_sha256"),
    ]:
        if sha256(root / name) != summary[key]:
            raise ValueError("Stale mission artifact binding")
    scenario = Scenario.from_dict(json.loads((root / "scenario.json").read_text()))
    metadata = json.loads((root / "metadata.json").read_text())
    payload = json.loads((root / "payload.json").read_text())
    side = payload["side_m"]
    samples = _rows(root / "samples.jsonl")
    events = _rows(root / "events.jsonl")
    decisions = _rows(root / "decisions.jsonl")
    predicates = _rows(root / "predicates.jsonl")
    explanations = _rows(root / "decision_explanations.jsonl")
    if not samples or samples[0]["kind"] != "INITIAL" or samples[0]["simulation_step"] != 0:
        raise ValueError("Missing actual initial record")
    physical = metadata["evidence_kind"] == "physical"
    if physical:
        if native is None:
            raise ValueError("Actual native case/model binding required")
        verify_case_scenario(scenario, native["case"])
        if (payload['side_m'] != native['case']['side_m']
                or payload['mass_ratio'] != native['case']['mass_ratio']
                or metadata['case_id'] != native['case']['case_id']
                or metadata['checkpoint_sha256'] != native['expected_hash']):
            raise ValueError('Actual mission physical case/model binding differs')
        verify_initial_sample(samples[0]["sample"], native["case"]["initial"])
    if [r["event_seq"] for r in events] != list(range(1, len(events) + 1)):
        raise ValueError("Missing/reordered actual events")
    step_rows = [r for r in samples[1:] if r["kind"] == "RAW_STEP"]
    if [r["simulation_step"] for r in step_rows] != list(range(1, len(step_rows) + 1)):
        raise ValueError("Missing/duplicate actual control intervals")
    if len(predicates) != len(step_rows):
        raise ValueError("Incomplete measured predicate coverage")
    actual_samples = [samples[0]["sample"], *[r["sample"] for r in step_rows]]
    previous_event_step = 0
    for event in events:
        step = event["simulation_step"]
        if (type(step) is not int or not previous_event_step <= step <= len(step_rows)
                or event["sim_time_s"] != step * .01
                or event["run_id"] != summary["run_id"]
                or event["evidence_kind"] != metadata["evidence_kind"]
                or not np.array_equal(event["physical_pose"], actual_samples[step]["drone_xyz"])):
            raise ValueError("Event time, order or measured pose differs")
        previous_event_step = step
    selected = [e for e in events if e["type"] == "TARGET_SELECTED"]
    if len(selected) != len(decisions) or len(explanations) != len(decisions):
        raise ValueError("Missing actual decisions/explanations")
    state = RoutingState(scenario)
    completed = []
    expected_events = []

    def expect_event(kind, step, sample, **fields):
        row = dict(type=kind, event_seq=len(expected_events) + 1, run_id=summary["run_id"],
                   simulation_step=step, sim_time_s=step * .01,
                   physical_pose=copy.deepcopy(sample["drone_xyz"]),
                   completed_order=completed.copy(), evidence_kind=metadata["evidence_kind"], **fields)
        expected_events.append(row)
        return row

    expect_event("MISSION_STARTED", 0, samples[0]["sample"])
    reached = []
    native_rows = [dict(kind="INITIAL", step=0, sample=samples[0]["sample"])]
    previous = samples[0]["sample"]
    selected_by_step = {d["simulation_step"]: d for d in decisions}
    if len(selected_by_step) != len(decisions):
        raise ValueError("Repeated decision boundary")
    active = None
    target = None
    intervals, previous_valid, leg_start = 0, False, 0
    total_distance = 0.0
    event_expected = None
    for i, row in enumerate(step_rows, 1):
        if i - 1 in selected_by_step:
            decision = selected_by_step[i - 1]
            if active is not None or state.terminal:
                raise ValueError("Retarget before shared completion")
            action = int(planner.predict(state.copy()))
            active = action
            target = scenario.coords[action + 1]
            expected_snapshot = snapshot(state)
            if (
                decision["selected_action"] != action
                or decision["target_id"] != scenario.checkpoint_ids[action]
                or decision["boundary_snapshot"] != expected_snapshot
                or decision["boundary_hash"] != content_hash(expected_snapshot)
                or decision["policy_hash"] != metadata["routing_model_hash"]
                or not np.array_equal(decision["feature_vector"], state.observation().tolist())
                or decision["mask"] != state.mask().tolist()
                or not np.array_equal(decision["physical_pose"], previous["drone_xyz"])
            ):
                raise ValueError("Actual learned routing decision differs from replay")
            event_row = next((e for e in selected if e["event_seq"] == decision["event_seq"]), None)
            expected_selected = expect_event("TARGET_SELECTED", i - 1, previous,
                                             target_id=scenario.checkpoint_ids[action], selected_action=action)
            if (event_row != expected_selected
                    or any(decision.get(k) != v for k, v in expected_selected.items())
                    or not np.array_equal(decision["target"], target)):
                raise ValueError("Decision/event boundary differs")
            expect_event("LEG_STARTED", i - 1, previous, target_id=scenario.checkpoint_ids[action])
            explanation = explanations[len(completed)]
            if _objective_fields(explain_payload_decision(decision, planner.predict)) != _objective_fields(
                explanation
            ):
                raise ValueError("Q/J/suffix or measured-pose explanation differs")
            after = copy.deepcopy(previous)
            after["effective_target"] = copy.deepcopy(target)
            native_rows.append(
                dict(
                    kind="RETARGET",
                    step=i - 1,
                    target_index=scenario.checkpoint_ids[action],
                    target=target,
                    sample=after,
                )
            )
            intervals, previous_valid, leg_start = 0, False, i - 1
        if (
            active is None
            or row["target_id"] != scenario.checkpoint_ids[active]
            or not np.array_equal(row["target"], target)
        ):
            raise ValueError("Measured leg differs from prescribed routing command")
        sample = row["sample"]
        if (
            sample["step"] != i
            or sample["time_s"] != i * 0.01
            or not np.array_equal(sample["effective_target"], target)
            or row["sim_time_s"] != i * 0.01
            or i > 12000
            or i - leg_start > 3000
        ):
            raise ValueError("Mission time/command/reset mismatch")
        _continuous(previous, sample)
        m = measured_predicates(sample, target, side)
        unsafe = m["unsafe"]
        intervals = intervals + 1 if m["valid"] and previous_valid else 0
        previous_valid = m["valid"]
        complete = intervals >= 100
        p = predicates[i - 1]
        if (
            p["simulation_step"] != i
            or p["target_id"] != row["target_id"]
            or p["valid_intervals"] != intervals
            or not math.isclose(p["dwell_s"], intervals * 0.01, abs_tol=1e-8)
            or p["shared_complete"] != complete
        ):
            raise ValueError("Claimed shared settling differs from measured replay")
        total_distance += float(np.linalg.norm(np.array(sample["drone_xyz"]) - previous["drone_xyz"]))
        native_rows.append(
            dict(
                kind="STEP",
                step=i,
                target_index=scenario.checkpoint_ids[active],
                target=target,
                sample=sample,
            )
        )
        previous = sample
        if unsafe:
            if i != len(step_rows) or summary["status"] == "SUCCEEDED":
                raise ValueError("Unsafe state passed as success or continued")
        if complete:
            departure = state.departure(active)
            state.complete(active, departure)
            completed.append(scenario.checkpoint_ids[active])
            reached.append((i, completed[-1], intervals))
            expect_event("SHARED_SETTLED", i, sample, target_id=completed[-1],
                         valid_intervals=intervals, dwell_s=intervals * .01)
            active = None
            if len(completed) == 1 and scenario.event:
                before = state.copy()
                comparison = event_comparison(state, planner.predict)
                old = int(planner.predict(state.copy()))
                update = state.apply_event()
                new = int(planner.predict(state.copy()))
                if not update or update["skipped"]:
                    raise ValueError("Exact unfinished-task event was not applied")
                event_expected = dict(step=i, comparison=comparison, update=update, old=old, new=new)
                pose = state.copy()
                pose.scenario.coords[pose.current] = copy.deepcopy(sample["drone_xyz"])
                retained = remaining_cost(pose, comparison["no_event_order"])
                replanned = remaining_cost(pose, comparison["post_event_order"])
                pose_comparison = dict(actual_pose=copy.deepcopy(sample["drone_xyz"]),
                    scope="Same actual pose and post-event remaining tasks; adopted policy suffixes; "
                          "nominal speed/service, no future events or physical optimality claim",
                    retained_order_post_event=retained, replanned_order_post_event=replanned,
                    improvement=retained["cost"] - replanned["cost"], exact=exact_remaining(pose),
                    forced_first=[dict(action=int(a), **exact_remaining(pose, first=int(a)))
                                  for a in np.flatnonzero(pose.mask())])
                expect_event("PRIORITY_UPDATED" if update["kind"] == "priority" else "COST_UPDATED",
                    i, sample, update=update, comparison=comparison, actual_action=new,
                    pose_cost_comparison=pose_comparison, counterfactual_action=old,
                    actual_features=state.observation().tolist(),
                    counterfactual_features=before.observation().tolist(), mask=state.mask().tolist())
    actual_reached = [
        (e["simulation_step"], e["target_id"], e["valid_intervals"])
        for e in events
        if e["type"] == "SHARED_SETTLED"
    ]
    if actual_reached != reached or summary["completed_order"] != completed:
        raise ValueError("Event/summary completions differ from actual shared intervals")
    updates = [e for e in events if e["type"] in ["PRIORITY_UPDATED", "COST_UPDATED"]]
    if len(updates) != (1 if event_expected else 0):
        raise ValueError("Missing/repeated scenario event")
    if event_expected:
        event = updates[0]
        expected = event_expected
        if (
            event["simulation_step"] != expected["step"]
            or event["update"] != expected["update"]
            or event["comparison"] != expected["comparison"]
            or event["actual_action"] != expected["new"]
            or event["counterfactual_action"] != expected["old"]
        ):
            raise ValueError("Actual post-completion event differs")
    wall = summary["wall_residence_s"]
    if type(wall) not in (float, int) or not math.isfinite(wall) or wall < 0:
        raise ValueError("Invalid measured wall clock")
    success = len(completed) == 10 and wall < 600 and len(step_rows) <= 12000
    if (summary["status"] == "SUCCEEDED") != success:
        raise ValueError("Summary success differs from all ten measured stops")
    if success:
        expect_event("MISSION_FINISHED", len(step_rows), previous,
                     termination_code=None, cause=None, capture_errors=[])
        if _objective_fields(events) != _objective_fields(expected_events):
            raise ValueError("Event stream differs from independent mission replay")
    proof = None
    if physical and success:
        from .payload_solver_trace import verify_native_sequence

        proof = verify_native_sequence(native_rows, **native)
    return dict(
        success=success,
        completed_count=len(completed),
        completed_order=completed,
        physical_acceptance=bool(physical and success and proof),
        native_proof=proof,
        termination_code=summary["termination_code"],
        measured_distance_m=total_distance,
        steps=len(step_rows),
        summary_sha256=sha256(root / "mission_summary.json"),
    )


def wilson(success, total):
    if total <= 0 or not 0 <= success <= total:
        raise ValueError("Valid fixed denominator required")
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def paired_statistics(results, cases):
    if (
        len(cases) != 127
        or len({c["case_id"] for c in cases}) != 127
        or Counter(c["suite"] for c in cases) != dict(basic=9, endpoints=18, heldout=100)
    ):
        raise ValueError("Exact original127-case denominator required")
    expected = {(m, c["case_id"]) for m in ["randomized", "fixed"] for c in cases}
    actual = [(r["method"], r["case_id"]) for r in results]
    if len(actual) != 254 or len(set(actual)) != 254 or set(actual) != expected:
        raise ValueError("All254 distinct paired trials required; failures remain in denominator")
    case_by_id = {c["case_id"]: c for c in cases}
    answer = {}
    for method in ["randomized", "fixed"]:
        selected = [r for r in results if r["method"] == method]
        groups = {}
        for suite, n in [("basic", 9), ("endpoints", 18), ("heldout", 100)]:
            rows = [r for r in selected if case_by_id[r["case_id"]]["suite"] == suite]
            k = sum(bool(r["success"] and r["physical_acceptance"]) for r in rows)
            groups[suite] = dict(successes=k, total=n, wilson_95=wilson(k, n))
        subgroups = {}
        for field in ["scenario", "side_m"]:
            for value in sorted({str(c[field]) for c in cases}):
                rows = [r for r in selected if str(case_by_id[r["case_id"]][field]) == value]
                k = sum(bool(r["success"] and r["physical_acceptance"]) for r in rows)
                subgroups[field + "=" + value] = dict(
                    successes=k, total=len(rows), wilson_95=wilson(k, len(rows))
                )
        for lo, hi in [
            (0.1, 0.16666666666666666),
            (0.16666666666666666, 0.23333333333333334),
            (0.23333333333333334, 0.30000000000000004),
        ]:
            rows = [r for r in selected if lo <= case_by_id[r["case_id"]]["mass_ratio"] < hi]
            k = sum(bool(r["success"] and r["physical_acceptance"]) for r in rows)
            subgroups[f"mass_ratio=[{lo},{hi})"] = dict(
                successes=k, total=len(rows), wilson_95=wilson(k, len(rows))
            )
        answer[method] = dict(
            suites=groups,
            subgroups=subgroups,
            failures=[r for r in selected if not (r["success"] and r["physical_acceptance"])],
        )
    g = answer["randomized"]["suites"]
    answer["randomized_acceptance"] = (
        g["basic"]["successes"] == 9 and g["endpoints"]["successes"] == 18 and g["heldout"]["successes"] >= 95
    )
    return answer
