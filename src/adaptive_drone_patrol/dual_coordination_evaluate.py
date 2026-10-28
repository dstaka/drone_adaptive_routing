"""Prospective validation capture with retained-input or actual native-replay evidence.

Only a verified learning capability admits worlds. Software tests of this module
are not physical evidence. Incomplete audit/replay receipts remain partial and
cannot enter checkpoint selection. Held-out cases are deliberately not opened.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

from .dual_coordination_campaign import (
    EXPERIMENT_ROLES,
    VerifiedLearningLock,
    _digest,
    _validate_terminal_checkpoint,
    _validation_cases,
)
from .dual_coordination_progress import atomic_json

METHODS = ("N42", "N43", "N44", "B-static", "B-dynamic")


def useful_transfer(state):
    """A previously assigned goal is reassigned after observed delay and served."""
    observed = [
        r["observed_tick"]
        for r in state.get("event_receipts", [])
        if r.get("event", {}).get("kind") == "availability-delay"
        and r.get("observed_tick") is not None
        and not r.get("no_op", False)
    ]
    if not observed:
        return False
    owners = [None] * len(state["goals"])
    for receipt in state.get("action_receipts", []):
        for i, owner in enumerate(receipt["owners"]):
            goal = state["goals"][i]
            if (
                owner is not None
                and owners[i] is not None
                and owner != owners[i]
                and receipt["tick"] >= min(observed)
                and goal["status"] == "DONE"
                and goal["owner"] == owner
                and goal["completed_tick"] >= receipt["tick"]
            ):
                return True
            if owner is not None:
                owners[i] = owner
    return False


def time_score(outcome):
    return float(outcome["team_time"]) if outcome["success"] else 600.0


def mission_outcome(
    case, state, *, decision_intervals, intervened_intervals, held_seconds, safety_violations, invalid_actions
):
    success = state["terminal_reason"] == "success"
    completed = sum(g["status"] == "DONE" for g in state["goals"])
    duplicate = max(0, sum(state["completed_counts"]) - completed)
    return dict(
        case_id=case["case_id"],
        scenario=case["scenario"],
        success=success,
        team_time=float(state["tick"] * 0.01) if success else None,
        safety_violations=int(safety_violations),
        invalid_actions=int(invalid_actions),
        duplicate_service=duplicate,
        decision_intervals=decision_intervals,
        intervened_intervals=intervened_intervals,
        supervisor_held_drone_seconds=float(held_seconds),
        useful_post_event_transfer=(case["scenario"] == "availability-delay" and useful_transfer(state)),
        failure_reason=None if success else state["terminal_reason"] or "incomplete",
    )


def build_audit_receipt(result, report, *, auditor_hash):
    coverage = dict(
        control_steps=report.get("control_steps", 0),
        rk_states=report.get("rk_states", 0),
        native_substep_intervals=report.get("native_poses", 0) - 1,
        swept_intervals=report.get("reconstructed_intervals", 0),
    )
    tick = result["terminal_tick"]
    expected = dict(
        control_steps=tick, rk_states=tick * 64, native_substep_intervals=tick * 16, swept_intervals=tick * 16
    )
    errors = list(report.get("errors", []))
    if coverage != expected or report.get("body_reconstructed_intervals") != tick * 16:
        errors.append("Incomplete internal-stage or reconstructed-interval coverage")
    gaps = list(report.get("proof_gaps", ["Missing proof-gap declaration"]))
    return dict(
        schema="dual-validation-audit-v1",
        identity=deepcopy(result["identity"]),
        result_sha256=_digest(result),
        auditor_sha256=auditor_hash,
        outcome=deepcopy(result["outcome"]),
        complete=not errors and not gaps,
        errors=errors,
        proof_gaps=gaps,
        coverage=coverage,
        report=deepcopy(report),
    )


def compare_replays(first, second):
    decisions = first["actions"] == second["actions"]
    terminal = (
        first["terminal_tick"] == second["terminal_tick"]
        and first["mission"]["terminal_reason"] == second["mission"]["terminal_reason"]
    )
    metrics = first["outcome"] == second["outcome"] and first["mission"] == second["mission"]
    for field in ("qpos", "qvel"):
        a = np.asarray(first["final_backend"].get(field, []), dtype=float)
        b = np.asarray(second["final_backend"].get(field, []), dtype=float)
        metrics = bool(
            metrics
            and a.size > 0
            and a.shape == b.shape
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.allclose(a, b, rtol=0, atol=1e-8)
        )
    return dict(
        passed=bool(decisions and terminal and metrics),
        decisions_match=decisions,
        terminal_match=terminal,
        metrics_within_tolerance=metrics,
    )


def propose_and_step(env, selector, actions_path):
    """Flush the actual proposed action before even a failed native step."""
    action = int(selector(env))
    mask = np.asarray(env.legal_mask(), dtype=bool)
    record = dict(tick=int(env.mission.tick), action=action, mask=mask.tolist())
    with Path(actions_path).open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return env.step(action)


class ActionReplay:
    """Replay retained proposals; reject missing, extra or differently masked decisions."""

    def __init__(self, actions):
        self.actions = deepcopy(actions)
        self.index = 0

    @property
    def exhausted(self):
        return self.index == len(self.actions)

    def __call__(self, env):
        if self.exhausted:
            raise ValueError("Replay requested an extra decision")
        row = self.actions[self.index]
        if row["tick"] != env.mission.tick:
            raise ValueError("Replay decision tick differs")
        if row["mask"] != np.asarray(env.legal_mask(), dtype=bool).tolist():
            raise ValueError("Replay structural mask differs")
        self.index += 1
        return row["action"]


def causal_conflict_predictor(env):
    """Bound candidate paths using current references and measured tracking error."""
    from .dual_coordination_baselines import ConflictPrediction
    from .dual_coordination_reference import KinematicState, ReferenceInfeasible
    from .dual_coordination_schema import WAIT

    rows = env.fixture.sample()["drones"]

    def predict(request):
        try:
            references = []
            radii = []
            for d, (reference, row, target) in enumerate(zip(request.references, rows, request.targets)):
                actual = KinematicState(row["drone_xyz"], row["drone_velocity"])
                radii.append(
                    float(
                        env.supervisor.assembly_radius
                        + env.supervisor.tracking_inflation
                        + np.linalg.norm(reference.sample(request.time)[0] - np.asarray(row["drone_xyz"]))
                    )
                )
                if target == env.mission.targets[d] and (
                    target == WAIT or reference.target == request.goals[target]
                ):
                    candidate = reference
                elif target == WAIT:
                    candidate = reference.brake(request.time, actual_state=actual)
                else:
                    candidate = reference.retarget(request.goals[target], request.time, actual_state=actual)
                references.append(candidate)
            horizon = max(2.0, *(r.end_time - request.time + 0.2 for r in references))
            conflict, _ = env.supervisor.predict(references, request.time, horizon, radii)
        except ReferenceInfeasible as error:
            # The accepted formula keeps all structural candidates. An
            # uncertified candidate is conservatively conflicting, never clear.
            conflict = True
            if not hasattr(env, "_baseline_prediction_diagnostics"):
                env._baseline_prediction_diagnostics = dict(uncertified_candidates=0, samples=[])
            diagnostics = env._baseline_prediction_diagnostics
            diagnostics["uncertified_candidates"] += 1
            if len(diagnostics["samples"]) < 16:
                diagnostics["samples"].append(
                    dict(time=float(request.time), targets=list(request.targets), reason=str(error)[:512])
                )
        return ConflictPrediction(
            bool(conflict),
            tuple(
                bool(conflict and target != WAIT and request.unavailable_remaining[1 - d] > 0)
                for d, target in enumerate(request.targets)
            ),
        )

    return predict


def capture_mission(env, selector, case, directory, *, heartbeat=None):
    """Capture one already-admitted world; close failures are retained as failures."""
    directory = Path(directory)
    actions_path = directory / "proposed-actions.jsonl"
    if actions_path.exists():
        raise ValueError("Refuse to overwrite an existing proposal stream")
    started = time.monotonic()
    errors = []
    held_seconds = 0.0
    intervened = invalid = 0
    try:
        atomic_json(directory / "reset-inputs.json", dict(case=case, initial_native=env.fixture.sample()))
        last_heartbeat = started
        while not env.terminated:
            _, _, _, info = propose_and_step(env, selector, actions_path)
            held = float(info["supervisor_held_drone_seconds"])
            held_seconds += held
            intervened += int(held > 0 or str(info.get("terminal_reason", "")).startswith("supervisor:"))
            if heartbeat and time.monotonic() - last_heartbeat >= 30:
                heartbeat(env.mission.tick)
                last_heartbeat = time.monotonic()
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        try:
            env.close()
        except Exception as error:
            errors.append(f"close: {type(error).__name__}: {error}")
    actions = (
        [json.loads(line) for line in actions_path.read_text().splitlines()] if actions_path.exists() else []
    )
    invalid = sum(not 0 <= r["action"] < len(r["mask"]) or not r["mask"][r["action"]] for r in actions)
    state = deepcopy(env.mission.to_dict()["state"])
    native_terminal_reason = state["terminal_reason"]
    if errors:
        state["terminal_reason"] = "infrastructure:" + errors[0]
        state["team_completion_tick"] = None
    report = deepcopy(env.online_auditor.report())
    if errors:
        report.setdefault("errors", []).extend(errors)
    final = env.fixture.sample()
    final_backend = {name: final.get(name) for name in ("qpos", "qvel")}
    for name, values in final_backend.items():
        if values is None or not np.isfinite(np.asarray(values, dtype=float)).all():
            final_backend[name] = None
            report.setdefault("errors", []).append(f"Nonfinite or missing final {name}")
    unsafe = str(native_terminal_reason).startswith(("safety:", "supervisor:")) or any(
        report.get(key) is False for key in ("safe", "retained_points_safe", "reconstructed_path_safe")
    )
    safety = max(
        int(report.get("violation_count", 0)),
        len(env.ledger.failures),
        len(report.get("reconstruction_failures", [])),
        int(unsafe),
    )
    if safety and state["terminal_reason"] == "success":
        state["terminal_reason"] = "safety:inconsistent_success_with_failed_audit"
        state["team_completion_tick"] = None
    outcome = mission_outcome(
        case,
        state,
        decision_intervals=len(actions),
        intervened_intervals=intervened,
        held_seconds=held_seconds,
        safety_violations=safety,
        invalid_actions=invalid,
    )
    capture = dict(
        schema="dual-validation-capture-v1",
        actions=actions,
        mission=state,
        terminal_tick=state["tick"],
        outcome=outcome,
        final_backend=final_backend,
        audit_report=report,
        infrastructure_errors=errors,
        replay_segments=deepcopy(env.replay_segments),
        wall_seconds=time.monotonic() - started,
        baseline_prediction_diagnostics=deepcopy(
            getattr(env, "_baseline_prediction_diagnostics", dict(uncertified_candidates=0, samples=[]))
        ),
    )
    atomic_json(directory / "capture.json", capture)
    return capture


def compare_compact_replay(first, second):
    """Stream both retained trajectories, including intermediate states/controls."""
    import math
    from itertools import zip_longest
    from .dual_coordination_raw import read_raw

    def records(capture):
        for segment in capture["replay_segments"]:
            yield from read_raw(segment["path"])

    def equal(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(a, float):
            return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-8
        return a == b

    fields = (
        "kind",
        "tick",
        "actual",
        "controls",
        "action",
        "action_tick",
        "action_receipt",
        "structural_mask",
        "mission",
        "reference",
        "requested_references",
        "executed_references",
        "supervisor",
        "event",
        "reward",
    )
    sentinel = object()
    count = 0
    for left, right in zip_longest(records(first), records(second), fillvalue=sentinel):
        count += 1
        if left is sentinel or right is sentinel:
            return False
        if not equal({k: left[k] for k in fields if k in left}, {k: right[k] for k in fields if k in right}):
            return False
    return count > 0


def file_record(path, root):
    from .util import sha256

    path = Path(path).resolve()
    return dict(
        path=str(path.relative_to(Path(root).resolve())), bytes=path.stat().st_size, sha256=sha256(path)
    )


def bundle_captures(path, captures, *, root):
    """Validate then retain both independent executions, without deleting sources."""
    from .dual_coordination_raw import read_raw
    from .util import sha256

    files, resets = [], []
    for i, (directory, capture) in enumerate(captures):
        directory = Path(directory).resolve()
        count = 0
        for segment in capture["replay_segments"]:
            source = Path(segment["path"]).resolve()
            if (
                not source.is_relative_to(directory)
                or source.is_symlink()
                or source.stat().st_size != segment["bytes"]
                or sha256(source) != segment["sha256"]
            ):
                raise ValueError("Compact replay checksum or path mismatch")
            for record in read_raw(source):
                count += record.get("kind") == "RESET"
            files.append((source, f"{i}/{source.name}"))
        if count != 1:
            raise ValueError("Exactly one initial RESET required for each uninterrupted mission")
        resets.append(count)
        for name in ("reset-inputs.json", "proposed-actions.jsonl", "capture.json"):
            source = directory / name
            if not source.is_file() or source.is_symlink():
                raise ValueError("Missing reset/actions/result capture")
            files.append((source, f"{i}/{name}"))
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
        for source, name in files:
            archive.write(source, name)
    return dict(record=file_record(path, root), reset_records=resets)


def retire_slot_captures(slot, record, *, root, token, capture_names=("capture", "replay")):
    """Prune only this run's new slot after every archived member is verified."""
    import shutil
    from .util import sha256

    slot = Path(slot)
    marker = slot / "ownership.json"
    expected = dict(schema="dual-evaluator-owned-slot-v1", token=token)
    if slot.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != expected:
        raise ValueError("Missing or mismatched evaluator slot ownership")
    archive_path = Path(root) / record["path"]
    if file_record(archive_path, root) != record:
        raise ValueError("Bundle checksum changed before duplicate cleanup")
    if (
        not capture_names
        or len(set(capture_names)) != len(capture_names)
        or any(name not in ("capture", "replay") for name in capture_names)
    ):
        raise ValueError("Only declared capture/replay directories may be retired")
    directories = [slot / name for name in capture_names]
    if any(p.is_symlink() or not p.is_dir() for p in directories):
        raise ValueError("Owned capture directories must be real directories")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Bundle member CRC mismatch")
        for member in archive.infolist():
            components = Path(member.filename).parts
            if (
                len(components) != 2
                or components[0] not in tuple(str(i) for i in range(len(directories)))
                or components[1] in (".", "..")
            ):
                raise ValueError("Unsafe bundle member path")
            source = directories[int(components[0])] / components[1]
            if source.is_symlink() or not source.is_file():
                raise ValueError("Bundle member source changed")
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != sha256(source):
                raise ValueError("Bundle member checksum differs from retained source")
    # Sources are derived fixture copies plus the exact archived capture files.
    # Immutable input assets live outside these newly created owned directories.
    for directory in directories:
        shutil.rmtree(directory)
    atomic_json(
        slot / "storage.json",
        dict(
            schema="dual-validation-storage-v1",
            bundle=record,
            duplicate_capture_directories_retired=True,
            immutable_inputs_retained=True,
        ),
    )


def _bound_file(lock, role, path):
    from .util import sha256

    path = Path(path).resolve()
    if not path.is_file() or not any(
        (path.stat().st_size, sha256(path)) == (r["bytes"], r["sha256"]) for r in lock.binding.files[role]
    ):
        raise ValueError(f"Actual {role} input is not bound to the learning lock")
    return path


def _selector(method, case, terminal, device):
    from .dual_coordination_baselines import StaticBaseline, DynamicBaseline

    if method == "B-static":
        baseline = StaticBaseline(case["starts"], case["goals"])
        return lambda env: baseline.select(
            env.mission, [d["drone_xyz"] for d in env.fixture.sample()["drones"]]
        ).action
    if method == "B-dynamic":
        baseline = DynamicBaseline(case["goals"])
        return lambda env: baseline.select(
            env.mission,
            [d["drone_xyz"] for d in env.fixture.sample()["drones"]],
            references=env.executed_references,
            conflict_predictor=causal_conflict_predictor(env),
        ).action
    import torch
    from .dual_coordination_policy import JointActorCritic

    with torch.random.fork_rng(devices=[]):
        model = JointActorCritic()
    state = torch.load(terminal, map_location="cpu", weights_only=False)
    model.load_state_dict(state["policy"], strict=True)
    model.to(device).eval()

    def choose(env):
        obs = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
            for k, v in env.observation().items()
        }
        mask = torch.as_tensor(env.legal_mask(), dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = model.act(obs, mask, deterministic=True)
        return int(action.item())

    return choose


def _mission_receipts(*, lock, identity, first, second, bundle):
    result = dict(
        schema="dual-validation-result-v1",
        identity=identity,
        outcome=first["outcome"],
        terminal_tick=first["terminal_tick"],
        time_score=time_score(first["outcome"]),
        mission=first["mission"],
        final_backend=first["final_backend"],
        wall_seconds=first["wall_seconds"],
        baseline_prediction_diagnostics=first.get("baseline_prediction_diagnostics", {}),
    )
    auditor_hash = _digest(lock.binding.files["audit"])
    audit = build_audit_receipt(result, first["audit_report"], auditor_hash=auditor_hash)
    replay_audit = build_audit_receipt(
        dict(result, terminal_tick=second["terminal_tick"], outcome=second["outcome"]),
        second["audit_report"],
        auditor_hash=auditor_hash,
    )
    comparison = compare_replays(first, second)
    compact_match = compare_compact_replay(first, second)
    comparison["metrics_within_tolerance"] = bool(comparison["metrics_within_tolerance"] and compact_match)
    comparison["passed"] = bool(comparison["passed"] and compact_match)
    inventory = dict.fromkeys(("reset", "actions", "compact_replay"), bundle["record"])
    replay = dict(
        schema="dual-validation-replay-v1",
        identity=identity,
        result_sha256=_digest(result),
        audit_sha256=_digest(audit),
        inventory=inventory,
        inventory_sha256=_digest(inventory),
        configuration_sha256=_digest(lock.binding.files["configuration"]),
        **comparison,
        control_steps=second["terminal_tick"],
        reset_records=bundle["reset_records"][0],
        outcome=first["outcome"],
        audit=replay_audit,
    )
    replay["passed"] = bool(
        replay["passed"]
        and replay_audit["complete"]
        and not first["infrastructure_errors"]
        and not second["infrastructure_errors"]
    )
    return dict(result=result, audit=audit, replay=replay)


def validate_retained_inputs(capture):
    """Validate one retained stream; this is not an independent native replay."""
    from .dual_coordination_raw import read_raw
    from .util import sha256

    errors, resets, count = [], 0, 0
    case_hash = None
    try:
        segments = capture["replay_segments"]
        if len(segments) != 1:
            raise ValueError("One uninterrupted compact segment required")
        segment = segments[0]
        path = Path(segment["path"])
        if path.stat().st_size != segment["bytes"] or sha256(path) != segment["sha256"]:
            raise ValueError("Retained compact checksum mismatch")
        directory = path.parent
        reset = json.loads((directory / "reset-inputs.json").read_text())
        case_hash = _digest(reset["case"])
        proposals = [
            json.loads(line) for line in (directory / "proposed-actions.jsonl").read_text().splitlines()
        ]
        if proposals != capture["actions"] or not proposals:
            raise ValueError("Retained proposal stream differs from capture")
        terminal = capture["terminal_tick"]
        if type(terminal) is not int or not 0 <= terminal <= 30000:
            raise ValueError("Invalid terminal tick")
        expected_ticks = {0}
        by_tick = {}
        for index, row in enumerate(proposals):
            action, tick, mask = row["action"], row["tick"], row["mask"]
            if (
                type(tick) is not int
                or tick != index * 20
                or tick > terminal
                or type(action) is not int
                or not 0 <= action < 242
                or len(mask) != 242
                or any(type(v) is not bool for v in mask)
                or not mask[action]
            ):
                raise ValueError("Invalid or missing canonical proposal/mask/decision boundary")
            by_tick[tick] = row
            if tick < terminal:
                expected_ticks.update((tick + 1, min(tick + 20, terminal)))
        observed_ticks = set()
        last = None
        for row in read_raw(path):
            count += 1
            tick = row["tick"]
            observed_ticks.add(tick)
            if row.get("case_id") != capture["outcome"]["case_id"] or row.get("replay_stride_ticks") != 20:
                raise ValueError("Compact case/profile identity mismatch")
            if not np.isclose(row["actual"]["world_time"], tick * 0.01, rtol=0, atol=1e-8):
                raise ValueError("Retained native time differs from committed control tick")
            if row["kind"] == "RESET":
                resets += 1
                if tick != 0:
                    raise ValueError("Initial reset must be at tick zero")
                for key in ("qpos", "qvel"):
                    if row["actual"][key] != reset["initial_native"][key]:
                        raise ValueError("Retained reset differs from initial native state")
            else:
                proposal = by_tick.get(row["action_tick"])
                if (
                    proposal is None
                    or row["action"] != proposal["action"]
                    or row["structural_mask"] != proposal["mask"]
                    or not proposal["tick"] < tick <= proposal["tick"] + 20
                ):
                    raise ValueError("Retained control differs from proposed action/mask/boundary")
            last = row
        if resets != 1 or observed_ticks != expected_ticks or last["tick"] != terminal:
            raise ValueError("Missing decision first/last/terminal state or reset")
        for key in ("qpos", "qvel"):
            actual = np.asarray(last["actual"][key], dtype=float)
            final = np.asarray(capture["final_backend"][key], dtype=float)
            if (
                actual.shape != final.shape
                or not np.isfinite(final).all()
                or not np.allclose(actual, final, rtol=0, atol=1e-8)
            ):
                raise ValueError("Final native state differs from compact terminal boundary")
        if capture["infrastructure_errors"]:
            raise ValueError("Capture infrastructure errors prevent retained-input verification")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    return dict(
        retained_inputs_verified=not errors,
        compact_boundary_consistency=not errors,
        errors=errors,
        reset_records=resets,
        action_records=len(capture.get("actions", [])),
        retained_state_records=count,
        control_steps=capture["terminal_tick"],
        reset_case_sha256=case_hash,
    )


def build_single_capture_receipts(cap, identity, capture, bundle, *, development_replay_reference):
    """Retained-input evidence referencing a separately qualified development replay."""
    result = dict(
        schema="dual-validation-result-v1",
        identity=identity,
        outcome=capture["outcome"],
        terminal_tick=capture["terminal_tick"],
        time_score=time_score(capture["outcome"]),
        mission=capture["mission"],
        final_backend=capture["final_backend"],
        wall_seconds=capture["wall_seconds"],
        baseline_prediction_diagnostics=capture.get("baseline_prediction_diagnostics", {}),
    )
    audit = build_audit_receipt(
        result, capture["audit_report"], auditor_hash=_digest(cap.binding.files["audit"])
    )
    verification = validate_retained_inputs(capture)
    if verification["reset_case_sha256"] != identity.get("case_sha256"):
        verification["errors"].append("Retained reset case differs from bound case identity")
        verification["retained_inputs_verified"] = verification["compact_boundary_consistency"] = False
    inventory = dict.fromkeys(("reset", "actions", "compact_replay"), bundle["record"])
    replay = dict(
        schema="dual-validation-replay-v2",
        identity=identity,
        result_sha256=_digest(result),
        audit_sha256=_digest(audit),
        inventory=inventory,
        inventory_sha256=_digest(inventory),
        configuration_sha256=_digest(cap.binding.files["configuration"]),
        execution_mode="retained-input-stream-validation",
        native_reexecution_performed=False,
        development_replay_reference=deepcopy(development_replay_reference),
        **verification,
        outcome=capture["outcome"],
    )
    return dict(result=result, audit=audit, replay=replay)


def run_validation(
    *,
    learning_lock,
    terminals,
    original,
    design_path,
    output,
    journal_root=None,
    case_id=None,
    method=None,
    native_reexecute=False,
):
    """Run requested prospective slots, never select from a partial campaign.

    All three terminal artifacts and the complete common validation manifest are
    checked before creating output. Existing slots are not rerun: retain failed
    attempts, inspect their diagnostics, and use a separately declared run.
    """
    from .dual_coordination_env import DualCoordinationEnv
    from .dual_coordination_progress import RunJournal
    from .dual_coordination_selection import validate_terminal_checkpoints
    from .util import sha256
    import torch

    lock = VerifiedLearningLock.load(learning_lock)
    lock.require_current()
    if not isinstance(terminals, dict) or set(terminals) != {42, 43, 44}:
        raise ValueError("Exactly all terminal seeds 42, 43, 44 are required")
    if lock.selected_backend not in ("native-cpu", "native-cuda"):
        raise ValueError("Qualified native backend is required; Warp adapter unavailable")
    device = "cuda" if lock.selected_backend == "native-cuda" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Qualified CUDA inference device is unavailable")
    manifests = {seed: _validate_terminal_checkpoint(Path(path), seed) for seed, path in terminals.items()}
    validate_terminal_checkpoints(manifests)
    original = _bound_file(lock, "assets", original)
    design_path = _bound_file(lock, "controller", design_path)
    for name in (
        "dual_coordination_evaluate.py",
        "dual_coordination_policy.py",
        "dual_coordination_train.py",
        "dual_coordination_baselines.py",
        "dual_coordination_selection.py",
        "dual_coordination_campaign.py",
    ):
        _bound_file(lock, "source", Path(__file__).with_name(name))
    for seed, path in terminals.items():
        verify_terminal_recovery(path, lock, seed=seed, original=original, design_path=design_path)
    development_reference = None
    if type(native_reexecute) is not bool:
        raise ValueError("Explicit boolean native reexecution mode required")
    if not native_reexecute:
        from .dual_coordination_campaign import verified_development_replay_reference

        development_reference = verified_development_replay_reference(
            lock.binding, lock.receipt["qualification"]
        )
    design = json.loads(design_path.read_text())
    cases = _validation_cases(lock.binding)
    for case in cases.values():
        lock.verify_case(case, "validation")
    if case_id is not None and case_id not in cases:
        raise ValueError("Requested case is not in declared validation300")
    if method is not None and method not in METHODS:
        raise ValueError("Unknown prospective validation method")
    output = Path(output).resolve()
    if not output.is_relative_to(lock.binding.root):
        raise ValueError("Validation output must remain inside the campaign binding root")
    declaration = dict(
        schema="dual-validation-declaration-v1",
        learning_lock_sha256=lock.digest,
        checkpoint_sha256={str(s): manifests[s]["sha256"] for s in (42, 43, 44)},
        case_sha256={k: _digest(c) for k, c in cases.items()},
        methods=list(METHODS),
        expected_missions=1500,
        backend=lock.selected_backend,
        replay_stride_ticks=20,
        independent_native_replay=native_reexecute,
        development_replay_reference=development_reference,
        baseline_sha256=_digest(lock.binding.files["baseline"]),
        experiment_inputs_sha256=_digest({r: lock.binding.files[r] for r in EXPERIMENT_ROLES}),
        auditor_sha256=_digest(lock.binding.files["audit"]),
        configuration_sha256=_digest(lock.binding.files["configuration"]),
    )
    if output.exists():
        if (
            not (output / "declaration.json").is_file()
            or json.loads((output / "declaration.json").read_text()) != declaration
        ):
            raise ValueError("Existing validation output has a different prospective declaration")
    else:
        output.mkdir(parents=True)
        atomic_json(output / "declaration.json", declaration)
    journal = RunJournal(journal_root or output)
    attempt = journal.start(
        "native-validation",
        "Verified prospective validation with declared evidence mode",
        category="validation",
    )
    try:
        selected_cases = [cases[case_id]] if case_id is not None else list(cases.values())
        selected_methods = [method] if method is not None else list(METHODS)
        total = len(selected_cases) * len(selected_methods)
        done = 0
        for current_method in selected_methods:
            for case in selected_cases:
                directory = (
                    output
                    / "missions"
                    / current_method
                    / hashlib.sha256(case["case_id"].encode()).hexdigest()[:20]
                )
                if directory.exists():
                    raise FileExistsError(
                        f"Preserved existing mission; automatic rerun forbidden: {directory}"
                    )
                import uuid

                token = uuid.uuid4().hex
                directory.mkdir(parents=True)
                atomic_json(
                    directory / "ownership.json", dict(schema="dual-evaluator-owned-slot-v1", token=token)
                )
                lock.require_current()
                for seed, path in terminals.items():
                    if sha256(path) != manifests[seed]["sha256"]:
                        raise ValueError("Terminal checkpoint changed during validation")
                identity = dict(
                    method=current_method,
                    case_id=case["case_id"],
                    case_sha256=_digest(case),
                    model_sha256=manifests[int(current_method[1:])]["sha256"]
                    if current_method.startswith("N")
                    else _digest(lock.binding.files["baseline"]),
                    experiment_inputs_sha256=_digest({r: lock.binding.files[r] for r in EXPERIMENT_ROLES}),
                )
                terminal = terminals[int(current_method[1:])] if current_method.startswith("N") else None
                choose = _selector(current_method, case, terminal, device)

                def beat(tick):
                    journal.heartbeat(
                        attempt,
                        completed=done,
                        total=total,
                        reason=f"{current_method} {case['case_id']}: native tick {tick}",
                        next_action="Complete capture and declared retained-input or native-replay verification",
                    )

                env = DualCoordinationEnv(
                    original,
                    directory / "capture",
                    case,
                    design,
                    mode="validation",
                    admission=lock,
                    retain_replay=True,
                    replay_stride_ticks=20,
                )
                first = capture_mission(env, choose, case, directory / "capture", heartbeat=beat)
                del env
                captures = [(directory / "capture", first)]
                if native_reexecute:
                    replay_selector = ActionReplay(first["actions"])
                    env = DualCoordinationEnv(
                        original,
                        directory / "replay",
                        case,
                        design,
                        mode="validation",
                        admission=lock,
                        retain_replay=True,
                        replay_stride_ticks=20,
                    )
                    second = capture_mission(env, replay_selector, case, directory / "replay", heartbeat=beat)
                    del env
                    if not replay_selector.exhausted:
                        second["infrastructure_errors"].append(
                            "Replay ended before all retained proposals were consumed"
                        )
                        atomic_json(directory / "replay" / "capture.json", second)
                    captures.append((directory / "replay", second))
                evidence_dir = output / "evidence"
                evidence_dir.mkdir(exist_ok=True)
                name = current_method + "-" + directory.name
                try:
                    bundle = bundle_captures(
                        evidence_dir / (name + ".zip"),
                        captures,
                        root=lock.binding.root,
                    )
                    if native_reexecute:
                        receipts = _mission_receipts(
                            lock=lock, identity=identity, first=first, second=second, bundle=bundle
                        )
                    else:
                        receipts = build_single_capture_receipts(
                            lock, identity, first, bundle, development_replay_reference=development_reference
                        )
                    atomic_json(
                        evidence_dir / (name + ".json"),
                        dict(schema="dual-validation-evidence-v1", missions=[receipts]),
                    )
                    retire_slot_captures(
                        directory,
                        bundle["record"],
                        root=lock.binding.root,
                        token=token,
                        capture_names=("capture", "replay") if native_reexecute else ("capture",),
                    )
                except (OSError, ValueError) as error:
                    atomic_json(
                        directory / "partial.json",
                        dict(identity=identity, reason=str(error), outcome=first["outcome"]),
                    )
                done += 1
                journal.heartbeat(
                    attempt,
                    completed=done,
                    total=total,
                    reason="Mission capture/replay retained",
                    next_action="Next declared mission or aggregate",
                )
        summary = aggregate_validation(output, declaration, root=lock.binding.root)
        journal.finish(
            attempt, status="done", reason=f"Validation slots finished; status={summary['status']}"
        )
        return summary
    except BaseException as error:
        journal.finish(attempt, status="failed", reason=f"{type(error).__name__}: {error}")
        raise


def aggregate_validation(output, declaration, *, root):
    """Check all1500 retained receipt chains before emitting bound selection."""
    from .dual_coordination_selection import select_candidate

    output, root = Path(output), Path(root).resolve()
    expected = {(method, case_id) for method in METHODS for case_id in declaration["case_sha256"]}
    seen, documents, inventory_paths, errors = {}, [], set(), []
    errors.extend(
        f"Preserved partial mission: {p.relative_to(output)}"
        for p in (output / "missions").glob("*/*/partial.json")
    )
    verified_files = {}
    for path in sorted((output / "evidence").glob("*.json")):
        try:
            document = json.loads(path.read_text())
            if document["schema"] != "dual-validation-evidence-v1" or not document["missions"]:
                raise ValueError("Invalid evidence document")
            documents.append(str(path.resolve().relative_to(root)))
            for receipts in document["missions"]:
                result, audit, replay = (receipts[k] for k in ("result", "audit", "replay"))
                identity = result["identity"]
                key = identity["method"], identity["case_id"]
                if key not in expected or key in seen:
                    raise ValueError("Duplicate or undeclared mission")
                seen[key] = result["outcome"]
                method, case_id = key
                model_hash = (
                    declaration["checkpoint_sha256"][method[1:]]
                    if method.startswith("N")
                    else declaration["baseline_sha256"]
                )
                required = dict(
                    method=method,
                    case_id=case_id,
                    case_sha256=declaration["case_sha256"][case_id],
                    model_sha256=model_hash,
                    experiment_inputs_sha256=declaration["experiment_inputs_sha256"],
                )
                if any(r["identity"] != required for r in (result, audit, replay)):
                    raise ValueError("Receipt identity differs from prospective declaration")
                tick = result["terminal_tick"]
                if type(tick) is not int or not 0 <= tick <= 30000:
                    raise ValueError("Invalid terminal control tick")
                if (
                    result["schema"] != "dual-validation-result-v1"
                    or result["outcome"]["case_id"] != case_id
                    or (
                        result["outcome"]["success"]
                        and not np.isclose(result["outcome"]["team_time"], tick * 0.01, rtol=0, atol=1e-9)
                    )
                ):
                    raise ValueError("Terminal outcome differs from retained identity/time")
                expected_coverage = dict(
                    control_steps=tick,
                    rk_states=tick * 64,
                    native_substep_intervals=tick * 16,
                    swept_intervals=tick * 16,
                )
                if (
                    audit["schema"] != "dual-validation-audit-v1"
                    or audit["result_sha256"] != _digest(result)
                    or audit["auditor_sha256"] != declaration["auditor_sha256"]
                    or audit["complete"] is not True
                    or audit["errors"] != []
                    or audit["proof_gaps"] != []
                    or audit["coverage"] != expected_coverage
                    or any(type(v) is not int for v in audit["coverage"].values())
                    or audit["outcome"] != result["outcome"]
                ):
                    raise ValueError("Incomplete or mismatched online audit")
                if (
                    replay["schema"] not in ("dual-validation-replay-v1", "dual-validation-replay-v2")
                    or replay["result_sha256"] != _digest(result)
                    or replay["audit_sha256"] != _digest(audit)
                    or replay["configuration_sha256"] != declaration["configuration_sha256"]
                    or replay["inventory_sha256"] != _digest(replay["inventory"])
                    or replay["outcome"] != result["outcome"]
                    or type(replay["control_steps"]) is not int
                    or replay["control_steps"] != tick
                    or type(replay["reset_records"]) is not int
                    or replay["reset_records"] != 1
                ):
                    raise ValueError("Incomplete or mismatched independent replay")
                if replay["schema"] == "dual-validation-replay-v1":
                    if any(
                        replay[k] is not True
                        for k in ("passed", "decisions_match", "terminal_match", "metrics_within_tolerance")
                    ):
                        raise ValueError("Actual independent native replay did not match")
                elif (
                    replay.get("execution_mode") != "retained-input-stream-validation"
                    or replay.get("native_reexecution_performed") is not False
                    or replay.get("development_replay_reference")
                    != declaration.get("development_replay_reference")
                    or replay.get("development_replay_reference") is None
                    or replay.get("retained_inputs_verified") is not True
                    or replay.get("compact_boundary_consistency") is not True
                    or replay.get("errors") != []
                    or type(replay.get("action_records")) is not int
                    or replay["action_records"] != result["outcome"]["decision_intervals"]
                    or type(replay.get("retained_state_records")) is not int
                    or replay["retained_state_records"] < 1
                    or replay.get("reset_case_sha256") != identity["case_sha256"]
                    or any(
                        k in replay
                        for k in ("passed", "decisions_match", "terminal_match", "metrics_within_tolerance")
                    )
                ):
                    raise ValueError("Retained-input evidence or development replay reference incomplete")
                if set(replay["inventory"]) != {"reset", "actions", "compact_replay"}:
                    raise ValueError("Missing replay inventory role")
                for record in replay["inventory"].values():
                    source = root / record["path"]
                    if source.is_symlink() or not source.resolve().is_relative_to(root):
                        raise ValueError("Replay inventory escapes campaign root")
                    if record["path"] not in verified_files:
                        verified_files[record["path"]] = file_record(source, root)
                    if record != verified_files[record["path"]]:
                        raise ValueError("Replay inventory checksum changed")
                    inventory_paths.add(record["path"])
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    complete = len(expected) == 1500 and set(seen) == expected and not errors
    selection_path = output / "selection.json"
    selection = None
    if complete:
        try:
            candidates = {
                seed: [seen[f"N{seed}", cid] for cid in declaration["case_sha256"]] for seed in (42, 43, 44)
            }
            static = [seen["B-static", cid] for cid in declaration["case_sha256"]]
            dynamic = [seen["B-dynamic", cid] for cid in declaration["case_sha256"]]
            numeric = select_candidate(candidates, static, dynamic)
            selection = dict(
                schema="dual-selection-bound-v1",
                selected_seed=numeric["selected"],
                candidates={str(s): rows for s, rows in candidates.items()},
                static=static,
                dynamic=dynamic,
                checkpoint_sha256=declaration["checkpoint_sha256"],
                validation_evidence=documents,
                numeric_report=numeric,
            )
            atomic_json(selection_path, selection)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"Numeric selection: {error}")
            complete = False
    if not complete and selection_path.exists():
        # Preserve previous receipt as invalidated evidence; never leave it selectable.
        selection_path.rename(output / f"invalidated-selection-{time.time_ns()}.json")
    summary = dict(
        schema="dual-validation-summary-v1",
        status="complete" if complete else "partial",
        expected_missions=1500,
        retained_missions=len(seen),
        missing_missions=len(expected - set(seen)),
        errors=errors,
        selected_seed=selection["selected_seed"] if complete else None,
        evidence_files=sorted(set(documents) | inventory_paths),
        selection=str(selection_path.relative_to(root)) if complete else None,
    )
    atomic_json(output / "summary.json", summary)
    return summary


def verify_terminal_recovery(path, lock, *, seed, original, design_path):
    """Validate complete trainer recovery binding while preserving caller RNGs."""
    import random
    import torch
    from .dual_coordination_campaign import expected_terminal_binding
    from .dual_coordination_train import load_recovery
    from .util import sha256

    binding = expected_terminal_binding(lock.binding, lock.digest, lock.receipt["benchmark"], seed)
    if binding["original_sha256"] != sha256(original) or binding["controller_sha256"] != sha256(design_path):
        raise ValueError("Actual evaluation assets/controller differ from terminal recovery binding")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            trainer, recovery = load_recovery(path, binding=binding, device="cpu")
            if not trainer.budget.selectable or recovery["rollout"]["rows"]:
                raise ValueError("Only complete terminal recovery is evaluable")
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-lock", required=True)
    for seed in (42, 43, 44):
        parser.add_argument(f"--terminal{seed}", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal-root")
    parser.add_argument("--case-id")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument(
        "--native-reexecute",
        action="store_true",
        help="Actually run a second native flight and emit v1 replay evidence",
    )
    args = parser.parse_args(argv)
    result = run_validation(
        learning_lock=args.learning_lock,
        terminals={s: getattr(args, f"terminal{s}") for s in (42, 43, 44)},
        original=args.original,
        design_path=args.design,
        output=args.output,
        journal_root=args.journal_root,
        case_id=args.case_id,
        method=args.method,
        native_reexecute=args.native_reexecute,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
