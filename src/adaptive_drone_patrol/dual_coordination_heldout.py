"""Finite post-freeze held-out evaluation; never generate cases or select a seed.

Production entry points require VerifiedHeldout. Every slot binds its exact case,
selected terminal or baseline, executable source, joint freeze and cohort file.
Started slots are immutable attempts, including failures; resume executes only
never-started slots. Missing/unverified outcomes remain in every denominator.
Final crossing/video evidence is a later phase and remains an explicit unmet gate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
import zipfile

import numpy as np

from .dual_coordination_campaign import EXPERIMENT_ROLES, VerifiedHeldout, _digest, _path
from .dual_coordination_evaluate import (
    _bound_file,
    _selector,
    build_single_capture_receipts,
    bundle_captures,
    capture_mission,
    file_record,
    retire_slot_captures,
)
from .dual_coordination_progress import RunJournal, atomic_json
from .dual_coordination_schema import WAIT, decode_action, encode_action
from .dual_coordination_selection import physical_report, routing_report
from .util import sha256

ROUTING_METHODS = ("B-static", "B-dynamic", "N-full", "N-no-transfer", "N-fixed-priority")
PHYSICAL_METHODS = ("N-full",)


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def time_score(outcome):
    value = outcome.get("team_time")
    return (
        float(value)
        if outcome.get("success") is True
        and type(value) in (int, float)
        and math.isfinite(value)
        and 0 <= value <= 300
        else 600.0
    )


def unverified_outcome(case, reason):
    return dict(
        case_id=case["case_id"],
        scenario=case["scenario"],
        success=False,
        team_time=None,
        safety_violations=1,
        invalid_actions=0,
        duplicate_service=0,
        decision_intervals=0,
        intervened_intervals=0,
        supervisor_held_drone_seconds=0.0,
        useful_post_event_transfer=False,
        failure_reason=str(reason),
    )


def _validate_outcome(outcome, case):
    """Reject malformed rows before any cohort reduction; retain a failed slot."""
    _require(
        isinstance(outcome, dict) and set(outcome) == set(unverified_outcome(case, "schema")),
        "Exact outcome fields required",
    )
    _require(
        type(outcome["success"]) is bool and type(outcome["useful_post_event_transfer"]) is bool,
        "Explicit boolean outcome flags required",
    )
    counts = (
        "safety_violations",
        "invalid_actions",
        "duplicate_service",
        "decision_intervals",
        "intervened_intervals",
    )
    _require(
        all(type(outcome[k]) is int and outcome[k] >= 0 for k in counts)
        and outcome["intervened_intervals"] <= outcome["decision_intervals"],
        "Invalid outcome counters",
    )
    for name in ("team_time", "supervisor_held_drone_seconds"):
        value = outcome[name]
        _require(
            (name == "team_time" and value is None and not outcome["success"])
            or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
            "Finite nonnegative outcome metrics required",
        )
    _require(
        (not outcome["useful_post_event_transfer"] or case["scenario"] == "availability-delay")
        and (
            outcome["failure_reason"] is None and outcome["decision_intervals"] > 0
            if outcome["success"]
            else isinstance(outcome["failure_reason"], str) and bool(outcome["failure_reason"])
        ),
        "Invalid outcome success/failure/transfer fields",
    )


class AblationSelector:
    """Fresh frozen NN forward each decision; only the permitted priority rewrite."""

    def __init__(self, policy, method, path):
        _require(method in ROUTING_METHODS[2:], "Exact NN method required")
        self.policy, self.method, self.path = policy, method, Path(path)
        _require(not self.path.exists(), "Preserve existing NN proposal evidence")
        self.counters = dict(forward_passes=0, priority_opportunities=0, priority_overwrites=0)

    def __call__(self, env):
        proposed = int(self.policy(env))
        self.counters["forward_passes"] += 1
        mask = np.asarray(env.legal_mask(), dtype=bool)
        _require(mask.shape == (242,), "Exact structural action mask required")
        legal = 0 <= proposed < 242 and bool(mask[proposed])
        action = proposed
        opportunity = False
        overwritten = False
        if legal and self.method == "N-fixed-priority":
            left, right, _ = decode_action(proposed)
            one, two = encode_action(left, right, 1), encode_action(left, right, 2)
            opportunity = left != WAIT and right != WAIT and left != right and bool(mask[one] and mask[two])
            if opportunity:
                action = one
                overwritten = action != proposed
        self.counters["priority_opportunities"] += int(opportunity)
        self.counters["priority_overwrites"] += int(overwritten)
        row = dict(
            tick=int(env.mission.tick),
            method=self.method,
            proposed_action=proposed,
            executed_action=action,
            legal=legal,
            priority_opportunity=opportunity,
            priority_overwrite=overwritten,
            mask=mask.tolist(),
        )
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _require(legal, "NN proposed an illegal action; no repair or canonicalization is allowed")
        return action


@dataclass(frozen=True)
class HeldoutIdentity:
    cohort: str
    method: str
    case_id: str
    case_sha256: str
    model_sha256: str
    model_kind: str
    selected_seed: int
    freeze_sha256: str
    heldout_manifest_sha256: str
    experiment_inputs_sha256: str
    no_transfer: bool

    def __post_init__(self):
        methods = (
            ROUTING_METHODS
            if self.cohort == "routing"
            else PHYSICAL_METHODS
            if self.cohort == "physical"
            else ()
        )
        _require(
            self.method in methods and type(self.selected_seed) is int and self.selected_seed in (42, 43, 44),
            "Exact frozen cohort/method/selected-seed identity required",
        )
        _require(
            type(self.no_transfer) is bool and self.no_transfer == (self.method == "N-no-transfer"),
            "Ownership ablation identity mismatch",
        )
        _require(
            self.model_kind == ("checkpoint" if self.method.startswith("N-") else "baseline"),
            "Typed model/baseline identity mismatch",
        )
        _require(isinstance(self.case_id, str) and bool(self.case_id), "Named exact case required")
        for name in (
            "case_sha256",
            "model_sha256",
            "freeze_sha256",
            "heldout_manifest_sha256",
            "experiment_inputs_sha256",
        ):
            value = getattr(self, name)
            _require(
                isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                "SHA256 identity required: " + name,
            )

    def to_dict(self):
        return asdict(self)


def claim_slot(path, identity):
    """Atomic finite slot claim. A prior incomplete attempt is never retried."""
    path = Path(path)
    if path.exists():
        _require(not path.is_symlink(), "Slot symlink is not an owned attempt")
        marker = path / "slot.json"
        _require(
            marker.is_file() and json.loads(marker.read_text()).get("identity") == identity,
            "Existing slot identity differs or is incomplete",
        )
        return None
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return claim_slot(path, identity)
    token = uuid.uuid4().hex
    atomic_json(path / "ownership.json", dict(schema="dual-evaluator-owned-slot-v1", token=token))
    atomic_json(path / "slot.json", dict(schema="dual-heldout-slot-v1", identity=identity, status="started"))
    return token


def _cohort(capability, cohort):
    _require(type(capability) is VerifiedHeldout, "VerifiedHeldout capability required")
    _require(cohort in ("routing", "physical"), "Exact routing or physical cohort required")
    capability.require_current()
    name = "routing300.json" if cohort == "routing" else "physical127.json"
    rows = json.loads(_path(capability.path.parent, name).read_text())
    cases = {}
    for item in rows:
        case = item if cohort == "routing" else item["case"]
        case = capability.verify_case(case, cohort + "-heldout")
        _require(case["case_id"] not in cases, "Duplicate heldout case")
        cases[case["case_id"]] = dict(case=case, group=None if cohort == "routing" else item["group"])
    _require(len(cases) == (300 if cohort == "routing" else 127), "Complete frozen heldout cohort required")
    return cases


def _identity(capability, cohort, method, case):
    freeze = capability.freeze
    seed = freeze.receipt["selected_seed"]
    neural = method.startswith("N-")
    model = (
        freeze.binding.files[f"terminal{seed}"][0]["sha256"]
        if neural
        else _digest(freeze.binding.files["baseline"])
    )
    return HeldoutIdentity(
        cohort,
        method,
        case["case_id"],
        _digest(case),
        model,
        "checkpoint" if neural else "baseline",
        seed,
        freeze.digest,
        capability.digest,
        _digest({r: freeze.binding.files[r] for r in EXPERIMENT_ROLES}),
        method == "N-no-transfer",
    ).to_dict()


def _development_reference(capability):
    from .dual_coordination_campaign import verified_development_replay_reference

    return verified_development_replay_reference(
        capability.binding, capability.freeze.receipt["qualification"]
    )


def _heldout_receipts(capability, identity, capture, bundle, development_reference):
    receipts = build_single_capture_receipts(
        capability, identity, capture, bundle, development_replay_reference=development_reference
    )
    result, audit, replay = (receipts[k] for k in ("result", "audit", "replay"))
    result["schema"] = "dual-heldout-result-v1"
    result["time_score"] = time_score(result["outcome"])
    audit["schema"] = "dual-heldout-audit-v1"
    audit["result_sha256"] = _digest(result)
    replay["schema"] = "dual-heldout-replay-v2"
    replay["result_sha256"] = _digest(result)
    replay["audit_sha256"] = _digest(audit)
    return receipts


def verify_receipts(
    receipts, identity, case, *, root, auditor_hash, configuration_hash, development_replay_reference=None
):
    """Authenticate completed evidence without converting a safety failure to success."""
    result, audit, replay = (receipts[k] for k in ("result", "audit", "replay"))
    _require(all(r["identity"] == identity for r in (result, audit, replay)), "Heldout identity mismatch")
    _require(
        result["schema"] == "dual-heldout-result-v1"
        and audit["schema"] == "dual-heldout-audit-v1"
        and replay["schema"] in ("dual-heldout-replay-v1", "dual-heldout-replay-v2"),
        "Heldout receipt schema required",
    )
    tick = result["terminal_tick"]
    outcome = result["outcome"]
    _validate_outcome(outcome, case)
    _require(
        result["mission"]["no_transfer"] is identity["no_transfer"],
        "Mission ownership ablation differs from typed identity",
    )
    _require(type(tick) is int and 0 <= tick <= 30000, "Invalid terminal clock")
    _require(
        outcome["case_id"] == case["case_id"] and outcome["scenario"] == case["scenario"],
        "Outcome case mismatch",
    )
    _require(result["time_score"] == time_score(outcome), "Invalid failure-penalized score")
    if outcome["success"]:
        _require(
            0 < tick <= 30000 and math.isclose(outcome["team_time"], tick * 0.01, rel_tol=0, abs_tol=1e-9),
            "Success beyond deadline or wrong actual clock",
        )
    coverage = dict(
        control_steps=tick, rk_states=tick * 64, native_substep_intervals=tick * 16, swept_intervals=tick * 16
    )
    single = replay["schema"] == "dual-heldout-replay-v2"
    for row in (audit,) if single else (audit, replay["audit"]):
        _require(
            row["identity"] == identity
            and row["auditor_sha256"] == auditor_hash
            and row["complete"] is True
            and row["errors"] == []
            and row["proof_gaps"] == []
            and row["coverage"] == coverage,
            "Incomplete independent native audit",
        )
        report = row["report"]
        _require(
            report["body_reconstructed_intervals"] == tick * 16, "Missing reconstructed body audit coverage"
        )
        unsafe = (
            report.get("violation_count", 0) > 0
            or bool(report.get("reconstruction_failures"))
            or any(
                report.get(k) is False for k in ("safe", "retained_points_safe", "reconstructed_path_safe")
            )
        )
        _require(
            not unsafe or outcome["safety_violations"] >= 1, "Failed safety audit disguised as safe outcome"
        )
    _require(
        audit["result_sha256"] == _digest(result) and audit["outcome"] == outcome, "Audit result link differs"
    )
    _require(
        replay["result_sha256"] == _digest(result)
        and replay["audit_sha256"] == _digest(audit)
        and replay["configuration_sha256"] == configuration_hash
        and replay["outcome"] == outcome
        and replay["inventory_sha256"] == _digest(replay["inventory"])
        and replay["control_steps"] == tick
        and replay["reset_records"] == 1,
        "Incomplete or mismatched retained evidence",
    )
    replay_claims = ("passed", "decisions_match", "terminal_match", "metrics_within_tolerance")
    if single:
        _require(
            development_replay_reference is not None
            and replay.get("development_replay_reference") == development_replay_reference,
            "Exact frozen development replay reference required",
        )
        reference_path = _path(Path(root).resolve(), development_replay_reference["path"])
        _require(
            file_record(reference_path, root) == development_replay_reference,
            "Frozen development replay reference changed",
        )
        _require(
            replay.get("execution_mode") == "retained-input-stream-validation"
            and replay.get("native_reexecution_performed") is False
            and replay.get("retained_inputs_verified") is True
            and replay.get("compact_boundary_consistency") is True
            and replay.get("errors") == []
            and replay.get("reset_case_sha256") == _digest(case) == identity["case_sha256"]
            and type(replay.get("action_records")) is int
            and replay["action_records"] == outcome["decision_intervals"]
            and type(replay.get("retained_state_records")) is int
            and replay["retained_state_records"] > 0
            and not any(key in replay for key in (*replay_claims, "audit")),
            "Incomplete single-capture evidence or false native replay claim",
        )
    else:
        _require(all(replay.get(k) is True for k in replay_claims), "Incomplete independent native replay")
    _require(
        set(replay["inventory"]) == {"reset", "actions", "compact_replay"}, "Replay inventory incomplete"
    )
    for record in replay["inventory"].values():
        path = _path(Path(root).resolve(), record["path"])
        _require(file_record(path, root) == record, "Replay bundle changed")
    return deepcopy(outcome)


def run_heldout(
    *, capability, cohort, original, design_path, output, journal_root=None, case_id=None, method=None
):
    """Execute only unstarted finite slots; caller must already possess joint freeze."""
    cases = _cohort(capability, cohort)
    from .dual_coordination_env import DualCoordinationEnv
    import torch

    capability.require_current()
    development_reference = _development_reference(capability)
    methods = ROUTING_METHODS if cohort == "routing" else PHYSICAL_METHODS
    _require(method is None or method in methods, "Method not declared for this cohort")
    _require(case_id is None or case_id in cases, "Case absent from verified heldout cohort")
    _require(
        capability.selected_backend in ("native-cpu", "native-cuda"), "Qualified native backend required"
    )
    device = "cuda" if capability.selected_backend == "native-cuda" else "cpu"
    _require(device != "cuda" or torch.cuda.is_available(), "Frozen CUDA inference device unavailable")
    original = _bound_file(capability, "assets", original)
    design_path = _bound_file(capability, "controller", design_path)
    for name in (
        "dual_coordination_heldout.py",
        "dual_coordination_evaluate.py",
        "dual_coordination_policy.py",
        "dual_coordination_baselines.py",
        "dual_coordination_selection.py",
        "dual_coordination_campaign.py",
    ):
        _bound_file(capability, "source", Path(__file__).with_name(name))
    seed = capability.freeze.receipt["selected_seed"]
    terminal_record = capability.binding.files[f"terminal{seed}"][0]
    terminal = _path(capability.binding.root, terminal_record["path"])
    _require(sha256(terminal) == terminal_record["sha256"], "Frozen selected model changed")
    output = Path(output).resolve()
    root = capability.binding.root
    _require(output.is_relative_to(root), "Heldout output must remain inside frozen campaign root")
    declaration = dict(
        schema="dual-heldout-evaluation-declaration-v1",
        cohort=cohort,
        freeze_sha256=capability.freeze.digest,
        heldout_manifest_sha256=capability.digest,
        selected_seed=seed,
        selected_model_sha256=terminal_record["sha256"],
        methods=list(methods),
        cases={k: dict(sha256=_digest(v["case"]), group=v["group"]) for k, v in cases.items()},
        expected_missions=len(cases) * len(methods),
        replay_stride_ticks=20,
        execution_mode="retained-input-stream-validation",
        development_replay_reference=development_reference,
        experiment_inputs_sha256=_digest({r: capability.binding.files[r] for r in EXPERIMENT_ROLES}),
    )
    if output.exists():
        _require(
            (output / "declaration.json").is_file()
            and json.loads((output / "declaration.json").read_text()) == declaration,
            "Existing heldout run has a different prospective declaration",
        )
    else:
        output.mkdir(parents=True)
        atomic_json(output / "declaration.json", declaration)
    design = json.loads(design_path.read_text())
    journal = RunJournal(journal_root or output)
    attempt = journal.start(
        "native-heldout-" + cohort,
        "Frozen selected model; no heldout reselection",
        category="heldout evaluation",
    )
    done = skipped = 0
    try:
        for current_method in [method] if method else methods:
            for key in [case_id] if case_id else cases:
                case = capability.verify_case(cases[key]["case"], cohort + "-heldout")
                capability.require_current()
                identity = _identity(capability, cohort, current_method, case)
                slot = output / "missions" / current_method / hashlib.sha256(key.encode()).hexdigest()[:20]
                token = claim_slot(slot, identity)
                if token is None:
                    skipped += 1
                    continue
                try:
                    choose = _selector(
                        f"N{seed}" if current_method.startswith("N-") else current_method,
                        case,
                        terminal,
                        device,
                    )
                    if current_method.startswith("N-"):
                        choose = AblationSelector(choose, current_method, slot / "nn-proposals.jsonl")

                    def beat(tick):
                        journal.heartbeat(
                            attempt,
                            completed=done,
                            total=declaration["expected_missions"],
                            reason=f"{current_method} {key}: native tick {tick}; skipped retained slots {skipped}",
                        )

                    options = dict(
                        mode=cohort + "-heldout",
                        admission=capability,
                        retain_replay=True,
                        replay_stride_ticks=20,
                        no_transfer=current_method == "N-no-transfer",
                    )
                    env = DualCoordinationEnv(original, slot / "capture", case, design, **options)
                    first = capture_mission(env, choose, case, slot / "capture", heartbeat=beat)
                    del env
                    evidence = output / "evidence"
                    evidence.mkdir(exist_ok=True)
                    bundle = bundle_captures(
                        evidence / (current_method + "-" + slot.name + ".zip"),
                        [(slot / "capture", first)],
                        root=root,
                    )
                    receipts = _heldout_receipts(capability, identity, first, bundle, development_reference)
                    outcome = verify_receipts(
                        receipts,
                        identity,
                        case,
                        root=root,
                        auditor_hash=_digest(capability.binding.files["audit"]),
                        configuration_hash=_digest(capability.binding.files["configuration"]),
                        development_replay_reference=development_reference,
                    )
                    ablation = (
                        dict(
                            counters=deepcopy(choose.counters),
                            proposals=file_record(slot / "nn-proposals.jsonl", root),
                        )
                        if isinstance(choose, AblationSelector)
                        else None
                    )
                    document = dict(
                        schema="dual-heldout-evidence-v1",
                        identity=identity,
                        receipts=receipts,
                        ablation=ablation,
                    )
                    path = evidence / (current_method + "-" + slot.name + ".json")
                    atomic_json(path, document)
                    retire_slot_captures(
                        slot, bundle["record"], root=root, token=token, capture_names=("capture",)
                    )
                    atomic_json(
                        slot / "slot.json",
                        dict(
                            schema="dual-heldout-slot-v1",
                            identity=identity,
                            status="complete",
                            evidence=file_record(path, root),
                            outcome=outcome,
                        ),
                    )
                except Exception as error:
                    reason = f"{type(error).__name__}: {error}"
                    atomic_json(
                        slot / "failure.json",
                        dict(
                            identity=identity,
                            reason=reason,
                            outcome=unverified_outcome(case, reason),
                            time_score=600.0,
                        ),
                    )
                    atomic_json(
                        slot / "slot.json",
                        dict(
                            schema="dual-heldout-slot-v1", identity=identity, status="failed", reason=reason
                        ),
                    )
                done += 1
                journal.heartbeat(
                    attempt,
                    completed=done,
                    total=declaration["expected_missions"],
                    reason=f"Retained finite slot; existing skipped={skipped}",
                )
        result = aggregate_heldout(capability=capability, cohort=cohort, output=output)
        journal.finish(
            attempt, status="done", reason=f"Heldout slots retained; {result['status']}; no reselection"
        )
        return result
    except BaseException as error:
        journal.finish(attempt, status="failed", reason=str(error))
        raise


def verify_ablation(document, *, root):
    """Recompute every allowed overwrite from retained proposed actions/masks."""
    identity = document["identity"]
    method = identity["method"]
    data = document.get("ablation")
    if not method.startswith("N-"):
        _require(data is None, "Baseline must not claim neural forwards")
        return dict(forward_passes=0, priority_opportunities=0, priority_overwrites=0)
    _require(isinstance(data, dict), "NN forward evidence missing")
    record = data["proposals"]
    path = _path(Path(root).resolve(), record["path"])
    _require(file_record(path, root) == record, "NN proposal evidence changed")
    counters = dict(forward_passes=0, priority_opportunities=0, priority_overwrites=0)
    action_record = document["receipts"]["replay"]["inventory"]["actions"]
    archive = _path(Path(root).resolve(), action_record["path"])
    _require(file_record(archive, root) == action_record, "Original action bundle changed")
    with zipfile.ZipFile(archive) as bundle:
        actions = [json.loads(line) for line in bundle.read("0/proposed-actions.jsonl").decode().splitlines()]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _require(len(actions) == len(rows), "Original proposal stream differs from neural forward count")
    previous = -20
    for row, action in zip(rows, actions, strict=True):
        _require(
            row["method"] == method and row["tick"] == previous + 20,
            "Missing or reordered fresh NN decisions",
        )
        previous = row["tick"]
        proposed = row["proposed_action"]
        mask = row["mask"]
        _require(
            len(mask) == 242
            and all(type(v) is bool for v in mask)
            and type(proposed) is int
            and 0 <= proposed < 242
            and mask[proposed]
            and row["legal"] is True,
            "Illegal retained NN action",
        )
        opportunity = False
        executed = proposed
        if method == "N-fixed-priority":
            left, right, _ = decode_action(proposed)
            one, two = encode_action(left, right, 1), encode_action(left, right, 2)
            opportunity = left != WAIT and right != WAIT and left != right and mask[one] and mask[two]
            if opportunity:
                executed = one
        _require(
            row["priority_opportunity"] == opportunity
            and row["executed_action"] == executed
            and row["priority_overwrite"] == (executed != proposed),
            "Unauthorized ablation or mask canonicalization",
        )
        _require(
            action["tick"] == row["tick"] and action["action"] == executed and action["mask"] == mask,
            "Actual action stream differs from fresh NN proposal/allowed ablation",
        )
        counters["forward_passes"] += 1
        counters["priority_opportunities"] += int(opportunity)
        counters["priority_overwrites"] += int(executed != proposed)
    result = document["receipts"]["result"]
    _require(
        counters == data["counters"]
        and counters["forward_passes"] == result["outcome"]["decision_intervals"],
        "NN forward/opportunity counters do not match actual decisions",
    )
    _require(
        result["mission"]["no_transfer"] is identity["no_transfer"], "Ablation mission ownership mode differs"
    )
    return counters


def numeric_reports(outcomes, cohort, *, crossing=None):
    """Existing unchanged §9 calculations only; no evidence authentication here.

    Crossing/media execution is a later phase. Missing crossing evidence remains
    an explicit failed gate; this function never invents a visibility approval.
    """
    if cohort == "physical":
        return physical_report(outcomes["N-full"])
    _require(
        cohort == "routing" and set(outcomes) == set(ROUTING_METHODS), "All five routing methods required"
    )
    crossing = (
        crossing
        if crossing is not None
        else dict(
            predeclared=False,
            success=False,
            safety_violations=1,
            intervened_intervals=0,
            nn_priority_yield_visible=False,
        )
    )
    return routing_report(
        outcomes["N-full"],
        outcomes["B-static"],
        outcomes["B-dynamic"],
        outcomes["N-no-transfer"],
        outcomes["N-fixed-priority"],
        crossing,
    )


def aggregate_heldout(*, capability, cohort, output):
    """Re-read exact finite slots. Failed/missing evidence never leaves denominators."""
    cases = _cohort(capability, cohort)
    development_reference = _development_reference(capability)
    output = Path(output).resolve()
    root = capability.binding.root
    _require(output.is_relative_to(root), "Heldout report must remain inside bound campaign root")
    declaration = json.loads(_path(output, "declaration.json").read_text())
    _require(
        declaration["freeze_sha256"] == capability.freeze.digest
        and declaration["heldout_manifest_sha256"] == capability.digest
        and declaration["selected_seed"] == capability.freeze.receipt["selected_seed"],
        "Heldout declaration changed",
    )
    _require(
        declaration.get("execution_mode") == "retained-input-stream-validation"
        and declaration.get("development_replay_reference") == development_reference,
        "Frozen development replay reference differs from declaration",
    )
    methods = ROUTING_METHODS if cohort == "routing" else PHYSICAL_METHODS
    _require(
        declaration["cohort"] == cohort
        and declaration["methods"] == list(methods)
        and declaration["cases"]
        == {k: dict(sha256=_digest(v["case"]), group=v["group"]) for k, v in cases.items()},
        "Heldout slots differ from frozen cases",
    )
    outcomes = {m: [] for m in methods}
    errors = []
    attempted = verified = 0
    ablation = {m: dict(forward_passes=0, priority_opportunities=0, priority_overwrites=0) for m in methods}
    records = []
    for method in methods:
        for key, item in cases.items():
            case = item["case"]
            identity = _identity(capability, cohort, method, case)
            slot = output / "missions" / method / hashlib.sha256(key.encode()).hexdigest()[:20]
            outcome = unverified_outcome(case, "heldout slot never started")
            independently_audited = False
            if slot.exists():
                attempted += 1
                try:
                    marker = json.loads(_path(slot, "slot.json").read_text())
                    _require(marker["identity"] == identity, "Retained slot identity differs")
                    _require(
                        marker["status"] == "complete",
                        "Preserved failed/incomplete attempt: " + str(marker.get("reason", "incomplete")),
                    )
                    evidence = marker["evidence"]
                    path = _path(root, evidence["path"])
                    _require(file_record(path, root) == evidence, "Heldout evidence changed")
                    document = json.loads(path.read_text())
                    _require(
                        document["schema"] == "dual-heldout-evidence-v1" and document["identity"] == identity,
                        "Wrong evidence identity/schema",
                    )
                    outcome = verify_receipts(
                        document["receipts"],
                        identity,
                        case,
                        root=root,
                        auditor_hash=_digest(capability.binding.files["audit"]),
                        configuration_hash=_digest(capability.binding.files["configuration"]),
                        development_replay_reference=development_reference,
                    )
                    counters = verify_ablation(document, root=root)
                    for name, value in counters.items():
                        ablation[method][name] += value
                    _require(marker["outcome"] == outcome, "Retained slot outcome changed")
                    independently_audited = True
                    verified += 1
                    records.append(evidence)
                except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, EOFError) as error:
                    reason = f"{method} {key}: {type(error).__name__}: {error}"
                    errors.append(reason)
                    outcome = unverified_outcome(case, reason)
            if cohort == "physical":
                outcome.update(group=item["group"], independently_audited=independently_audited)
            outcomes[method].append(outcome)
    expected = len(cases) * len(methods)
    numeric = numeric_reports(outcomes, cohort)
    summary = dict(
        schema="dual-heldout-summary-v1",
        cohort=cohort,
        freeze_sha256=capability.freeze.digest,
        heldout_manifest_sha256=capability.digest,
        selected_seed=capability.freeze.receipt["selected_seed"],
        status="complete" if verified == expected and not errors else "partial",
        expected_missions=expected,
        attempted_missions=attempted,
        verified_missions=verified,
        unstarted_missions=expected - attempted,
        unverified_missions=expected - verified,
        outcomes=outcomes,
        ablation_counters=ablation,
        errors=errors,
        evidence_files=records,
        numeric_report=numeric,
        cohort_acceptance_passed=verified == expected and not errors and numeric["passed"],
        crossing_evidence_pending=cohort == "routing",
        selection_repeated=False,
    )
    atomic_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    import argparse
    from .dual_coordination_campaign import VerifiedJointFreeze

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--heldout-root", required=True)
    parser.add_argument("--cohort", choices=["routing", "physical"], required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal-root")
    parser.add_argument("--case-id")
    parser.add_argument("--method")
    args = parser.parse_args(argv)
    freeze = VerifiedJointFreeze.load(args.freeze)
    capability = VerifiedHeldout.load(args.heldout_root, freeze=freeze)
    result = run_heldout(
        capability=capability,
        cohort=args.cohort,
        original=args.original,
        design_path=args.design,
        output=args.output,
        journal_root=args.journal_root,
        case_id=args.case_id,
        method=args.method,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "expected_missions",
                    "verified_missions",
                    "unstarted_missions",
                    "selected_seed",
                )
            }
        )
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
