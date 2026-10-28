"""Spec §7 post-selection validation diagnostics, never selection or freeze inputs.

Exactly validation300 × two interventions on the already selected terminal model.
Each intervention runs fresh inference on its own world. Failed/incomplete slots
are preserved, and a subsequent invocation executes only never-started slots.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .dual_coordination_campaign import (
    BoundFiles,
    EXPERIMENT_ROLES,
    VerifiedLearningLock,
    _digest,
    _path,
    _validate_terminal_checkpoint,
    _validation_cases,
    _verify_validation_evidence,
    _verify_retained_validation_bundle,
    verified_development_replay_reference,
)
from .dual_coordination_evaluate import (
    _bound_file,
    _selector,
    build_single_capture_receipts,
    bundle_captures,
    capture_mission,
    file_record,
    retire_slot_captures,
    verify_terminal_recovery,
)
from .dual_coordination_heldout import (
    AblationSelector,
    claim_slot,
    unverified_outcome,
    verify_ablation,
    _validate_outcome,
)
from .dual_coordination_progress import RunJournal, atomic_json
from .dual_coordination_selection import _summary, select_candidate
from .util import sha256

METHODS = ("N-no-transfer", "N-fixed-priority")


def check_selected_terminal(selection, terminal, numeric):
    """Authenticate the existing choice; never choose another model for diagnostics."""
    seed = selection.get("selected_seed")
    if (
        selection.get("schema") != "dual-selection-bound-v1"
        or type(seed) is not int
        or seed not in (42, 43, 44)
        or numeric.get("selected") != seed
    ):
        raise ValueError("Existing eligible selection required; diagnostics cannot reselect")
    if selection.get("checkpoint_sha256", {}).get(str(seed)) != sha256(terminal):
        raise ValueError("Selected terminal hash differs from existing selection")
    return seed


def authenticate_selection(lock, selection_path, terminal, original, design_path):
    """Revalidate the existing complete selection using its original evidence.

    This in-memory binding adds only the original validation evidence to the
    learning lock. It creates no new lock, selection, or joint-freeze artifact.
    """
    root = lock.binding.root
    selection_path = _path(root, str(Path(selection_path).resolve().relative_to(root)))
    selection = json.loads(selection_path.read_text())
    numeric = select_candidate(
        {int(k): v for k, v in selection["candidates"].items()}, selection["static"], selection["dynamic"]
    )
    seed = check_selected_terminal(selection, terminal, numeric)
    _validate_terminal_checkpoint(terminal, seed)
    verify_terminal_recovery(terminal, lock, seed=seed, original=original, design_path=design_path)
    cases = _validation_cases(lock.binding)
    for case in cases.values():
        lock.verify_case(case, "validation")
    records = {}
    for relative in selection["validation_evidence"]:
        path = _path(root, relative)
        records[relative] = file_record(path, root)
        document = json.loads(path.read_text())
        for mission in document["missions"]:
            for record in mission["replay"]["inventory"].values():
                if file_record(_path(root, record["path"]), root) != record:
                    raise ValueError("Original selection evidence changed")
                records[record["path"]] = record
    files = deepcopy(lock.binding.files)
    files["validation_evidence"] = list(records.values())
    bound = BoundFiles(root, files)
    manifests = {int(k): {"sha256": value} for k, value in selection["checkpoint_sha256"].items()}
    _verify_validation_evidence(bound, selection, cases, manifests)
    return seed, cases, bound, file_record(selection_path, root)


def slot_path(output, method, case_id):
    return Path(output) / "missions" / method / hashlib.sha256(case_id.encode()).hexdigest()[:20]


def summarize(output, cases, identity_for, verify):
    """Keep all600 slots, including missing/failed, without an acceptance/selection gate."""
    output = Path(output)
    outcomes = {method: [] for method in METHODS}
    attempted = verified = 0
    errors = []
    counters = {
        method: dict(forward_passes=0, priority_opportunities=0, priority_overwrites=0) for method in METHODS
    }
    for method in METHODS:
        for case in cases.values():
            identity = identity_for(method, case)
            slot = slot_path(output, method, case["case_id"])
            outcome = unverified_outcome(case, "diagnostic slot never started")
            if slot.exists():
                attempted += 1
                try:
                    marker = json.loads((slot / "slot.json").read_text())
                    if marker["identity"] != identity or marker["status"] != "complete":
                        raise ValueError("Preserved incomplete or failed diagnostic slot")
                    path = slot / "diagnostic.json"
                    if sha256(path) != marker["diagnostic_sha256"]:
                        raise ValueError("Diagnostic evidence changed")
                    document = json.loads(path.read_text())
                    outcome, counts = verify(document, identity, case)
                    if marker["outcome"] != outcome:
                        raise ValueError("Diagnostic slot outcome changed")
                    for key, value in counts.items():
                        counters[method][key] += value
                    verified += 1
                except (OSError, ValueError, KeyError, TypeError) as error:
                    reason = f"{method} {case['case_id']}: {type(error).__name__}: {error}"
                    errors.append(reason)
                    outcome = unverified_outcome(case, reason)
            outcomes[method].append(outcome)
    expected = 2 * len(cases)
    return dict(
        schema="dual-validation-diagnostics-summary-v1",
        diagnostic_only=True,
        selection_modified=False,
        joint_freeze_depends_on_diagnostics=False,
        expected_missions=expected,
        attempted_missions=attempted,
        verified_missions=verified,
        unstarted_missions=expected - attempted,
        status="complete" if verified == expected else "partial",
        outcomes=outcomes,
        metrics={m: _summary(rows) for m, rows in outcomes.items()},
        ablation_counters=counters,
        errors=errors,
    )


def verify_document(document, identity, case, *, lock, development_reference):
    if document.get("schema") != "dual-validation-diagnostic-evidence-v1" or document["identity"] != identity:
        raise ValueError("Diagnostic identity differs")
    receipts = document["receipts"]
    if any(row["identity"] != identity for row in receipts.values()):
        raise ValueError("Diagnostic receipt identity differs")
    for record in receipts["replay"]["inventory"].values():
        if file_record(_path(lock.binding.root, record["path"]), lock.binding.root) != record:
            raise ValueError("Diagnostic retained bundle changed")
    _verify_retained_validation_bundle(lock.binding, receipts, development_reference)
    result, audit = receipts["result"], receipts["audit"]
    outcome = result["outcome"]
    _validate_outcome(outcome, case)
    if (
        outcome["case_id"] != case["case_id"]
        or outcome["scenario"] != case["scenario"]
        or result["mission"]["no_transfer"] is not identity["no_transfer"]
        or audit["complete"] is not True
        or audit["errors"]
        or audit["proof_gaps"]
    ):
        raise ValueError("Incomplete diagnostic case/ownership/audit evidence")
    return deepcopy(outcome), verify_ablation(document, root=lock.binding.root)


def run_diagnostics(
    *,
    learning_lock,
    selection,
    terminal,
    original,
    design_path,
    output,
    journal_root=None,
    case_id=None,
    method=None,
):
    """Run600 post-selection diagnostics with a qualified learning capability.

    The selected seed/hash and prior validation evidence are authenticated before
    output. Results remain diagnostic-only regardless of success or failure.
    """
    from .dual_coordination_env import DualCoordinationEnv
    import torch

    lock = VerifiedLearningLock.load(learning_lock)
    lock.require_current()
    if method is not None and method not in METHODS:
        raise ValueError("Only the two post-selection diagnostic interventions are allowed")
    original = _bound_file(lock, "assets", original)
    design_path = _bound_file(lock, "controller", design_path)
    for name in (
        "dual_coordination_diagnostics.py",
        "dual_coordination_heldout.py",
        "dual_coordination_evaluate.py",
        "dual_coordination_policy.py",
        "dual_coordination_campaign.py",
        "dual_coordination_selection.py",
    ):
        _bound_file(lock, "source", Path(__file__).with_name(name))
    terminal = Path(terminal).resolve()
    seed, cases, selection_binding, selection_record = authenticate_selection(
        lock, selection, terminal, original, design_path
    )
    if case_id is not None and case_id not in cases:
        raise ValueError("Diagnostic case must belong to the original validation300")
    development_reference = verified_development_replay_reference(lock.binding, lock.receipt["qualification"])
    if lock.selected_backend not in ("native-cpu", "native-cuda"):
        raise ValueError("Qualified native backend required")
    device = "cuda" if lock.selected_backend == "native-cuda" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Frozen CUDA inference device unavailable")
    model_hash = sha256(terminal)
    root = lock.binding.root
    output = Path(output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Diagnostic output must remain inside the campaign root")
    declaration = dict(
        schema="dual-validation-diagnostics-declaration-v1",
        diagnostic_only=True,
        learning_lock_sha256=lock.digest,
        selection=selection_record,
        selected_seed=seed,
        selected_terminal_sha256=model_hash,
        methods=list(METHODS),
        expected_missions=600,
        cases={key: _digest(value) for key, value in cases.items()},
        development_replay_reference=development_reference,
        replay_stride_ticks=20,
        selection_modified=False,
        joint_freeze_depends_on_diagnostics=False,
    )
    if output.exists():
        if json.loads((output / "declaration.json").read_text()) != declaration:
            raise ValueError("Existing diagnostic declaration differs")
    else:
        output.mkdir(parents=True)
        atomic_json(output / "declaration.json", declaration)
    design = json.loads(design_path.read_text())

    def identity_for(current, case):
        return dict(
            method=current,
            case_id=case["case_id"],
            case_sha256=_digest(case),
            model_sha256=model_hash,
            selected_seed=seed,
            selection_sha256=selection_record["sha256"],
            learning_lock_sha256=lock.digest,
            no_transfer=current == "N-no-transfer",
            diagnostic_only=True,
            experiment_inputs_sha256=_digest({r: lock.binding.files[r] for r in EXPERIMENT_ROLES}),
        )

    def verify(document, identity, case):
        return verify_document(
            document, identity, case, lock=lock, development_reference=development_reference
        )

    journal = RunJournal(journal_root or output)
    attempt = journal.start(
        "post-selection-validation-diagnostics",
        "Fixed selected weights;600 diagnostic slots; no reselection",
        category="validation",
    )
    done = 0
    try:
        for current in (method,) if method else METHODS:
            for key in (case_id,) if case_id else cases:
                lock.require_current()
                if (
                    file_record(_path(root, selection_record["path"]), root) != selection_record
                    or sha256(terminal) != model_hash
                ):
                    raise ValueError("Pinned selection or terminal changed during diagnostics")
                case = lock.verify_case(cases[key], "validation")
                identity = identity_for(current, case)
                slot = slot_path(output, current, key)
                token = claim_slot(slot, identity)
                if token is None:
                    continue
                try:
                    choose = AblationSelector(
                        _selector(f"N{seed}", case, terminal, device), current, slot / "nn-proposals.jsonl"
                    )

                    def heartbeat(tick):
                        journal.heartbeat(
                            attempt, completed=done, total=600, reason=f"{current} {key}: native tick {tick}"
                        )

                    env = DualCoordinationEnv(
                        original,
                        slot / "capture",
                        case,
                        design,
                        mode="validation",
                        admission=lock,
                        retain_replay=True,
                        replay_stride_ticks=20,
                        no_transfer=current == "N-no-transfer",
                    )
                    capture = capture_mission(env, choose, case, slot / "capture", heartbeat=heartbeat)
                    del env
                    bundle = bundle_captures(slot / "capture.zip", [(slot / "capture", capture)], root=root)
                    receipts = build_single_capture_receipts(
                        lock, identity, capture, bundle, development_replay_reference=development_reference
                    )
                    document = dict(
                        schema="dual-validation-diagnostic-evidence-v1",
                        identity=identity,
                        receipts=receipts,
                        ablation=dict(
                            counters=deepcopy(choose.counters),
                            proposals=file_record(slot / "nn-proposals.jsonl", root),
                        ),
                    )
                    outcome, _ = verify(document, identity, case)
                    atomic_json(slot / "diagnostic.json", document)
                    retire_slot_captures(
                        slot, bundle["record"], root=root, token=token, capture_names=("capture",)
                    )
                    atomic_json(
                        slot / "slot.json",
                        dict(
                            identity=identity,
                            status="complete",
                            outcome=outcome,
                            diagnostic_sha256=sha256(slot / "diagnostic.json"),
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
                    atomic_json(slot / "slot.json", dict(identity=identity, status="failed", reason=reason))
                done += 1
                journal.heartbeat(attempt, completed=done, total=600, reason="Diagnostic slot retained")
        selection_binding.verify()
        if (
            file_record(_path(root, selection_record["path"]), root) != selection_record
            or sha256(terminal) != model_hash
        ):
            raise ValueError("Pinned selection or terminal changed")
        summary = summarize(output, cases, identity_for, verify)
        summary.update(selection=selection_record, selected_seed=seed, selected_terminal_sha256=model_hash)
        atomic_json(output / "summary.json", summary)
        journal.finish(
            attempt, status="done", reason="Diagnostics retained; selection and joint freeze unchanged"
        )
        return summary
    except BaseException as error:
        journal.finish(attempt, status="failed", reason=str(error))
        raise


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("learning-lock", "selection", "terminal", "original", "design-path", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--journal-root")
    parser.add_argument("--case-id")
    parser.add_argument("--method", choices=METHODS)
    result = run_diagnostics(**vars(parser.parse_args(argv)))
    print(
        json.dumps(
            {k: result[k] for k in ("status", "expected_missions", "verified_missions", "selected_seed")}
        )
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
