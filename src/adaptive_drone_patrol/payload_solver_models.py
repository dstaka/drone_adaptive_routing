"""Check matched policy identity/provenance before evaluation or capture."""

import json
from pathlib import Path
from .util import sha256


def verify_pair_metadata(infos, plan):
    if set(infos) != {"randomized", "fixed"}:
        raise ValueError("Both distinct final methods required")
    common = plan["common"]
    if (
        common["seed"] != 7
        or common["N"] != 1
        or common["T"] != 100
        or common["pilot"]
        or common["S"] != common["N"] * common["T"] * common["K"]
    ):
        raise ValueError("Final common training plan differs")
    for method, info in infos.items():
        if info.get("method") != method:
            raise ValueError("Checkpoint method differs from declared policy label")
        if (
            info.get("training_split") != "final"
            or info.get("common_training_plan") != common
            or info.get("valid_environment_transitions") != common["S"]
        ):
            raise ValueError("Checkpoint final plan/sample provenance differs")
        for role in ["actor", "critic"]:
            key = f"initial_{role}_tensor_sha256"
            if not isinstance(plan.get(key), str) or len(plan[key]) != 64 or info.get(key) != plan[key]:
                raise ValueError("Final initial tensor differs from frozen plan")
    return True


def audit_final_training(method, item, plan):
    """Require the adopted checkpoint's complete closed final learning stream."""
    import torch
    from .payload_checkpoint import actor_signature
    from .payload_retry_checkpoint import transfer_state, initialization_comparison
    from .payload_solver_exposure import audit_training_prefix, require_exposure

    if not item.get("run_root"):
        raise ValueError("Complete actual final training run root required")
    root = Path(item["run_root"])
    common = plan["common"]
    steps = common["S"]
    if Path(item["checkpoint"]).resolve() != (root / f"checkpoint_s{steps}.pt").resolve():
        raise ValueError("Adopted checkpoint is outside actual final training run")
    manifest = json.loads((root / "manifest.json").read_text())
    completed = json.loads((root / "completed.json").read_text())
    if (manifest["method"] != method or manifest["split"] != "final"
            or manifest["common"] != common or manifest["fresh_optimizer"] is not True
            or manifest["initial_iteration"] != 0 or completed["counters"]["valid"] != steps):
        raise ValueError("Actual final training declaration/sample closure differs")
    parent = Path(plan["parent_path"])
    if sha256(parent) != common["parent_sha256"]:
        raise ValueError("Actual original4296 final parent differs")
    original = torch.load(parent, map_location="cpu", weights_only=False)
    expected = transfer_state(original)
    comparison = initialization_comparison(original, expected)
    init = root / "initialization.pt"
    if sha256(init) != manifest["initialization_sha256"]:
        raise ValueError("Actual final initialization artifact changed")
    actual = torch.load(init, map_location="cpu", weights_only=False)
    if (actual.get("iter") != 0 or "optimizer_state_dict" in actual
            or manifest["initialization_comparison"] != comparison
            or actual["infos"].get("transfer") != expected["infos"]["transfer"]
            or actual["infos"].get("training_split") != "initialization-only"
            or actual["infos"].get("actual_solver_binding") != common["actual_binding"]):
        raise ValueError("Actual final fresh initialization differs")
    for role in ["actor", "critic"]:
        key = f"initial_{role}_tensor_sha256"
        if (actor_signature(actual[role + "_state_dict"]) != comparison[key]
                or comparison[key] != plan[key]):
            raise ValueError("Actual final initial tensors differ from fresh4296 transfer")
    proof = audit_training_prefix(root, steps)
    require_exposure(proof)
    artifacts = {str(root / name): digest for name, digest in proof["artifact_sha256"].items()}
    artifacts.update({str(p): sha256(p) for p in [init, root / "completed.json", Path(item["checkpoint"])]})
    return dict(method=method, run_root=str(root), exposure=proof, artifact_sha256=artifacts)


def inspect_models(models, *, final, plan=None, training_proofs=None):
    """Hash exact actual artifacts and read their embedded training provenance."""
    import torch
    from .payload_retry_checkpoint import _validate_tensors

    if not models or set(models) - {"randomized", "fixed"}:
        raise ValueError("Explicit known methods required")
    infos = {}
    for method, item in models.items():
        path = Path(item["checkpoint"])
        if sha256(path) != item["sha256"]:
            raise ValueError("Actual policy artifact differs")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for role in ["actor", "critic"]:
            _validate_tensors(checkpoint[role + "_state_dict"], role, 29)
        info = checkpoint["infos"]
        if info.get("method") != method:
            raise ValueError("Checkpoint method differs from declared policy label")
        if info.get("training_split") not in (["final"] if final else ["development", "final"]):
            raise ValueError("Initialization-only or unsupported evaluation policy")
        infos[method] = info
    if final:
        if plan is None:
            raise ValueError("Final paired evaluation requires common frozen training plan")
        verify_pair_metadata(infos, plan)
        if len({str(Path(item.get("run_root", ".")).resolve()) for item in models.values()}) != 2:
            raise ValueError("Distinct actual final training roots required")
        for method, item in models.items():
            proof = audit_final_training(method, item, plan)
            if training_proofs is not None:
                training_proofs[method] = proof
    return infos
