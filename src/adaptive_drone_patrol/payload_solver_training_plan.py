"""Freeze equal final training only from independently replayed development.

The selector is arithmetic, not physical admission. The writer replays every
selected candidate's actual learning prefix and real development sequences.
"""

import copy
import json
from pathlib import Path

import numpy as np

from .payload_solver_exposure import audit_training_prefix, require_exposure
from .util import sha256, load_config


PLAN_SCHEMA = "payload-solver-final-training-plan-v1"


def select_common_rung(candidates, *, ladder=(300000, 1000000, 3000000)):
    samples = [row["S"] for row in candidates]
    if not samples or samples != list(ladder[: len(samples)]) or len(samples) > len(ladder):
        raise ValueError("All earlier distinct predeclared common rungs required")
    if any(row["conditions"] != dict(randomized=6, fixed=6) for row in candidates):
        raise ValueError("Both unchanged development denominators required")
    for row in candidates:
        if row["successes"]["randomized"] != 6:
            continue
        try:
            for method in ["randomized", "fixed"]:
                require_exposure(row["exposure"][method])
        except ValueError:
            continue
        return row["S"]
    raise ValueError("No qualifying common development rung; final training remains blocked")


def replay_development(root, models, binding):
    """Replay fixed development seeds and actual native policy/body sequences."""
    from .payload_cases import initial_state
    from .payload_solver_validation import verify_sequence

    root = Path(root)
    record = json.loads((root / "binding.json").read_text())
    if record["mode"] != "development" or record["models"] != models or record["actual_binding"] != binding:
        raise ValueError("Actual development model/physics split differs")
    if any(sha256(p) != h for p, h in record["source_binding"].items()):
        raise ValueError("Actual development checker/source changed")
    cfg = load_config("configs/payload_validation.yaml")
    if sha256('configs/payload_validation.yaml') != '5820da816e284e34c0f5586dcee2721414b097dfba07406bb832b18066e85490':
        raise ValueError('Prescribed validation/development geometry declaration changed')
    expected = [
        dict(
            case_id=f"development-{seed}-{i}",
            **load,
            initial=initial_state(
                np.random.default_rng(np.random.SeedSequence([seed, 720, i])), load["side_m"]
            ),
        )
        for seed in [83021, 83022]
        for i, load in enumerate(cfg["loads"])
    ]
    if record["cases"] != expected or record["targets"] != cfg["targets"]:
        raise ValueError("Actual development seeds/loads/targets differ")
    results = json.loads((root / "results.json").read_text())
    if sorted((r["method"], r["case_id"]) for r in results) != sorted(
        (method, case["case_id"]) for method in ["randomized", "fixed"] for case in expected
    ):
        raise ValueError("Missing or repeated actual development cases")
    successes = dict(randomized=0, fixed=0)
    failures = []
    artifacts = {}
    for method in ["randomized", "fixed"]:
        item = models[method]
        for case in expected:
            path = root / method / case["case_id"]
            if json.loads((path / "case.json").read_text()) != case:
                raise ValueError("Recorded development case changed")
            native = dict(
                native_path=path / "flight/runtime/raw.jsonl.gz",
                properties_path=path / "flight/runtime-properties.json",
                checkpoint=item["checkpoint"],
                expected_hash=item["sha256"],
                case=case,
                binding=binding,
            )
            if not native["native_path"].is_file():
                raise ValueError("Actual failed or completed development case raw missing")
            proof = verify_sequence(path / "sequence", cfg["targets"], native=native, allow_incomplete=True)
            if proof['evidence_kind'] != 'physical' or not proof['native_proof']:
                raise ValueError('Actual physical development native proof required')
            if proof['passed']:
                successes[method] += 1
            else:
                failures.append(dict(method=method, case_id=case["case_id"], proof=proof))
            for artifact in path.rglob("*"):
                if artifact.is_file():
                    artifacts[str(artifact)] = sha256(artifact)
    for name in ["binding.json", "results.json", "independent-check.json"]:
        artifacts[str(root / name)] = sha256(root / name)
    return dict(
        conditions=dict(randomized=6, fixed=6),
        successes=successes,
        failures=failures,
        artifact_sha256=artifacts,
    )


def freeze_final_plan(index, admission, *, parent):
    """Construct a final seed7 plan after full development evidence replay."""
    admission.require_current()
    return _reconstruct_final_plan(index, binding=admission.binding, parent=parent)


def _reconstruct_final_plan(index, *, binding, parent):
    """Recompute the complete evidence graph without trusting saved verdicts."""
    import torch
    from .payload_checkpoint import actor_signature
    from .payload_retry_checkpoint import transfer_state
    from .payload_solver_models import inspect_models

    if index.get("schema") != "payload-solver-development-index-v1":
        raise ValueError("Explicit development index schema required")
    if Path(parent).is_absolute() or '..' in Path(parent).parts:
        raise ValueError('Portable actual parent reference required')
    roots = index["run_roots"]
    if (
        set(roots) != {"randomized", "fixed"}
        or Path(roots["randomized"]).resolve() == Path(roots["fixed"]).resolve()
    ):
        raise ValueError("Distinct actual development methods required")
    manifests = {
        method: json.loads((Path(root) / "manifest.json").read_text()) for method, root in roots.items()
    }
    common = manifests["randomized"]["common"]
    if (
        common != manifests["fixed"]["common"]
        or common["pilot"]
        or common["actual_binding"] != binding
    ):
        raise ValueError("Both actual development methods need the identical nonpilot recipe")
    if sha256(parent) != common["parent_sha256"]:
        raise ValueError("Actual initialization parent differs")
    initial = transfer_state(torch.load(parent, map_location="cpu", weights_only=False))
    expected_initial = {
        f"initial_{role}_tensor_sha256": actor_signature(initial[role + "_state_dict"])
        for role in ["actor", "critic"]
    }
    for method, manifest in manifests.items():
        if (
            manifest["method"] != method
            or manifest["split"] != "development"
            or not manifest["fresh_optimizer"]
            or manifest["initial_iteration"] != 0
        ):
            raise ValueError("Only fresh correctly labeled actual development runs allowed")
        init_path = Path(roots[method]) / "initialization.pt"
        if sha256(init_path) != manifest["initialization_sha256"]:
            raise ValueError("Actual development initialization differs")
        actual = torch.load(init_path, map_location="cpu", weights_only=False)
        if any(
            actor_signature(actual[role + "_state_dict"]) != expected_initial[f"initial_{role}_tensor_sha256"]
            for role in ["actor", "critic"]
        ):
            raise ValueError("Initial tensors were not the exact fresh model4296 transfer")
    if any(sha256(p) != h for p, h in common["source_binding"].items()):
        raise ValueError("Development recipe source changed before final freeze")
    candidates, artifacts = [], {str(parent): sha256(parent)}
    artifacts.update({str(Path(root) / 'manifest.json'): sha256(Path(root) / 'manifest.json')
                      for root in roots.values()})
    for stage in index["stages"]:
        steps, models = stage["S"], stage["models"]
        infos = inspect_models(models, final=False)
        if any(
            info["common_training_plan"] != common
            or info["training_split"] != "development"
            or info["valid_environment_transitions"] != steps
            for info in infos.values()
        ):
            raise ValueError("Actual rung checkpoint recipe/sample identity differs")
        for method, item in models.items():
            if (
                Path(item["checkpoint"]).resolve()
                != (Path(roots[method]) / f"checkpoint_s{steps}.pt").resolve()
            ):
                raise ValueError("Rung checkpoint is outside the declared actual run")
        exposure = {method: audit_training_prefix(root, steps) for method, root in roots.items()}
        evaluation = replay_development(stage["evaluation"], models, binding)
        candidates.append(dict(S=steps, exposure=exposure, **evaluation))
        artifacts.update(evaluation["artifact_sha256"])
        for method, result in exposure.items():
            artifacts.update({str(Path(roots[method]) / p): h for p, h in result["artifact_sha256"].items()})
            artifacts[models[method]["checkpoint"]] = models[method]["sha256"]
    chosen = select_common_rung(candidates, ladder=common["development_evaluation"]["ladder"])
    frozen = copy.deepcopy(common)
    frozen.update(seed=7, S=chosen, K=chosen // (common["N"] * common["T"]), pilot=False)
    return dict(
        schema=PLAN_SCHEMA,
        status="FINAL_PLAN_FROZEN_BEFORE_BOTH_RUNS",
        common=frozen,
        **expected_initial,
        development_candidates=candidates,
        development_index=copy.deepcopy(index),
        parent_path=str(parent),
        artifact_sha256=artifacts,
        heldout_opened=False,
        final_initialization="Fresh original4296 transfer, never any development checkpoint",
    )


def require_final_plan(plan):
    """Bind independently replayed development records before either final run."""
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "FINAL_PLAN_FROZEN_BEFORE_BOTH_RUNS"
        or plan.get("heldout_opened") is not False
        or not plan.get("artifact_sha256")
    ):
        raise ValueError("Independently frozen final training plan required")
    if not plan.get('development_index') or not plan.get('parent_path'):
        raise ValueError('Complete development replay inputs required before final training')
    if any(sha256(p) != h for p, h in plan["artifact_sha256"].items()):
        raise ValueError("Actual final-plan development evidence changed")
    from .payload_solver_qualification import current_solver_binding

    binding = current_solver_binding()
    reconstructed = _reconstruct_final_plan(plan['development_index'],
        binding=binding, parent=plan['parent_path'])
    if any(plan.get(key) != value for key, value in reconstructed.items()):
        raise ValueError('Actual complete development replay differs from final plan')
    chosen = select_common_rung(
        reconstructed["development_candidates"], ladder=reconstructed["common"]["development_evaluation"]["ladder"]
    )
    common = plan["common"]
    if (
        common["S"] != chosen
        or common["seed"] != 7
        or common["pilot"]
        or common["S"] != common["N"] * common["T"] * common["K"]
    ):
        raise ValueError("Final samples/seed differ from actual common development selection")
    return True
