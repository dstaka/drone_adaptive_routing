"""Freeze actual final models, real validation and all acceptance/media sources."""

import json
from pathlib import Path
import runpy

import numpy as np

from .payload_cases import initial_state
from .payload_solver_models import inspect_models
from .payload_solver_training_plan import require_final_plan
from .payload_solver_validation import verify_sequence
from .util import sha256, load_config

FREEZE_SCHEMA = "payload-solver-acceptance-freeze-v1"
ROUTING_SHA = "a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef"
VALIDATION_SHA = "5820da816e284e34c0f5586dcee2721414b097dfba07406bb832b18066e85490"


def verify_file_snapshot(snapshot):
    """Hash exact portable explicit files; never trust a stored status instead."""
    if not snapshot:
        raise ValueError("Complete relative artifact snapshot required")
    for name, digest in snapshot.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Freeze artifact labels must be relative")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("Frozen artifact changed or disappeared: " + name)
    return True


def replay_final_validation(root, models, plan, binding):
    """Reconstruct all12 actual conditions and120 physical target transitions."""
    root = Path(root)
    if sha256("configs/payload_validation.yaml") != VALIDATION_SHA:
        raise ValueError("Prescribed validation declaration changed")
    training = {}
    inspect_models(models, final=True, plan=plan, training_proofs=training)
    declaration = json.loads((root / "binding.json").read_text())
    if (
        declaration["mode"] != "validation"
        or declaration["models"] != models
        or declaration["actual_binding"] != binding
    ):
        raise ValueError("Actual final validation models/physics differ")
    verify_file_snapshot(declaration["source_binding"])
    cfg = load_config("configs/payload_validation.yaml")
    cases = [
        dict(
            case_id=f"validation-{seed}-{i}",
            **load,
            initial=initial_state(
                np.random.default_rng(np.random.SeedSequence([seed, 720, i])), load["side_m"]
            ),
        )
        for seed in cfg["initial_seeds"]
        for i, load in enumerate(cfg["loads"])
    ]
    if (
        len(cases) != 6
        or len(cfg["targets"]) != 10
        or declaration["cases"] != cases
        or declaration["targets"] != cfg["targets"]
    ):
        raise ValueError("Actual prescribed validation12/120 cases/targets differ")
    proofs, artifacts = [], {}
    for result in training.values():
        artifacts.update(result["artifact_sha256"])
    for method in ["randomized", "fixed"]:
        item = models[method]
        for case in cases:
            path = root / method / case["case_id"]
            if json.loads((path / "case.json").read_text()) != case:
                raise ValueError("Recorded actual final case differs")
            proof = verify_sequence(
                path / "sequence",
                cfg["targets"],
                native=dict(
                    native_path=path / "flight/runtime/raw.jsonl.gz",
                    properties_path=path / "flight/runtime-properties.json",
                    checkpoint=item["checkpoint"],
                    expected_hash=item["sha256"],
                    case=case,
                    binding=binding,
                ),
            )
            if proof['evidence_kind'] != 'physical' or not proof['native_proof']:
                raise ValueError('Actual physical native validation proof required')
            proofs.append(dict(method=method, case_id=case["case_id"], **proof))
            artifacts.update({str(p): sha256(p) for p in path.rglob("*") if p.is_file()})
    if len(proofs) != 12 or sum(p["completed_targets"] for p in proofs) != 120:
        raise ValueError("Full real validation12/120 required")
    artifacts.update(
        {str(root / n): sha256(root / n) for n in ["binding.json", "results.json", "independent-check.json"]}
    )
    return dict(conditions=12, completed_targets=120, proofs=proofs,
                final_training_proofs=training, artifact_sha256=artifacts)


def freeze_acceptance(
    *, models, plan_path, validation, cases_path, routing, camera_assets, qualification, admission
):
    """Create the complete immutable pre-heldout record; no trials run here."""
    from .payload_solver_heldout import require_unopened_stream

    require_unopened_stream()
    admission.require_current()
    plan = json.loads(Path(plan_path).read_text())
    require_final_plan(plan)
    proof = replay_final_validation(validation, models, plan, admission.binding)
    inherited = runpy.run_path("scripts/prepare_payload_retry_cases.py")["verify_original_cases"](
        cases_path, Path.cwd()
    )
    if len(inherited["cases"]) != 127:
        raise ValueError("Untouched original127 conditions required")
    if sha256(routing) != ROUTING_SHA:
        raise ValueError("Actual adopted seed44 routing model differs")
    required = [
        "scripts/evaluate_payload_recovery.py",
        "scripts/check_payload_recovery_trials.py",
        "scripts/capture_payload_recovery.py",
        "configs/payload_solver_camera.json",
    ]
    if any(not Path(p).is_file() for p in required):
        raise ValueError("Complete evaluation/checker/camera source required before heldout")
    files = [
        *Path("src/adaptive_drone_patrol").glob("*.py"),
        *Path("scripts").glob("*.py"),
        *Path("requirements").glob("*.lock.txt"),
        Path("configs/payload_validation.yaml"),
        *Path("configs/demo_scenarios_dispersed").glob("*.yaml"),
        Path("configs/payload_solver_camera.json"),
        Path(plan_path),
        Path(cases_path),
        Path(routing),
        Path(routing).with_suffix(".schema.json"),
        Path(qualification),
        *map(Path, camera_assets),
    ]
    snapshot = {str(p): sha256(p) for p in files}
    snapshot.update(proof["artifact_sha256"])
    for item in models.values():
        snapshot[item["checkpoint"]] = item["sha256"]
    verify_file_snapshot(snapshot)
    result = dict(
        schema=FREEZE_SCHEMA,
        status="FROZEN_BEFORE_HELDOUT",
        models=models,
        training_plan=plan_path,
        validation=str(validation),
        cases=str(cases_path),
        routing_model=str(routing),
        camera_assets=list(map(str, camera_assets)),
        qualification=str(qualification),
        actual_binding=admission.binding,
        qualified_raw_manifest_sha256=admission.proof["raw_manifest_sha256"],
        validation_proof=proof,
        artifact_sha256=snapshot,
        heldout_opened=False,
        required_trials=254,
        required_randomized=dict(basic=9, endpoints=18, heldout=95),
        scope="Both fixed127 and randomized127 fresh trials; no method-specific tuning or selective rescue",
    )
    validate_freeze_references(result)
    return result


def require_acceptance_freeze(freeze, admission):
    """Rehash all frozen inputs and revalidate real transitions before opening."""
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("status") != "FROZEN_BEFORE_HELDOUT"
        or freeze.get("heldout_opened") is not False
        or freeze.get("required_trials") != 254
    ):
        raise ValueError("Complete original pre-heldout freeze required")
    validate_freeze_references(freeze)
    verify_file_snapshot(freeze["artifact_sha256"])
    admission.require_current()
    if (
        freeze["actual_binding"] != admission.binding
        or freeze["qualified_raw_manifest_sha256"] != admission.proof["raw_manifest_sha256"]
    ):
        raise ValueError("Actual heldout physical binding differs")
    plan = json.loads(Path(freeze["training_plan"]).read_text())
    require_final_plan(plan)
    replay = replay_final_validation(freeze["validation"], freeze["models"], plan, admission.binding)
    if replay != freeze["validation_proof"]:
        raise ValueError("Replayed final validation differs from pre-heldout freeze")
    return runpy.run_path("scripts/prepare_payload_retry_cases.py")["verify_original_cases"](
        freeze["cases"], Path.cwd()
    )


def validate_freeze_references(freeze):
    """Require every consumed pointer and prescribed source to be snapshotted."""
    snapshot = freeze['artifact_sha256']
    routing = freeze['routing_model']
    if routing not in snapshot or snapshot[routing] != ROUTING_SHA:
        raise ValueError('Actual routing reference must match prescribed snapshot')
    required = [
        freeze['training_plan'], freeze['cases'], freeze['qualification'],
        str(Path(routing).with_suffix('.schema.json')),
        'configs/payload_solver_camera.json', 'configs/payload_validation.yaml',
        'scripts/evaluate_payload_recovery.py', 'scripts/check_payload_recovery_trials.py',
        'scripts/capture_payload_recovery.py',
        *map(str, Path('src/adaptive_drone_patrol').glob('*.py')),
        *map(str, Path('requirements').glob('*.lock.txt')),
        *map(str, Path('configs/demo_scenarios_dispersed').glob('*.yaml')),
        *freeze['camera_assets'],
    ]
    if any(name not in snapshot for name in required):
        raise ValueError('Required source/camera/input reference missing from snapshot')
    if (len(freeze['camera_assets']) != 2
            or sorted(snapshot[p] for p in freeze['camera_assets']) != sorted([
                'c2f3b4d463500a2ddcd3849cded1fceeb9fd6d1c32e6cbecd568453ba50fc68f',
                '1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9'])):
        raise ValueError('Actual Japanese font and original license must be frozen')
    for item in freeze['models'].values():
        if snapshot.get(item['checkpoint']) != item['sha256']:
            raise ValueError('Actual model reference escaped snapshot')
    return True
