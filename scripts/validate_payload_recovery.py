#!/usr/bin/env python
"""Evaluate actual continuous policy targets after complete raw admission.

Pilot/development records cannot establish final validation or held-out
acceptance. Validation uses the unchanged prescribed12 conditions/120 targets.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_cases import initial_state
from adaptive_drone_patrol.payload_solver_flight import SolverAdmission, SolverFlight
from adaptive_drone_patrol.payload_solver_models import inspect_models
from adaptive_drone_patrol.payload_solver_validation import (
    record_sequence,
    verify_sequence,
    verify_initial_sample,
)
from adaptive_drone_patrol.util import load_config, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["qualification", "locations", "models", "original", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=["pilot", "development", "validation"], required=True)
    parser.add_argument("--training-plan", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve evaluation attempts")
    models = json.loads(args.models.read_text())
    if (
        not models
        or set(models) - {"randomized", "fixed"}
        or (args.mode == "validation" and set(models) != {"randomized", "fixed"})
    ):
        raise ValueError("Final validation requires both unchanged final policy artifacts")
    admission = SolverAdmission.from_receipt(
        json.loads(args.qualification.read_text()), json.loads(args.locations.read_text())
    )
    plan = json.loads(args.training_plan.read_text()) if args.training_plan else None
    inspect_models(models, final=args.mode == "validation", plan=plan)
    if (
        sha256("configs/payload_validation.yaml")
        != "5820da816e284e34c0f5586dcee2721414b097dfba07406bb832b18066e85490"
    ):
        raise ValueError("Prescribed validation12/120 declaration changed")
    cfg = load_config("configs/payload_validation.yaml")
    targets = cfg["targets"]
    if args.mode == "pilot":
        targets = [[0.0, 0.0, 1.0], [0.15, 0.0, 1.0], [0.0, 0.0, 1.0]]
        draws = [(83011, 0, dict(side_m=0.045, mass_ratio=0.2))]
    else:
        seeds = cfg["initial_seeds"] if args.mode == "validation" else [83021, 83022]
        draws = [(seed, i, load) for seed in seeds for i, load in enumerate(cfg["loads"])]
    cases = [
        dict(
            case_id=f"{args.mode}-{seed}-{i}",
            **load,
            initial=initial_state(
                np.random.default_rng(np.random.SeedSequence([seed, 720, i])), load["side_m"]
            ),
        )
        for seed, i, load in draws
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    source_paths = [
        Path("scripts/validate_payload_recovery.py"),
        *Path("src/adaptive_drone_patrol").glob("payload_solver_*.py"),
        Path("configs/payload_validation.yaml"),
    ]
    binding = dict(
        mode=args.mode,
        models=models,
        qualification_sha256=sha256(args.qualification),
        cases=cases,
        targets=targets,
        actual_binding=admission.binding,
        source_binding={str(p): sha256(p) for p in source_paths},
        scope="Only final validation can qualify12/120; pilot/development cannot select by held-out cases",
    )
    write_json(args.out / "binding.json", binding)
    results = []
    for method, item in models.items():
        for case in cases:
            if any(sha256(p) != h for p, h in binding["source_binding"].items()):
                raise ValueError("Evaluation source changed")
            root = args.out / method / case["case_id"]
            flight = SolverFlight.from_checkpoint(
                admission=admission,
                checkpoint=item["checkpoint"],
                expected_hash=item["sha256"],
                original=args.original,
                case=case,
                output=root / "flight",
                allow_development=args.mode != "validation",
                expected_method=method,
            )
            verify_initial_sample(flight.sample(), case["initial"])
            write_json(root / "case.json", case)
            try:
                execution = record_sequence(
                    flight,
                    targets,
                    root / "sequence",
                    dict(
                        evidence_kind="physical",
                        method=method,
                        case=case,
                        checkpoint_sha256=item["sha256"],
                        actual_binding=admission.binding,
                        binding_sha256=sha256(args.out / "binding.json"),
                        physical_evidence=dict(solver_admission=admission, actual_binding=admission.binding),
                    ),
                )
            finally:
                flight.close()
            try:
                proof = verify_sequence(
                    root / "sequence",
                    targets,
                    native=dict(
                        native_path=root / "flight/runtime/raw.jsonl.gz",
                        properties_path=root / "flight/runtime-properties.json",
                        checkpoint=item["checkpoint"],
                        expected_hash=item["sha256"],
                        case=case,
                        binding=admission.binding,
                    ),
                )
                result = dict(passed=True, proof=proof)
            except ValueError as exc:
                result = dict(passed=False, replay_error=repr(exc))
            result.update(method=method, case_id=case["case_id"], execution=execution)
            results.append(result)
            write_json(args.out / "results.json", results)
            print(method, case["case_id"], result["passed"], len(execution["completed"]), flush=True)
    proof = dict(
        mode=args.mode,
        conditions=len(results),
        prescribed_targets=len(results) * len(targets),
        completed_targets=sum(len(r["execution"]["completed"]) for r in results),
        passed=all(r["passed"] for r in results),
        results_sha256=sha256(args.out / "results.json"),
        binding_sha256=sha256(args.out / "binding.json"),
        physical_acceptance=False,
    )
    if args.mode == "validation" and (proof["conditions"] != 12 or proof["completed_targets"] != 120):
        proof["passed"] = False
    write_json(args.out / "independent-check.json", proof)
    return 0 if proof["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
