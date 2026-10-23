#!/usr/bin/env python
"""Compare frozen development checkpoints on one identical real pilot case."""

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_cases import initial_state
from adaptive_drone_patrol.payload_solver_flight import SolverAdmission, SolverFlight
from adaptive_drone_patrol.payload_solver_trace import verify_native_sequence
from adaptive_drone_patrol.payload_solver_validation import record_sequence, verify_sequence
from adaptive_drone_patrol.util import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["qualification", "locations", "index", "original", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    if not index or len({item["id"] for item in index}) != len(index):
        raise ValueError("Distinct predeclared development checkpoints required")
    if args.out.exists():
        raise FileExistsError("Preserve every development comparison")
    for item in index:
        if (
            item["method"] not in ["randomized", "fixed"]
            or "/" in item["id"]
            or item["id"] in [".", ".."]
            or sha256(item["checkpoint"]) != item["sha256"]
        ):
            raise ValueError("Exact frozen development checkpoint identity required")
    admission = SolverAdmission.from_receipt(
        json.loads(args.qualification.read_text()), json.loads(args.locations.read_text())
    )
    import torch

    torch.set_num_threads(1)
    case = dict(
        case_id="pilot-83011-0",
        side_m=0.045,
        mass_ratio=0.2,
        initial=initial_state(np.random.default_rng(np.random.SeedSequence([83011, 720, 0])), 0.045),
    )
    targets = [[0, 0, 1], [0.15, 0, 1], [0, 0, 1]]
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "binding.json",
        dict(
            index=index,
            case=case,
            targets=targets,
            actual_binding=admission.binding,
            source_sha256=sha256(__file__),
            scope="Development comparison only; report actual displacement separately from overlapping goal radii",
        ),
    )
    results = []
    for item in index:
        root = args.out / item["id"]
        flight = SolverFlight.from_checkpoint(
            admission=admission,
            checkpoint=item["checkpoint"],
            expected_hash=item["sha256"],
            original=args.original,
            case=case,
            output=root / "flight",
            allow_development=True,
            expected_method=item["method"],
        )
        try:
            execution = record_sequence(
                flight,
                targets,
                root / "sequence",
                dict(
                    evidence_kind="physical",
                    case=case,
                    checkpoint_sha256=item["sha256"],
                    actual_binding=admission.binding,
                    physical_evidence=dict(solver_admission=admission, actual_binding=admission.binding),
                ),
            )
        finally:
            flight.close()
        native = dict(
            native_path=root / "flight/runtime/raw.jsonl.gz",
            properties_path=root / "flight/runtime-properties.json",
            checkpoint=item["checkpoint"],
            expected_hash=item["sha256"],
            case=case,
            binding=admission.binding,
        )
        rows = [json.loads(line) for line in (root / "sequence/samples.jsonl").read_text().splitlines()]
        native_proof = None
        try:
            native_proof = verify_native_sequence(rows, **native)
            proof = verify_sequence(root / "sequence", targets, native=native)
            passed, error = True, None
        except ValueError as exc:
            passed, error = False, str(exc)
            proof = None
        samples = [r["sample"] for r in rows if r["kind"] == "STEP"]
        metric_names = [
            "drone_error_m",
            "drone_speed_m_s",
            "package_speed_m_s",
            "swing_angle_deg",
            "swing_rate_deg_s",
            "package_spin_deg_s",
        ]
        result = dict(
            id=item["id"],
            passed=passed,
            execution=execution,
            replay_error=error,
            native_proof=native_proof,
            proof=proof,
            physical_acceptance=False,
            last500_mean={k: float(np.mean([s["metrics"][k] for s in samples[-500:]])) for k in metric_names}
            if samples
            else {},
        )
        results.append(result)
        write_json(args.out / "results.json", results)
        print(json.dumps(result), flush=True)
    write_json(
        args.out / "completed.json",
        dict(
            trials=len(results),
            passed=sum(r["passed"] for r in results),
            results_sha256=sha256(args.out / "results.json"),
            physical_acceptance=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
