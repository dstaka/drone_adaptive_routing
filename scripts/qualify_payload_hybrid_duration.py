#!/usr/bin/env python3
"""Record seven continuous330s C-only traces on the unchanged qualified reference."""

import argparse
import json
from pathlib import Path

from adaptive_drone_patrol.payload_hybrid_duration import (
    DurationReference,
    duration_cases,
    duration_declaration,
    run_case,
    sources,
    write_json,
)
from adaptive_drone_patrol.util import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output",
        "original",
        "design",
        "reference-rebinding",
        "historical-receipt",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--qualification-locations", type=Path)
    parser.add_argument("--raw-verification", type=Path)
    parser.add_argument("--raw-verification-sha256")
    parser.add_argument(
        "--qualification-root",
        type=Path,
        help="Root for relative stage paths in the historical locations JSON",
    )
    parser.add_argument("--case-id", action="append", choices=[c["case_id"] for c in duration_cases()])
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    locations = json.loads(args.qualification_locations.read_text()) if args.qualification_locations else None
    if args.qualification_root is not None:
        locations = {
            name: str(path if (path := Path(value)).is_absolute() else args.qualification_root / path)
            for name, value in locations.items()
        }
    reference = DurationReference.from_files(
        args.reference_rebinding,
        args.historical_receipt,
        locations,
        raw_verification=args.raw_verification,
        expected_receipt_sha256=args.raw_verification_sha256,
    )
    design = json.loads(args.design.read_text())
    declaration = dict(
        duration_declaration(),
        reference_proof=reference.proof,
        design_sha256=sha256(args.design),
        design=design,
        original=str(args.original.resolve()),
        original_sha256=sha256(args.original),
        source_binding=sources(args.source_root),
    )
    if args.output.exists():
        if not args.case_id or json.loads((args.output / "inputs.json").read_text()) != declaration:
            raise ValueError("Reuse only an identical declared group with explicit unexecuted case IDs")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "inputs.json", declaration)
    if args.declare_only:
        print("PROSPECTIVE_CONTINUOUS_DURATION_ONLY; seven future traces, no learner", flush=True)
        return
    selected = set(args.case_id or [c["case_id"] for c in duration_cases()])
    failed = []
    for case in duration_cases():
        if case["case_id"] not in selected:
            continue
        destination = args.output / case["case_id"]
        if destination.exists():
            raise ValueError("Never overwrite or retry a recorded duration case")
        try:
            result = run_case(
                original=args.original,
                output=destination,
                case=case,
                design=design,
                reference=reference,
                source_root=args.source_root,
            )
            print(
                json.dumps(
                    dict(
                        case_id=case["case_id"],
                        success=result["success"],
                        failure=result["failure"],
                        actual_intervals=result["steps"],
                        raw_audit="PENDING",
                    )
                ),
                flush=True,
            )
            if not result["success"]:
                failed.append(case["case_id"])
        except Exception as exc:
            failed.append(case["case_id"])
            write_json(
                args.output / (case["case_id"] + "-failed-job.json"),
                dict(case_id=case["case_id"], error=repr(exc), raw_retained=True, learning_admitted=False),
            )
            print("DURATION_JOB_FAILED", case["case_id"], repr(exc), flush=True)
    if failed:
        raise SystemExit("Recorded duration failures: " + ", ".join(failed))


if __name__ == "__main__":
    main()
