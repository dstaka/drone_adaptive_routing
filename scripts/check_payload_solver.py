#!/usr/bin/env python
"""Independently replay full actual qualification; never trust a status label."""

import argparse
from pathlib import Path
from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_solver_qualification import issue_qualification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ["passive", "powered", "refinement", "motor", "reset", "initial-reset", "engine-provenance"]:
        parser.add_argument("--" + stage, required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve qualification attempts")
    locations = {
        name: str(getattr(args, name))
        for name in [
            "passive",
            "powered",
            "refinement",
            "motor",
            "reset",
            "initial_reset",
            "engine_provenance",
        ]
    }
    proof = issue_qualification(locations)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "locations.json", locations)
    write_json(args.out / "qualification.json", proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
