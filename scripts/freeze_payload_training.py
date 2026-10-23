#!/usr/bin/env python
"""Freeze equal fresh final training from complete real development evidence."""

import argparse
import json
from pathlib import Path

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_solver_flight import SolverAdmission
from adaptive_drone_patrol.payload_solver_training_plan import freeze_final_plan
from adaptive_drone_patrol.util import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["qualification", "locations", "index", "parent", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve all frozen plans")
    index = json.loads(args.index.read_text())
    admission = SolverAdmission.from_receipt(
        json.loads(args.qualification.read_text()), json.loads(args.locations.read_text())
    )
    plan = freeze_final_plan(index, admission, parent=args.parent)
    plan.update(
        development_index_sha256=sha256(args.index),
        qualification_sha256=sha256(args.qualification),
        qualified_raw_manifest_sha256=admission.proof["raw_manifest_sha256"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
