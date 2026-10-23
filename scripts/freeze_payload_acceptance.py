#!/usr/bin/env python
"""Freeze real final models, validation12/120, original cases and camera sources."""

import argparse
import json
from pathlib import Path

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_solver_flight import SolverAdmission
from adaptive_drone_patrol.payload_solver_freeze import freeze_acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['qualification', 'locations', 'models', 'plan', 'validation', 'cases', 'routing', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--camera-assets', nargs=2, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve every pre-heldout freeze')
    admission = SolverAdmission.from_receipt(json.loads(args.qualification.read_text()),
                                           json.loads(args.locations.read_text()))
    freeze = freeze_acceptance(models=json.loads(args.models.read_text()), plan_path=args.plan,
        validation=args.validation, cases_path=args.cases, routing=args.routing,
        camera_assets=args.camera_assets, qualification=args.qualification, admission=admission)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, freeze)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
