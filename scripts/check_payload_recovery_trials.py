#!/usr/bin/env python
"""Independently reconstruct all254 original trials and report fixed denominators."""

import argparse
import json
from pathlib import Path

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_solver_flight import SolverAdmission
from adaptive_drone_patrol.payload_solver_trials import recheck_paired_trials
from adaptive_drone_patrol.payload_solver_freeze import ROUTING_SHA
from adaptive_drone_patrol.planner_ipc import PlannerClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['qualification', 'locations', 'freeze', 'trials', 'routing-python', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve independent checker attempts')
    freeze = json.loads(args.freeze.read_text())
    admission = SolverAdmission.from_receipt(json.loads(args.qualification.read_text()),
                                           json.loads(args.locations.read_text()))
    router = freeze['routing_model']
    schema = json.loads(Path(router).with_suffix('.schema.json').read_text())
    command = [str(args.routing_python), '-m', 'adaptive_drone_patrol.planner_ipc',
               '--model', router, '--hash', ROUTING_SHA]
    with PlannerClient(command, ROUTING_SHA, schema=schema) as planner:
        proof = recheck_paired_trials(freeze_path=args.freeze, output=args.trials,
                                     admission=admission, planner=planner)
    write_json(args.out, proof)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
