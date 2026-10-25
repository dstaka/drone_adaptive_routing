#!/usr/bin/env python
"""Capture all three accepted nominal randomizedseed1 missions as full1x media."""

import argparse
import json
import os
from pathlib import Path



def main():
    # MuJoCo binds its GL backend at import time, including during raw audits.
    # Set the camera-only process environment before importing any flight modules.
    for _key, _value in dict(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                             LIBGL_ALWAYS_SOFTWARE='true', GALLIUM_DRIVER='llvmpipe').items():
        os.environ[_key] = _value

    from adaptive_drone_patrol.payload_solver_flight import SolverAdmission
    from adaptive_drone_patrol.payload_solver_capture import capture_all_cases
    from adaptive_drone_patrol.payload_solver_freeze import ROUTING_SHA
    from adaptive_drone_patrol.planner_ipc import PlannerClient
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['qualification', 'locations', 'freeze', 'trials', 'routing-python', 'font', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve every capture attempt')
    freeze = json.loads(args.freeze.read_text())
    admission = SolverAdmission.from_receipt(json.loads(args.qualification.read_text()),
                                           json.loads(args.locations.read_text()))
    router = freeze['routing_model']
    schema = json.loads(Path(router).with_suffix('.schema.json').read_text())
    command = [str(args.routing_python), '-m', 'adaptive_drone_patrol.planner_ipc',
               '--model', router, '--hash', ROUTING_SHA]
    with PlannerClient(command, ROUTING_SHA, schema=schema) as planner:
        capture_all_cases(freeze_path=args.freeze, trials=args.trials, admission=admission,
                          planner=planner, font=args.font, output=args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
