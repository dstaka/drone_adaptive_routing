#!/usr/bin/env python3
"""One predeclared full development mission, independently checked before media."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_cases import initial_state
from adaptive_drone_patrol.payload_hybrid_flight import HybridFlight, ReferenceAdmission
from adaptive_drone_patrol.payload_recovery_mission import execute_payload_mission
from adaptive_drone_patrol.payload_solver_acceptance import verify_mission
from adaptive_drone_patrol.routing_agent import RoutingAgent
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.util import sha256,load_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-rebinding',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--design',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--routing',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenario',choices=('normal','emergency','cost_shift'),default='normal')
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    seed=manifest['development_video_seeds'][args.scenario]
    case=dict(case_id=f'development-video-{args.scenario}-{seed}',suite='development-video',
        scenario=args.scenario,seed=seed,side_m=.045,mass_ratio=.2,
        initial=initial_state(np.random.default_rng(np.random.SeedSequence([seed,8262])),.045))
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'case.json',case)
    design=json.loads(args.design.read_text())
    admission=ReferenceAdmission.from_rebinding(json.loads(args.reference_rebinding.read_text()))
    planner=RoutingAgent(args.routing,expected_hash='a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef')
    scenario=Scenario.from_dict(load_config(f'configs/demo_scenarios_dispersed/{args.scenario}.yaml'))
    import adaptive_drone_patrol.payload_hybrid_flight as flight_module
    source=Path(flight_module.__file__).resolve().parent
    files=['payload_hybrid_flight.py','payload_hybrid_controller.py','payload_hybrid_runtime.py',
        'payload_hybrid_contract.py','payload_recovery_mission.py','payload_solver_acceptance.py',
        'payload_solver_validation.py','payload_hybrid_native.py','payload_solver_runtime.py']
    write_json(root/'source-binding.json',{f:sha256(source/f) for f in files})
    setup=time.monotonic()
    flight=HybridFlight.create(admission=admission,original=args.original,output=root/'flight',case=case,design=design)
    checkpoint=root/'flight/design.json'
    controller_hash=sha256(checkpoint)
    write_json(root/'setup.json',dict(wall_s=flight.reset_wall_s-setup))
    try:
        result=execute_payload_mission(flight,planner,scenario,root/'mission',
            dict(evidence_kind='physical',case_id=case['case_id'],checkpoint_sha256=controller_hash,
                controller='C-local-lqr-v1',backend='MuJoCo3.15.0/float64/RK4/substeps16/N1',
                actual_binding=admission.binding,routing_model_hash=planner.hash,
                development_label='開発実証・動画優先静定基準 / 頑健性評価前',
                flight_nn='NOT_USED',scenario=args.scenario),
            physical_evidence=dict(hybrid_admission=admission),revision2=True)
    finally:
        flight.close()
    flush_elapsed=time.monotonic()-flight.reset_wall_s
    write_json(root/'final-flush.json',dict(flushed=True,elapsed_wall_s=flush_elapsed))
    print('MISSION',result['status'],result['completed_count'],result['sim_time_s'],
          result['wall_residence_s'],result['termination_code'],flush=True)
    native=dict(native_path=root/'flight/runtime/raw.jsonl.gz',properties_path=root/'flight/runtime-properties.json',
        checkpoint=checkpoint,expected_hash=controller_hash,case=case,binding=admission.binding,
        control_path=root/'flight/control.jsonl.gz',admission=admission)
    checked=verify_mission(root/'mission',planner,native=native,revision2=True)
    write_json(root/'independent-check.json',checked)
    print('INDEPENDENT',checked['success'],checked['physical_acceptance'],flush=True)


if __name__=='__main__':
    main()
