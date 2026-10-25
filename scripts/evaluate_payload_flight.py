#!/usr/bin/env python
"""Run complete manifest-driven loaded trials, retaining every failed attempt."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from adaptive_drone_patrol.payload_runtime import PayloadFlight
from adaptive_drone_patrol.payload_contract import Payload, SharedSettling, payload_metrics, settled, require_physical_admission
from adaptive_drone_patrol.payload_cases import initial_state, verify_case_manifest
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.objective import RoutingState, schema_for
from adaptive_drone_patrol.planner_ipc import PlannerClient
from adaptive_drone_patrol.util import sha256, json_write, load_config


def run(flight, case, path, targets=None, planner=None, scenario=None, execution_binding_sha256=None):
    path.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    payload = Payload(flight.bank.reference_mass,case['mass_ratio'],case['side_m'])
    initial = flight.reset_case(payload,case['initial'])
    json_write(path/'case.json',case)
    json_write(path/'provenance.json',dict(checkpoint_sha256=flight.checkpoint_sha256,
        parent_cfg_sha256=flight.config_sha256,actor_sha256=flight.actor_sha256,
        execution_binding_sha256=execution_binding_sha256))
    json_write(path/'payload.json',dict(reference_mass_kg=payload.reference_mass_kg,mass_kg=payload.mass,
               effective_mass_kg=flight.bank.package.get_mass().cpu().tolist(),side_m=payload.side_m,
               inertia_kg_m2=flight.bank.drone.get_links_inertia([flight.bank.package.idx-flight.bank.drone.link_start]).cpu().tolist()))
    state = RoutingState(scenario) if scenario else None
    completed,events,decisions = [],[],[]
    step,leg_start = 0,0
    status,error = 'FAILED',None
    rows = path/'samples.jsonl'
    with rows.open('x') as stream:
        stream.write(json.dumps(dict(step=0,time_s=0,initial=True,**initial))+'\n')
        try:
            while len(completed)<len(targets or range(10)):
                if targets is None:
                    action = planner.predict(state)
                    target = scenario.coords[action+1]
                    target_id = scenario.checkpoint_ids[action]
                    snapshot = state.departure(action)
                    decisions.append(dict(step=step,time_s=step*.01,target_id=target_id,action=action,
                                          physical_xyz=flight.sample['drone_xyz'],features=state.observation().tolist(),
                                          mask=state.mask().tolist(),nominal_planner_elapsed_s=state.elapsed))
                else:
                    action = len(completed)
                    target,target_id = targets[action],f'transition-{action}'
                flight.set_target(target)
                dwell = SharedSettling(.01)
                leg_start = step
                # Initial command instant earns no dwell interval.
                dwell.update(step*.01,False,target_id)
                while True:
                    sample = flight.advance()
                    step += 1
                    metrics = payload_metrics(sample,target,payload)
                    unsafe = (sample['contacts'] or metrics['package_floor_m']<=.1 or metrics['drone_floor_m']<=.1 or
                              abs(metrics['drone_roll_deg'])>60 or abs(metrics['drone_pitch_deg'])>60 or
                              metrics['constraint_residual_m']>2e-5)
                    complete = dwell.update(step*.01,bool(settled(metrics) and not unsafe),target_id)
                    row = dict(step=step,time_s=step*.01,target_id=target_id,target=target,
                               dwell_s=dwell.dwell_s,complete=complete,metrics=metrics,**sample)
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                    if unsafe:
                        raise RuntimeError('Contact, clearance, attitude or constraint safety gate')
                    if time.monotonic()-started>600 or step*.01>120:
                        raise RuntimeError('Mission/wall timeout')
                    if complete:
                        completed.append(target_id)
                        events.append(dict(kind='SHARED_SETTLED',step=step,time_s=step*.01,target_id=target_id))
                        if state:
                            state.complete(action,snapshot)
                            if len(completed)==1 and scenario.event:
                                before=state.copy()
                                no_event=planner.predict(before)
                                update=state.apply_event()
                                after=planner.predict(state)
                                events.append(dict(kind='SCENARIO_EVENT',step=step,time_s=step*.01,update=update,
                                                   counterfactual_next=no_event,actual_next=after,
                                                   physical_xyz=sample['drone_xyz']))
                        break
                    if (step-leg_start)*.01>=30:
                        raise RuntimeError('Leg timeout')
            status='SUCCEEDED'
        except Exception as exc:
            error=repr(exc)
            (path/'failure-sample.json').write_text(json.dumps(flight.sample,allow_nan=True)+'\n')
    json_write(path/'events.json',events)
    json_write(path/'decisions.json',decisions)
    result=dict(status=status,error=error,case_id=case['case_id'],completed=completed,steps=step,
                sim_time_s=step*.01,wall_s=time.monotonic()-started,telemetry_sha256=sha256(rows),
                case_sha256=sha256(path/'case.json'),payload_sha256=sha256(path/'payload.json'),
                provenance_sha256=sha256(path/'provenance.json'),
                events_sha256=sha256(path/'events.json'),decisions_sha256=sha256(path/'decisions.json'))
    json_write(path/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',required=True)
    p.add_argument('--mode',choices=['validation','acceptance'],required=True)
    p.add_argument('--cases',required=True)
    p.add_argument('--cases-sha256',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    models=json.loads(Path(a.models).read_text())
    if sha256(models['preflight'])!=models['preflight_sha256']:
        raise ValueError('Physical preflight hash mismatch')
    require_physical_admission(json.loads(Path(models['preflight']).read_text()),
                               load_config('configs/payload_preflight.yaml'),sha256('configs/payload_preflight.yaml'))
    manifest=verify_case_manifest(a.cases,a.cases_sha256,Path.cwd()) if a.mode=='acceptance' else None
    if a.mode=='acceptance' and not models.get('selection_frozen_before_heldout'):
        raise ValueError('Model selection must be frozen before acceptance')
    import genesis as gs
    gs.init(backend=gs.gpu,precision='32',logging_level='warning',seed=7)
    sources={str(p):sha256(p) for p in sorted(Path('src/adaptive_drone_patrol').glob('payload*.py'))}
    sources['scripts/evaluate_payload_flight.py']=sha256(__file__)
    binding=dict(models_sha256=sha256(a.models),cases_sha256=sha256(a.cases),sources=sources,
                 validation_declaration_sha256=sha256('configs/payload_validation.yaml'),
                 preflight_sha256=models['preflight_sha256'],mode=a.mode,started_epoch=time.time(),run_id=out.name)
    json_write(out/'binding.json',binding)
    results=[]
    with PlannerClient([models['planner_python'],'-m','adaptive_drone_patrol.planner_ipc',
                        '--model',models['routing_model'],'--hash',models['routing_sha256']],
                        models['routing_sha256'],schema=schema_for(Scenario.from_dict(load_config('configs/demo_scenarios_dispersed/normal.yaml')))) as planner:
        for method in ('randomized','fixed'):
            item=models['policies'][method]
            flight=PayloadFlight(item['checkpoint'],item['sha256'],models['parent_cfg'],out/f'{method}_scenes')
            if a.mode=='validation':
                cases=[]
                cfg=load_config('configs/payload_validation.yaml')
                for seed in cfg['initial_seeds']:
                    for i,load in enumerate(cfg['loads']):
                        rng=np.random.default_rng(np.random.SeedSequence([seed,720,i]))
                        cases.append(dict(case_id=f'validation-{seed}-{i}',suite='validation',**load,
                                          initial=initial_state(rng,load['side_m'])))
            else:
                cases=manifest['cases']
            for case in cases:
                scenario=None if a.mode=='validation' else Scenario.from_dict(load_config(f"configs/demo_scenarios_dispersed/{case['scenario']}.yaml"))
                result=run(flight,case,out/method/case['case_id'],targets=cfg['targets'] if a.mode=='validation' else None,
                           planner=planner,scenario=scenario,execution_binding_sha256=sha256(out/'binding.json'))
                results.append(dict(method=method,suite=case['suite'],**result))
                json_write(out/'results.json',results)
    json_write(out/'completed.json',dict(trials=len(results),succeeded=sum(r['status']=='SUCCEEDED' for r in results),
               results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':
    main()
