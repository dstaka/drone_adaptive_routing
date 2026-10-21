#!/usr/bin/env python
"""Independent payload settling replay; summaries cannot establish completion."""

import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import numpy as np
from verify_payload_probes import rot, digest


def check_target(row,targets):
    if row['target_id'] not in targets or not np.array_equal(row['effective_target'],targets[row['target_id']]):
        raise ValueError('Command differs from declared target')
    if not np.array_equal(row['target'],targets[row['target_id']]):
        raise ValueError('Target label/command swapped')


def check_hash(path,expected):
    if digest(path)!=expected:
        raise ValueError('Stale evidence binding')


def measures(row,target,side):
    d,p = np.array(row['drone_xyz']),np.array(row['package_xyz'])
    vd,vp = np.array(row['drone_velocity']),np.array(row['package_velocity'])
    wd,wp = np.array(row['drone_angular_velocity']),np.array(row['package_angular_velocity'])
    rd,rp = rot(row['drone_quaternion_wxyz']),rot(row['package_quaternion_wxyz'])
    od,op = rd@[0,0,-.02],rp@[0,0,side/2]
    r=p+op-d-od
    length=np.linalg.norm(r)
    if length<=0 or not all(np.isfinite(v).all() for v in (d,p,vd,vp,wd,wp)):
        raise ValueError('Invalid measured state')
    v=vp+np.cross(wp,op)-vd-np.cross(wd,od)
    angle=math.degrees(math.acos(float(np.clip(-r[2]/length,-1,1))))
    rate=math.degrees(np.linalg.norm(np.cross(r,v))/length**2)
    speed=float(np.linalg.norm(vp))
    floor=float(p[2]-side/2*sum(abs(x) for x in rp[2]))
    roll=math.degrees(math.atan2(rd[2,1],rd[2,2]))
    pitch=math.degrees(math.asin(float(np.clip(-rd[2,0],-1,1))))
    drone_error=float(np.linalg.norm(d-target))
    package_error=float(np.linalg.norm(p-(np.array(target)+[0,0,-.02-.3-side/2])))
    unsafe=bool(row['contacts'] or floor<=.1 or d[2]<=.1 or abs(roll)>60 or abs(pitch)>60 or abs(length-.3)>2e-5)
    valid=bool(not unsafe and drone_error<=.2 and package_error<=.2 and
               np.linalg.norm(vd)<=.3 and speed<=.1 and angle<=5 and rate<=10 and
               math.degrees(np.linalg.norm(wp))<=10)
    return dict(valid=valid,unsafe=unsafe,floor_m=floor,swing_deg=angle,swing_rate_deg_s=rate,
                drone_error_m=drone_error,package_error_m=package_error,package_speed_m_s=speed)


def verify_trial(path,case,targets,planner=None,scenario=None,expected_provenance=None):
    path=Path(path)
    summary=json.loads((path/'summary.json').read_text())
    for name,key in [('samples.jsonl','telemetry_sha256'),('case.json','case_sha256'),
                     ('payload.json','payload_sha256'),('events.json','events_sha256'),('decisions.json','decisions_sha256')]:
        check_hash(path/name,summary[key])
    if json.loads((path/'case.json').read_text())!=case:
        raise ValueError('Trial does not match frozen physical case')
    if expected_provenance is not None:
        check_hash(path/'provenance.json',summary['provenance_sha256'])
        if json.loads((path/'provenance.json').read_text())!=expected_provenance:
            raise ValueError('Trial actor/configuration/batch differs from frozen execution')
    properties=json.loads((path/'payload.json').read_text())
    reference=.02700047753751278
    expected_mass=reference*case['mass_ratio']
    if (abs(properties['reference_mass_kg']-reference)>1e-9 or properties['side_m']!=case['side_m'] or
        abs(properties['mass_kg']-expected_mass)>1e-9 or
        not np.allclose(properties['effective_mass_kg'],expected_mass,rtol=1e-5,atol=1e-10) or
        not np.allclose(np.array(properties['inertia_kg_m2']).reshape(3,3),
                        np.eye(3)*expected_mass*case['side_m']**2/6,rtol=1e-5,atol=1e-12)):
        raise ValueError('Effective payload properties differ from frozen physical condition')
    rows=[json.loads(line) for line in (path/'samples.jsonl').read_text().splitlines()]
    if rows[0]['step']!=0 or not rows[0]['initial'] or len(rows)!=summary['steps']+1:
        raise ValueError('Missing initial/final physical coverage')
    for key in ('drone_xyz','package_xyz','drone_velocity','package_velocity',
                'drone_angular_velocity','package_angular_velocity'):
        if not np.allclose(rows[0][key],case['initial'][key],atol=1e-6,rtol=0):
            raise ValueError('Initial perturbation mismatch')
    for body in ('drone','package'):
        actual=np.array(rows[0][f'{body}_quaternion_wxyz'])
        expected=np.array(case['initial'][f'{body}_quaternion_wxyz'])
        sign=1 if np.dot(actual,expected)>=0 else -1
        if not np.allclose(actual,sign*expected,atol=1e-6,rtol=0):
            raise ValueError('Initial orientation differs from frozen perturbation')
    completed=[]
    expected_events=[]
    expected_decisions=[]
    intervals=0
    previous_valid=False
    current=None
    leg_start=0
    unsafe=False
    metrics=[]
    from adaptive_drone_patrol.objective import RoutingState
    state=RoutingState(scenario) if scenario else None
    for i,row in enumerate(rows[1:],1):
        if row['step']!=i or abs(row['time_s']-i*.01)>1e-8:
            raise ValueError('Missing/duplicate physics sample')
        check_target(row,targets)
        if current!=row['target_id']:
            if current is not None and (not completed or completed[-1]!=current):
                raise ValueError('Target switched before shared settling')
            current=row['target_id']
            intervals=0
            previous_valid=False
            leg_start=i-1
            if state:
                action=planner.predict(state)
                if scenario.checkpoint_ids[action]!=current:
                    raise ValueError('Command differs from frozen learned routing policy')
                expected_decisions.append(dict(step=i-1,time_s=(i-1)*.01,target_id=current,action=action,
                    physical_xyz=rows[i-1]['drone_xyz'],features=state.observation().tolist(),
                    mask=state.mask().tolist(),nominal_planner_elapsed_s=state.elapsed))
                snapshot=state.departure(action)
            elif current!=list(targets)[len(completed)]:
                raise ValueError('Validation transition reordered')
        expected_preclip=(np.array(targets[current])-row['drone_xyz'])/3
        if (not np.allclose(row['relative_position_preclip'],expected_preclip,atol=1e-8) or
            not np.allclose(row['relative_position_clipped'],np.clip(expected_preclip,-1,1),atol=1e-8) or
            np.array(row['actions']).shape!=(4,) or not np.isfinite(row['actions']).all() or
            np.max(np.abs(row['actions']))>1):
            raise ValueError('Observation/motor telemetry differs from actual measured condition')
        previous=rows[i-1]
        for body in ('drone','package'):
            displacement=np.linalg.norm(np.array(row[f'{body}_xyz'])-previous[f'{body}_xyz'])
            limit=.01*max(np.linalg.norm(row[f'{body}_velocity']),np.linalg.norm(previous[f'{body}_velocity']))+.005
            if displacement>limit:
                raise ValueError('Physical discontinuity/hidden reset between consecutive samples')
            q=np.array(row[f'{body}_quaternion_wxyz'])
            previous_q=np.array(previous[f'{body}_quaternion_wxyz'])
            dot=abs(np.dot(q,previous_q)/(np.linalg.norm(q)*np.linalg.norm(previous_q)))
            turn=2*math.acos(float(np.clip(dot,0,1)))
            angular_limit=.01*max(np.linalg.norm(row[f'{body}_angular_velocity']),
                                  np.linalg.norm(previous[f'{body}_angular_velocity']))+.02
            if turn>angular_limit:
                raise ValueError('Unmeasured angular discontinuity/attitude reset')
        m=measures(row,targets[current],case['side_m'])
        metrics.append(m)
        intervals=intervals+1 if m['valid'] and previous_valid else 0
        previous_valid=m['valid']
        reached=intervals>=100
        if bool(row['complete'])!=reached or abs(row['dwell_s']-intervals*.01)>1e-8:
            raise ValueError('Shared settling differs from independent measured replay')
        if reached:
            if current in completed:
                raise ValueError('Duplicate completion')
            completed.append(current)
            expected_events.append(dict(kind='SHARED_SETTLED',step=i,time_s=i*.01,target_id=current))
            if state:
                state.complete(action,snapshot)
                if len(completed)==1 and scenario.event:
                    no_event=planner.predict(state.copy())
                    update=state.apply_event()
                    after=planner.predict(state)
                    expected_events.append(dict(kind='SCENARIO_EVENT',step=i,time_s=i*.01,update=update,
                        counterfactual_next=no_event,actual_next=after,physical_xyz=row['drone_xyz']))
        unsafe|=m['unsafe']
        if m['unsafe'] and i!=len(rows)-1:
            raise ValueError('Continued after safety failure')
        if (i-leg_start)*.01>30 or i*.01>120:
            if summary['status']=='SUCCEEDED':
                raise ValueError('Timeout passed as success')
    if summary['completed']!=completed:
        raise ValueError('Summary completion differs from physical samples')
    success=len(completed)==len(targets) and not unsafe
    if (summary['status']=='SUCCEEDED')!=success:
        raise ValueError('Summary success differs from measured ten-stop completion')
    events=json.loads((path/'events.json').read_text())
    if events!=expected_events:
        raise ValueError('Events differ from reconstructed physical completion and scenario update')
    decisions=json.loads((path/'decisions.json').read_text())
    if len(decisions)!=len(expected_decisions):
        raise ValueError('Decisions differ from actual commanded transitions')
    for actual,expected in zip(decisions,expected_decisions):
        if actual.keys()!=expected.keys():
            raise ValueError('Decision schema mismatch')
        for key,value in expected.items():
            if key in ('features','physical_xyz'):
                if not np.allclose(actual[key],value,rtol=0,atol=1e-7):
                    raise ValueError('Decision feature/pose mismatch')
            elif actual[key]!=value:
                raise ValueError('Decision differs from independently reconstructed routing boundary')
    return dict(success=success,completed=len(completed),steps=len(rows)-1,unsafe=unsafe,
                saturation_samples=sum(any(abs(x)>1 for x in r['relative_position_preclip']) for r in rows[1:]),
                minimum_floor_m=min((m['floor_m'] for m in metrics),default=None),
                max_swing_deg=max((m['swing_deg'] for m in metrics),default=None),
                telemetry_sha256=summary['telemetry_sha256'])


def wilson(success,total):
    if total<=0:
        raise ValueError('Confidence interval requires actual trials')
    z=1.959963984540054
    p=success/total
    denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    delta=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [center-delta,center+delta]


def verify_validation(root,policies,parent_cfg_sha256,candidate_path):
    """Reconstruct every declared pre-acceptance transition with actual actors."""
    from adaptive_drone_patrol.util import load_config
    from adaptive_drone_patrol.payload_cases import initial_state
    root=Path(root)
    declaration=load_config('configs/payload_validation.yaml')
    binding=json.loads((root/'binding.json').read_text())
    check_hash(candidate_path,binding['models_sha256'])
    candidate=json.loads(Path(candidate_path).read_text())
    if candidate['policies']!=policies or binding['mode']!='validation':
        raise ValueError('Validation used another policy pair or execution mode')
    if candidate['created_epoch']>binding['started_epoch']:
        raise ValueError('Validation predates declared candidate checkpoints')
    check_hash('configs/payload_validation.yaml',binding['validation_declaration_sha256'])
    for name,expected in binding['sources'].items():
        check_hash(name,expected)
    completed=json.loads((root/'completed.json').read_text())
    if completed['trials']!=12:
        raise ValueError('Incomplete declared paired validation')
    check_hash(root/'binding.json',completed['binding_sha256'])
    check_hash(root/'results.json',completed['results_sha256'])
    receipt=[]
    for method in ('randomized','fixed'):
        expected_cases=[]
        for seed in declaration['initial_seeds']:
            for i,load in enumerate(declaration['loads']):
                rng=np.random.default_rng(np.random.SeedSequence([seed,720,i]))
                expected_cases.append(dict(case_id=f'validation-{seed}-{i}',suite='validation',**load,
                                           initial=initial_state(rng,load['side_m'])))
        actual={p.parent.name for p in (root/method).glob('*/summary.json')}
        if actual!={c['case_id'] for c in expected_cases}:
            raise ValueError('Validation cases differ from the declared transition suite')
        provenance=dict(checkpoint_sha256=policies[method]['sha256'],parent_cfg_sha256=parent_cfg_sha256,
                        actor_sha256=policies[method]['actor_sha256'],execution_binding_sha256=digest(root/'binding.json'))
        for case in expected_cases:
            targets={f'transition-{i}':t for i,t in enumerate(declaration['targets'])}
            result=verify_trial(root/method/case['case_id'],case,targets,expected_provenance=provenance)
            if not result['success']:
                raise ValueError('Loaded physical transition validation failed; held-out suite must remain closed')
            receipt.append(dict(method=method,case_id=case['case_id'],**result))
    return receipt


def verify_execution(root,models_path,execution_root):
    """Recheck actual frozen artifacts and provenance instead of a status label."""
    import pickle
    from adaptive_drone_patrol.payload_checkpoint import validate_parent_config, actor_signature
    from adaptive_drone_patrol.payload_contract import OBSERVATION_CONTRACT, require_physical_admission
    from adaptive_drone_patrol.util import load_config
    import torch
    root,execution_root=Path(root),Path(execution_root)
    models=json.loads(Path(models_path).read_text())
    if not models.get('selection_frozen_before_heldout'):
        raise ValueError('Models were not selected before held-out evaluation')
    binding=json.loads((root/'binding.json').read_text())
    check_hash(models_path,binding['models_sha256'])
    if binding['started_epoch']<models['created_epoch']:
        raise ValueError('Evaluation started before model-selection freeze')
    if binding['mode']!='acceptance' or binding['preflight_sha256']!=models['preflight_sha256']:
        raise ValueError('Execution profile/preflight differs from frozen selection')
    for path,expected in binding['sources'].items():
        check_hash(path,expected)
        if expected!=models['source_binding'].get(path):
            raise ValueError('Execution source differs from model freeze')
    for path,expected in models['source_binding'].items():
        check_hash(path,expected)
    check_hash('configs/payload_validation.yaml',binding['validation_declaration_sha256'])
    check_hash(execution_root/models['parent_cfg'],models['parent_cfg_sha256'])
    validate_parent_config(pickle.load(open(execution_root/models['parent_cfg'],'rb')))
    training_manifests=[]
    for method in ('randomized','fixed'):
        policy=models['policies'][method]
        path=execution_root/policy['checkpoint']
        check_hash(path,policy['sha256'])
        if path.stat().st_size!=policy['bytes']:
            raise ValueError('Actual checkpoint size differs from freeze')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if actor_signature(checkpoint['actor_state_dict'])!=policy['actor_sha256']:
            raise ValueError('Actual actor tensors differ from frozen signature')
        if checkpoint['infos']['observation_contract']!=OBSERVATION_CONTRACT:
            raise ValueError('Actual checkpoint observation contract mismatch')
        check_hash(execution_root/policy['training_manifest'],policy['training_manifest_sha256'])
        training=json.loads((execution_root/policy['training_manifest']).read_text())
        training_manifests.append(training)
        check_hash(execution_root/policy['training_completed'],policy['training_completed_sha256'])
        training_completed=json.loads((execution_root/policy['training_completed']).read_text())
        if training_completed['checkpoint_sha256']!=policy['sha256'] or training_completed['actual_environment_steps']!=models['matched_environment_steps']:
            raise ValueError('Training completion differs from actual model/budget')
        if training['method']!=method or training['preflight_sha256']!=models['preflight_sha256']:
            raise ValueError('Policy labels differ from actual training condition')
    from adaptive_drone_patrol.payload_checkpoint import validate_matched_manifests
    validate_matched_manifests(*training_manifests)
    for path,expected in models['validation_binding'].items():
        check_hash(execution_root/path,expected)
    fresh_validation=verify_validation(execution_root/models['validation_root'],models['policies'],
        models['parent_cfg_sha256'],execution_root/models['validation_models'])
    if fresh_validation!=models['validation_receipt']:
        raise ValueError('Stale pre-acceptance transition receipt')
    preflight_path=execution_root/models['preflight']
    check_hash(preflight_path,models['preflight_sha256'])
    preflight=json.loads(preflight_path.read_text())
    require_physical_admission(preflight,load_config('configs/payload_preflight.yaml'),
                               digest('configs/payload_preflight.yaml'))
    for path,expected in preflight['source_binding'].items():
        check_hash(path,expected)
    check_hash('configs/payload_preflight.yaml',preflight['thresholds_sha256'])
    from verify_payload_probes import verify
    fresh=verify(execution_root/models['preflight_probes'],load_config('configs/payload_preflight.yaml'))
    if fresh['checks']!=preflight['checks'] or fresh['timestep_comparisons']!=preflight['timestep_comparisons']:
        raise ValueError('Stale physical diagnostic receipt')
    for name,expected in preflight['bank_binding'].items():
        check_hash(execution_root/models['preflight_bank']/name,expected)
    completed=json.loads((root/'completed.json').read_text())
    check_hash(root/'binding.json',completed['binding_sha256'])
    check_hash(root/'results.json',completed['results_sha256'])
    if completed['trials']!=254:
        raise ValueError('Incomplete paired trial denominator')
    if binding['cases_sha256']!=models['cases_sha256']:
        raise ValueError('Evaluation cases differ from frozen selection')
    return models,binding


def main():
    from adaptive_drone_patrol.util import load_config
    from adaptive_drone_patrol.scenario import Scenario
    from adaptive_drone_patrol.routing_agent import RoutingAgent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--cases',required=True)
    p.add_argument('--cases-sha256',required=True)
    p.add_argument('--routing-model',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--models',required=True)
    p.add_argument('--execution-root',default='.')
    a=p.parse_args()
    out=Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    root=Path(a.root)
    check_hash(a.cases,a.cases_sha256)
    from adaptive_drone_patrol.payload_cases import verify_case_manifest
    manifest=verify_case_manifest(a.cases,a.cases_sha256,Path.cwd())
    cases=manifest['cases']
    if len(cases)!=127 or len({c['case_id'] for c in cases})!=127 or Counter(c['suite'] for c in cases)!=dict(basic=9,endpoints=18,heldout=100):
        raise ValueError('Invalid declared suite denominator')
    models,binding=verify_execution(root,a.models,a.execution_root)
    if binding['cases_sha256']!=a.cases_sha256:
        raise ValueError('Provided cases differ from actual execution')
    planner=RoutingAgent(a.routing_model,'a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef')
    results=[]
    for method in ('randomized','fixed'):
        actual={p.parent.name for p in (root/method).glob('*/summary.json')}
        if actual!={c['case_id'] for c in cases}:
            raise ValueError('Declared 127-case denominator differs from actual trials')
        for c in cases:
            scenario=Scenario.from_dict(load_config(f"configs/demo_scenarios_dispersed/{c['scenario']}.yaml"))
            targets=dict(zip(scenario.checkpoint_ids,scenario.coords[1:]))
            provenance=dict(checkpoint_sha256=models['policies'][method]['sha256'],
                            parent_cfg_sha256=models['parent_cfg_sha256'],actor_sha256=models['policies'][method]['actor_sha256'],
                            execution_binding_sha256=digest(root/'binding.json'))
            r=verify_trial(root/method/c['case_id'],c,targets,planner,scenario,provenance)
            results.append(dict(method=method,case_id=c['case_id'],suite=c['suite'],side_m=c['side_m'],
                                mass_bin='[.10,.20)' if c['mass_ratio']<.2 else '[.20,.30]',**r))
    counts=defaultdict(Counter)
    for r in results:
        counts[r['method']][r['suite']]+=int(r['success'])
    gates={method:dict(basic_9_of_9=counts[method]['basic']==9,
                        endpoints_18_of_18=counts[method]['endpoints']==18,
                        heldout_95_of_100=counts[method]['heldout']>=95,
                        successes=dict(counts[method]),heldout_wilson_95=wilson(counts[method]['heldout'],100))
           for method in ('randomized','fixed')}
    receipt=dict(status='PASS' if all(gates['randomized'][k] for k in ('basic_9_of_9','endpoints_18_of_18','heldout_95_of_100')) else 'FAIL',
                 actual_trials=len(results),denominators=dict(basic=9,endpoints=18,heldout=100),gates=gates,
                 checker_sha256=digest(__file__),cases_sha256=digest(a.cases),trials=results,
                 models_sha256=digest(a.models),execution_binding_sha256=digest(root/'binding.json'),
                 confidence_method='Wilson score 95%; empirical threshold is not a population reliability proof')
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],gates)


if __name__=='__main__':
    main()
