#!/usr/bin/env python
"""Validate fixed physical trials and regenerate same-boundary exact explanations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from adaptive_drone_patrol.evidence import validate_run
from adaptive_drone_patrol.util import unique_dir, json_write, sha256
from adaptive_drone_patrol.objective import RoutingState
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.analytical import snapshot, content_hash
from adaptive_drone_patrol.physical_evidence import physical_explanation, verify_physical_explanation, verify_event_response, verify_frozen_inputs, verify_media_index
from verify_analytical_costs import recompute


def verify_trial(run, agent, require_media=True):
    summary = validate_run(run, require_media=require_media)
    if summary['evidence_kind'] != 'physical':
        raise ValueError('CPU/mock evidence cannot pass physical acceptance')
    s = Scenario.from_dict(json.loads((run/'scenario.json').read_text()))
    if summary['routing_model_hash']!=agent.hash:
        raise ValueError('Logged routing policy differs from loaded verification checkpoint')
    state = RoutingState(s)
    rows={int(r['simulation_step']):r for r in csv.DictReader((run/'telemetry.csv').open())}
    initial=json.loads((run/'initial_state.json').read_text())['xyz']
    decisions=[json.loads(x) for x in (run/'decisions.jsonl').read_text().splitlines()]
    explanations=[]
    events=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
    verify_event_response(events,decisions)
    expected_updates = int(s.event is not None)
    if sum(e['type'] in ['PRIORITY_UPDATED','COST_UPDATED'] for e in events) != expected_updates:
        raise ValueError('Configured event was not delivered exactly once')
    for decision in decisions:
        if decision['policy_hash'] != agent.hash:
            raise ValueError('Physical decision policy hash mismatch')
        if (decision['boundary_snapshot'] != snapshot(state)
                or decision['boundary_hash'] != content_hash(snapshot(state))
                or not np.allclose(decision['feature_vector'],state.observation(),atol=1e-7)):
            raise ValueError('Actual boundary state cannot be replayed')
        step=decision['simulation_step']
        pose=initial if step==0 else [float(rows[step][k]) for k in ['x','y','z']]
        if not np.allclose(decision['physical_pose'],pose,atol=1e-9):
            raise ValueError('Boundary pose is not from same-run physical telemetry')
        value=physical_explanation(decision,agent.predict)
        verify_physical_explanation(value,decision)
        for order in [value['policy_rollout'],value['exact'],*value['forced_first']]:
            costs=recompute(value['snapshot'],order['order'])
            if any(abs(costs[k]-order[k])>value['equality_tolerance'] for k in costs):
                raise ValueError('Independent physical-boundary objective replay failed')
        explanations.append(value)
        # Advance only through verified physical completion events, never predicted suffixes.
        action=decision['selected_action']
        if s.checkpoint_ids[action] in summary['completed_order']:
            state.complete(action,state.departure(action))
            if state.visited.sum()==1 and state.scenario.event:
                state.apply_event()
    if summary['status']=='SUCCEEDED' and not state.terminal:
        raise ValueError('Physical completion does not match decision sequence')
    if require_media:
        frames=list(csv.DictReader((run/'frames.csv').open()))
        if int(frames[0]['simulation_step'])!=0 or int(frames[-1]['simulation_step'])!=len(rows):
            raise ValueError('Missing initial/final recorded physical state')
        import subprocess
        subprocess.run(['ffmpeg','-v','error','-i',str(run/'raw.mp4'),'-f','null','-'],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                         '-show_streams','-show_frames','-show_entries',
                         'stream=width,height:frame=best_effort_timestamp_time','-of','json',str(run/'raw.mp4')]))
        verify_media_index(frames,probe)
    return summary,explanations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--model-hash',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--font')
    p.add_argument('--annotate-seed',type=int,default=1)
    a=p.parse_args()
    from adaptive_drone_patrol.routing_agent import RoutingAgent
    agent=RoutingAgent(a.model,a.model_hash)
    out=unique_dir(a.out)
    runs=sorted(Path(a.root).glob('*/mission_summary.json'))
    summaries,explanations,inputs=[],[],[]
    for path in runs:
        summary,values=verify_trial(path.parent,agent)
        inputs.append({'summary':summary,'scenario':json.loads((path.parent/'scenario.json').read_text()),
                       'scene':json.loads((path.parent/'scene_manifest.json').read_text()),
                       'initial_pose':json.loads((path.parent/'initial_state.json').read_text())})
        summaries.append(summary)
        explanations.extend(values)
        if a.font and summary['execution_seed']==a.annotate_seed:
            from adaptive_drone_patrol.annotations import annotate_run
            annotate_run(path.parent,out/f'{summary["run_id"]}.annotated.mp4',a.font,values)
    verify_frozen_inputs(inputs,a.model_hash)
    pairs={(r['scenario_id'],r['execution_seed']) for r in summaries}
    expected={(kind,seed) for kind in ['normal','emergency','cost_shift'] for seed in [1,2,3]}
    if len(summaries)!=9 or pairs!=expected:
        raise ValueError('Fixed acceptance denominator requires exactly nine declared trials')
    for key in ['flight_checkpoint_hash','flight_config_hash','routing_model_hash','flight_acceptance_config_hash']:
        if len({r[key] for r in summaries})!=1:
            raise ValueError(f'Comparative suite changed frozen {key}')
    qualifies=[{'run_id':v['run_id'],'decision_id':v['decision_id'],'L':v['selected_L'],'H':v['selected_H'],
                'physical_reference':v['physical_reference']} for v in explanations if v['nearer_lower_priority_gate']]
    json_write(out/'decision_explanations.json',{'schema_version':2,'evidence_kind':'physical','decisions':explanations})
    result={'schema_version':1,'evidence_kind':'physical','missions_attempted':len(summaries),
            'missions_succeeded':sum(r['status']=='SUCCEEDED' for r in summaries),
            'inspections_completed':sum(r['completed_count'] for r in summaries),
            'by_scenario':{kind:{'succeeded':sum(r['status']=='SUCCEEDED' for r in summaries if r['scenario_id']==kind),
                          'attempted':3} for kind in ['normal','emergency','cost_shift']},
            'nearer_lower_priority_decisions':qualifies,
            'policy_hash':a.model_hash,'explanations_sha256':sha256(out/'decision_explanations.json'),
            'runs':summaries}
    result['status']='PASS' if result['missions_succeeded']==9 and result['inspections_completed']==90 and qualifies else 'INCOMPLETE'
    json_write(out/'acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','nearer_lower_priority_decisions']}))
    raise SystemExit(0 if result['status']=='PASS' else 2)


if __name__=='__main__':
    main()
