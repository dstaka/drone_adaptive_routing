"""Offline exact comparisons bound to actual physical mission decisions."""
import copy
import numpy as np
from .analytical import restore, explain_decision, verify_explanation, content_hash


def physical_explanation(decision, chooser):
    if decision.get('evidence_kind') != 'physical':
        raise ValueError('Physical explanation requires physical decision provenance')
    state = restore(decision['boundary_snapshot'])
    result = explain_decision(state, chooser, decision['policy_hash'], decision['run_id'], decision['decision_id'])
    if (result['policy_action'] != decision['selected_action']
            or result['snapshot'] != decision['boundary_snapshot']
            or content_hash(decision['boundary_snapshot']) != decision['boundary_hash']):
        raise ValueError('Physical decision action/boundary does not reproduce')
    # Scenario validation normalizes integer costs to floats. Preserve the exact
    # recorded numeric encoding so provenance hashes bind to the actual log bytes.
    result['snapshot'] = copy.deepcopy(decision['boundary_snapshot'])
    result['boundary_hash'] = decision['boundary_hash']
    result['evidence_kind'] = 'physical'
    result['physical_reference'] = {k: copy.deepcopy(decision[k]) for k in
                                    ['simulation_step', 'event_seq', 'physical_pose', 'boundary_hash']}
    for item in result['candidate_priorities_distances']:
        item['pose_distance_m'] = float(np.linalg.norm(
            np.array(state.scenario.coords[item['action']+1])-decision['physical_pose']))
    for key in ['selected_L', 'selected_H', 'nearer_lower_priority_gate']:
        result['planning_'+key] = result[key]
    low,high = physical_pair(result)
    result.update(selected_L=low,selected_H=high,nearer_lower_priority_gate=high is not None)
    return result


def physical_pair(value):
    candidates = value['candidate_priorities_distances']
    low = next(c for c in candidates if c['action']==value['policy_action'])
    maximum = max(c['priority'] for c in candidates)
    costs = {q['action']:q['cost'] for q in value['forced_first']}
    best,tol = value['exact']['cost'],value['equality_tolerance']
    if len(candidates)<3 or abs(costs[low['action']]-best)>tol:
        return None,None
    high = next((c for c in candidates if c['priority']==maximum and low['priority']<maximum
                 and low['distance_m']<c['distance_m'] and low['pose_distance_m']<c['pose_distance_m']
                 and costs[c['action']]>best+tol),None)
    return (low['id'],high['id']) if high else (None,None)


def verify_physical_explanation(value, decision):
    if (value.get('evidence_kind') != 'physical' or decision.get('evidence_kind') != 'physical'
            or value['policy_action'] != decision['selected_action']
            or value['snapshot'] != decision['boundary_snapshot']
            or value['physical_reference'] != {k: decision[k] for k in
                ['simulation_step', 'event_seq', 'physical_pose', 'boundary_hash']}):
        raise ValueError('Physical explanation provenance mismatch')
    low,high = physical_pair(value)
    if (value['selected_L']!=low or value['selected_H']!=high
            or value['nearer_lower_priority_gate']!=(high is not None)):
        raise ValueError('Physical nearer/lower-priority condition is unsupported by measured pose')
    logical = copy.deepcopy(value)
    for key in ['selected_L', 'selected_H', 'nearer_lower_priority_gate']:
        logical[key] = logical['planning_'+key]
    logical['evidence_kind'] = 'analytical'
    state = restore(value['snapshot'])
    for item in logical['candidate_priorities_distances']:
        measured = item.pop('pose_distance_m')
        expected = np.linalg.norm(np.array(state.scenario.coords[item['action']+1])-decision['physical_pose'])
        if not np.isfinite(measured) or abs(measured-expected) > 1e-9:
            raise ValueError('Physical pose distance mismatch')
    return verify_explanation(logical, decision['policy_hash'], decision['run_id'], decision['decision_id'])


def verify_event_response(events, decisions):
    """Check event conditions against the following actual learned decision."""
    updates = [e for e in events if e['type'] in ['PRIORITY_UPDATED','COST_UPDATED']]
    if len(updates) > 1:
        raise ValueError('Demo event was delivered more than once')
    for event in updates:
        update,comparison = event['update'],event['comparison']
        following = next((d for d in decisions if d['event_seq'] > event['event_seq']),None)
        old,new = comparison['no_event_order'],comparison['post_event_order']
        if (update['boundary'] != 1 or update['skipped'] or not following
                or following['selected_action'] != event['actual_action']
                or following['selected_action'] != new[0]
                or event['actual_action'] == event['counterfactual_action'] or old == new
                or comparison['improvement'] <= 1e-6
                or abs(comparison['improvement'] - (comparison['retained_order_post_event']['cost']
                       - comparison['replanned_order_post_event']['cost'])) > 1e-8):
            raise ValueError('Physical event response does not meet frozen comparison conditions')
        if event['type']=='PRIORITY_UPDATED':
            if update['checkpoint_id']!='C' or update['value']!=5 or new.index(2)>=old.index(2):
                raise ValueError('Emergency did not promote unvisited checkpoint C')
    return True


def verify_frozen_inputs(records, policy_hash):
    """Reject mixed model, geometry, camera or source versions in paired trials."""
    from .analytical import content_hash
    if not records:
        raise ValueError('No frozen physical inputs')
    signatures=[]
    scenarios_by_kind={}
    for record in records:
        summary,scenario,scene = record['summary'],record['scenario'],record['scene']
        if summary['routing_model_hash']!=policy_hash:
            raise ValueError('Physical summary does not bind to the loaded routing model')
        kind = scenario['scenario_id']
        scenario_hash = content_hash(scenario)
        if kind in scenarios_by_kind and scenarios_by_kind[kind] != scenario_hash:
            raise ValueError('Scenario changed between paired execution seeds')
        scenarios_by_kind[kind] = scenario_hash
        shared={k:scenario[k] for k in ['coords','checkpoint_ids','priorities','multipliers','service_duration_s',
                                       'nominal_speed_mps','alpha','beta','return_to_base']}
        signatures.append(content_hash({'models':{k:summary[k] for k in
                ['flight_checkpoint_hash','flight_config_hash','routing_model_hash',
                 'flight_acceptance_config_hash','execution_source_hash','hover_source_hash']},
                 'initial_scenario':shared,'scene':scene,'initial_pose':record['initial_pose']}))
    if len(set(signatures))!=1:
        raise ValueError('Frozen physical inputs changed across comparative trials')
    return True


def verify_media_index(frames, probe):
    """Bind every retained camera frame to a real decoded MP4 presentation timestamp."""
    stream = probe['streams'][0]
    decoded = probe['frames']
    if (not frames or len(decoded)!=len(frames) or stream['width']!=1280 or stream['height']!=720):
        raise ValueError('Decoded camera size/frame count differs from retained capture index')
    for i,(frame,actual) in enumerate(zip(frames,decoded)):
        pts=float(actual['best_effort_timestamp_time'])
        recorded=float(frame['pts_s'])
        if (int(frame['frame_index'])!=i or not np.isfinite([pts,recorded]).all()
                or abs(pts-recorded)>1e-6):
            raise ValueError('Decoded camera PTS differs from retained frame index')
    return True


def evidence_frame_indices(events, frames):
    """Retain first/final frames and the first frame displaying every critical event."""
    if not frames:
        raise ValueError('No frames to inspect')
    selected={0,len(frames)-1}
    for event in events:
        if event['type'] not in ['TARGET_SELECTED','PRIORITY_UPDATED','COST_UPDATED']:
            continue
        index=next((i for i,f in enumerate(frames) if int(f['event_seq'])>=event['event_seq']),None)
        if index is None:
            raise ValueError('Critical state change has no recorded display frame')
        selected.add(index)
    return sorted(selected)
