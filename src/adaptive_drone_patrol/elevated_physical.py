"""New-height development definitions and measured vertical/safety evidence."""
import copy
import json
from pathlib import Path
import numpy as np
from .payload_cases import initial_state
from .util import load_config


def vertical_cases():
    goals=load_config('configs/demo_scenarios_payload_elevated/normal.yaml')['coords']
    by_id=dict(zip('SABCDEFGHIJ',goals)); targets=[by_id[k] for k in 'AGIDBHFJEC']
    loads=[(.03,.13),(.045,.20),(.06,.27)]
    return [dict(case_id=f'elevated-vertical-{seed}-{i}',suite='elevated-vertical',execution_seed=seed,
                 side_m=side,mass_ratio=mass,targets=copy.deepcopy(targets),target_ids=list('AGIDBHFJEC'),
                 initial=initial_state(np.random.default_rng(np.random.SeedSequence([seed,720,i])),side))
            for seed in (11,12,13,14) for i,(side,mass) in enumerate(loads)]


def clearance_envelope():
    # Bounds apply to commanded segments only under the explicitly unproved
    # tracking assumption. Reset z is exactly1; quintic segments are convex.
    z=1.; tracking=.20; attach=.02; length=.30
    values=[]
    for side in (.03,.045,.06):
        # Arbitrary box orientation: center-to-top offset side/2 and the
        # circumscribed corner radius sqrt(3)*side/2, including full box extent.
        box_radius=np.sqrt(3)*side/2
        floor=z-tracking-attach-length-side/2-box_radius
        values.append(dict(side_m=side,box_corner_radius_m=float(box_radius),package_floor_bound_m=float(floor)))
    drone_support=float(np.hypot(.06,.0125))
    return dict(schema='elevated-conservative-clearance-envelope-v1',initial_drone_z_m=z,allowed_tracking_error_m=tracking,
                drone_attachment_radius_m=attach,link_m=length,loads=values,
                minimum_package_floor_bound_m=min(v['package_floor_bound_m'] for v in values),
                minimum_drone_floor_bound_m=z-tracking-drone_support,
                measured_flight_safety_claim=False,
                assumptions=['Tracking error <=0.20m throughout every segment is an assumption requiring native/substep measurement; dwell thresholds alone do not prove it.',
                             'Actual initial poses/orientations are retained; commanded quintic trajectories stay within endpoint convex envelopes.',
                             'All descending transfers and initial S climb included; raw oriented support overrides this planning estimate.'])


def vertical_report(stops):
    if len(stops)!=10 or len({s['id'] for s in stops})!=10:raise ValueError('All ten settled destinations required')
    previous=dict(id='S',target_z_m=1.,actual_z_m=1.,step=0); legs=[]
    for stop in stops:
        if not np.isfinite([stop['target_z_m'],stop['actual_z_m']]).all() or stop['step']<=previous['step']:
            raise ValueError('Invalid settled-height chronology')
        legs.append(dict(**{'from':previous['id'],'to':stop['id']},commanded_dz_m=stop['target_z_m']-previous['target_z_m'],
                         measured_dz_m=stop['actual_z_m']-previous['actual_z_m'],start_step=previous['step'],end_step=stop['step']))
        previous=stop
    asc=[l for l in legs[1:] if l['commanded_dz_m']>=.75 and l['measured_dz_m']>=.50]
    desc=[l for l in legs[1:] if l['commanded_dz_m']<=-.75 and l['measured_dz_m']<=-.50]
    return dict(passed=bool(asc and desc),stops=stops,legs=legs,ascents=asc,descents=desc,initial_climb_excluded_from_display_gate=True)


def mission_vertical_report(trial):
    trial=Path(trial)
    scenario=json.loads((trial/'mission/scenario.json').read_text()); goals=dict(zip(scenario['checkpoint_ids'],scenario['coords'][1:]))
    events=[json.loads(line) for line in (trial/'mission/events.jsonl').read_text().splitlines()]
    return vertical_report([dict(id=e['target_id'],target_z_m=goals[e['target_id']][2],actual_z_m=e['physical_pose'][2],step=e['simulation_step'])
                            for e in events if e['type']=='SHARED_SETTLED'])
