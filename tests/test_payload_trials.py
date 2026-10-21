"""Fail-closed independent loaded-trial evidence checks, using CPU fixtures."""

import importlib.util
from pathlib import Path
import pytest


def checker():
    path=Path(__file__).parents[1]/'scripts/check_payload_trials.py'
    import sys
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('payload_trial_checker',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_swapped_commands_and_stale_hashes_are_rejected(tmp_path):
    mod=checker()
    row=dict(target_id='A',target=[0,0,1],effective_target=[0,0,1])
    mod.check_target(row,{'A':[0,0,1]})
    with pytest.raises(ValueError):
        mod.check_target(row,{'A':[2,0,1]})
    p=tmp_path/'sample.json'
    p.write_text('{}')
    old=mod.digest(p)
    p.write_text('{"step":1}')
    with pytest.raises(ValueError):
        mod.check_hash(p,old)


def test_confidence_interval_has_declared_denominator():
    low,high=checker().wilson(95,100)
    assert .88<low<.90 and .97<high<.98
    with pytest.raises(ValueError):
        checker().wilson(0,0)


def fixture_trial(tmp_path):
    import json
    mod=checker()
    path=tmp_path/'synthetic'
    path.mkdir()
    state=dict(drone_xyz=[0,0,1],drone_quaternion_wxyz=[1,0,0,0],drone_velocity=[0,0,0],
               drone_angular_velocity=[0,0,0],package_xyz=[0,0,.6575],package_quaternion_wxyz=[1,0,0,0],
               package_velocity=[0,0,0],package_angular_velocity=[0,0,0])
    case=dict(case_id='fixture',side_m=.045,mass_ratio=.2,initial=state.copy())
    rows=[dict(step=0,time_s=0,initial=True,**state)]
    for i in range(1,102):
        rows.append(dict(step=i,time_s=i*.01,target_id='transition-0',target=[0,0,1],effective_target=[0,0,1],
                         contacts=False,complete=i==101,dwell_s=(i-1)*.01,actions=[0]*4,
                         relative_position_preclip=[0]*3,relative_position_clipped=[0]*3,**state))
    (path/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    mass=.02700047753751278*.2
    (path/'payload.json').write_text(json.dumps(dict(reference_mass_kg=.02700047753751278,mass_kg=mass,
          effective_mass_kg=[mass],side_m=.045,inertia_kg_m2=[[[mass*.045**2/6 if i==j else 0 for j in range(3)] for i in range(3)]])))
    (path/'case.json').write_text(json.dumps(case))
    (path/'events.json').write_text(json.dumps([dict(kind='SHARED_SETTLED',step=101,time_s=1.01,target_id='transition-0')]))
    (path/'decisions.json').write_text('[]')
    summary=dict(status='SUCCEEDED',steps=101,completed=['transition-0'])
    for name,key in [('samples.jsonl','telemetry_sha256'),('case.json','case_sha256'),('payload.json','payload_sha256'),('events.json','events_sha256'),('decisions.json','decisions_sha256')]:
        summary[key]=mod.digest(path/name)
    (path/'summary.json').write_text(json.dumps(summary))
    return mod,path,case,summary


@pytest.mark.parametrize('corruption',['mass','event_time','decisions'])
def test_rehashed_semantic_corruption_does_not_pass(tmp_path,corruption):
    import json
    mod,path,case,summary=fixture_trial(tmp_path)
    assert mod.verify_trial(path,case,{'transition-0':[0,0,1]})['success']
    if corruption=='mass':
        p=path/'payload.json'
        d=json.loads(p.read_text())
        d['effective_mass_kg']=[999.]
        p.write_text(json.dumps(d))
        summary['payload_sha256']=mod.digest(p)
    elif corruption=='event_time':
        p=path/'events.json'
        d=json.loads(p.read_text())
        d[0]['step']=-10
        d[0]['time_s']=-100
        p.write_text(json.dumps(d))
        summary['events_sha256']=mod.digest(p)
    else:
        p=path/'decisions.json'
        p.write_text('[{"corrupted":true}]')
        summary['decisions_sha256']=mod.digest(p)
    (path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        mod.verify_trial(path,case,{'transition-0':[0,0,1]})


@pytest.mark.parametrize('at_initial',[True,False])
def test_wrong_initial_yaw_or_unmeasured_angular_jump_is_rejected(tmp_path,at_initial):
    import json
    mod,path,case,summary=fixture_trial(tmp_path)
    rows=[json.loads(line) for line in (path/'samples.jsonl').read_text().splitlines()]
    rows[0 if at_initial else 50]['drone_quaternion_wxyz']=[0,0,0,1]
    (path/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary['telemetry_sha256']=mod.digest(path/'samples.jsonl')
    (path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        mod.verify_trial(path,case,{'transition-0':[0,0,1]})
