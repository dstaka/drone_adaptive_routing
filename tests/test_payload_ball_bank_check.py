"""Independent CPU reconstruction of reset evidence, never simulator success."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/check_payload_ball_bank.py'


def api():
    assert SCRIPT.is_file(), 'independent reset checker must exist'
    spec=importlib.util.spec_from_file_location('bank_check',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_seed_reconstruction_matches_frozen_diagnostic_draws():
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('draws',ROOT/'scripts/probe_payload_ball_bank.py')
    runner=module_from_spec(spec)
    spec.loader.exec_module(runner)
    expected=api().expected_resets()
    assert expected == runner.reset_sequence(730)
    assert len(expected)==30
    assert expected[0]['side_m']==.03 and expected[0]['mass_ratio']==.1
    assert expected[8]['side_m']==.06 and expected[8]['mass_ratio']==.3


def snapshot_from_draw(draw,reference=.02700047753751278):
    """Analytical state at reset; fixture poses use independent quaternion algebra."""
    import numpy as np
    initial=draw['initial']
    side=draw['side_m']
    mass=reference*draw['mass_ratio']
    def r(q):
        w,x,y,z=q
        return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                         [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                         [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    # This simple vertical fixture is used for measurement-negative tests; the
    # complete random draw schedule is checked separately above.
    from adaptive_drone_patrol.payload_ball_bank import ball_joint_configuration
    upper,lower=ball_joint_configuration(initial,side)
    rd=r(initial['drone_quaternion_wxyz'])
    rp=r(initial['package_quaternion_wxyz'])
    drone_q=initial['drone_quaternion_wxyz']
    rod=np.r_[drone_q[0]*upper[0]-np.dot(drone_q[1:],upper[1:]),
              drone_q[0]*np.asarray(upper[1:])+upper[0]*np.asarray(drone_q[1:])+np.cross(drone_q[1:],upper[1:])].tolist()
    anchor=np.asarray(initial['drone_xyz'])+rd@[0,0,-.02]
    center=np.asarray(initial['package_xyz'])
    extent=np.abs(rp)@np.full(3,side/2)
    snapshot=dict(align_root=True,reference_original_mass_kg=reference,
        links=[dict(name='base_link',mass_kg=.027000000700354576)]
             +[dict(name=f'prop{i}_link',mass_kg=1.1920928955078125e-7) for i in range(4)]
             +[dict(name='suspension_rod',mass_kg=.0001),dict(name='payload',mass_kg=mass)],
        inertia_kg_m2=[[[mass*side**2/6 if i==j else 0 for j in range(3)] for i in range(3)]]*7,
        package_aabb=[(center-extent).tolist(),(center+extent).tolist()],
        links_com_body_m=[[0,0,0]]*7,links_pos=[[0,0,0]]*7,links_quat=[[1,0,0,0]]*7,
        damping=[0]*12,armature=[0]*12,
        qpos=[0,0,1,1,0,0,0]+upper+lower,dofs_velocity=[0]*12,
        joints=[dict(name='vehicle_free',q_start_local=0,n_qs=7,n_dofs=6),
                dict(name='suspension_ball',q_start_local=7,n_qs=4,n_dofs=3),
                dict(name='payload_ball',q_start_local=11,n_qs=4,n_dofs=3)],
        drone_xyz=initial['drone_xyz'],drone_quaternion_wxyz=initial['drone_quaternion_wxyz'],
        package_xyz=initial['package_xyz'],package_quaternion_wxyz=initial['package_quaternion_wxyz'],
        rod_xyz=anchor.tolist(),rod_quaternion_wxyz=rod)
    for name in ('drone','rod','package'):
        snapshot[name+'_velocity']=[0,0,0]
        snapshot[name+'_angular_velocity']=[0,0,0]
    return snapshot


def test_independent_measured_properties_pass_only_diagnostic_screen():
    draw=api().expected_resets()[0]
    failures,metrics=api().check_reset(draw,snapshot_from_draw(draw),.02700047753751278,True)
    assert failures==[]
    assert metrics['constraint_residual_m'] < 1e-12


@pytest.mark.parametrize(('mode','code'),[('mass','PACKAGE_MASS'),('inertia','PACKAGE_INERTIA'),
    ('aabb','COLLISION_AABB'),('position','INITIAL_POSE'),('velocity','NONZERO_RESET_VELOCITY'),
    ('joint','NATIVE_BALL_FK'),('nonfinite','NONFINITE_STATE')])
def test_raw_properties_and_joint_states_are_independently_rejected(mode,code):
    draw=api().expected_resets()[0]
    snapshot=snapshot_from_draw(draw)
    if mode=='mass':
        snapshot['links'][6]['mass_kg']=1
    elif mode=='inertia':
        snapshot['inertia_kg_m2'][6][0][0]=1
    elif mode=='aabb':
        snapshot['package_aabb'][1][0]+=1
    elif mode=='position':
        snapshot['package_xyz']=list(snapshot['package_xyz'])
        snapshot['package_xyz'][0]+=1
    elif mode=='velocity':
        snapshot['dofs_velocity'][1]=1
    elif mode=='joint':
        snapshot['qpos'][7:11]=[1,0,0,0]
    else:
        snapshot['package_velocity']=[float('nan'),0,0]
    failures,_=api().check_reset(draw,snapshot,.02700047753751278,True)
    assert code in failures


def test_missing_files_and_archived_sources_block_without_absolute_path_leak(tmp_path):
    result=api().verify_bank(tmp_path,source_root=tmp_path)
    assert result['status']=='QUALIFICATION_BLOCKED'
    assert not result['physical_training_admitted']
    assert 'MISSING_EVIDENCE' in result['failure_codes']
    assert str(tmp_path.parent) not in json.dumps(result)


def test_cli_refuses_overwrite_even_for_missing_evidence(tmp_path):
    out=tmp_path/'receipt.json'
    command=[sys.executable,str(SCRIPT),'--probe',str(tmp_path/'absent'),'--source-root',str(tmp_path),'--out',str(out)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==2,result.stderr
    before=hashlib.sha256(out.read_bytes()).hexdigest()
    assert subprocess.run(command,capture_output=True).returncode!=0
    assert hashlib.sha256(out.read_bytes()).hexdigest()==before


@pytest.fixture
def complete_recording(tmp_path):
    import copy
    probe=tmp_path/'probe'
    probe.mkdir()
    module=api()
    sequence=module.expected_resets()
    def save(name,data):
        (probe/name).write_text(json.dumps(data)+'\n')
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in module.SOURCES}
    limits=dict(mass_relative_error=1e-5,inertia_relative_error=1e-5,collision_extent_error_m=1e-5,
                attachment_residual_m=2e-5,initial_pose_error=1e-6,zero_velocity_abs=1e-7)
    declaration=dict(schema='payload-ball-bank-reset-declaration-v1',seed=730,rng_streams=dict(size=731,mass=732,pose=733),
        dt_s=.01,substeps=2,planned_resets=30,reset_sequence=sequence,limits=limits,align_root=True,
        source_binding=sources,manifest_sha256=hashlib.sha256((ROOT/'configs/payload_retry_probe_manifest.json').read_bytes()).hexdigest(),
        physical_training_admitted=False)
    save('declaration.json',declaration)
    states={f"{draw['side_m']:.3f}":snapshot_from_draw(draw) for draw in sequence[:3]}
    initial=copy.deepcopy(states)
    records=[]
    for draw in sequence:
        before=copy.deepcopy(states)
        key=f"{draw['side_m']:.3f}"
        states[key]=snapshot_from_draw(draw)
        after=copy.deepcopy(states)
        record=dict(index=draw['index'],draw=draw,before=before,after=after)
        if draw['motor_step_after_reset']:
            states[key]['dofs_velocity'][0]=.001
            states[key]['drone_velocity']=[0,0,-.001]
            record['after_motor_step']=copy.deepcopy(states)
        records.append(record)
    save('reset-samples.json',records)
    assets={}
    for side in (.03,.045,.06):
        path=probe/f'bank/load_{side:.3f}/candidate.xml'
        path.parent.mkdir(parents=True)
        path.write_text('<test-fixture/>')
        assets[str(path.relative_to(probe))]=hashlib.sha256(path.read_bytes()).hexdigest()
    construction=dict(reference_mass_kg=.02700047753751278,align_root=True,asset_hashes=assets,initial_scenes=initial)
    save('runtimeproperties.json',construction)
    summary=dict(status='EXPERIMENTAL_DIAGNOSTICS_ONLY',physical_training_admitted=False,source_binding=sources,
                 completed_resets=30,asset_hashes=assets,failures=[])
    for name,key in [('declaration.json','declaration_sha256'),('runtimeproperties.json','runtimeproperties_sha256'),
                     ('reset-samples.json','reset_samples_sha256')]:
        summary[key]=hashlib.sha256((probe/name).read_bytes()).hexdigest()
    save('summary.json',summary)
    return probe


def rewrite_records(probe,records):
    (probe/'reset-samples.json').write_text(json.dumps(records)+'\n')
    summary=json.loads((probe/'summary.json').read_text())
    summary['reset_samples_sha256']=hashlib.sha256((probe/'reset-samples.json').read_bytes()).hexdigest()
    (probe/'summary.json').write_text(json.dumps(summary)+'\n')


def test_full_analytic_fixture_screens_reset_evidence_without_admission(complete_recording):
    result=api().verify_bank(complete_recording)
    assert result['status']=='EXPERIMENTAL_DIAGNOSTICS_ONLY'
    assert result['reconstructed_moving_inactive_comparisons'] > 0
    assert result['reconstructed_motor_steps']==29
    assert result['expected_motor_rpm']==pytest.approx([15047.166350840727,14468.429183500699,14468.429183500699,14468.429183500699])
    assert not result['physical_training_admitted']
    assert not result['actual_kernel_rpm_observed']


@pytest.mark.parametrize('mode',['terminal','draw','inactive_reset','inactive_motor','unrecorded_reset','missing_motor'])
def test_sequence_and_inactive_evidence_cannot_be_forged_by_summary_flags(complete_recording,mode):
    probe=complete_recording
    records=json.loads((probe/'reset-samples.json').read_text())
    if mode=='terminal':
        records=records[:-1]
    elif mode=='draw':
        records[0]['draw']['mass_ratio']=.2
    elif mode=='inactive_reset':
        records[1]['after']['0.030']['dofs_velocity'][0]=9
    elif mode=='inactive_motor':
        records[0]['after_motor_step']['0.045']['qpos'][0]=9
    elif mode=='unrecorded_reset':
        records[3]['before']['0.030']['qpos'][0]=9
    else:
        del records[0]['after_motor_step']
    rewrite_records(probe,records)
    assert api().verify_bank(probe)['status']=='QUALIFICATION_BLOCKED'


def test_changed_actual_source_and_raw_file_hashes_rejected(complete_recording,tmp_path):
    probe=complete_recording
    assert 'SOURCE_BINDING_MISMATCH' in api().verify_bank(probe,source_root=tmp_path)['failure_codes']
    with (probe/'reset-samples.json').open('a') as stream:
        stream.write(' ')
    assert 'RAW_HASH_MISMATCH' in api().verify_bank(probe)['failure_codes']


def test_omitted_effective_property_channels_cannot_satisfy_isolation(complete_recording):
    probe=complete_recording
    records=json.loads((probe/'reset-samples.json').read_text())
    for record in records:
        for channel in ('before','after','after_motor_step'):
            for snapshot in record.get(channel,{}).values():
                snapshot.pop('damping',None)
    rewrite_records(probe,records)
    construction=json.loads((probe/'runtimeproperties.json').read_text())
    for snapshot in construction['initial_scenes'].values():
        snapshot.pop('damping',None)
    (probe/'runtimeproperties.json').write_text(json.dumps(construction)+'\n')
    summary=json.loads((probe/'summary.json').read_text())
    summary['runtimeproperties_sha256']=hashlib.sha256((probe/'runtimeproperties.json').read_bytes()).hexdigest()
    (probe/'summary.json').write_text(json.dumps(summary)+'\n')
    assert api().verify_bank(probe)['status']=='QUALIFICATION_BLOCKED'


def test_archived_profile_must_match_root_aligned_v3(complete_recording,tmp_path):
    probe=complete_recording
    source=tmp_path/'source'
    for name in (*api().SOURCES,'configs/payload_retry_probe_manifest.json'):
        target=source/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    manifest=source/'configs/payload_retry_probe_manifest.json'
    data=json.loads(manifest.read_text())
    data['profile']='cf2x-two-native-ball-mjcf-v2'
    manifest.write_text(json.dumps(data)+'\n')
    declaration=json.loads((probe/'declaration.json').read_text())
    declaration['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    (probe/'declaration.json').write_text(json.dumps(declaration)+'\n')
    summary=json.loads((probe/'summary.json').read_text())
    summary['declaration_sha256']=hashlib.sha256((probe/'declaration.json').read_bytes()).hexdigest()
    (probe/'summary.json').write_text(json.dumps(summary)+'\n')
    assert 'PHYSICAL_PROFILE_MISMATCH' in api().verify_bank(probe,source_root=source)['failure_codes']


def test_derived_position_roundoff_does_not_change_a_physical_draw():
    import copy
    expected=api().expected_resets()[8]
    remote=copy.deepcopy(expected)
    remote['initial']['package_xyz'][1]=-0.02324638871112723
    assert api().same_draw(expected,remote)
    remote['initial']['package_xyz'][1]+=.000001
    assert not api().same_draw(expected,remote)
    remote=copy.deepcopy(expected)
    remote['mass_ratio']=float.fromhex(expected['mass_ratio'].hex())+1e-16
    assert not api().same_draw(expected,remote)
