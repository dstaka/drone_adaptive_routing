"""Independent aggregate physics gates; synthetic fixtures certify no dynamics."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/check_payload_retry_qualification.py'


def api():
    assert SCRIPT.is_file(), 'aggregate checker must exist'
    spec=importlib.util.spec_from_file_location('qualification_check',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(duration=30.):
    return [dict(step=i,time_s=i*.01,drone_xyz=[0,0,1],package_xyz=[0,0,.6575],
                 drone_quaternion_wxyz=[1,0,0,0],package_quaternion_wxyz=[1,0,0,0])
            for i in range(round(duration/.01)+1)]


def test_refinement_compares_all_matched_positions_and_orientations():
    result=api().compare_refinement(rows(),rows())
    assert result['passed']
    assert result['samples']==3001
    assert result['maximum_position_difference_m']==0


@pytest.mark.parametrize(('body','field','value'),[
    ('drone','xyz',[.02001,0,1]),('package','xyz',[0,0,.678]),
    ('package','quaternion_wxyz',[float(np.cos(np.deg2rad(1.1))),float(np.sin(np.deg2rad(1.1))),0,0])])
def test_refinement_rejects_vehicle_or_package_limit_excursions(body,field,value):
    a,b=rows(),rows()
    b[100][body+'_'+field]=value
    assert not api().compare_refinement(a,b)['passed']


def test_refinement_quaternion_sign_is_not_orientation_error():
    a,b=rows(),rows()
    b[100]['package_quaternion_wxyz']=[-1,0,0,0]
    assert api().compare_refinement(a,b)['passed']


@pytest.mark.parametrize('mode',['terminal','time','nonfinite'])
def test_refinement_requires_full_30_seconds_and_terminal_state(mode):
    a,b=rows(),rows()
    if mode=='terminal':
        b=b[:-1]
    elif mode=='time':
        b[100]['time_s']+=.001
    else:
        b[100]['package_xyz'][0]=float('nan')
    assert not api().compare_refinement(a,b)['passed']


def test_control_endpoint_work_guard_has_fixed_declared_allowance():
    result=api().energy_growth_guard([0,.1,.2],[1,2,3],dt=.01)
    assert result['work_guard_j']==pytest.approx(.02+1.1*.01*(2+3))
    assert result['guard_exceeded']
    assert not result['conservative_substep_bound_established']
    assert result['classification']=='ACTUAL_ENERGY_GUARD_EXCURSION'


def test_endpoint_guard_success_does_not_invent_unobserved_substep_bound():
    result=api().energy_growth_guard([.1,.1,.1],[0,0,0],dt=.01)
    assert not result['guard_exceeded']
    assert result['classification']=='INSUFFICIENT_WORK_BOUND_EVIDENCE'
    assert not result['conservative_substep_bound_established']


def test_full_case_inventory_is_55_with_six_refinement_pairs():
    manifest=json.loads((ROOT/'configs/payload_retry_probe_manifest.json').read_text())
    result=api().inventory(manifest)
    assert result['valid']
    assert result['kind_counts']=={'anchor':9,'large_angle':16,'powered':18,'refinement':12}
    assert len(result['refinement_pairs'])==6


def test_shortened_or_omitted_case_cannot_be_hidden_by_manifest_status():
    manifest=json.loads((ROOT/'configs/payload_retry_probe_manifest.json').read_text())
    manifest['cases'][0]['duration_s']=119.99
    assert not api().inventory(manifest)['valid']
    manifest['cases']=manifest['cases'][1:]
    assert not api().inventory(manifest)['valid']


def test_missing_aggregate_sources_and_stage_evidence_are_blocked(tmp_path):
    result=api().verify_qualification(tmp_path,tmp_path,tmp_path,source_root=tmp_path,initial_cases=tmp_path/'missing.json')
    assert result['status']=='QUALIFICATION_BLOCKED'
    assert not result['physical_training_admitted']
    assert 'SOURCE_OR_DECLARATION_MISSING' in result['failure_codes']
    assert str(tmp_path.parent) not in json.dumps(result)


def test_static_layout_check_preserves_all127_cases_and_all11_locations():
    result=api().static_geometry(ROOT,ROOT/'outputs/suspended_20261021_v1/cases-v1.json')
    assert result['passed']
    assert result['initial_cases']==127
    assert result['locations']==11
    assert result['minimum_initial_package_floor_m'] > .62
    assert result['minimum_nominal_package_floor_m']==pytest.approx(.62)
    assert not result['dynamic_clearance_certified']


def initial_case(theta=10.,phi=0.):
    angle=np.deg2rad(theta)
    azimuth=phi
    quaternion=[float(np.cos(angle/2)),float(np.sin(azimuth)*np.sin(angle/2)),float(-np.cos(azimuth)*np.sin(angle/2)),0.]
    direction=np.array([np.sin(angle)*np.cos(azimuth),np.sin(angle)*np.sin(azimuth),-np.cos(angle)])
    # Independent Rodrigues construction for the same unit axis.
    axis=np.array([np.sin(azimuth),-np.cos(azimuth),0.])
    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    orientation=np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
    anchor=np.array([0,0,.98])
    center=anchor+.3*direction-orientation@[0,0,.0225]
    row=dict(drone_xyz=[0,0,1],drone_quaternion_wxyz=[1,0,0,0],rod_xyz=anchor.tolist(),
             rod_quaternion_wxyz=quaternion,package_xyz=center.tolist(),package_quaternion_wxyz=quaternion,
             drone_velocity=[0,0,0],drone_angular_velocity=[0,0,0],rod_velocity=[0,0,0],
             rod_angular_velocity=[0,0,0],package_velocity=[0,0,0],package_angular_velocity=[0,0,0])
    case=dict(initial_swing_deg=theta,initial_direction_rad=phi,side_m=.045)
    return row,case


@pytest.mark.parametrize('theta',[0.,5.,90.,170.])
def test_predeclared_initial_direction_and_inherited_package_orientation(theta):
    row,case=initial_case(theta)
    assert api().initial_case_errors(row,case)==[]


def test_old_upper_rx_direction_is_rejected_even_with_correct_swing_angle():
    row,_=initial_case(10.,np.pi/2)
    case=dict(initial_swing_deg=10.,initial_direction_rad=0.,side_m=.045)
    assert 'INITIAL_DIRECTION_MISMATCH' in api().initial_case_errors(row,case)


def test_initial_cube_rotation_and_stationary_root_are_not_assumed():
    row,case=initial_case(10.)
    row['package_quaternion_wxyz']=[1,0,0,0]
    assert 'INITIAL_PACKAGE_ORIENTATION_MISMATCH' in api().initial_case_errors(row,case)
    row,case=initial_case(10.)
    row['drone_velocity']=[.01,0,0]
    assert 'INITIAL_NONZERO_VELOCITY' in api().initial_case_errors(row,case)


def test_actual_damping_and_armature_are_part_of_physical_signature():
    runtime=dict(links=[dict(name='base_link',mass_kg=.027000000700354576)]
                      +[dict(name=f'prop{i}_link',mass_kg=1.1920928955078125e-7) for i in range(4)]
                      +[dict(name='suspension_rod',mass_kg=.0001),dict(name='payload',mass_kg=.005400095507502556)],
                 damping_after=[0,0,0]+[(.027+.0001+.005400095507502556)*1e-5]*3+[0]*6,armature=[0]*12)
    case=dict(kind='powered',mass_ratio=.2)
    assert api().physical_parameters_errors(runtime,case)==[]
    runtime['damping_after'][7]=.1
    assert 'UNDECLARED_PASSIVE_DAMPING' in api().physical_parameters_errors(runtime,case)
    runtime['armature'][7]=.1
    assert 'ARMATURE_MISMATCH' in api().physical_parameters_errors(runtime,case)


def test_endpoint_bound_uses_same_interval_input_at_both_endpoints():
    result=api().energy_growth_guard([0,.01,.02],[[2,3],[4,5]],dt=.01)
    assert result['work_guard_j']==pytest.approx(.02+1.1*.01*(3+5))


def work_path():
    records=[]
    for i in range(2):
        q0=[i*.01,0,0,float(np.cos(i*.005/2)),0,0,float(np.sin(i*.005/2)),1,0,0,0,1,0,0,0]
        q1=[(i+1)*.01,0,0,float(np.cos((i+1)*.005/2)),0,0,float(np.sin((i+1)*.005/2)),1,0,0,0,1,0,0,0]
        force=np.zeros((8,1,3))
        moment=np.zeros((8,1,3))
        force[2,0,0]=-1
        moment[2,0,2]=-2
        records.append(dict(control_step=0,substep=i,time_s=i*.005,dt_s=.005,qpos_before=q0,qpos_after=q1,
            dofs_velocity_next=[2,0,0,0,0,1,0,0,0,0,0,0],root_com_before=[i*.01,0,0],
            root_com_after=[(i+1)*.01,0,0],negative_external_force_field=force.tolist(),
            negative_external_force_field_after=force.tolist(),negative_external_torque_about_root_com_field=moment.tolist(),
            negative_external_torque_about_root_com_field_after=moment.tolist()))
    return records


def test_source_justified_substep_path_can_establish_conservative_work_bound():
    bound=api().substep_path_bound(work_path(),control_steps=1,substeps=2)
    assert bound['established']
    assert bound['cumulative_bound_j']==pytest.approx([0,.045])
    result=api().energy_growth_guard([0,.01],[[2,2]],justified_work_bound=bound['cumulative_bound_j'])
    assert result['conservative_substep_bound_established']
    assert result['classification']=='SOURCE_JUSTIFIED_WORK_BOUND_SCREEN'
    assert result['work_guard_j']==pytest.approx(.042)


@pytest.mark.parametrize('mode',['missing','nonconstant_wrench','nonrotor_force','wrong_qpos','hidden_rotation'])
def test_unverifiable_substep_path_is_never_promoted_to_work_bound(mode):
    data=work_path()
    if mode=='missing':
        data=data[:-1]
    elif mode=='nonconstant_wrench':
        data[0]['negative_external_force_field_after'][2][0][0]=-2
    elif mode=='nonrotor_force':
        data[0]['negative_external_force_field'][6][0][0]=-1
    elif mode=='wrong_qpos':
        data[0]['qpos_after'][0]=1
    else:
        data[0]['dofs_velocity_next'][5]=0
    assert not api().substep_path_bound(data,control_steps=1,substeps=2)['established']


def test_physical_signature_changes_with_actual_engine_source_bytes():
    a=api()
    args=dict(bindings={'asset_builder.py':'a'*64},settings={'dt':.005},manifest_sha256='b'*64,
              assets={'cube':'c'*64})
    first=a.physical_signature(engine={'solver.py':'d'*64},**args)
    second=a.physical_signature(engine={'solver.py':'e'*64},**args)
    assert first!=second


@pytest.mark.parametrize('mode',['root_speed','root_spin','joint_speed','joint_quaternion'])
def test_finite_substep_excursions_cannot_get_work_certificate(mode):
    records=work_path()
    if mode=='root_speed':
        for i,r in enumerate(records):
            r['qpos_before'][0]=i*.5
            r['qpos_after'][0]=(i+1)*.5
            r['root_com_before'][0]=i*.5
            r['root_com_after'][0]=(i+1)*.5
            r['dofs_velocity_next'][0]=100.
    elif mode=='root_spin':
        records[0]['dofs_velocity_next'][5]=300.
    elif mode=='joint_speed':
        records[0]['dofs_velocity_next'][6]=1000.
    else:
        records[0]['qpos_before'][7]=2.
    assert not api().substep_path_bound(records,control_steps=1,substeps=2)['established']
