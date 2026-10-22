"""Predeclared reset draws and independent runtime-property checks, CPU only."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/probe_payload_ball_bank.py'


def api():
    assert SCRIPT.is_file(), 'reset probe must exist'
    spec=importlib.util.spec_from_file_location('bank_reset_probe',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thirty_diagnostic_draws_cover_grid_and_independent_random_episodes():
    plan=api().reset_sequence(730)
    assert len(plan) == 30
    assert {(r['side_m'],r['mass_ratio']) for r in plan[:9]} == {
        (side,ratio) for side in (.03,.045,.06) for ratio in (.1,.2,.3)}
    assert all(plan[i]['side_m'] != plan[i+1]['side_m'] for i in range(8))
    assert all(.1 <= r['mass_ratio'] <= .3 for r in plan)
    assert len({r['mass_ratio'] for r in plan[9:]}) == 21
    assert plan == api().reset_sequence(730)
    assert plan != api().reset_sequence(731)
    assert all(r['motor_step_after_reset'] for r in plan[:-1])


def test_inactive_comparison_covers_generalized_state_and_physical_properties():
    before={'0.030':dict(qpos=[1],mass=[.1],aabb=[[0],[1]],damping=[0]),
            '0.045':dict(qpos=[2],mass=[.2],aabb=[[0],[2]],damping=[0]),
            '0.060':dict(qpos=[3],mass=[.3],aabb=[[0],[3]],damping=[0])}
    after=copy.deepcopy(before)
    after['0.030']['qpos']=[99]
    assert api().inactive_changes(before,after,.03) == []
    for key in ('qpos','mass','aabb','damping'):
        changed=copy.deepcopy(after)
        changed['0.045'][key]=[999]
        assert api().inactive_changes(before,changed,.03) == ['0.045']


def snapshot():
    mass=.02700047753751278*.2
    inertia=mass*.045**2/6
    return dict(reference_original_mass_kg=.02700047753751278,align_root=True,
        links=[dict(name='base_link',mass_kg=.027000000700354576)]
              +[dict(name=f'prop{i}_link',mass_kg=1.1920928955078125e-7) for i in range(4)]
              +[dict(name='suspension_rod',mass_kg=.0001),dict(name='payload',mass_kg=mass)],
        inertia_kg_m2=[[[inertia if i==j else 0 for j in range(3)] for i in range(3)]]*7,
        package_aabb=[[-.0225,-.0225,.635],[.0225,.0225,.68]],
        qpos=[0,0,1,1,0,0,0,1,0,0,0,1,0,0,0],dofs_velocity=[0]*12,
        joints=[dict(name='vehicle_free',q_start_local=0,n_qs=7,n_dofs=6),
                dict(name='suspension_ball',q_start_local=7,n_qs=4,n_dofs=3),
                dict(name='payload_ball',q_start_local=11,n_qs=4,n_dofs=3)],
        drone_xyz=[0,0,1],drone_quaternion_wxyz=[1,0,0,0],
        rod_xyz=[0,0,.98],rod_quaternion_wxyz=[1,0,0,0],
        package_xyz=[0,0,.6575],package_quaternion_wxyz=[1,0,0,0],
        drone_velocity=[0,0,0],drone_angular_velocity=[0,0,0],
        package_velocity=[0,0,0],package_angular_velocity=[0,0,0],
        rod_velocity=[0,0,0],rod_angular_velocity=[0,0,0])


def draw():
    return dict(side_m=.045,mass_ratio=.2,initial=dict(drone_xyz=[0,0,1],drone_quaternion_wxyz=[1,0,0,0],
          package_xyz=[0,0,.6575],package_quaternion_wxyz=[1,0,0,0]))


def test_actual_reset_properties_and_native_ball_fk_are_screened():
    assert api().reset_failures(draw(),snapshot(),.02700047753751278,True) == []


@pytest.mark.parametrize(('mutation','reason'),[
    ('mass','PACKAGE_MASS'),('inertia','PACKAGE_INERTIA'),('aabb','COLLISION_AABB'),
    ('pose','INITIAL_POSE'),('velocity','NONZERO_RESET_VELOCITY'),('ball','NATIVE_BALL_FK'),
    ('reference','ORIGINAL_MASS'),('align','ROOT_ALIGNMENT'),('nonfinite','NONFINITE_STATE')])
def test_actual_measurements_cannot_be_replaced_by_declared_values(mutation,reason):
    state=snapshot()
    if mutation=='mass':
        state['links'][6]['mass_kg']=.1
    elif mutation=='inertia':
        state['inertia_kg_m2'][6][0][0]=1.
    elif mutation=='aabb':
        state['package_aabb'][1][0]=.5
    elif mutation=='pose':
        state['package_xyz'][0]=.01
    elif mutation=='velocity':
        state['dofs_velocity'][0]=.01
    elif mutation=='ball':
        state['qpos'][7:11]=[.7071067811865476,.7071067811865476,0,0]
    elif mutation=='reference':
        state['reference_original_mass_kg']=.01
    elif mutation=='align':
        state['align_root']=False
    else:
        state['package_velocity'][0]=float('nan')
    assert reason in api().reset_failures(draw(),state,.02700047753751278,True)
