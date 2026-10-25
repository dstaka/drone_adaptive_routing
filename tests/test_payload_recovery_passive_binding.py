"""Reject relabelled passive cases and incomplete measured physical observations."""
import copy
import importlib
import math
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / 'src/adaptive_drone_patrol/payload_recovery_passive_binding.py'


def api():
    assert MODULE.is_file(), 'Strict physical/state binding helper is not implemented'
    return importlib.import_module('adaptive_drone_patrol.payload_recovery_passive_binding')


def fixture(side=.06, ratio=.1, angle=90., azimuth=0., substeps=128):
    case = dict(kind='large_angle', side_m=side, mass_ratio=ratio, initial_swing_deg=angle,
                initial_direction_rad=azimuth, substeps=substeps, sample_dt_s=.01)
    names = ['base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link', 'suspension_rod', 'payload']
    mass = .02700047753751278 * ratio
    inertia = [np.diag([1.4e-5, 1.4e-5, 2.17e-5]).tolist()] + [np.zeros((3, 3)).tolist() for _ in range(4)]
    inertia += [np.diag([7.50025e-7, 7.50025e-7, 5e-11]).tolist(), (np.eye(3)*mass*side**2/6).tolist()]
    positions = [[0., 0., 1.]] * 5 + [[0., 0., .98], [0., 0., .68-side/2]]
    runtime = dict(reference_mass_kg=.02700047753751278, fixture_original_mass_kg=.027,
                   links=[dict(name=n, idx=i+1, mass_kg=m) for i, (n, m) in enumerate(
                       zip(names, [.027, 0., 0., 0., 0., .0001, mass]))], inertia_kg_m2=inertia,
                   links_com_body_m=[[0., 0., 0.], [.028, -.028, 0.], [-.028, -.028, 0.],
                                     [-.028, .028, 0.], [.028, .028, 0.], [0., 0., -.15], [0., 0., 0.]],
                   n_qs=8, n_dofs=6, joints=[dict(name='suspension_ball', type='3', n_qs=4, n_dofs=3,
                                                q_start=0, dof_start=0),
                                               dict(name='payload_ball', type='3', n_qs=4, n_dofs=3,
                                                    q_start=4, dof_start=3)],
                   initial_entity_pos=[0., 0., 1.], initial_base_pos=[0., 0., 1.],
                   initial_links_pos=positions, initial_links_quat=[[1., 0., 0., 0.]]*7,
                   links_inertial_quaternion_wxyz=[[1., 0., 0., 0.]]*7,
                   links_frame_offset_quaternion_wxyz=[[1., 0., 0., 0.]]*7,
                   actual_solver_settings=dict(dt='0.01', substep_dt=str(.01/substeps),
                                               _substep_dt=str(.01/substeps), _integrator='2'),
                   rotor_indices=[2, 3, 4, 5], rotor_spin=[-1, 1, -1, 1], kf=3.16e-10, km=7.94e-12,
                   rpm_mapping='(1+0.8*action)*14468.429183500699', damping_before=[0.]*6,
                   damping_after=[0.]*6, armature=[0.]*6, root_aligned=False,
                   package_aabb=[[-side/2, -side/2, .68-side], [side/2, side/2, .68]])
    t, p = math.radians(angle), azimuth
    q = [math.cos(t/2), math.sin(p)*math.sin(t/2), -math.cos(p)*math.sin(t/2), 0.]
    direction = np.array([math.sin(t)*math.cos(p), math.sin(t)*math.sin(p), -math.cos(t)])
    row = dict(drone_xyz=[0., 0., 1.], drone_quaternion_wxyz=[1., 0., 0., 0.],
               rod_xyz=[0., 0., .98], rod_quaternion_wxyz=q, package_quaternion_wxyz=q,
               package_xyz=(np.array([0., 0., .98])+(.3+side/2)*direction).tolist())
    for body in ('drone', 'rod', 'package'):
        for key in ('velocity', 'angular_velocity'):
            row[f'{body}_{key}'] = [0., 0., 0.]
    wrench = dict(rotor_indices=[2, 3, 4, 5], rpm=[0.]*4,
                  negative_external_force_field=np.zeros((8, 1, 3)).tolist(),
                  negative_external_torque_about_root_com_field=np.zeros((8, 1, 3)).tolist())
    return runtime, case, row, wrench


@pytest.mark.parametrize('side,ratio,angle,azimuth', [(.03,.1,5.,0.), (.045,.2,170.,1.2), (.06,.3,90.,-2.)])
def test_authored_load_and_initial_pose_are_bound_for_all_sizes(side, ratio, angle, azimuth):
    runtime, case, row, wrench = fixture(side, ratio, angle, azimuth)
    assert api().physical_errors(runtime, case) == []
    assert api().initial_errors(row, case) == []
    assert api().wrench_errors(wrench, runtime) == []


@pytest.mark.parametrize('field,value', [
    ('reference_mass_kg', .027), ('fixture_original_mass_kg', .028), ('n_qs', 15), ('n_dofs', 12),
    ('links', []), ('inertia_kg_m2', 0.), ('links_com_body_m', 0.),
    ('links_inertial_quaternion_wxyz', [1.,0.,0.,0.]),
    ('links_frame_offset_quaternion_wxyz', [1.,0.,0.,0.]),
    ('initial_entity_pos', 0.), ('initial_base_pos', [0.,0.,1.1]), ('initial_links_pos', []),
    ('initial_links_quat', [1.,0.,0.,0.]), ('package_aabb', 0.),
    ('damping_after', 0.), ('damping_before', [1e-5]*6), ('armature', [0.]*5),
    ('rotor_indices', [1,2,3,4]), ('rotor_spin', [1,1,-1,1]), ('kf', 4e-10),
    ('km', float('nan')), ('rpm_mapping', 'other'), ('root_aligned', True),
])
def test_physical_scalar_broadcast_and_changed_properties_fail_closed(field, value):
    runtime, case, _, _ = fixture()
    runtime[field] = value
    assert api().physical_errors(runtime, case)


@pytest.mark.parametrize('mutation', ['cube_mass', 'rod_mass', 'base_mass', 'rotor_mass', 'duplicate_name',
                                      'duplicate_index', 'cube_inertia', 'rod_inertia', 'base_inertia',
                                      'cube_COM', 'rod_COM', 'inertial_frame', 'frame_offset', 'aabb',
                                      'joint_type', 'joint_start', 'joint_count', 'damping', 'armature',
                                      'dt', 'substep_dt', '_substep_dt', '_integrator', 'wrong_side', 'wrong_ratio'])
def test_cases_cannot_be_relabelled_with_wrong_effective_load_or_timestep(mutation):
    rt, case, _, _ = fixture()
    if mutation in ('cube_mass', 'rod_mass', 'base_mass', 'rotor_mass'):
        rt['links'][{'cube_mass':6,'rod_mass':5,'base_mass':0,'rotor_mass':1}[mutation]]['mass_kg'] += .0001
    elif mutation == 'duplicate_name':
        rt['links'][6]['name'] = 'suspension_rod'
    elif mutation == 'duplicate_index':
        rt['links'][6]['idx'] = 6
    elif mutation.endswith('_inertia'):
        rt['inertia_kg_m2'][{'cube_inertia':6,'rod_inertia':5,'base_inertia':0}[mutation]][0][0] *= 1.01
    elif mutation.endswith('_COM'):
        rt['links_com_body_m'][6 if mutation == 'cube_COM' else 5][2] += .001
    elif mutation in ('inertial_frame','frame_offset'):
        key = 'links_inertial_quaternion_wxyz' if mutation == 'inertial_frame' else 'links_frame_offset_quaternion_wxyz'
        rt[key] = copy.deepcopy(rt[key])
        rt[key][5] = [.70710678, .70710678, 0., 0.]
    elif mutation == 'aabb':
        rt['package_aabb'][1][2] += .001
    elif mutation.startswith('joint_'):
        rt['joints'][0][{'joint_type':'type','joint_start':'q_start','joint_count':'n_dofs'}[mutation]] = 2
    elif mutation in ('damping','armature'):
        rt['damping_after' if mutation == 'damping' else 'armature'][0] = 1e-6
    elif mutation in ('dt','substep_dt','_substep_dt','_integrator'):
        rt['actual_solver_settings'][mutation] = '0.005' if mutation == 'dt' else '1'
    elif mutation == 'wrong_side':
        case['side_m'] = .03
    else:
        case['mass_ratio'] = .3
    assert api().physical_errors(rt, case)


@pytest.mark.parametrize('field,value', [('drone_xyz', 0.), ('drone_xyz', [0.,0.,1.001]),
    ('drone_quaternion_wxyz', [2.,0.,0.,0.]), ('rod_xyz', [0.,0.,.981]),
    ('package_xyz', [0.,0.,.68]), ('rod_quaternion_wxyz', [1.,0.,0.,0.]),
    ('package_quaternion_wxyz', [1.,0.,0.,0.]), ('rod_velocity', 0.),
    ('package_velocity', [float('nan'),0.,0.]), ('drone_angular_velocity', [0.,0.,1e-6])])
def test_initial_missing_shapes_nonfinite_or_wrong_pose_are_rejected(field, value):
    _, case, row, _ = fixture()
    row[field] = value
    assert api().initial_errors(row, case)


def test_azimuth_relabelling_and_missing_velocity_cannot_pass():
    _, case, row, _ = fixture(azimuth=1.2)
    assert api().initial_errors(row, {**case, 'initial_direction_rad':0.})
    del row['rod_velocity']
    assert api().initial_errors(row, case)


@pytest.mark.parametrize('value', [[], 0., [[[0.,0.,0.]]], np.zeros((7,1,3)).tolist(),
                                  np.zeros((8,2,3)).tolist(), np.zeros((8,1,1)).tolist(),
                                  np.full((8,1,3),float('nan')).tolist(), np.ones((8,1,3)).tolist()])
def test_empty_truncated_broadcast_and_nonzero_wrench_fields_fail(value):
    rt, _, _, row = fixture()
    row['negative_external_force_field'] = value
    assert api().wrench_errors(row, rt)


def test_wrench_observation_binds_rotor_indices_rpm_and_both_global_fields():
    rt, _, _, row = fixture()
    row['rotor_indices'] = [2,3,4,6]
    assert api().wrench_errors(row, rt)
    row = fixture()[3]
    row['rpm'] = 0.
    assert api().wrench_errors(row, rt)
    row = fixture()[3]
    del row['negative_external_torque_about_root_com_field']
    assert api().wrench_errors(row, rt)
