"""Strict CPU bindings for the inherited precision32 fixed-anchor passive model.

These checks validate measured metadata and raw observations. They never grant
physical or optimizer admission and cannot replace the actual source/asset hash
bindings or independent energy, period, contact and constraint screens.
"""
import math

import numpy as np

from .payload_passive_check import rotation

REFERENCE_MASS = .02700047753751278
NAMES = ('base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link',
         'suspension_rod', 'payload')


def _array(value, shape):
    """Require the actual shape before arithmetic, preventing scalar broadcasting."""
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Missing, malformed or nonfinite measured array')
    return array


def _near(value, expected, *, atol=1e-7, rtol=0):
    target = np.asarray(expected, dtype=float)
    return bool(np.allclose(_array(value, target.shape), target, atol=atol, rtol=rtol))


def _scalar(value):
    if isinstance(value, (bool, list, dict)):
        raise ValueError('Finite scalar required')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Finite scalar required')
    return result


def _case(case):
    side, ratio = _scalar(case['side_m']), _scalar(case['mass_ratio'])
    if side not in (.03, .045, .06) or not .1 <= ratio <= .3:
        raise ValueError('Inherited load domain required')
    return side, ratio


def physical_errors(runtime, case):
    """Bind a fixed-anchor runtime to its declared load and actual timestep.

    Runtime link snapshots precede the declared swing initialization in the
    inherited probe. Fixed rotors have authored zero mass; the free-vehicle
    epsilon-regularized reference remains separately fixed at REFERENCE_MASS.
    """
    errors = []
    try:
        side, ratio = _case(case)
        if case['kind'] not in ('anchor', 'large_angle'):
            raise ValueError('Only fixed-anchor passive fixtures are supported')
        links = runtime['links']
        if (not isinstance(links, list) or len(links) != 7
                or [link['name'] for link in links] != list(NAMES)
                or any(type(link['idx']) is not int for link in links)
                or [link['idx'] for link in links] != list(range(1, 8))):
            raise ValueError('Exact measured link names, order and indices required')
        mass = REFERENCE_MASS*ratio
        masses = _array([link['mass_kg'] for link in links], (7,))
        if not np.allclose(masses, [.027, 0, 0, 0, 0, .0001, mass], atol=0, rtol=1e-5):
            errors.append('EFFECTIVE_LINK_MASS_MISMATCH')
        if abs(_scalar(runtime['reference_mass_kg'])-REFERENCE_MASS) > 1e-8:
            errors.append('REFERENCE_MASS_DISCREPANCY')
        if not math.isclose(_scalar(runtime['fixture_original_mass_kg']), sum(masses[:5]), rel_tol=1e-5):
            errors.append('FIXTURE_MASS_MISMATCH')
        wanted_inertia = np.zeros((7, 3, 3))
        wanted_inertia[0] = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
        wanted_inertia[5] = np.diag([7.50025e-7, 7.50025e-7, 5e-11])
        wanted_inertia[6] = np.eye(3)*mass*side**2/6
        if not _near(runtime['inertia_kg_m2'], wanted_inertia, atol=1e-15, rtol=1e-5):
            errors.append('EFFECTIVE_INERTIA_MISMATCH')
        coms = [[0,0,0], [.028,-.028,0], [-.028,-.028,0], [-.028,.028,0],
                [.028,.028,0], [0,0,-.15], [0,0,0]]
        if not _near(runtime['links_com_body_m'], coms):
            errors.append('COM_FRAME_MISMATCH')
        for field in ('links_inertial_quaternion_wxyz', 'links_frame_offset_quaternion_wxyz'):
            if not _near(runtime[field], [[1,0,0,0]]*7):
                errors.append('INERTIAL_FRAME_MISMATCH')
        for field in ('initial_entity_pos', 'initial_base_pos'):
            if not _near(runtime[field], [0,0,1], atol=1e-6):
                errors.append('INITIAL_ROOT_METADATA_MISMATCH')
        positions = [[0,0,1]]*5+[[0,0,.98], [0,0,.68-side/2]]
        if (not _near(runtime['initial_links_pos'], positions, atol=1e-6)
                or not _near(runtime['initial_links_quat'], [[1,0,0,0]]*7, atol=1e-6)):
            errors.append('INITIAL_LINK_METADATA_MISMATCH')
        aabb = [[-side/2,-side/2,.68-side], [side/2,side/2,.68]]
        if not _near(runtime['package_aabb'], aabb, atol=1e-5):
            errors.append('PACKAGE_COLLISION_EXTENT')
        if (type(runtime['n_qs']) is not int or runtime['n_qs'] != 8
                or type(runtime['n_dofs']) is not int or runtime['n_dofs'] != 6):
            errors.append('NATIVE_BALL_DOF')
        joints = runtime['joints']
        expected = [('suspension_ball',0,0), ('payload_ball',4,3)]
        if not isinstance(joints, list) or len(joints) != 2:
            raise ValueError('Both measured native ball joints required')
        for joint, (name, qstart, dstart) in zip(joints, expected):
            if (joint['name'] != name or str(joint['type']) != '3'
                    or any(type(joint[key]) is not int for key in ('n_qs','n_dofs','q_start','dof_start'))
                    or (joint['n_qs'],joint['n_dofs'],joint['q_start'],joint['dof_start']) != (4,3,qstart,dstart)):
                errors.append('NATIVE_BALL_DOF')
        for field in ('damping_before', 'damping_after', 'armature'):
            if np.any(_array(runtime[field], (6,)) != 0):
                errors.append('UNDECLARED_PASSIVE_DAMPING_OR_ARMATURE')
        if runtime['root_aligned'] is not False:
            errors.append('ROOT_FRAME_MISMATCH')
        if (runtime['rotor_indices'] != [2,3,4,5] or runtime['rotor_spin'] != [-1,1,-1,1]
                or _scalar(runtime['kf']) != 3.16e-10 or _scalar(runtime['km']) != 7.94e-12
                or runtime['rpm_mapping'] != '(1+0.8*action)*14468.429183500699'):
            errors.append('ROTOR_IDENTITY')
        substeps = case['substeps']
        if type(substeps) is not int or not 1 <= substeps <= 256 or case['sample_dt_s'] != .01:
            raise ValueError('Actual finite integer substeps and inherited control interval required')
        settings = runtime['actual_solver_settings']
        for field, wanted in [('dt',.01), ('substep_dt',.01/substeps), ('_substep_dt',.01/substeps)]:
            if not math.isclose(_scalar(settings[field]), wanted, abs_tol=1e-15, rel_tol=1e-12):
                errors.append('ACTUAL_TIMESTEP_MISMATCH')
        if str(settings['_integrator']) != '2':
            errors.append('ACTUAL_INTEGRATOR_MISMATCH')
    except (KeyError, ValueError, TypeError, IndexError, OverflowError):
        errors.append('PHYSICAL_METADATA_INCOMPLETE_OR_INVALID')
    return list(dict.fromkeys(errors))


def initial_errors(row, case):
    """Preserve inherited pose/direction and zero-velocity thresholds with shapes."""
    errors = []
    try:
        side, _ = _case(case)
        theta = math.radians(_scalar(case['initial_swing_deg']))
        phi = _scalar(case['initial_direction_rad'])
        direction = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), -math.cos(theta)])
        q = [math.cos(theta/2), math.sin(phi)*math.sin(theta/2), -math.cos(phi)*math.sin(theta/2), 0]
        expected_rotation = rotation(q)
        positions, rotations = {}, {}
        for body in ('drone', 'rod', 'package'):
            positions[body] = _array(row[f'{body}_xyz'], (3,))
            quat = _array(row[f'{body}_quaternion_wxyz'], (4,))
            if abs(np.linalg.norm(quat)-1) > 1e-4:
                errors.append('INITIAL_INVALID_QUATERNION')
            rotations[body] = rotation(quat)
            for field in ('velocity', 'angular_velocity'):
                if np.linalg.norm(_array(row[f'{body}_{field}'], (3,))) > 1e-7:
                    errors.append('INITIAL_NONZERO_VELOCITY')
        anchor = positions['drone']+rotations['drone'] @ [0,0,-.02]
        endpoint = positions['package']+rotations['package'] @ [0,0,side/2]
        if (not _near(positions['drone'], [0,0,1], atol=1e-6)
                or not _near(rotations['drone'], np.eye(3), atol=1e-6)
                or not _near(positions['rod'], anchor, atol=1e-6)):
            errors.append('INITIAL_ROOT_OR_ANCHOR_POSE_MISMATCH')
        length = float(np.linalg.norm(endpoint-anchor))
        if length <= 0 or abs(length-.3) > 2e-5:
            errors.append('INITIAL_LINK_LENGTH_MISMATCH')
        if length <= 0 or not _near((endpoint-anchor)/length, direction, atol=1e-6):
            errors.append('INITIAL_DIRECTION_MISMATCH')
        for body in ('rod', 'package'):
            if not _near(rotations[body], expected_rotation, atol=1e-6):
                errors.append(f'INITIAL_{body.upper()}_ORIENTATION_MISMATCH')
        wanted_center = anchor+.3*direction-expected_rotation @ [0,0,side/2]
        if not _near(positions['package'], wanted_center, atol=1e-6):
            errors.append('INITIAL_PACKAGE_CENTER_MISMATCH')
    except (KeyError, ValueError, TypeError, IndexError, OverflowError):
        errors.append('INITIAL_STATE_INCOMPLETE_OR_INVALID')
    return list(dict.fromkeys(errors))


def wrench_errors(row, runtime):
    """Require a complete env1 global external wrench field, not a zero subset."""
    errors = []
    try:
        links = runtime['links']
        indices = [link['idx'] for link in links]
        if (len(links) != 7 or any(type(i) is not int for i in indices)
                or indices != list(range(1,8))
                or runtime['rotor_indices'] != [2,3,4,5]
                or row['rotor_indices'] != runtime['rotor_indices']):
            raise ValueError('Actual global link/rotor indices required')
        if np.any(_array(row['rpm'], (4,)) != 0):
            errors.append('PASSIVE_MOTOR_INPUT')
        shape = (max(indices)+1, 1, 3)
        for field in ('negative_external_force_field', 'negative_external_torque_about_root_com_field'):
            if np.any(_array(row[field], shape) != 0):
                errors.append('NONZERO_EXTERNAL_WRENCH')
    except (KeyError, ValueError, TypeError, IndexError, OverflowError):
        errors.append('EXTERNAL_WRENCH_INCOMPLETE_OR_INVALID')
    return list(dict.fromkeys(errors))
