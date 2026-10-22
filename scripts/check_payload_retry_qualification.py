#!/usr/bin/env python
"""Aggregate actual native-ball qualification evidence, with explicit work limits.

No simulator, optimizer, policy, producer or admission-switch code is imported.
An endpoint power guard is not a proof about unobserved substep work.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'configs/payload_retry_probe_manifest.json'
BACKEND = 'src/adaptive_drone_patrol/payload_ball_bank.py'
CONTROLLER = 'src/adaptive_drone_patrol/payload_probe_controller.py'
MOTOR_PLAN = 'configs/payload_retry_motor_audit.json'
CASE_HASH = 'f98e0632d96a4d7b40feda667bff6e9ca831fc7f67e1734f2438e3c527ecf01c'
SCENARIO_HASHES = {'normal': 'dd5f8d7aeca20d9e238581098e91a6d7cc0e6d02dd42d6b55882e59ac833d3aa',
    'emergency': '7f0629c872837ebf3f53afdd572d07e89832a8c13afd74325d89d3af28889c5d',
    'cost_shift': '3a408c294b308def450a953e763f9ecca1c50782a0feca0439826a28391128ff'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_checker(name):
    path = ROOT/'scripts'/name
    spec = importlib.util.spec_from_file_location('aggregate_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, (float, int)) else True


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-4:
        raise ValueError('Invalid unit quaternion')
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def angle(a, b):
    rotation(a)
    rotation(b)
    a, b = np.asarray(a, float), np.asarray(b, float)
    dot = abs(float(a @ b/(np.linalg.norm(a)*np.linalg.norm(b))))
    return math.degrees(2*math.acos(float(np.clip(dot, -1, 1))))


def inventory(manifest):
    expected = {}
    for side in (.03, .045, .06):
        for ratio in (.1, .2, .3):
            suffix = f'{side:.3f}_{ratio:.2f}'
            expected['anchor_'+suffix] = ('anchor', side, ratio, 120., 5., 2)
            expected['large_'+suffix] = ('large_angle', side, ratio, 30., 90., 2)
            for swing in (0, 10):
                expected[f'powered_{suffix}_{swing}'] = ('powered', side, ratio, 120., swing, 2)
        for ratio in (.1, .3):
            for substeps in (2, 4):
                expected[f'refine_{side:.3f}_{ratio:.2f}_{substeps}'] = ('refinement', side, ratio, 30., 10., substeps)
    for swing in (5, 45, 80, 89, 91, 120, 170):
        expected[f'sweep_{swing}'] = ('large_angle', .045, .2, 30., swing, 2)
    counts = {kind: 0 for kind in ('anchor', 'large_angle', 'powered', 'refinement')}
    valid = True
    try:
        cases = manifest['cases']
        valid = len(cases) == 55 and {c['id'] for c in cases} == set(expected)
        for case in cases:
            kind, side, ratio, minimum, swing, steps = expected[case['id']]
            counts[kind] += 1
            if ((case['kind'], case['side_m'], case['mass_ratio'], case['initial_swing_deg'], case['substeps'])
                    != (kind, side, ratio, swing, steps) or case['duration_s'] < minimum
                    or not math.isfinite(case['duration_s']) or case['sample_dt_s'] != .01
                    or case['initial_direction_rad'] != 0 or case['reset_during_run']
                    or not 0 < case['timeout_wall_s'] <= 300):
                valid = False
    except (KeyError, ValueError, TypeError):
        valid = False
    pairs = [(f'refine_{side:.3f}_{ratio:.2f}_2', f'refine_{side:.3f}_{ratio:.2f}_4')
             for side in (.03, .045, .06) for ratio in (.1, .3)]
    return dict(valid=valid, kind_counts=counts, refinement_pairs=pairs, required_cases=55)


def compare_refinement(left, right):
    failures = []
    result = dict(passed=False, samples=min(len(left), len(right)), maximum_position_difference_m=0.,
                  maximum_orientation_difference_deg=0., failures=failures)
    if len(left) != 3001 or len(right) != 3001:
        failures.append('INCOMPLETE_30_SECOND_PAIR')
    try:
        for i, (a, b) in enumerate(zip(left, right)):
            if (a['step'] != i or b['step'] != i or abs(a['time_s']-i*.01) > 1e-8
                    or abs(b['time_s']-i*.01) > 1e-8):
                raise ValueError('Unmatched timestamps')
            for body in ('drone', 'package'):
                xyz_a, xyz_b = np.asarray(a[body+'_xyz']), np.asarray(b[body+'_xyz'])
                if xyz_a.shape != (3,) or xyz_b.shape != (3,) or not np.isfinite([xyz_a, xyz_b]).all():
                    raise ValueError('Invalid actual matched positions')
                error = float(np.linalg.norm(xyz_a-xyz_b))
                if not math.isfinite(error):
                    raise ValueError('Finite coordinates produced unusable displacement')
                turn = angle(a[body+'_quaternion_wxyz'], b[body+'_quaternion_wxyz'])
                result['maximum_position_difference_m'] = max(result['maximum_position_difference_m'], error)
                result['maximum_orientation_difference_deg'] = max(result['maximum_orientation_difference_deg'], turn)
        if result['maximum_position_difference_m'] > .02 or result['maximum_orientation_difference_deg'] > 2.:
            failures.append('REFINEMENT_DIFFERENCE_EXCEEDED')
    except (ValueError, KeyError, TypeError, IndexError):
        failures.append('INCOMPLETE_OR_NONFINITE_MATCHED_STATE')
    result['passed'] = not failures
    return result


def energy_growth_guard(energies, endpoint_powers, *, dt=.01, justified_work_bound=None):
    energies = np.asarray(energies, float)
    powers = np.asarray(endpoint_powers, float)
    if (energies.ndim != 1 or len(energies) < 2 or not np.isfinite(energies).all()
            or not np.isfinite(powers).all()):
        raise ValueError('Complete finite energy and endpoint power traces required')
    if powers.shape == energies.shape:
        interval_maximum = np.maximum(np.abs(powers[:-1]), np.abs(powers[1:]))
    elif powers.shape == (len(energies)-1,2):
        interval_maximum = np.max(np.abs(powers),axis=1)
    else:
        raise ValueError('Each control interval needs both endpoint powers')
    bound = .02+1.1*np.r_[0., np.cumsum(dt*interval_maximum)]
    growth = energies-energies[0]
    residual = growth-bound
    if not np.isfinite(bound).all() or not np.isfinite(residual).all():
        raise ValueError('Unusable finite-input energy arithmetic')
    exceeded = bool(np.any(residual > 0))
    established = justified_work_bound is not None
    source_exceeded = None
    if established:
        justified = np.asarray(justified_work_bound,float)
        if (justified.shape != energies.shape or not np.isfinite(justified).all() or justified[0] != 0
                or np.any(np.diff(justified) < 0)):
            raise ValueError('A finite cumulative independently justified work bound is required')
        source_exceeded = bool(np.any(growth > .02+justified))
    return dict(work_guard_j=float(bound[-1]), maximum_energy_growth_j=float(np.max(growth)),
                maximum_growth_above_guard_j=float(np.max(residual)), guard_exceeded=exceeded,
                first_guard_exceedance_index=int(np.flatnonzero(residual > 0)[0]) if exceeded else None,
                conservative_substep_bound_established=established, source_work_bound_exceeded=source_exceeded,
                classification='ACTUAL_ENERGY_GUARD_EXCURSION' if exceeded or source_exceeded else
                    'SOURCE_JUSTIFIED_WORK_BOUND_SCREEN' if established else 'INSUFFICIENT_WORK_BOUND_EVIDENCE',
                interpretation=('Endpoint guard plus separately verified conservative substep path work; not an exact energy balance.'
                    if established else 'Endpoint guard alone cannot bound hidden substep work.'))


def substep_path_bound(records, *, control_steps, substeps):
    """Conservative path bound for a separately declared/logged integration hook.

    The pinned articulated integrator advances the free root by vel_next*h and
    its quaternion by the corresponding angular rotation vector. World forces
    and root-COM moments are constant during that substep. Transforming the
    moment to the authored root P gives M_P=M_C+(C-P)xF. Every positive-mass COM
    lies within .5m of P for the audited .02+.30+.03m fixed-link geometry.
    Thus |F dot deltaP|+(|M_C|+.5|F|)*|omega_next|*h bounds external work.

    This function is used only after the hook, engine and main raw bindings
    are verified. It does not infer hidden substeps from control endpoints.
    """
    failures, cumulative = [], [0.]
    result = dict(established=False,failures=failures,cumulative_bound_j=cumulative,
                  root_com_radius_bound_m=.5,method='constant-world-cfrc+semiimplicit-root-path-v1')
    if len(records) != control_steps*substeps:
        failures.append('INCOMPLETE_SUBSTEP_PATH')
        return result
    previous = None
    total = 0.
    try:
        for i,record in enumerate(records):
            if (record['control_step'] != i//substeps or record['substep'] != i%substeps
                    or record['dt_s'] != .01/substeps or abs(record['time_s']-i*.01/substeps) > 1e-8
                    or not finite(record)):
                raise ValueError('Invalid substep path continuity')
            q0,q1,vel = [np.asarray(record[key],float) for key in ('qpos_before','qpos_after','dofs_velocity_next')]
            if q0.shape != (15,) or q1.shape != (15,) or vel.shape != (12,):
                raise ValueError('Missing actual integration coordinates')
            if (np.linalg.norm(vel[:3]) > 20 or
                    any(np.linalg.norm(vel[j:j+3]) > 200 for j in (3,6,9)) or
                    any(abs(np.linalg.norm(q[j:j+4])-1) > 1e-4
                        for q in (q0,q1) for j in (3,7,11))):
                raise ValueError('Finite substep numerical excursion')
            if previous is not None and not np.array_equal(q0,previous):
                raise ValueError('Unrecorded substep state change')
            h = record['dt_s']
            delta = q1[:3]-q0[:3]
            turn = float(np.linalg.norm(vel[3:6])*h)
            if (not np.allclose(delta,vel[:3]*h,atol=1e-6,rtol=1e-6)
                    or math.radians(angle(q0[3:7],q1[3:7])) > turn+1e-6):
                raise ValueError('Logged velocity is not the integration path')
            for key,q in [('root_com_before',q0),('root_com_after',q1)]:
                if np.linalg.norm(np.asarray(record[key])-q[:3]) > .5:
                    raise ValueError('Authored geometry radius bound violated')
            fields = [np.asarray(record[key],float) for key in ('negative_external_force_field',
                'negative_external_torque_about_root_com_field','negative_external_force_field_after',
                'negative_external_torque_about_root_com_field_after')]
            if any(field.ndim != 3 or field.shape[1:] != (1,3) or field.shape[0] < 8 for field in fields):
                raise ValueError('Missing full actual world wrench fields')
            if not np.array_equal(fields[0],fields[2]) or not np.array_equal(fields[1],fields[3]):
                raise ValueError('World wrench not constant inside substep')
            for field in fields:
                others = [index for index in range(len(field)) if index not in (2,3,4,5)]
                if np.any(np.abs(field[others]) > 1e-7):
                    raise ValueError('Direct force on nonrotor')
            force = -fields[0][[2,3,4,5],0].sum(axis=0)
            moment = -fields[1][[2,3,4,5],0].sum(axis=0)
            total += abs(float(force @ delta))+(np.linalg.norm(moment)+.5*np.linalg.norm(force))*turn
            if not math.isfinite(total):
                raise ValueError('Unusable finite-input work arithmetic')
            previous = q1
            if i%substeps == substeps-1:
                cumulative.append(float(total))
    except (ValueError,KeyError,TypeError,IndexError):
        failures.append('SUBSTEP_PATH_OR_WRENCH_UNVERIFIED')
    result['established'] = not failures
    return result


def work_evidence(path,source_root,case,manifest,metadata,summary,rows,inputs,engine_verified):
    method = 'constant-world-cfrc+semiimplicit-root-path-v1'
    result = dict(established=False,failures=['SUBSTEP_WORK_EVIDENCE_NOT_RECORDED'])
    if (manifest['energy'].get('substep_work_bound_method') != method
            or metadata.get('substep_work_method') != method or not engine_verified):
        return result
    try:
        path = Path(path)/'substep-work.jsonl'
        if digest(path) != summary['substep_work_sha256']:
            raise ValueError('Substep work raw hash mismatch')
        binding = metadata['substep_work_source_binding']
        if (set(binding) != {'scripts/probe_payload_ball.py',
                             'src/adaptive_drone_patrol/payload_substep_probe.py'} or
                any(Path(name).is_absolute() or '..' in Path(name).parts
                              or digest(Path(source_root)/name) != wanted for name,wanted in binding.items())):
            raise ValueError('Actual recording hook source is not bound')
        records = [json.loads(line) for line in path.read_text().splitlines()]
        result = substep_path_bound(records,control_steps=len(rows)-1,substeps=case['substeps'])
        for i,input_record in enumerate(inputs):
            first,last = records[i*case['substeps']],records[(i+1)*case['substeps']-1]
            if (not np.array_equal(first['qpos_before'],rows[i]['qpos'])
                    or not np.array_equal(last['qpos_after'],rows[i+1]['qpos'])
                    or not np.array_equal(first['negative_external_force_field'],input_record['negative_external_force_field'])
                    or not np.array_equal(first['negative_external_torque_about_root_com_field'],
                                          input_record['negative_external_torque_about_root_com_field'])):
                raise ValueError('Substep path differs from actual main input/state')
        result['raw_sha256'] = digest(path)
    except (OSError,ValueError,KeyError,TypeError,IndexError):
        result = dict(established=False,failures=['SUBSTEP_WORK_BINDING_OR_PATH_MISMATCH'])
    return result


def initial_case_errors(row, case):
    """Reconstruct declared world swing direction and zero lower-ball pose."""
    errors = []
    try:
        theta, phi = math.radians(case['initial_swing_deg']), case['initial_direction_rad']
        expected_direction = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), -math.cos(theta)])
        q = [math.cos(theta/2), math.sin(phi)*math.sin(theta/2), -math.cos(phi)*math.sin(theta/2), 0.]
        expected_rod = rotation(q)
        rd, rr, rp = [rotation(row[body+'_quaternion_wxyz']) for body in ('drone','rod','package')]
        drone = np.asarray(row['drone_xyz'])
        anchor = drone+rd @ [0,0,-.02]
        endpoint = np.asarray(row['package_xyz'])+rp @ [0,0,case['side_m']/2]
        if (not np.allclose(drone,[0,0,1],atol=1e-6,rtol=0) or not np.allclose(rd,np.eye(3),atol=1e-6,rtol=0)
                or not np.allclose(row['rod_xyz'],anchor,atol=1e-6,rtol=0)):
            errors.append('INITIAL_ROOT_OR_ANCHOR_POSE_MISMATCH')
        direction = (endpoint-anchor)/np.linalg.norm(endpoint-anchor)
        if not np.allclose(direction,expected_direction,atol=1e-6,rtol=0):
            errors.append('INITIAL_DIRECTION_MISMATCH')
        if not np.allclose(rr,expected_rod,atol=1e-6,rtol=0):
            errors.append('INITIAL_ROD_ORIENTATION_MISMATCH')
        if not np.allclose(rp,expected_rod,atol=1e-6,rtol=0):
            errors.append('INITIAL_PACKAGE_ORIENTATION_MISMATCH')
        wanted_center = anchor+.3*expected_direction-expected_rod @ [0,0,case['side_m']/2]
        if not np.allclose(row['package_xyz'],wanted_center,atol=1e-6,rtol=0):
            errors.append('INITIAL_PACKAGE_CENTER_MISMATCH')
        if any(np.linalg.norm(row[f'{body}_{field}']) > 1e-7 for body in ('drone','rod','package')
               for field in ('velocity','angular_velocity')):
            errors.append('INITIAL_NONZERO_VELOCITY')
    except (ValueError, TypeError, KeyError, IndexError):
        errors.append('INITIAL_STATE_INCOMPLETE')
    return errors


def physical_parameters_errors(runtime,case):
    errors = []
    try:
        links = {link['name']:link for link in runtime['links']}
        mass = .02700047753751278*case['mass_ratio']
        fixed = case['kind'] in ('anchor','large_angle')
        if (set(links) != {'base_link','prop0_link','prop1_link','prop2_link','prop3_link','suspension_rod','payload'}
                or any(not math.isfinite(link['mass_kg']) or link['mass_kg'] < 0 for link in links.values())
                or not math.isclose(links['base_link']['mass_kg'],.027,rel_tol=1e-5)
                or any((links[f'prop{i}_link']['mass_kg'] != 0 if fixed else not math.isclose(
                    links[f'prop{i}_link']['mass_kg'],1.1920928955078125e-7,rel_tol=1e-5)) for i in range(4))
                or not math.isclose(links['suspension_rod']['mass_kg'],.0001,rel_tol=1e-5)
                or not math.isclose(links['payload']['mass_kg'],mass,rel_tol=1e-5)):
            errors.append('EFFECTIVE_LINK_MASS_MISMATCH')
        expected = np.zeros(6 if fixed else 12)
        if not fixed:
            expected[3:6] = (.027+mass+.0001)*1e-5
        damping = np.asarray(runtime['damping_after'],float)
        armature = np.asarray(runtime['armature'],float)
        if (damping.shape != expected.shape or not np.isfinite(damping).all()
                or not np.allclose(damping,expected,atol=1e-13,rtol=1e-5)):
            errors.append('UNDECLARED_PASSIVE_DAMPING')
        if armature.shape != expected.shape or not np.isfinite(armature).all() or np.any(armature != 0):
            errors.append('ARMATURE_MISMATCH')
    except (KeyError,ValueError,TypeError):
        errors.append('EFFECTIVE_PHYSICAL_PARAMETERS_UNVERIFIED')
    return errors


def static_geometry(source_root, initial_cases):
    result = dict(passed=False, initial_cases=0, locations=0, dynamic_clearance_certified=False,
                  minimum_initial_package_floor_m=None, minimum_nominal_package_floor_m=None, failures=[])
    try:
        initial_cases = Path(initial_cases)
        if digest(initial_cases) != CASE_HASH:
            raise ValueError('The preserved exact 127-case draws are required')
        bundle = read(initial_cases)
        cases = bundle['cases']
        if len(cases) != 127 or bundle['link_m'] != .3 or bundle['drone_attachment_body_m'] != [0., 0., -.02]:
            raise ValueError('Changed inherited geometry contract')
        layouts = []
        for name, expected in SCENARIO_HASHES.items():
            path = Path(source_root)/f'configs/demo_scenarios_dispersed/{name}.yaml'
            if digest(path) != expected:
                raise ValueError('Changed inherited destination file')
            layouts.append(yaml.safe_load(path.read_text())['coords'])
        if any(layout != layouts[0] for layout in layouts) or len(layouts[0]) != 11 or layouts[0][0] != [0., 0., 1.]:
            raise ValueError('Initial S or target coordinates changed')
        floors = []
        for case in cases:
            state, side = case['initial'], case['side_m']
            rd, rp = rotation(state['drone_quaternion_wxyz']), rotation(state['package_quaternion_wxyz'])
            anchor = np.asarray(state['drone_xyz'])+rd @ [0, 0, -.02]
            center = np.asarray(state['package_xyz'])
            endpoint = center+rp @ [0, 0, side/2]
            if abs(np.linalg.norm(endpoint-anchor)-.3) > 2e-5:
                raise ValueError('Initial attachment inconsistency')
            floors.append(float(center[2]-side/2*np.abs(rp[2]).sum()))
        nominal = [float(point[2]-.02-.3-side) for point in layouts[0] for side in (.03, .045, .06)]
        result.update(initial_cases=127, locations=11, minimum_initial_package_floor_m=min(floors),
                      minimum_nominal_package_floor_m=min(nominal), initial_cases_sha256=digest(initial_cases),
                      nominal_size_location_checks=len(nominal))
        if min(floors+nominal) <= .1:
            raise ValueError('Initial or nominal oriented cube floor failed')
        result['passed'] = True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        result['failures'].append('STATIC_GEOMETRY_EVIDENCE_MISSING_OR_CHANGED')
    return result


def diagnostic_rpm(row, target, mass):
    """Independently reconstruct declared vehicle geometric PD, no load feedback."""
    R = rotation(row['drone_quaternion_wxyz'])
    acceleration = 1.2*(np.asarray(target)-row['drone_xyz'])-2*np.asarray(row['drone_velocity'])+[0, 0, 9.81]
    acceleration[:2] = np.clip(acceleration[:2], -5., 5.)
    acceleration[2] = np.clip(acceleration[2], 4., 15.)
    z = acceleration/np.linalg.norm(acceleration)
    y = np.cross(z, [1, 0, 0])
    y /= np.linalg.norm(y)
    desired = np.column_stack([np.cross(y, z), y, z])
    skew = .5*(desired.T @ R-R.T @ desired)
    attitude = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    omega = R.T @ np.asarray(row['drone_angular_velocity'])
    inertia = np.array([1.4e-5, 1.4e-5, 2.17e-5])
    torque = -100*inertia*attitude-20*inertia*omega+np.cross(omega, inertia*omega)
    thrust = max(0., mass*float(acceleration @ R[:, 2]))
    xy, kf, km = np.array([[.028,-.028],[-.028,-.028],[-.028,.028],[.028,.028]]), 3.16e-10, 7.94e-12
    allocation = np.array([[kf]*4, xy[:, 1]*kf, -xy[:, 0]*kf, km*np.array([-1,1,-1,1])])
    return np.sqrt(np.clip(np.linalg.solve(allocation, np.r_[thrust,torque]),
                           (.2*14468.429183500699)**2, (1.8*14468.429183500699)**2))


def system_energy(row, runtime, authored):
    """All seven rigid link COM energies in their measured authored frames."""
    by_name = {link['name']: link for link in runtime['links']}
    value = 0.
    for body, name, offset in [('drone','base_link',[0,0,0]), ('rod','suspension_rod',[0,0,-.15]), ('package','payload',[0,0,0])]:
        R = rotation(row[body+'_quaternion_wxyz'])
        arm = R @ offset
        xyz = np.asarray(row[body+'_xyz'])+arm
        omega = np.asarray(row[body+'_angular_velocity'])
        velocity = np.asarray(row[body+'_velocity'])+np.cross(omega, arm)
        mass = by_name[name]['mass_kg']
        value += .5*mass*float(velocity @ velocity)+.5*float(omega @ R @ authored[name] @ R.T @ omega)+mass*9.81*xyz[2]
    R = rotation(row['drone_quaternion_wxyz'])
    for i in range(4):
        name = f'prop{i}_link'
        mass = by_name[name]['mass_kg']
        velocity, omega = np.asarray(row['rotor_com_velocity'][i]), np.asarray(row['rotor_angular_velocity'][i])
        value += .5*mass*float(velocity @ velocity)+.5*float(omega @ R @ authored[name] @ R.T @ omega)
        value += mass*9.81*row['rotor_com_world'][i][2]
    if not math.isfinite(value):
        raise ValueError('Nonfinite measured total energy')
    return value


def physical_signature(*, bindings, settings, manifest_sha256, engine, assets):
    """Bind effective physics to actual engine bytes, not its version alone."""
    return hashlib.sha256(json.dumps(dict(source_binding=bindings, settings=settings,
        manifest_sha256=manifest_sha256, engine_source_binding=engine, assets=assets),
        sort_keys=True).encode()).hexdigest()


def verify_qualification(probes, motor_probes, reset_probe, *, source_root=ROOT, initial_cases=None, engine_root=None,
                         geometry_root=ROOT):
    probes, source_root = Path(probes), Path(source_root)
    failures, missing, stages, outcomes, refinement_rows, engine_sets = [], [], {}, {}, {}, []
    result = dict(schema='payload-operating-domain-independent-v3', status='QUALIFICATION_BLOCKED',
                  physical_training_admitted=False, policy_admitted=False, failure_codes=failures,
                  missing_evidence=missing, checklist=stages, cases=outcomes, refinement={},
                  checker_sha256=digest(__file__), selected_profile=None)

    def fail(code):
        if code not in failures:
            failures.append(code)

    try:
        manifest, plan = read(source_root/MANIFEST), read(source_root/MOTOR_PLAN)
        expected = inventory(manifest)
        if not expected['valid']:
            raise ValueError('Incomplete 55-case declaration')
        bound_paths = (BACKEND, CONTROLLER, MOTOR_PLAN, MANIFEST,
                       'scripts/probe_payload_ball.py', 'scripts/probe_payload_ball_bank.py',
                       'src/adaptive_drone_patrol/payload_ball_asset.py',
                       'src/adaptive_drone_patrol/payload_contract.py',
                       'src/adaptive_drone_patrol/payload_qualification.py',
                       'src/adaptive_drone_patrol/payload_substep_probe.py')
        bindings = {path: digest(source_root/path) for path in bound_paths}
        if (manifest['profile'] != 'cf2x-two-native-ball-root-aligned-v3'
                or manifest['settings']['root_alignment_requested'] is not True
                or manifest['settings']['control_dt_s'] != .01 or manifest['settings']['rigid_dt_s'] != .005
                or manifest['energy']['powered_positive_residual_guard_j'] != .02):
            raise ValueError('Profile, timing or predeclared energy guard differs')
        result.update(selected_profile=manifest['profile'], inventory=expected, source_binding=bindings,
                      settings=manifest['settings'])
    except (OSError, ValueError, KeyError, TypeError):
        fail('SOURCE_OR_DECLARATION_MISSING')
        return result
    ball = load_checker('check_payload_ball.py')
    motor = load_checker('check_payload_motor_audit.py')
    reset = load_checker('check_payload_ball_bank.py')
    passive_spec = importlib.util.spec_from_file_location('aggregate_passive', ROOT/'src/adaptive_drone_patrol/payload_passive_check.py')
    passive = importlib.util.module_from_spec(passive_spec)
    passive_spec.loader.exec_module(passive)
    stage = motor.verify_motor_audit(motor_probes, source_root)
    observed = stage.get('actual_solver_wrench_observed', {})
    stages['construction_motor_and_reciprocal_coupling'] = dict(passed=stage.get('motor_audit_passed') is True
        and len(observed) == 15 and all(observed.values()), evidence=stage)
    reset_result = reset.verify_bank(reset_probe, source_root=source_root)
    stages['thirty_effective_partial_resets'] = dict(passed=reset_result['status']=='EXPERIMENTAL_DIAGNOSTICS_ONLY', evidence=reset_result)
    geometry = static_geometry(geometry_root, initial_cases or ROOT/'outputs/suspended_20261021_v1/cases-v1.json')
    stages['S_all_ten_nominal_targets_and_127_initial_draws_static_geometry'] = geometry
    paths = {}
    for path in sorted(probes.rglob('declaration.json')):
        try:
            metadata = read(path)
            case = metadata.get('case')
            if not case:
                continue
            key = case['id']
            if key in paths:
                fail('DUPLICATE_DECLARED_CASE')
            paths[key] = path.parent
        except (OSError, ValueError, KeyError, TypeError):
            fail('MALFORMED_CASE_DECLARATION')
    required_ids = {case['id'] for case in manifest['cases']}
    if set(paths) != required_ids:
        fail('MISSING_OR_UNDECLARED_OPERATING_CASES')
        missing.extend(sorted(required_ids-set(paths)))
    raw_fields = ['step','time_s','rpm','qpos','dofs_velocity']+[f'{body}_{field}'
        for body in ('drone','rod','package') for field in ('xyz','quaternion_wxyz','velocity','angular_velocity')]
    for case in manifest['cases']:
        key = case['id']
        if key not in paths:
            continue
        path = paths[key]
        errors, case_result = [], dict(case=case, passed=False)
        outcomes[key] = case_result
        try:
            bound = ball.verify_probe(path, source_root=source_root)
            case_result['independent_raw_state_screen'] = bound
            if bound['status'] != 'EXPERIMENTAL_DIAGNOSTICS_ONLY':
                errors.append('RAW_PHYSICAL_SCREEN_FAILED')
            declaration, initial, summary = [read(path/name) for name in ('declaration.json','runtime-initial.json','summary.json')]
            runtime = initial['runtime']
            errors.extend(physical_parameters_errors(runtime,case))
            for metadata in (declaration,initial,summary):
                if (metadata.get('case') != case or metadata.get('backend_source_sha256') != bindings[BACKEND]
                        or metadata.get('controller_source_sha256') != bindings[CONTROLLER]
                        or metadata.get('motor_plan_sha256') != bindings[MOTOR_PLAN]
                        or metadata.get('align_root') is not True):
                    errors.append('SHARED_BACKEND_CONTROLLER_SIGNATURE_MISMATCH')
            solver = runtime['actual_solver_settings']
            if (float(solver['dt']) != .01 or float(solver['substep_dt']) != .01/case['substeps']
                    or float(solver['_substep_dt']) != .01/case['substeps']):
                errors.append('ACTUAL_PHYSICS_TIMESTEP_MISMATCH')
            if initial['engine_source'] != summary['engine_source']:
                errors.append('ENGINE_BINDING_MISMATCH')
            engine_sets.append(initial['engine_source'])
            asset = path/'candidate.xml'
            if (initial['asset']['generated_asset_sha256'] != digest(asset)
                    or initial['asset']['original_asset_sha256'] != plan['original_asset_sha256']
                    or initial['asset'].get('align_root') is not True):
                errors.append('ASSET_BINDING_MISMATCH')
            xml = ET.parse(asset).getroot()
            joints = xml.findall('.//joint')
            if (len([j for j in joints if j.get('type') == 'ball']) != 2
                    or any(float(j.get('damping',0)) != 0 or float(j.get('armature',0)) != 0 for j in joints)
                    or xml.find('actuator') is not None):
                errors.append('PASSIVE_ASSET_CONTRACT_MISMATCH')
            rows = [json.loads(line) for line in (path/'samples.jsonl').read_text().splitlines()]
            if rows:
                errors.extend(initial_case_errors(rows[0],case))
            else:
                errors.append('INITIAL_STATE_INCOMPLETE')
            raw = [json.loads(line) for line in (path/'raw-states.jsonl').read_text().splitlines()]
            if digest(path/'raw-states.jsonl') != summary['raw_state_sha256']:
                errors.append('RAW_HASH_MISMATCH')
            if len(raw) != len(rows) or any(any(a.get(field) != b.get(field) for field in raw_fields) for a,b in zip(raw,rows)):
                errors.append('PRECLIP_RAW_STATE_MISMATCH')
            if summary['status'] != 'CONSTRUCTION_RECORDED' or summary.get('failures'):
                errors.append('ACTUAL_CASE_FAILED')
            actual_tensors, irrelevant = motor.authored_inertias(runtime)
            actual_tensors = {name:np.asarray(value) for name,value in actual_tensors.items()}
            authored_runtime = copy.deepcopy(runtime)
            authored_runtime['inertia_kg_m2'] = [actual_tensors[link['name']].tolist() for link in runtime['links']]
            case_result.update(samples=len(rows), measured_duration_s=rows[-1]['time_s'] if rows else 0,
                raw_sha256=digest(path/'raw-states.jsonl'), runtime_sha256=digest(path/'runtime-initial.json'),
                asset_sha256=digest(asset), zero_inertia_orientation_irrelevant=irrelevant)
            if case['kind'] == 'anchor':
                passive_result = passive.analyse_passive(rows, authored_runtime, case)
                case_result['passive'] = passive_result
                if passive_result['status'] != 'PASSIVE_SCREEN_PASSED':
                    errors.append('PASSIVE_ENERGY_OR_PERIOD_FAILURE')
            elif case['kind'] == 'large_angle':
                anchor = np.asarray(rows[0]['drone_xyz'])+rotation(rows[0]['drone_quaternion_wxyz']) @ [0,0,-.02]
                energies = [passive.mechanical_energy(row, authored_runtime, case['side_m'], anchor[2]) for row in rows]
                e0 = energies[0]
                growth = (max(energies)-e0)/e0 if e0 > 1e-12 else None
                case_result['large_angle_diagnostic'] = dict(initial_energy_j=e0, positive_energy_fraction=growth)
                if growth is None or growth > .05 or any(np.any(np.asarray(row['rpm']) != 0) for row in rows):
                    errors.append('LARGE_ANGLE_PASSIVE_ENERGY_FAILURE')
            else:
                inputs_path = path/'motor-inputs.jsonl'
                if digest(inputs_path) != summary['motor_inputs_sha256']:
                    errors.append('ACTUAL_MOTOR_INPUT_HASH_MISMATCH')
                inputs = [json.loads(line) for line in inputs_path.read_text().splitlines()]
                if len(inputs) != len(rows)-1:
                    errors.append('INCOMPLETE_ACTUAL_MOTOR_INPUTS')
                powers, energies = [], []
                mass = sum(link['mass_kg'] for link in runtime['links'])
                for i,row in enumerate(rows):
                    if row.get('contact') is not False:
                        errors.append('ACTUAL_CONTACT_OR_CONTACT_UNOBSERVED')
                    R = rotation(row['drone_quaternion_wxyz'])
                    floor = float(row['drone_xyz'][2]-.0125*abs(R[2,2])-.06*np.linalg.norm(R[2,:2]))
                    roll = math.degrees(math.atan2(R[2,1],R[2,2]))
                    pitch = math.degrees(math.asin(float(np.clip(-R[2,0],-1,1))))
                    if floor <= .1 or abs(roll) > 60 or abs(pitch) > 60:
                        errors.append('ORIENTED_VEHICLE_CLEARANCE_OR_ATTITUDE_FAILURE')
                    if i < len(inputs):
                        errors.extend(motor.check_solver_wrench(inputs[i],row,plan))
                        target = manifest['commands'][int(i*.01/manifest['command_interval_s']) % len(manifest['commands'])]
                        expected_rpm = diagnostic_rpm(row,target,mass)
                        if not np.allclose(inputs[i]['rpm'],expected_rpm,rtol=1e-6,atol=1e-7):
                            errors.append('UNDECLARED_CONTROLLER_INPUT')
                        if i+1 < len(rows) and not np.allclose(inputs[i]['rpm'],rows[i+1]['rpm'],rtol=1e-6,atol=1e-7):
                            errors.append('ACTUAL_INPUT_SAMPLE_CONTINUITY')
                    if i < len(inputs) and i+1 < len(rows):
                        rpm = np.asarray(inputs[i]['rpm'])
                        endpoints = []
                        for endpoint in (row,rows[i+1]):
                            axis = rotation(endpoint['drone_quaternion_wxyz'])[:,2]
                            forces = rpm[:,None]**2*plan['kf']*axis
                            torques = rpm[:,None]**2*plan['km']*np.asarray(plan['spin'])[:,None]*axis
                            endpoints.append(float(np.sum(forces*np.asarray(endpoint['rotor_com_velocity']))
                                +np.sum(torques*np.asarray(endpoint['rotor_angular_velocity']))))
                        powers.append(endpoints)
                    energies.append(system_energy(row,runtime,actual_tensors))
                engine_verified = engine_root is not None and all((Path(engine_root)/name).is_file()
                    and digest(Path(engine_root)/name)==wanted for name,wanted in initial['engine_source'].items())
                work = work_evidence(path,source_root,case,manifest,declaration,summary,rows,inputs,engine_verified)
                case_result['source_justified_work_evidence'] = work
                guard = energy_growth_guard(energies,powers,
                    justified_work_bound=work.get('cumulative_bound_j') if work['established'] else None)
                case_result['powered_energy'] = guard
                if guard['guard_exceeded']:
                    errors.append('ACTUAL_ENERGY_GUARD_EXCURSION')
                if guard['source_work_bound_exceeded']:
                    errors.append('ACTUAL_SOURCE_WORK_BOUND_EXCEEDED')
                if not guard['conservative_substep_bound_established']:
                    errors.append('ENERGY_WORK_BOUND_INSUFFICIENT_EVIDENCE')
                if case['kind'] == 'refinement':
                    refinement_rows[key] = rows
            case_result['passed'] = not errors
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, ET.ParseError):
            errors.append('INCOMPLETE_OR_MALFORMED_PHYSICAL_EVIDENCE')
        case_result['failures'] = sorted(set(errors))
    for left,right in expected['refinement_pairs']:
        result['refinement'][left+'__'+right] = compare_refinement(refinement_rows.get(left,[]),refinement_rows.get(right,[]))
    stages['nine_full_small_angle_120s_energy_period'] = dict(passed=sum(c.get('passed',False) for c in outcomes.values()
        if c['case']['kind']=='anchor')==9, required=9)
    stages['sixteen_complete_large_angle_diagnostics'] = dict(passed=sum(c.get('passed',False) for c in outcomes.values()
        if c['case']['kind']=='large_angle')==16, required=16)
    stages['eighteen_powered_120s_zero_and_10deg'] = dict(passed=sum(c.get('passed',False) for c in outcomes.values()
        if c['case']['kind']=='powered')==18, required=18)
    stages['six_matched_30s_dt_and_halfdt_pairs'] = dict(passed=all(c['passed'] for c in result['refinement'].values()), required=6)
    stages['all_powered_oriented_dynamic_clearance_no_contact'] = dict(passed=all(
        c.get('passed',False) for c in outcomes.values() if c['case']['kind'] in ('powered','refinement'))
        and sum(c['case']['kind'] in ('powered','refinement') for c in outcomes.values())==30)
    engine_ok = bool(engine_sets) and all(e==engine_sets[0] for e in engine_sets)
    if engine_root is not None and engine_ok:
        engine_ok = all((Path(engine_root)/name).is_file() and digest(Path(engine_root)/name)==wanted
                        for name,wanted in engine_sets[0].items())
    else:
        engine_ok = False
    stages['actual_pinned_engine_source_bytes'] = dict(passed=engine_ok, observed_source_count=len(engine_sets[0]) if engine_sets else 0)
    result['engine_source_binding'] = engine_sets[0] if engine_sets else {}
    work_cases = [c for c in outcomes.values() if c['case']['kind'] in ('powered','refinement')]
    stages['justified_powered_work_bound'] = dict(passed=len(work_cases)==30 and all(
        c.get('powered_energy',{}).get('conservative_substep_bound_established',False) for c in work_cases),
        reason='Requires complete verified substep path work as well as the independent endpoint guard.')
    for key,stage in stages.items():
        if not stage.get('passed',False):
            fail('REQUIRED_STAGE_INCOMPLETE:'+key)
    if len(outcomes) != 55 or any(not c.get('passed') for c in outcomes.values()):
        fail('OPERATING_DOMAIN_CONDITIONS_NOT_QUALIFIED')
    if any('ACTUAL_ENERGY_GUARD_EXCURSION' in c.get('failures',[]) for c in outcomes.values()):
        fail('ACTUAL_ENERGY_GUARD_EXCURSION')
    if any('ENERGY_WORK_BOUND_INSUFFICIENT_EVIDENCE' in c.get('failures',[]) for c in outcomes.values()):
        fail('ENERGY_WORK_BOUND_INSUFFICIENT_EVIDENCE')
    result['physical_signature_sha256'] = physical_signature(bindings=bindings,
        settings=manifest['settings'],manifest_sha256=bindings[MANIFEST],
        engine=result['engine_source_binding'], assets={key:c.get('asset_sha256') for key,c in outcomes.items()})
    result['checker_bindings'] = {name:digest(ROOT/'scripts'/name) for name in (
        'check_payload_ball.py','check_payload_ball_bank.py','check_payload_motor_audit.py')}
    result['checker_bindings']['src/adaptive_drone_patrol/payload_passive_check.py'] = digest(ROOT/'src/adaptive_drone_patrol/payload_passive_check.py')
    if not failures:
        result.update(status='QUALIFIED_FOR_TRAINING',physical_training_admitted=True)
    result['interpretation_ja'] = ('既存admission入口は変更しない。全実測条件とsourceを再構成。固定anchor/static幾何/短motorをpowered合格へ転用しない。'
        'energy guard超過と、guard内だがsubstep仕事上界の証拠不足を区別する。')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probes',required=True)
    parser.add_argument('--motor-probes',required=True)
    parser.add_argument('--reset-probe',required=True)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--initial-cases',default=str(ROOT/'outputs/suspended_20261021_v1/cases-v1.json'))
    parser.add_argument('--engine-root')
    parser.add_argument('--geometry-root',default=str(ROOT))
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():
        raise FileExistsError('Qualification output already exists')
    result=verify_qualification(args.probes,args.motor_probes,args.reset_probe,source_root=args.source_root,
                                initial_cases=args.initial_cases,engine_root=args.engine_root,geometry_root=args.geometry_root)
    with out.open('x') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(result['status'])
    return 0 if result['physical_training_admitted'] else 2


if __name__=='__main__':
    raise SystemExit(main())
