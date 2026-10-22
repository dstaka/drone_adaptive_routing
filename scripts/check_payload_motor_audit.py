#!/usr/bin/env python
"""Independently replay the fifteen declared short motor audits on CPU.

A successful receipt covers this construction/motor comparison only. It never
admits training or proves operating-domain qualification, a complete energy
balance, measured constraint reaction forces, or tension-only rope behavior.
No simulator, producer, controller or asset-builder module is imported.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SOURCES = ('scripts/probe_payload_ball.py', 'src/adaptive_drone_patrol/payload_ball_asset.py',
           'src/adaptive_drone_patrol/payload_qualification.py', 'src/adaptive_drone_patrol/payload_contract.py')
CONTROLLER = 'src/adaptive_drone_patrol/payload_probe_controller.py'
BACKEND = 'src/adaptive_drone_patrol/payload_ball_bank.py'
MANIFEST = 'configs/payload_retry_probe_manifest.json'
PLAN = 'configs/payload_retry_motor_audit.json'
VEHICLE = ('base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link')
FIXTURES = ('original', 'unloaded', 'loaded')
TESTS = ('hover', 'rotor0', 'rotor1', 'rotor2', 'rotor3')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rotation(quaternion):
    q = array(quaternion, (4,))
    if abs(float(np.linalg.norm(q))-1) > 1e-4:
        raise ValueError('Invalid measured quaternion')
    # Normalize only for orientation comparison; an invalid raw norm is rejected above.
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def array(value, shape):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Missing, nonfinite or incorrectly shaped measurement')
    return a


def angle_degrees(left, right):
    a, b = array(left, (4,)), array(right, (4,))
    rotation(a)
    rotation(b)
    dot = abs(float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))))
    return math.degrees(2*math.acos(float(np.clip(dot, -1, 1))))



def authored_inertias(runtime):
    """Transform measured principal/inertial tensors into authored link axes.

    Eigenvalues alone do not determine the body tensor. All three measured
    frame rotations are retained; a missing inertial orientation is irrelevant
    only when every entry of that measured tensor is exactly zero.
    """
    links = runtime['links']
    tensors = array(runtime['inertia_kg_m2'], (len(links), 3, 3))
    authored = array(runtime['initial_links_quat'], (len(links), 4))
    internal = array(runtime['initial_links_internal_quaternion_wxyz'], (len(links), 4))
    inertial = runtime['links_inertial_quaternion_wxyz']
    if not isinstance(inertial, list) or len(inertial) != len(links):
        raise ValueError('Missing per-link inertial frame measurements')
    result, irrelevant = {}, []
    for i, link in enumerate(links):
        r_auth, r_internal = rotation(authored[i]), rotation(internal[i])
        if inertial[i] is None:
            if np.any(tensors[i] != 0):
                raise ValueError('A nonzero inertia tensor requires its measured inertial orientation')
            result[link['name']] = np.zeros((3, 3)).tolist()
            irrelevant.append(link['name'])
            continue
        transform = r_auth.T @ r_internal @ rotation(inertial[i])
        result[link['name']] = (transform @ tensors[i] @ transform.T).tolist()
    return result, irrelevant


def valid_plan(plan, manifest):
    if (plan['schema'] != 'payload-retry-motor-audit-plan-v1' or
            plan['status'] != 'EXPERIMENTAL_DIAGNOSTICS_ONLY' or
            plan['declared_before_execution'] is not True or plan['no_operational_admission'] is not True or
            plan['fixtures'] != list(FIXTURES) or plan['single_rotors'] != [0, 1, 2, 3] or
            plan['hover_duration_s'] != 1. or plan['single_rotor_duration_s'] != .05 or
            plan['spin'] != [-1, 1, -1, 1] or plan['kf'] != 3.16e-10 or plan['km'] != 7.94e-12 or
            plan['single_rotor_rpm'] != 14468.429183500699 or
            plan['rotor_com_body_m'] != [[.028, -.028, 0], [-.028, -.028, 0],
                                       [-.028, .028, 0], [.028, .028, 0]]):
        raise ValueError('Unsupported motor declaration')
    ceilings = {'unloaded_equivalence_position_max_m': 1e-5, 'unloaded_equivalence_angle_max_deg': .001,
                'unloaded_equivalence_velocity_max_m_s': .001,
                'unloaded_equivalence_angular_velocity_max_rad_s': .001,
                'algebra_absolute_tolerance': 1e-7}
    if any(not math.isfinite(plan[k]) or not 0 < plan[k] <= limit for k, limit in ceilings.items()):
        raise ValueError('Undeclared or relaxed motor tolerance')
    if not math.isfinite(plan['coupled_velocity_delta_min_m_s']) or plan['coupled_velocity_delta_min_m_s'] < 1e-8:
        raise ValueError('Relaxed coupling observation floor')
    if (manifest['profile'] not in ('cf2x-two-native-ball-mjcf-v2', 'cf2x-two-native-ball-root-aligned-v3') or
            manifest['status'] != 'EXPERIMENTAL_DIAGNOSTICS_ONLY' or
            manifest['settings']['genesis_version'] != '1.4.3' or
            manifest['settings']['control_dt_s'] != .01 or manifest['settings']['rigid_dt_s'] != .005):
        raise ValueError('Unsupported physical profile or timing')
    if (manifest['profile'] == 'cf2x-two-native-ball-root-aligned-v3' and
            manifest['settings'].get('root_alignment_requested') is not True):
        raise ValueError('Explicit root alignment request must be declared')
    for key in ('mass_relative_error', 'inertia_relative_error'):
        if not 0 < manifest['limits'][key] <= 1e-5:
            raise ValueError('Relaxed property tolerance')


def check_solver_wrench(record, row, plan):
    """Compare actual pre-step link fields, including arm moments, with RPM."""
    failures = []
    try:
        if record['step'] != row['step'] or record['time_s'] != row['time_s']:
            failures.append('MOTOR_INPUT_CONTINUITY')
        indices = record['rotor_indices']
        if indices != [2, 3, 4, 5]:
            raise ValueError('Changed global motor links')
        fields = [np.asarray(record[k], dtype=float) for k in (
            'negative_external_force_field', 'negative_external_torque_about_root_com_field',
            'root_com_world_field')]
        if any(a.ndim != 3 or a.shape[1:] != (1, 3) or a.shape[0] <= 5 or
               not np.isfinite(a).all() for a in fields) or len({a.shape for a in fields}) != 1:
            raise ValueError('Malformed solver fields')
        negative_force, negative_torque, root_com = [a[:, 0] for a in fields]
        rpm = array(record['rpm'], (4,))
        axis = rotation(row['drone_quaternion_wxyz'])[:, 2]
        force = rpm[:, None]**2*plan['kf']*axis
        drag = rpm[:, None]**2*plan['km']*np.asarray(plan['spin'])[:, None]*axis
        com = array(row['rotor_com_world'], (4, 3))
        moment = np.cross(com-root_com[indices], force)+drag
        tol = plan['algebra_absolute_tolerance']
        if (not np.allclose(negative_force[indices], -force, atol=tol, rtol=0) or
                not np.allclose(negative_torque[indices], -moment, atol=tol, rtol=0)):
            failures.append('ACTUAL_SOLVER_WRENCH_MISMATCH')
        others = [i for i in range(len(negative_force)) if i not in indices]
        if np.any(np.abs(negative_force[others]) > tol) or np.any(np.abs(negative_torque[others]) > tol):
            failures.append('DIRECT_NONROTOR_WRENCH')
    except (KeyError, TypeError, ValueError, IndexError):
        failures.append('ACTUAL_SOLVER_WRENCH_MALFORMED')
    return failures


def verify_motor_audit(probes, source_root):
    probes, source_root = Path(probes), Path(source_root)
    failures, traces, decoded, runtimes, equivalence, coupling = [], {}, {}, {}, {}, {}
    authored_tensors, backend_observations, observed_wrenches = {}, {}, {}
    def fail(code):
        if code not in failures:
            failures.append(code)
    result = dict(schema='payload-retry-motor-audit-check-v1', status='QUALIFICATION_BLOCKED',
                  motor_audit_passed=False, physical_training_admitted=False,
                  operational_qualification='NOT_RUN', failure_codes=failures, traces=traces,
                  unloaded_equivalence=equivalence, coupling=coupling, missing_evidence=[],
                  checker_sha256=digest(__file__),
                  limitations=[
                      'Short motor evidence only; long qualification, resets, refinement and learned control remain unverified.',
                      'Recorded endpoint wrenches are independently reconstructed from RPM and COM frames; not force sensors.',
                      'Actual first-interval vehicle reactions are checked by direction, not a full discretized momentum proof.',
                      'Inertia tensors are reconstructed in authored link axes; missing nonzero-inertia frames block the audit.',
                      'Control-step work only; solver substep work and internal reaction forces are unobserved.',
                      'Coupled motion is evidence of interaction, not sufficient proof of correct constraint dynamics or rope tension.',
                  ])
    try:
        plan, manifest = read(source_root/PLAN), read(source_root/MANIFEST)
        valid_plan(plan, manifest)
        bindings = {name: digest(source_root/name) for name in SOURCES}
        manifest_hash, plan_hash = digest(source_root/MANIFEST), digest(source_root/PLAN)
        controller_hash = digest(source_root/CONTROLLER)
        result.update(source_binding=bindings, controller_source_sha256=controller_hash,
                      manifest_sha256=manifest_hash, motor_plan_sha256=plan_hash)
    except (OSError, ValueError, TypeError, KeyError):
        fail('SOURCE_OR_DECLARATION_INVALID')
        return result
    tol, dt = plan['algebra_absolute_tolerance'], .01
    engine_sources = None
    for fixture in FIXTURES:
        for test in TESTS:
            name = f'motor-{fixture}-{test}-v4'
            path = probes/name
            required = ('declaration.json', 'runtime-initial.json', 'summary.json', 'samples.jsonl', 'raw-states.jsonl')
            missing = [f'{name}/{f}' for f in required if not (path/f).is_file()]
            if missing:
                result['missing_evidence'].extend(missing)
                fail('MISSING_EVIDENCE')
                continue
            rows = []
            try:
                declaration, initial, summary = [read(path/f) for f in required[:3]]
                rotor = None if test == 'hover' else int(test[-1])
                for record in (declaration, initial, summary):
                    expected = dict(schema='payload-ball-probe-v2', fixture=fixture, case=None,
                                    manifest_sha256=manifest_hash, motor_plan_sha256=plan_hash,
                                    source_binding=bindings, controller_source_sha256=controller_hash,
                                    settings=manifest['settings'], physical_training_admitted=False,
                                    rotor_excitation=rotor)
                    if any(record.get(k) != value for k, value in expected.items()):
                        fail('BINDING_MISMATCH')
                records = (declaration, initial, summary)
                backend_observations[name] = any('backend_source_sha256' in record for record in records)
                if backend_observations[name]:
                    try:
                        actual_backend_hash = digest(source_root/BACKEND)
                        if any(record.get('backend_source_sha256') != actual_backend_hash for record in records):
                            fail('BACKEND_BINDING_MISMATCH')
                        result['backend_source_sha256'] = actual_backend_hash
                    except OSError:
                        fail('BACKEND_BINDING_MISMATCH')
                if (summary['status'] != 'CONSTRUCTION_RECORDED' or summary.get('failures') != [] or
                        summary.get('policy_admitted') is not False or
                        summary.get('operational_qualification') != 'NOT_RUN'):
                    fail('MOTOR_PROBE_FAILED')
                runtime = initial['runtime']
                if summary['runtime'] != runtime:
                    fail('RUNTIME_BINDING_MISMATCH')
                for record in (initial, summary):
                    if record['original_asset_sha256'] != plan['original_asset_sha256']:
                        fail('ASSET_BINDING_MISMATCH')
                    if not record['engine_source']:
                        fail('ENGINE_BINDING_MISMATCH')
                    if engine_sources is None:
                        engine_sources = record['engine_source']
                    if record['engine_source'] != engine_sources:
                        fail('ENGINE_BINDING_MISMATCH')
                if fixture != 'original':
                    for record in (initial, summary):
                        asset = record['asset']
                        if (asset['original_asset_sha256'] != plan['original_asset_sha256'] or
                                asset['generated_asset_sha256'] != digest(path/'candidate.xml') or
                                asset['loaded'] != (fixture == 'loaded') or asset['fixed'] is not False):
                            fail('ASSET_BINDING_MISMATCH')
                if digest(path/'samples.jsonl') != summary['samples_sha256']:
                    fail('SAMPLES_HASH_MISMATCH')
                if digest(path/'raw-states.jsonl') != summary['raw_state_sha256']:
                    fail('RAW_HASH_MISMATCH')
                rows = [json.loads(line) for line in (path/'samples.jsonl').read_text().splitlines()]
                raw = [json.loads(line) for line in (path/'raw-states.jsonl').read_text().splitlines()]
                observed_wrenches[name] = False
                if 'motor_inputs_sha256' in summary:
                    input_path = path/'motor-inputs.jsonl'
                    if digest(input_path) != summary['motor_inputs_sha256']:
                        fail('MOTOR_INPUT_HASH_MISMATCH')
                    inputs = [json.loads(line) for line in input_path.read_text().splitlines()]
                    if len(inputs) != len(rows)-1:
                        fail('INCOMPLETE_MOTOR_INPUTS')
                    for i, record in enumerate(inputs):
                        if i >= len(rows)-1:
                            break
                        for code in check_solver_wrench(record, rows[i], plan):
                            fail(code)
                        if not np.allclose(record['rpm'], rows[i+1]['rpm'], atol=tol, rtol=1e-6):
                            fail('MOTOR_INPUT_CONTINUITY')
                    observed_wrenches[name] = len(inputs) == len(rows)-1
                count = 101 if rotor is None else 6
                duration = plan['hover_duration_s'] if rotor is None else plan['single_rotor_duration_s']
                if len(rows) != count or len(raw) != count or summary['steps'] != count-1:
                    fail('INCOMPLETE_TRACE')
                if not math.isfinite(summary['measured_duration_s']) or abs(summary['measured_duration_s']-duration) > 1e-10:
                    fail('SUMMARY_DURATION_MISMATCH')
                links = runtime['links']
                names = [link['name'] for link in links]
                if len(set(names)) != len(names) or any(name not in names for name in VEHICLE):
                    raise ValueError('Missing vehicle links')
                if set(names) != set(VEHICLE) | ({'suspension_rod', 'payload'} if fixture == 'loaded' else set()):
                    fail('VEHICLE_PROPERTY_MISMATCH')
                positions = {link['name']: i for i, link in enumerate(links)}
                expected_indices = [links[positions[prop]]['idx'] for prop in VEHICLE[1:]]
                if (runtime['rotor_indices'] != expected_indices or runtime['rotor_spin'] != plan['spin'] or
                        expected_indices != [2, 3, 4, 5] or links[positions['base_link']]['idx'] != 1):
                    fail('ROTOR_IDENTITY')
                if runtime['kf'] != plan['kf'] or runtime['km'] != plan['km']:
                    fail('MOTOR_CONSTANTS')
                if runtime['rpm_mapping'] != '(1+0.8*action)*14468.429183500699':
                    fail('MOTOR_INPUT')
                array(runtime['inertia_kg_m2'], (len(links), 3, 3))
                array(runtime['links_com_body_m'], (len(links), 3))
                initial_positions = array(runtime['initial_links_pos'], (len(links), 3))
                array(runtime['initial_links_quat'], (len(links), 4))
                if (not np.allclose(runtime['initial_entity_pos'], [0, 0, 1], atol=tol, rtol=0) or
                        not np.allclose(runtime['initial_base_pos'], [0, 0, 1], atol=tol, rtol=0) or
                        not np.allclose(initial_positions[positions['base_link']], [0, 0, 1], atol=tol, rtol=0)):
                    fail('ROOT_SEMANTICS')
                try:
                    common_tensors, irrelevant_orientation = authored_inertias(runtime)
                except (KeyError, TypeError, ValueError):
                    fail('INERTIAL_FRAME_UNVERIFIED')
                    raise ValueError('Inertia cannot be transformed into a common physical frame')
                base_index = positions['base_link']
                root_internal_from_authored = (
                    rotation(runtime['initial_links_quat'][base_index]).T @
                    rotation(runtime['initial_links_internal_quaternion_wxyz'][base_index]))
                original_mass = sum(links[positions[name]]['mass_kg'] for name in VEHICLE)
                if (not original_mass > 0 or not math.isclose(original_mass, runtime['reference_mass_kg'],
                        rel_tol=manifest['limits']['mass_relative_error'], abs_tol=0) or
                        not math.isclose(original_mass, runtime['fixture_original_mass_kg'],
                                         rel_tol=manifest['limits']['mass_relative_error'], abs_tol=0)):
                    fail('VEHICLE_PROPERTY_MISMATCH')
                work, moments = 0., []
                for i, row in enumerate(rows):
                    if any(not np.isfinite(np.asarray(value)).all() for value in row.values()
                           if isinstance(value, (int, float, list))):
                        fail('NONFINITE_STATE')
                        raise ValueError('Nonfinite raw measurement')
                    if row['step'] != i or type(row['step']) is not int or abs(row['time_s']-i*dt) > 1e-9:
                        fail('SAMPLE_CONTINUITY')
                    # Raw was flushed before derived metrics/work were added. Compare original state, not old work.
                    raw_keys = ['step', 'time_s', 'rpm', 'qpos', 'dofs_velocity', 'drone_internal_quaternion_wxyz']
                    for body in ('drone', 'rod', 'package') if fixture == 'loaded' else ('drone',):
                        raw_keys += [body+'_'+field for field in ('xyz', 'quaternion_wxyz', 'velocity', 'angular_velocity')]
                    if i >= len(raw) or any(raw[i].get(k) != row.get(k) for k in raw_keys):
                        fail('RAW_STATE_MISMATCH')
                    R = rotation(row['drone_quaternion_wxyz'])
                    xyz, v, omega = [array(row['drone_'+key], (3,)) for key in ('xyz', 'velocity', 'angular_velocity')]
                    qpos = array(row['qpos'], (runtime['n_qs'],))
                    array(row['dofs_velocity'], (runtime['n_dofs'],))
                    internal_q = row['drone_internal_quaternion_wxyz']
                    expected_internal = R @ root_internal_from_authored
                    internal_error = rotation(internal_q)-expected_internal
                    rotation_chord_limit = 2*math.sqrt(2)*math.sin(math.radians(
                        plan['unloaded_equivalence_angle_max_deg'])/2)
                    if (not np.allclose(qpos[:3], xyz, atol=tol, rtol=0) or
                            angle_degrees(qpos[3:7], internal_q) > plan['unloaded_equivalence_angle_max_deg'] or
                            np.linalg.norm(internal_error) > rotation_chord_limit):
                        fail('ROOT_SEMANTICS')
                    if fixture == 'loaded':
                        for body in ('rod', 'package'):
                            rotation(row[body+'_quaternion_wxyz'])
                            for key in ('xyz', 'velocity', 'angular_velocity'):
                                array(row[body+'_'+key], (3,))
                    rpm = array(row['rpm'], (4,))
                    expected_rpm = np.zeros(4)
                    if i:
                        if rotor is None:
                            total_mass = sum(link['mass_kg'] for link in links)
                            expected_rpm[:] = plan['single_rotor_rpm']*math.sqrt(total_mass/original_mass)
                        else:
                            expected_rpm[rotor] = plan['single_rotor_rpm']
                    if not np.allclose(rpm, expected_rpm, rtol=1e-6, atol=tol):
                        fail('MOTOR_INPUT')
                    com, vel, ang, forces, torques = [array(row[key], (4, 3)) for key in
                        ('rotor_com_world', 'rotor_com_velocity', 'rotor_angular_velocity',
                         'rotor_force_world_n', 'rotor_torque_world_nm')]
                    expected_com = xyz+(R @ np.array(plan['rotor_com_body_m']).T).T
                    if (not np.allclose(com, expected_com, atol=tol, rtol=0) or
                            not np.allclose(vel, v+np.cross(omega, com-xyz), atol=tol, rtol=0) or
                            not np.allclose(ang, omega[None, :], atol=tol, rtol=0)):
                        fail('ROTOR_COM_FRAME')
                    expected_force = rpm[:, None]**2*plan['kf']*R[:, 2]
                    expected_torque = rpm[:, None]**2*plan['km']*np.array(plan['spin'])[:, None]*R[:, 2]
                    if (not np.allclose(forces, expected_force, atol=tol, rtol=0) or
                            not np.allclose(torques, expected_torque, atol=tol, rtol=0)):
                        fail('MOTOR_WRENCH')
                    moment = (np.cross(com-xyz, forces)+torques).sum(axis=0)
                    moments.append(moment.tolist())
                    power = float(np.sum(forces*vel)+np.sum(torques*ang))
                    work += dt*power if i else 0.
                    if (not math.isfinite(row['motor_power_w']) or abs(row['motor_power_w']-power) > tol or
                            not math.isfinite(row['motor_work_j']) or abs(row['motor_work_j']-work) > tol):
                        fail('MOTOR_WORK_RECONSTRUCTION')
                    if row['contact'] is not False:
                        fail('MOTOR_PROBE_FAILED')
                reaction = None
                if rotor is not None and len(rows) >= 2 and fixture in ('original', 'unloaded'):
                    start, end = rows[:2]
                    R = rotation(start['drone_quaternion_wxyz'])
                    force = plan['single_rotor_rpm']**2*plan['kf']*R[:, 2]
                    arm = (R @ np.array(plan['rotor_com_body_m'][rotor]))
                    moment = np.cross(arm, force)+plan['single_rotor_rpm']**2*plan['km']*plan['spin'][rotor]*R[:, 2]
                    kick = array(end['drone_velocity'], (3,))-array(start['drone_velocity'], (3,))
                    kick -= np.array(manifest['settings']['gravity_m_s2'])*dt
                    delta_omega = array(end['drone_angular_velocity'], (3,))-array(start['drone_angular_velocity'], (3,))
                    # All three nonzero original arm/spin moment components must have an actual reaction.
                    if float(kick @ force) <= 0 or any(delta_omega[j]*moment[j] <= 0 for j in range(3) if abs(moment[j]) > tol):
                        fail('MOTOR_REACTION')
                    reaction = dict(gravity_corrected_velocity_kick_m_s=kick.tolist(),
                                    angular_velocity_change_rad_s=delta_omega.tolist(),
                                    first_interval_input_moment_nm=moment.tolist(),
                                    expected_rigid_linear_kick_m_s=(force*dt/original_mass).tolist())
                traces[name] = dict(samples=len(rows), measured_duration_s=rows[-1]['time_s'] if rows else 0,
                                    samples_sha256=digest(path/'samples.jsonl'),
                                    raw_state_sha256=digest(path/'raw-states.jsonl'),
                                    summary_sha256=digest(path/'summary.json'),
                                    runtime_sha256=digest(path/'runtime-initial.json'),
                                    declaration_sha256=digest(path/'declaration.json'),
                                    motor_work_j=work, total_moment_world_nm=moments,
                                    first_interval_reaction=reaction,
                                    vehicle_inertia_authored_kg_m2={key: common_tensors[key] for key in VEHICLE},
                                    zero_inertia_orientation_irrelevant=irrelevant_orientation,
                                    root_aligned=runtime.get('root_aligned', 'NOT_RECORDED'))
                decoded[(fixture, test)] = rows
                runtimes[(fixture, test)] = runtime
                authored_tensors[(fixture, test)] = common_tensors
            except (OSError, KeyError, TypeError, ValueError, IndexError, OverflowError):
                fail('NONFINITE_STATE' if 'rows' in locals() and any(
                    'NaN' in json.dumps(r) or 'Infinity' in json.dumps(r) for r in rows) else 'MALFORMED_EVIDENCE')
    # Compare name-indexed original properties; loaded extras cannot hide modified original links.
    if ('original', 'hover') in runtimes:
        original = runtimes[('original', 'hover')]
        def properties(runtime, tensors, name):
            i = next(i for i, link in enumerate(runtime['links']) if link['name'] == name)
            return (runtime['links'][i]['mass_kg'], tensors[name], runtime['links_com_body_m'][i],
                    runtime['initial_links_pos'][i], runtime['initial_links_quat'][i])
        for key, runtime in runtimes.items():
            for name in VEHICLE:
                left = properties(original, authored_tensors[('original', 'hover')], name)
                right = properties(runtime, authored_tensors[key], name)
                tensor_difference = np.linalg.norm(np.array(left[1])-right[1])
                tensor_limit = manifest['limits']['inertia_relative_error']*np.linalg.norm(left[1])
                if (not np.allclose(left[0], right[0], rtol=manifest['limits']['mass_relative_error'], atol=0) or
                        tensor_difference > tensor_limit or
                        not np.allclose(left[2], right[2], rtol=0, atol=tol) or
                        not np.allclose(left[3], right[3], rtol=0, atol=tol) or
                        angle_degrees(left[4], right[4]) > plan['unloaded_equivalence_angle_max_deg']):
                    fail('VEHICLE_PROPERTY_MISMATCH')
    for test in TESTS:
        left, right = decoded.get(('original', test)), decoded.get(('unloaded', test))
        if left is None or right is None or len(left) != len(right):
            fail('INCOMPLETE_COMPARISON')
            continue
        errors = dict(position_m=0., angle_deg=0., velocity_m_s=0., angular_velocity_rad_s=0.)
        for a, b in zip(left, right):
            if a['time_s'] != b['time_s'] or a['step'] != b['step']:
                fail('SAMPLE_CONTINUITY')
                continue
            for key, field in [('position_m', 'drone_xyz'), ('velocity_m_s', 'drone_velocity'),
                               ('angular_velocity_rad_s', 'drone_angular_velocity')]:
                errors[key] = max(errors[key], float(np.linalg.norm(np.array(a[field])-b[field])))
            errors['angle_deg'] = max(errors['angle_deg'], angle_degrees(a['drone_quaternion_wxyz'], b['drone_quaternion_wxyz']))
        equivalence[test] = errors
        for field, limit in [('position_m', 'unloaded_equivalence_position_max_m'),
                             ('angle_deg', 'unloaded_equivalence_angle_max_deg'),
                             ('velocity_m_s', 'unloaded_equivalence_velocity_max_m_s'),
                             ('angular_velocity_rad_s', 'unloaded_equivalence_angular_velocity_max_rad_s')]:
            if errors[field] > plan[limit]:
                fail('UNLOADED_EQUIVALENCE')
        if test == 'hover':
            continue
        loaded = decoded.get(('loaded', test))
        if loaded is None or len(loaded) != len(right):
            fail('INCOMPLETE_COMPARISON')
            continue
        velocity_deltas, relative_package_velocities, package_displacements = [], [], []
        for a, b in zip(loaded, right):
            if a['time_s'] != b['time_s'] or not np.allclose(a['rpm'], b['rpm'], atol=tol, rtol=0):
                fail('PAIRED_INPUT_MISMATCH')
            velocity_deltas.append((np.array(a['drone_velocity'])-b['drone_velocity']).tolist())
            relative_package_velocities.append(float(np.linalg.norm(np.array(a['package_velocity'])-a['drone_velocity'])))
            package_displacements.append(float(np.linalg.norm(np.array(a['package_xyz'])-loaded[0]['package_xyz'])))
        peak = max(float(np.linalg.norm(v)) for v in velocity_deltas)
        minimum = plan['coupled_velocity_delta_min_m_s']
        if peak <= minimum or max(relative_package_velocities) <= minimum or max(package_displacements) <= 0:
            fail('COUPLING_UNRESOLVED')
        coupling[test] = dict(root_velocity_delta_m_s=velocity_deltas,
                              maximum_root_velocity_delta_m_s=peak,
                              maximum_package_relative_velocity_m_s=max(relative_package_velocities),
                              maximum_package_displacement_m=max(package_displacements),
                              interpretation='Measured paired interaction only; no constraint force or rope tension measured')
    if any(backend_observations.values()) and not all(backend_observations.values()):
        fail('BACKEND_BINDING_MISMATCH')
    result['engine_source_recorded'] = engine_sources
    result['actual_solver_wrench_observed'] = observed_wrenches
    result['engine_source_recomputed'] = False
    result['motor_audit_passed'] = not failures and len(traces) == 15
    if result['motor_audit_passed']:
        result['status'] = 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probes', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Output already exists; retain the earlier audit')
    result = verify_motor_audit(args.probes, args.source_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'], ','.join(result['failure_codes']))
    raise SystemExit(0 if result['motor_audit_passed'] else 2)


if __name__ == '__main__':
    main()
