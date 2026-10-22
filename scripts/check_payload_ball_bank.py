#!/usr/bin/env python
"""Independently reconstruct 30 reset draws, runtime properties and isolation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('scripts/probe_payload_ball_bank.py', 'src/adaptive_drone_patrol/payload_ball_bank.py',
           'src/adaptive_drone_patrol/payload_ball_asset.py', 'src/adaptive_drone_patrol/payload_contract.py',
           'src/adaptive_drone_patrol/payload_cases.py', 'src/adaptive_drone_patrol/util.py')
SIZES = (.03, .045, .06)
KEYS = {'0.030', '0.045', '0.060'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-5:
        raise ValueError('Invalid measured unit quaternion')
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, (float, int)) else True


def expected_resets():
    """Reconstruct seed730 streams and initial poses without importing runner helpers."""
    sizes = np.random.default_rng(np.random.SeedSequence([730, 731]))
    masses = np.random.default_rng(np.random.SeedSequence([730, 732]))
    poses = np.random.default_rng(np.random.SeedSequence([730, 733]))
    grid = [(side, ratio) for ratio in (.1, .2, .3) for side in SIZES]
    result = []
    for i in range(30):
        side, ratio = grid[i] if i < 9 else (float(sizes.choice(SIZES)), float(masses.uniform(.1, .3)))
        roll, pitch = np.radians(poses.uniform(-2, 2, 2))
        cr, sr, cp, sp = math.cos(roll/2), math.sin(roll/2), math.cos(pitch/2), math.sin(pitch/2)
        quat = [cr*cp, sr*cp, cr*sp, -sr*sp]
        angle = float(poses.uniform(0, 10))
        direction = float(poses.uniform(0, 2*math.pi))
        theta = math.radians(angle)
        offset = .3*np.array([math.sin(theta)*math.cos(direction), math.sin(theta)*math.sin(direction), -math.cos(theta)])
        drone = np.array([*poses.uniform(-.02, .02, 2), 1.])
        anchor = drone+rotation(quat) @ [0, 0, -.02]
        package = anchor+offset-np.array([0, 0, side/2])
        initial = dict(drone_xyz=drone.tolist(), drone_quaternion_wxyz=quat, package_xyz=package.tolist(),
                       package_quaternion_wxyz=[1., 0., 0., 0.], drone_velocity=[0., 0., 0.],
                       drone_angular_velocity=[0., 0., 0.], package_velocity=[0., 0., 0.],
                       package_angular_velocity=[0., 0., 0.], swing_angle_deg=angle, swing_direction_rad=direction)
        result.append(dict(index=i, side_m=side, mass_ratio=ratio, initial=initial,
                           motor_step_after_reset=i < 29, motor_action=[.05, 0., 0., 0.]))
    return result


def same_draw(expected, observed):
    """Keep RNG draws exact; allow only float64 arithmetic in derived center.

    NumPy builds may round the final matrix/vector sum by a few ULPs. This
    allowance is below 2e-15 m and never changes a physical check tolerance.
    """
    try:
        left, right = dict(expected), dict(observed)
        left['initial'], right['initial'] = dict(left['initial']), dict(right['initial'])
        a = np.asarray(left['initial'].pop('package_xyz'), dtype=float)
        b = np.asarray(right['initial'].pop('package_xyz'), dtype=float)
        rounding = 8*np.finfo(float).eps*np.maximum(1., np.maximum(np.abs(a), np.abs(b)))
        return left == right and a.shape == b.shape == (3,) and bool(np.all(np.abs(a-b) <= rounding))
    except (KeyError, ValueError, TypeError):
        return False


def check_reset(draw, state, reference, alignment):
    failures, metrics = [], {}
    if not finite(state):
        return ['NONFINITE_STATE'], metrics
    try:
        required = {'align_root', 'reference_original_mass_kg', 'qpos', 'dofs_velocity', 'inertia_kg_m2',
                    'links_com_body_m', 'links_pos', 'links_quat', 'package_aabb', 'damping', 'armature',
                    'links', 'joints'}
        required.update(f'{body}_{field}' for body in ('drone', 'rod', 'package')
                        for field in ('xyz', 'quaternion_wxyz', 'velocity', 'angular_velocity'))
        if (not required.issubset(state) or len(state['qpos']) != 15 or len(state['dofs_velocity']) != 12
                or len(state['damping']) != 12 or len(state['armature']) != 12):
            raise ValueError('Missing effective state/property channel')
        rotation(state['qpos'][3:7])
        names = {link['name']: i for i, link in enumerate(state['links'])}
        original = sum(state['links'][names[name]]['mass_kg'] for name in
                       ('base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link'))
        if (abs(reference-.02700047753751278) > 1e-8
                or not math.isclose(original, reference, rel_tol=1e-5)
                or not math.isclose(state['reference_original_mass_kg'], reference, rel_tol=1e-5)):
            failures.append('ORIGINAL_MASS')
        if state['align_root'] is not alignment:
            failures.append('ROOT_ALIGNMENT')
        side, mass = draw['side_m'], reference*draw['mass_ratio']
        package_mass = state['links'][names['payload']]['mass_kg']
        mass_error = abs(package_mass-mass)/mass
        metrics['mass_relative_error'] = mass_error
        if mass_error > 1e-5:
            failures.append('PACKAGE_MASS')
        inertia = np.asarray(state['inertia_kg_m2'][names['payload']], dtype=float)
        expected = np.eye(3)*mass*side**2/6
        inertia_error = float(np.max(np.abs(inertia-expected))/(mass*side**2/6))
        metrics['inertia_relative_error'] = inertia_error
        if inertia.shape != (3, 3) or inertia_error > 1e-5:
            failures.append('PACKAGE_INERTIA')
        matrices = {body: rotation(state[f'{body}_quaternion_wxyz']) for body in ('drone', 'rod', 'package')}
        for body in ('drone', 'package'):
            wanted = draw['initial']
            if (not np.allclose(state[f'{body}_xyz'], wanted[f'{body}_xyz'], atol=1e-6, rtol=0)
                    or not np.allclose(matrices[body], rotation(wanted[f'{body}_quaternion_wxyz']), atol=1e-6, rtol=0)):
                failures.append('INITIAL_POSE')
        maximum_velocity = max([float(np.max(np.abs(state['dofs_velocity'])))]
            + [float(np.linalg.norm(state[f'{body}_{field}'])) for body in matrices for field in ('velocity', 'angular_velocity')])
        metrics['maximum_reset_velocity'] = maximum_velocity
        if maximum_velocity > 1e-7:
            failures.append('NONZERO_RESET_VELOCITY')
        joints = {joint['name']: joint for joint in state['joints']}
        qpos = np.asarray(state['qpos'])
        upper, lower = joints['suspension_ball'], joints['payload_ball']
        a, b = upper['q_start_local'], lower['q_start_local']
        if (upper['n_qs'] != 4 or lower['n_qs'] != 4 or upper['n_dofs'] != 3 or lower['n_dofs'] != 3
                or not np.allclose(matrices['drone'] @ rotation(qpos[a:a+4]), matrices['rod'], atol=1e-6, rtol=0)
                or not np.allclose(matrices['rod'] @ rotation(qpos[b:b+4]), matrices['package'], atol=1e-6, rtol=0)):
            failures.append('NATIVE_BALL_FK')
        anchor = np.asarray(state['drone_xyz'])+matrices['drone'] @ [0, 0, -.02]
        rod_a = np.asarray(state['rod_xyz'])
        rod_b = rod_a+matrices['rod'] @ [0, 0, -.3]
        center = np.asarray(state['package_xyz'])
        endpoint = center+matrices['package'] @ [0, 0, side/2]
        residual = float(max(np.linalg.norm(anchor-rod_a), np.linalg.norm(endpoint-rod_b), abs(np.linalg.norm(endpoint-anchor)-.3)))
        metrics['constraint_residual_m'] = residual
        if residual > 2e-5:
            failures.append('ATTACHMENT_CONSTRAINT')
        extent = np.abs(matrices['package']) @ np.full(3, side/2)
        aabb_error = float(np.max(np.abs(np.asarray(state['package_aabb'])-[center-extent, center+extent])))
        metrics['aabb_max_error_m'] = aabb_error
        metrics['package_floor_m'] = float(center[2]-extent[2])
        if aabb_error > 1e-5:
            failures.append('COLLISION_AABB')
    except (ValueError, TypeError, KeyError, IndexError):
        failures.append('INCOMPLETE_ACTUAL_PROPERTIES')
    return sorted(set(failures)), metrics


def verify_bank(probe, *, source_root=ROOT):
    probe, source_root = Path(probe), Path(source_root)
    failures, missing, files, outcomes = [], [], {}, []

    def fail(code):
        if code not in failures:
            failures.append(code)

    def read(name, kind=dict):
        path = probe/name
        if not path.is_file():
            missing.append(name)
            fail('MISSING_EVIDENCE')
            return kind()
        files[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, kind):
                raise ValueError('Wrong raw shape')
            return data
        except (ValueError, UnicodeError):
            fail('MALFORMED_EVIDENCE')
            return kind()

    declaration, construction = read('declaration.json'), read('runtimeproperties.json')
    records, summary = read('reset-samples.json', list), read('summary.json')
    actual_sources = {name: digest(source_root/name) if (source_root/name).is_file() else None for name in SOURCES}
    manifest = source_root/'configs/payload_retry_probe_manifest.json'
    actual_manifest = digest(manifest) if manifest.is_file() else None
    try:
        profile = json.loads(manifest.read_text())
        settings = profile['settings']
        if (profile.get('profile') != 'cf2x-two-native-ball-root-aligned-v3'
                or profile.get('schema') != 'payload-operating-domain-declaration-v2'
                or profile.get('status') != 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
                or settings.get('root_alignment_requested') is not True
                or settings.get('genesis_version') != '1.4.3'
                or settings.get('control_dt_s') != .01 or settings.get('rigid_dt_s') != .005):
            fail('PHYSICAL_PROFILE_MISMATCH')
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        fail('PHYSICAL_PROFILE_MISMATCH')
    if (declaration.get('source_binding') != actual_sources or any(v is None for v in actual_sources.values())
            or summary.get('source_binding') != actual_sources or actual_manifest is None
            or declaration.get('manifest_sha256') != actual_manifest):
        fail('SOURCE_BINDING_MISMATCH')
    required_limits = dict(mass_relative_error=1e-5, inertia_relative_error=1e-5, collision_extent_error_m=1e-5,
                           attachment_residual_m=2e-5, initial_pose_error=1e-6, zero_velocity_abs=1e-7)
    expected = expected_resets()
    if (declaration.get('schema') != 'payload-ball-bank-reset-declaration-v1'
            or declaration.get('seed') != 730 or declaration.get('rng_streams') != dict(size=731, mass=732, pose=733)
            or declaration.get('dt_s') != .01 or declaration.get('substeps') != 2
            or declaration.get('planned_resets') != 30 or len(declaration.get('reset_sequence', [])) != 30
            or any(not same_draw(a, b) for a, b in zip(expected, declaration.get('reset_sequence', [])))
            or declaration.get('limits') != required_limits or declaration.get('align_root') is not True):
        fail('DECLARATION_MISMATCH')
    if len(records) != 30:
        fail('INCOMPLETE_30_RESETS')
    for name, key in [('declaration.json', 'declaration_sha256'), ('runtimeproperties.json', 'runtimeproperties_sha256'),
                      ('reset-samples.json', 'reset_samples_sha256')]:
        if summary.get(key) != files.get(name, {}).get('sha256'):
            fail('RAW_HASH_MISMATCH')
    if summary.get('physical_training_admitted') or declaration.get('physical_training_admitted'):
        fail('UNSUPPORTED_ADMISSION_CLAIM')
    reference = construction.get('reference_mass_kg', 0)
    if (not isinstance(reference, (float, int)) or not math.isfinite(reference)
            or abs(reference-.02700047753751278) > 1e-8 or construction.get('align_root') is not True):
        fail('ORIGINAL_MASS')
        reference = .02700047753751278
    assets = construction.get('asset_hashes', {})
    if len(assets) != 3 or summary.get('asset_hashes') != assets:
        fail('ASSET_BINDING_MISMATCH')
    for name, wanted in assets.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not (probe/path).is_file() or digest(probe/path) != wanted:
            fail('ASSET_BINDING_MISMATCH')
    moving_count, motor_steps = 0, 0
    previous = construction.get('initial_scenes')
    for i, record in enumerate(records):
        errors, metrics, moving = [], {}, []
        try:
            draw = expected[i]
            if (record['index'] != i or not same_draw(draw, record['draw'])
                    or record['draw'] != declaration['reset_sequence'][i]):
                errors.append('DRAW_OR_ORDER_MISMATCH')
            before, after = record['before'], record['after']
            if set(before) != KEYS or set(after) != KEYS:
                errors.append('MISSING_SCENE')
            if before != previous:
                errors.append('UNRECORDED_INTERVENING_STATE_CHANGE')
            if not finite(before) or not finite(after):
                errors.append('NONFINITE_STATE')
            selected = f"{draw['side_m']:.3f}"
            for key in KEYS-{selected}:
                if before.get(key) != after.get(key):
                    errors.append('INACTIVE_SCENE_CHANGED')
                if np.linalg.norm(before[key]['dofs_velocity']) > 1e-8:
                    moving.append(key)
            moving_count += len(moving)
            active_errors, metrics = check_reset(draw, after[selected], reference, True)
            errors.extend(active_errors)
            if draw['motor_step_after_reset']:
                motor = record['after_motor_step']
                if set(motor) != KEYS or not finite(motor):
                    errors.append('INVALID_AFTER_MOTOR_RAW')
                for key in KEYS-{selected}:
                    if motor.get(key) != after.get(key):
                        errors.append('INACTIVE_SCENE_STEPPED')
                active = motor[selected]
                if (active['qpos'] == after[selected]['qpos']
                        and active['dofs_velocity'] == after[selected]['dofs_velocity']):
                    errors.append('NO_ACTIVE_MOTOR_RESPONSE')
                for field in ('links', 'inertia_kg_m2', 'damping', 'armature', 'align_root'):
                    if active.get(field) != after[selected].get(field):
                        errors.append('MOTOR_STEP_CHANGED_PROPERTIES')
                for body in ('drone', 'rod', 'package'):
                    rotation(active[f'{body}_quaternion_wxyz'])
                    if (np.linalg.norm(active[f'{body}_velocity']) > 20
                            or np.linalg.norm(active[f'{body}_angular_velocity']) > 200):
                        errors.append('FINITE_NUMERICAL_EXCURSION')
                previous = motor
                motor_steps += 1
            else:
                if 'after_motor_step' in record:
                    errors.append('UNDECLARED_EXTRA_MOTOR_STEP')
                previous = after
        except (ValueError, TypeError, KeyError, IndexError):
            errors.append('INCOMPLETE_RESET_RECORD')
        if errors:
            fail('RESET_RECONSTRUCTION_FAILED')
        outcomes.append(dict(index=i, failures=sorted(set(errors)), moving_inactive=moving, **metrics))
    if moving_count == 0:
        fail('NO_MOVING_INACTIVE_EVIDENCE')
    if motor_steps != 29:
        fail('INCOMPLETE_MOTOR_STEPS')
    if (summary.get('completed_resets') != 30 or summary.get('status') != 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
            or summary.get('failures')):
        fail('RUNNER_INCOMPLETE_OR_FAILED')
    return dict(schema='payload-ball-bank-independent-check-v1',
        status='QUALIFICATION_BLOCKED' if failures else 'EXPERIMENTAL_DIAGNOSTICS_ONLY',
        physical_training_admitted=False, policy_admitted=False, probe=probe.name,
        failure_codes=failures, missing_evidence=missing, files=files, actual_source_binding=actual_sources,
        manifest_sha256=actual_manifest, checker_sha256=digest(__file__), reset_count=len(records),
        reconstructed_moving_inactive_comparisons=moving_count, reconstructed_motor_steps=motor_steps,
        expected_motor_rpm=((1+.8*np.array([.05,0,0,0]))*14468.429183500699).tolist(),
        actual_kernel_rpm_observed=False, outcomes=outcomes,
        limitations_ja='RPMは宣言actionとhash固定bank.step sourceから計算した期待値。実kernel引数の測定値はreset rawにない。motor等価性は別の独立suiteを要する。30 reset/property/FK/isolationだけでは学習許可を出さない。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--source-root', default=str(ROOT))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Independent checker output already exists')
    receipt = verify_bank(args.probe, source_root=args.source_root)
    with out.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(receipt['status'])
    return 2 if receipt['status'] == 'QUALIFICATION_BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
