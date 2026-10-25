#!/usr/bin/env python
"""Independently screen native-ball raw evidence; never issue training admission."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'configs/payload_retry_probe_manifest.json'
SOURCES = ('scripts/probe_payload_ball.py', 'src/adaptive_drone_patrol/payload_ball_asset.py',
           'src/adaptive_drone_patrol/payload_qualification.py', 'src/adaptive_drone_patrol/payload_contract.py')
POLICY_VERSION = 'native-ball-construction-screen-v1'
POLICY_LIMITS = {'constraint_residual_max_m': 2e-5, 'mass_relative_error': 1e-5,
                 'inertia_relative_error': 1e-5, 'collision_extent_error_m': 1e-5,
                 'speed_max_m_s': 20., 'angular_speed_max_rad_s': 200., 'quaternion_norm_error': 1e-4}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(q):
    """Body-to-world matrix derived independently from raw wxyz quaternion."""
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def verify_probe(probe, *, source_root=ROOT):
    """Read local bound sources plus actual raw samples, without Genesis or PPO."""
    probe = Path(probe)
    source_root = Path(source_root)
    actual_sources = {name: digest(source_root / name) if (source_root / name).is_file() else None
                      for name in SOURCES}
    codes, missing = [], []
    checks = {}
    files = {}

    def fail(code):
        if code not in codes:
            codes.append(code)

    manifest_path = source_root / MANIFEST
    manifest_hash = digest(manifest_path) if manifest_path.is_file() else None
    try:
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('schema') != 'payload-operating-domain-declaration-v2'
                or manifest.get('status') != 'EXPERIMENTAL_DIAGNOSTICS_ONLY'):
            fail('MANIFEST_POLICY_MISMATCH')
        for key, ceiling in POLICY_LIMITS.items():
            if not 0 < float(manifest['limits'][key]) <= ceiling:
                fail('MANIFEST_POLICY_MISMATCH')
        limits = {key: min(float(manifest['limits'][key]), ceiling) for key, ceiling in POLICY_LIMITS.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        fail('BINDING_MISMATCH')
        fail('MANIFEST_POLICY_MISMATCH')
        if manifest_hash is None:
            missing.append(MANIFEST)
        manifest = json.loads((ROOT / MANIFEST).read_text())
        limits = dict(POLICY_LIMITS)

    def load(name):
        path = probe / name
        if not path.is_file():
            missing.append(name)
            return {}
        files[name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('Receipt must be an object')
            return value
        except (ValueError, UnicodeError):
            fail('MALFORMED_EVIDENCE')
            return {}

    declaration = load('declaration.json')
    summary = load('summary.json')
    construction_failed = summary.get('status') == 'CONSTRUCTION_FAILED'
    if construction_failed:
        fail('CONSTRUCTION_FAILURE')
        traceback = probe / 'traceback.txt'
        if traceback.is_file():
            files['traceback.txt'] = {'bytes': traceback.stat().st_size, 'sha256': digest(traceback)}
        else:
            missing.append('traceback.txt')
    initial = load('runtime-initial.json')
    for receipt in (declaration, summary, initial):
        if not receipt:
            continue
        if receipt.get('schema') != 'payload-ball-probe-v2' or receipt.get('physical_training_admitted'):
            fail('OLD_OR_UNSUPPORTED_RECEIPT')
        if manifest_hash is None or receipt.get('manifest_sha256') != manifest_hash:
            fail('BINDING_MISMATCH')
        binding = receipt.get('source_binding', {})
        if (not isinstance(binding, dict) or set(binding) != set(SOURCES)
                or any(actual_sources[name] is None or binding.get(name) != actual_sources[name] for name in SOURCES)):
            fail('BINDING_MISMATCH')
        if receipt.get('settings') != manifest['settings']:
            fail('BINDING_MISMATCH')
    if summary and summary.get('status') not in ('CONSTRUCTION_RECORDED', 'CONSTRUCTION_FAILED', 'PHYSICAL_FAILURE'):
        fail('OLD_OR_UNSUPPORTED_RECEIPT')
    fixture = declaration.get('fixture')
    if fixture not in ('original', 'unloaded', 'loaded', 'anchor'):
        fail('INVALID_FIXTURE')
    case = declaration.get('case')
    if case is not None:
        expected = next((c for c in manifest['cases'] if isinstance(case, dict) and c['id'] == case.get('id')), None)
        if case != expected or expected is None:
            fail('BINDING_MISMATCH')
        if expected is not None and (fixture == 'anchor') != (expected['kind'] in ('anchor', 'large_angle')):
            fail('INVALID_FIXTURE')
        case = expected
    for receipt in (initial, summary):
        if receipt and (receipt.get('case') != case or receipt.get('fixture') != fixture):
            fail('BINDING_MISMATCH')
    loaded = fixture in ('loaded', 'anchor')
    side = case['side_m'] if case else .045
    ratio = case['mass_ratio'] if case else .2
    duration = case['duration_s'] if case else 1.
    dt = case['sample_dt_s'] if case else manifest['settings']['control_dt_s']
    expected_steps = round(duration/dt)
    runtime = initial.get('runtime', {})
    if runtime:
        try:
            links = runtime['links']
            by_name = {link['name']: i for i, link in enumerate(links)}
            prop_indices = [links[by_name[f'prop{i}_link']]['idx'] for i in range(4)]
            if runtime['rotor_indices'] != prop_indices or prop_indices != [2, 3, 4, 5]:
                fail('ROTOR_IDENTITY')
            if runtime['rotor_spin'] != [-1, 1, -1, 1] or runtime['kf'] != 3.16e-10 or runtime['km'] != 7.94e-12:
                fail('ROTOR_IDENTITY')
            if runtime.get('rpm_mapping') != '(1+0.8*action)*14468.429183500699':
                fail('ROTOR_IDENTITY')
            if not np.allclose(runtime['initial_entity_pos'], runtime['initial_base_pos'], atol=1e-6, rtol=0):
                fail('ROOT_SEMANTICS')
            if not np.allclose(runtime['initial_base_pos'], [0, 0, 1], atol=1e-6, rtol=0):
                fail('ROOT_SEMANTICS')
            measured_reference = sum(links[by_name[name]]['mass_kg'] for name in
                                     ('base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link'))
            reference = float(runtime['reference_mass_kg'])
            if not math.isfinite(reference) or abs(reference-.02700047753751278) > 1e-8:
                fail('REFERENCE_MASS_DISCREPANCY')
            if fixture == 'anchor' and 'fixture_original_mass_kg' in runtime:
                # Fixed fixtures have zero authored rotor masses; only the free
                # importer regularizes those masses. Bind the free reference to
                # an actual attached audit instead of changing the load domain.
                fixture_mass = float(runtime['fixture_original_mass_kg'])
                checks['fixture_original_mass_kg'] = fixture_mass if math.isfinite(fixture_mass) else None
                audit_binding = initial.get('reference_mass_audit')
                if (not math.isfinite(fixture_mass)
                        or not math.isclose(fixture_mass, measured_reference, rel_tol=1e-5)
                        or not isinstance(audit_binding, dict)
                        or audit_binding.get('path') != 'reference-free-runtime.json'
                        or any(r.get('reference_mass_audit') != audit_binding for r in (declaration, summary))):
                    fail('REFERENCE_MASS_AUDIT_INVALID')
                else:
                    audit = load('reference-free-runtime.json')
                    try:
                        if files.get('reference-free-runtime.json', {}).get('sha256') != audit_binding.get('sha256'):
                            raise ValueError('Free audit hash differs')
                        if audit.get('schema') != 'payload-ball-probe-v2' or audit.get('fixture') not in ('original', 'unloaded', 'loaded'):
                            raise ValueError('Actual free construction audit required')
                        free = audit['runtime']
                        free_links = {link['name']: link for link in free['links']}
                        original_names = ('base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link')
                        free_mass = sum(float(free_links[name]['mass_kg']) for name in original_names)
                        expected_dims = (15, 12) if audit['fixture'] == 'loaded' else (7, 6)
                        if (not math.isfinite(free_mass) or abs(free_mass-.02700047753751278) > 1e-8
                                or not math.isclose(reference, free_mass, rel_tol=1e-5)
                                or not math.isclose(free['reference_mass_kg'], free_mass, rel_tol=1e-5)
                                or (free['n_qs'], free['n_dofs']) != expected_dims
                                or [free_links[f'prop{i}_link']['idx'] for i in range(4)] != [2, 3, 4, 5]):
                            raise ValueError('Measured free original properties differ')
                        checks['free_audit_original_mass_kg'] = free_mass
                    except (ValueError, TypeError, KeyError, IndexError):
                        fail('REFERENCE_MASS_AUDIT_INVALID')
            elif not math.isclose(reference, measured_reference, rel_tol=1e-5):
                # Preserve the v2 mislabeled-reference failure as recorded.
                fail('REFERENCE_MASS_DISCREPANCY')
            checks['reference_mass_kg'] = reference if math.isfinite(reference) else None
            if loaded:
                index = by_name['payload']
                expected_mass = .02700047753751278*ratio
                if not math.isclose(links[index]['mass_kg'], expected_mass, rel_tol=limits['mass_relative_error']):
                    fail('PACKAGE_MASS')
                inertia = np.asarray(runtime['inertia_kg_m2'][index], dtype=float)
                wanted = np.eye(3)*expected_mass*side**2/6
                if inertia.shape != (3, 3) or not np.isfinite(inertia).all() or not np.allclose(
                        inertia, wanted, atol=1e-15, rtol=limits['inertia_relative_error']):
                    fail('PACKAGE_INERTIA')
                aabb = np.asarray(runtime['package_aabb'], dtype=float)
                quat = np.asarray(runtime['initial_links_quat'][index], dtype=float)
                center = np.asarray(runtime['initial_links_pos'][index], dtype=float)
                extent = np.abs(rotation(quat)) @ np.full(3, side/2)
                if aabb.shape != (2, 3) or not np.isfinite(aabb).all() or not np.allclose(
                        aabb, [center-extent, center+extent], rtol=0, atol=limits['collision_extent_error_m']):
                    fail('PACKAGE_COLLISION_EXTENT')
                joints = {j['name']: j for j in runtime['joints']}
                for name in ('suspension_ball', 'payload_ball'):
                    if joints[name]['n_qs'] != 4 or joints[name]['n_dofs'] != 3:
                        fail('NATIVE_BALL_DOF')
                expected_dofs = 6 if fixture == 'anchor' else 12
                if runtime['n_dofs'] != expected_dofs or runtime['n_qs'] != (8 if fixture == 'anchor' else 15):
                    fail('NATIVE_BALL_DOF')
        except (KeyError, TypeError, ValueError, IndexError):
            fail('INCOMPLETE_RUNTIME_PROPERTIES')
    elif not construction_failed:
        fail('INCOMPLETE_RUNTIME_PROPERTIES')
    rows = []
    samples = probe / 'samples.jsonl'
    if not samples.is_file():
        missing.append('samples.jsonl')
    else:
        files['samples.jsonl'] = {'bytes': samples.stat().st_size, 'sha256': digest(samples)}
        if summary.get('samples_sha256') != digest(samples):
            fail('SAMPLES_HASH_MISMATCH')
        try:
            rows = [json.loads(line) for line in samples.read_text().splitlines()]
        except (ValueError, UnicodeError):
            fail('MALFORMED_EVIDENCE')
    checks.update(sample_count=len(rows), required_sample_count=expected_steps+1, required_duration_s=duration)
    if len(rows) != expected_steps+1:
        fail('INCOMPLETE_TRACE')
    max_constraint, min_floor, max_speed, max_ang = 0., None, 0., 0.
    for i, row in enumerate(rows):
        try:
            for key, size in [('qpos', runtime.get('n_qs')), ('dofs_velocity', runtime.get('n_dofs')), ('rpm', 4)]:
                if len(row[key]) != size:
                    raise ValueError('Missing generalized state or motor command')
            if row['step'] != i or not math.isclose(row['time_s'], i*dt, abs_tol=1e-8):
                fail('SAMPLE_CONTINUITY')
            numeric = [np.asarray(v, dtype=float) for v in row.values() if isinstance(v, (list, int, float))]
            if not all(np.isfinite(v).all() for v in numeric):
                fail('NONFINITE_STATE')
                continue
            bodies = ('drone', 'rod', 'package') if loaded else ('drone',)
            matrices = {}
            for body in bodies:
                q = np.asarray(row[body+'_quaternion_wxyz'], dtype=float)
                xyz = np.asarray(row[body+'_xyz'], dtype=float)
                velocity = np.asarray(row[body+'_velocity'], dtype=float)
                angular = np.asarray(row[body+'_angular_velocity'], dtype=float)
                if q.shape != (4,) or any(v.shape != (3,) for v in (xyz, velocity, angular)):
                    raise ValueError('Raw body state shape')
                if abs(np.linalg.norm(q)-1) > limits['quaternion_norm_error']:
                    fail('BAD_QUATERNION')
                matrices[body] = rotation(q)
                speed, ang = float(np.linalg.norm(velocity)), float(np.linalg.norm(angular))
                max_speed, max_ang = max(max_speed, speed), max(max_ang, ang)
                if speed > limits['speed_max_m_s'] or ang > limits['angular_speed_max_rad_s']:
                    fail('FINITE_NUMERICAL_EXCURSION')
            if loaded:
                a = np.asarray(row['drone_xyz'])+matrices['drone'] @ [0, 0, -.02]
                rod_a = np.asarray(row['rod_xyz'])
                rod_b = rod_a+matrices['rod'] @ [0, 0, -.3]
                b = np.asarray(row['package_xyz'])+matrices['package'] @ [0, 0, side/2]
                residual = max(np.linalg.norm(a-rod_a), np.linalg.norm(b-rod_b), abs(np.linalg.norm(b-a)-.3))
                max_constraint = max(max_constraint, float(residual))
                floor = float(row['package_xyz'][2]-side/2*sum(abs(matrices['package'][2])))
                min_floor = floor if min_floor is None else min(min_floor, floor)
                if residual > limits['constraint_residual_max_m']:
                    fail('ATTACHMENT_CONSTRAINT')
                if not math.isclose(row['constraint_residual_m'], abs(np.linalg.norm(b-a)-.3), abs_tol=1e-7):
                    fail('DERIVED_METRIC_MISMATCH')
                if not math.isclose(row['package_floor_m'], floor, abs_tol=1e-7):
                    fail('DERIVED_METRIC_MISMATCH')
                for name, wanted in [('attachment_a_world', a), ('attachment_b_world', b)]:
                    if not np.allclose(row[name], wanted, atol=1e-7, rtol=0):
                        fail('DERIVED_METRIC_MISMATCH')
                if fixture != 'anchor' and floor <= manifest['settings']['floor_m']:
                    fail('PHYSICAL_FLOOR_FAILURE')
            if fixture != 'anchor' and row['drone_xyz'][2] <= manifest['settings']['floor_m']:
                fail('PHYSICAL_FLOOR_FAILURE')
            if fixture != 'anchor':
                tilt = np.degrees(np.arccos(np.clip(matrices['drone'][2, 2], -1, 1)))
                if tilt > manifest['settings']['tilt_max_deg']:
                    fail('PHYSICAL_ATTITUDE_FAILURE')
        except (KeyError, TypeError, ValueError, IndexError):
            fail('INCOMPLETE_RAW_STATE')
    if summary and rows:
        try:
            duration_matches = math.isclose(float(summary.get('measured_duration_s', -1)),
                                           float(rows[-1].get('time_s', -2)), abs_tol=1e-8)
        except (TypeError, ValueError):
            duration_matches = False
        if summary.get('steps') != rows[-1].get('step') or not duration_matches:
            fail('SUMMARY_DURATION_MISMATCH')
    if summary.get('status') == 'PHYSICAL_FAILURE' or summary.get('failures'):
        fail('RUNNER_REPORTED_FAILURE')
    if missing:
        fail('MISSING_EVIDENCE')
    checks.update(max_constraint_residual_m=max_constraint, minimum_package_floor_m=min_floor,
                  max_speed_m_s=max_speed, max_angular_speed_rad_s=max_ang)
    classification = 'CONSTRUCTION_FAILURE' if construction_failed else ('INVALID_OR_FAILED_EVIDENCE' if codes else 'CONSTRUCTION_SCREEN_ONLY')
    if construction_failed and 'traceback.txt' in missing:
        classification = 'CONSTRUCTION_FAILURE_WITH_MISSING_EVIDENCE'
    elif missing and not construction_failed:
        classification = 'MISSING_EVIDENCE'
    return dict(schema='payload-ball-independent-check-v2',
                status='QUALIFICATION_BLOCKED' if codes else 'EXPERIMENTAL_DIAGNOSTICS_ONLY',
                physical_training_admitted=False, policy_admitted=False,
                classification=classification, failure_codes=codes, missing_evidence=missing,
                probe=probe.name, manifest_sha256=manifest_hash, checker_sha256=digest(__file__),
                checker_policy_version=POLICY_VERSION, checker_policy_limits=POLICY_LIMITS,
                actual_source_binding=actual_sources,
                files=files, checks=checks,
                unverified=['full operating-domain suite', 'motor wrench equivalence and reciprocal dynamics',
                            'passive period and energy', 'powered work balance', 'all reset isolation',
                            'matched-time refinement', 'all target and perturbation clearance'],
                interpretation_ja='constructor/短時間screenは運用域資格ではない。実rawを再構成し、process成功や旧PASSから学習許可を推定しない。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-root', default=str(ROOT),
                        help='Explicit measured source snapshot containing declared sources and manifest')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Checker output already exists')
    receipt = verify_probe(args.probe, source_root=args.source_root)
    with out.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(receipt['status'])
    return 2 if receipt['status'] == 'QUALIFICATION_BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
