#!/usr/bin/env python
"""Predeclared 30 selective native-ball episode resets; diagnostics only."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from adaptive_drone_patrol.payload_ball_bank import BallSceneBank
from adaptive_drone_patrol.payload_cases import initial_state
from adaptive_drone_patrol.payload_contract import Payload, rotation
from adaptive_drone_patrol.util import sha256

SIZES = (.03, .045, .06)
SOURCES = ('scripts/probe_payload_ball_bank.py', 'src/adaptive_drone_patrol/payload_ball_bank.py',
           'src/adaptive_drone_patrol/payload_ball_asset.py', 'src/adaptive_drone_patrol/payload_contract.py',
           'src/adaptive_drone_patrol/payload_cases.py', 'src/adaptive_drone_patrol/util.py')


def write(path, value, *, raw=False):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=raw)+'\n')


def reset_sequence(seed=730):
    size_rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
    mass_rng = np.random.default_rng(np.random.SeedSequence([seed, 732]))
    pose_rng = np.random.default_rng(np.random.SeedSequence([seed, 733]))
    grid = [(side, ratio) for ratio in (.1, .2, .3) for side in SIZES]
    rows = []
    for i in range(30):
        side, ratio = grid[i] if i < 9 else (float(size_rng.choice(SIZES)), float(mass_rng.uniform(.1, .3)))
        rows.append(dict(index=i, side_m=side, mass_ratio=ratio, initial=initial_state(pose_rng, side),
                         motor_step_after_reset=i < 29, motor_action=[.05, 0., 0., 0.]))
    return rows


def inactive_changes(before, after, selected_side):
    key = f'{selected_side:.3f}'
    return [side for side in before if side != key and before[side] != after.get(side)]


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, (float, int)) else True


def reset_failures(draw, measured, reference_mass, align_root):
    """Screen actual runtime values; declarations alone cannot satisfy checks."""
    failures = []
    if not finite(measured):
        return ['NONFINITE_STATE']
    try:
        if measured['align_root'] != align_root:
            failures.append('ROOT_ALIGNMENT')
        by_name = {link['name']: i for i, link in enumerate(measured['links'])}
        original = sum(measured['links'][by_name[name]]['mass_kg'] for name in
                       ('base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link'))
        if (not math.isclose(original, reference_mass, rel_tol=1e-5)
                or not math.isclose(measured['reference_original_mass_kg'], reference_mass, rel_tol=1e-5)):
            failures.append('ORIGINAL_MASS')
        index = by_name['payload']
        side, mass = draw['side_m'], reference_mass*draw['mass_ratio']
        if not math.isclose(measured['links'][index]['mass_kg'], mass, rel_tol=1e-5):
            failures.append('PACKAGE_MASS')
        expected_inertia = np.eye(3)*mass*side**2/6
        if not np.allclose(measured['inertia_kg_m2'][index], expected_inertia, rtol=1e-5, atol=1e-15):
            failures.append('PACKAGE_INERTIA')
        rd, rr, rp = [rotation(measured[f'{name}_quaternion_wxyz']) for name in ('drone', 'rod', 'package')]
        for name, actual_rotation in [('drone', rd), ('package', rp)]:
            wanted_rotation = rotation(draw['initial'][f'{name}_quaternion_wxyz'])
            if (not np.allclose(measured[f'{name}_xyz'], draw['initial'][f'{name}_xyz'], rtol=0, atol=1e-6)
                    or not np.allclose(actual_rotation, wanted_rotation, rtol=0, atol=1e-6)):
                failures.append('INITIAL_POSE')
        if (np.max(np.abs(measured['dofs_velocity'])) > 1e-7
                or any(np.linalg.norm(measured[f'{name}_{field}']) > 1e-7
                       for name in ('drone', 'rod', 'package') for field in ('velocity', 'angular_velocity'))):
            failures.append('NONZERO_RESET_VELOCITY')
        qpos = np.asarray(measured['qpos'])
        joints = {joint['name']: joint for joint in measured['joints']}
        upper, lower = joints['suspension_ball'], joints['payload_ball']
        qu = qpos[upper['q_start_local']:upper['q_start_local']+4]
        ql = qpos[lower['q_start_local']:lower['q_start_local']+4]
        if (upper['n_qs'] != 4 or lower['n_qs'] != 4 or upper['n_dofs'] != 3 or lower['n_dofs'] != 3
                or not np.allclose(rd @ rotation(qu), rr, rtol=0, atol=1e-6)
                or not np.allclose(rr @ rotation(ql), rp, rtol=0, atol=1e-6)):
            failures.append('NATIVE_BALL_FK')
        anchor = np.asarray(measured['drone_xyz'])+rd @ [0, 0, -.02]
        rod_a = np.asarray(measured['rod_xyz'])
        rod_b = rod_a+rr @ [0, 0, -.3]
        package = np.asarray(measured['package_xyz'])
        endpoint = package+rp @ [0, 0, side/2]
        if max(np.linalg.norm(anchor-rod_a), np.linalg.norm(endpoint-rod_b), abs(np.linalg.norm(endpoint-anchor)-.3)) > 2e-5:
            failures.append('ATTACHMENT_CONSTRAINT')
        extent = np.abs(rp) @ np.full(3, side/2)
        if not np.allclose(measured['package_aabb'], [package-extent, package+extent], rtol=0, atol=1e-5):
            failures.append('COLLISION_AABB')
    except (KeyError, ValueError, TypeError, IndexError):
        failures.append('INCOMPLETE_ACTUAL_PROPERTIES')
    return sorted(set(failures))


def snapshot(bank, side):
    _, entity, package, _ = bank.banks[side]
    masses = entity.get_links_mass()[0].detach().cpu().tolist()
    state = dict(align_root=bank.fixtures[side]['align_root'],
                 reference_original_mass_kg=bank.fixtures[side]['original_mass_kg'],
                 qpos=entity.get_qpos()[0].detach().cpu().tolist(),
                 dofs_velocity=entity.get_dofs_velocity()[0].detach().cpu().tolist(),
                 inertia_kg_m2=entity.get_links_inertia()[0].detach().cpu().tolist(),
                 links_com_body_m=entity.get_links_COM()[0].detach().cpu().tolist(),
                 links_pos=entity.get_links_pos()[0].detach().cpu().tolist(),
                 links_quat=entity.get_links_quat()[0].detach().cpu().tolist(),
                 package_aabb=package.get_AABB()[0].detach().cpu().tolist(),
                 damping=entity.get_dofs_damping().detach().cpu().tolist(),
                 armature=entity.get_dofs_armature().detach().cpu().tolist(),
                 links=[dict(name=link.name, idx=link.idx, mass_kg=masses[link.idx-entity.link_start]) for link in entity.links],
                 joints=[dict(name=j.name, type=str(j.type), n_qs=j.n_qs, n_dofs=j.n_dofs,
                              q_start_local=j.q_start-entity.q_start, dof_start_local=j.dof_start-entity.dof_start)
                         for j in entity.joints])
    for name, body in [('drone', entity.get_link('base_link')), ('rod', entity.get_link('suspension_rod')),
                       ('package', package)]:
        for field, getter in [('xyz', 'get_pos'), ('quaternion_wxyz', 'get_quat'),
                              ('velocity', 'get_vel'), ('angular_velocity', 'get_ang')]:
            state[f'{name}_{field}'] = getattr(body, getter)()[0].detach().cpu().tolist()
    return state


def all_snapshots(bank):
    return {f'{side:.3f}': snapshot(bank, side) for side in SIZES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--align-root', choices=('true', 'false'), default='true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    alignment = args.align_root == 'true'
    plan = reset_sequence(730)
    declaration = dict(schema='payload-ball-bank-reset-declaration-v1', status='EXPERIMENTAL_DIAGNOSTICS_ONLY',
        physical_training_admitted=False, seed=730, rng_streams=dict(size=731, mass=732, pose=733),
        dt_s=.01, substeps=2, timeout_wall_s=300, align_root=alignment, planned_resets=30, reset_sequence=plan,
        limits=dict(mass_relative_error=1e-5, inertia_relative_error=1e-5, collision_extent_error_m=1e-5,
                    attachment_residual_m=2e-5, initial_pose_error=1e-6, zero_velocity_abs=1e-7),
        source_binding={name: sha256(name) for name in SOURCES},
        manifest_sha256=sha256('configs/payload_retry_probe_manifest.json'),
        interpretation='Diagnostic resets only; not acceptance, training or operational admission')
    write(out/'declaration.json', declaration)
    started = time.monotonic()
    records, failures = [], []
    result = dict(schema='payload-ball-bank-reset-result-v1', status='FAILED', physical_training_admitted=False,
                  declaration_sha256=sha256(out/'declaration.json'), completed_resets=0, source_binding=declaration['source_binding'])
    try:
        if importlib.metadata.version('genesis-world') != '1.4.3':
            raise ValueError('Pinned Genesis 1.4.3 required')
        import genesis as gs
        gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=730)
        bank = BallSceneBank(out/'bank', dt=.01, substeps=2, align_root=alignment)
        assets = {str(path.relative_to(out)): sha256(path) for path in sorted((out/'bank').rglob('*.xml'))}
        construction = dict(reference_mass_kg=bank.reference_mass, align_root=alignment, asset_hashes=assets,
                            asset_manifests={f'{s:.3f}': bank.fixtures[s]['asset_manifest'] for s in SIZES},
                            initial_scenes=all_snapshots(bank))
        write(out/'runtimeproperties.json', construction, raw=True)
        moving_inactive_seen = 0
        for draw in plan:
            before = all_snapshots(bank)
            record = dict(index=draw['index'], draw=draw, before=before)
            records.append(record)
            write(out/'reset-samples.json', records, raw=True)
            payload = Payload(bank.reference_mass, draw['mass_ratio'], draw['side_m'])
            bank.reset(payload, draw['initial'])
            after = all_snapshots(bank)
            record['after'] = after
            # Every measured state is saved before finite/property/FK checks.
            write(out/'reset-samples.json', records, raw=True)
            changed = inactive_changes(before, after, draw['side_m'])
            active = after[f"{draw['side_m']:.3f}"]
            errors = reset_failures(draw, active, bank.reference_mass, alignment)
            if changed:
                errors.append('INACTIVE_SCENE_CHANGED')
            if not finite(before) or not finite(after):
                errors.append('NONFINITE_STATE')
            moving = [key for key, state in before.items() if key != f"{draw['side_m']:.3f}"
                      and np.linalg.norm(state['dofs_velocity']) > 1e-8]
            record.update(failures=errors, inactive_changed=changed, moving_inactive_before=moving)
            moving_inactive_seen += len(moving)
            write(out/'reset-samples.json', records, raw=True)
            if errors:
                failures.extend(f"reset-{draw['index']}:{error}" for error in errors)
                break
            result['completed_resets'] += 1
            if draw['motor_step_after_reset']:
                bank.step(draw['motor_action'])
                record['after_motor_step'] = all_snapshots(bank)
                write(out/'reset-samples.json', records, raw=True)
                if not finite(record['after_motor_step']):
                    failures.append(f"reset-{draw['index']}:NONFINITE_MOTOR_STEP")
                    break
                if inactive_changes(after, record['after_motor_step'], draw['side_m']):
                    failures.append(f"reset-{draw['index']}:INACTIVE_SCENE_STEPPED")
                    break
        if result['completed_resets'] != 30:
            failures.append('INCOMPLETE_30_RESETS')
        if moving_inactive_seen == 0:
            failures.append('NO_MOVING_INACTIVE_ISOLATION_EVIDENCE')
        result.update(moving_inactive_comparisons=moving_inactive_seen, failures=failures,
                      reference_mass_kg=bank.reference_mass, asset_hashes=assets,
                      runtimeproperties_sha256=sha256(out/'runtimeproperties.json'))
        if not failures:
            result['status'] = 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
            result['reset_screen_completed'] = True
    except Exception as exc:
        result.update(status='FAILED', error_type=type(exc).__name__, error=repr(exc), failures=failures)
        (out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        if (out/'reset-samples.json').exists():
            result['reset_samples_sha256'] = sha256(out/'reset-samples.json')
        result['elapsed_s'] = time.monotonic()-started
        result['source_tree_signature'] = hashlib.sha256(json.dumps(result['source_binding'], sort_keys=True).encode()).hexdigest()
        write(out/'summary.json', result)
    print(result['status'], result.get('failures', []))
    return 2 if result['status'] == 'FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
