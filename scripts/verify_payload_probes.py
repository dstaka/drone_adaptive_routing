#!/usr/bin/env python
"""Independently recompute payload probe geometry and passive energy from samples."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(root):
    root = Path(root)
    return {p.name: digest(p) for p in sorted(root.iterdir())
            if p.name in ('summary.json', 'samples.jsonl', 'runtime-initial.json', 'asset.json', 'loaded.urdf')}


def rot(q):
    w, x, y, z = q
    if not math.isclose(sum(v*v for v in q), 1, abs_tol=1e-5):
        raise ValueError('Invalid measured unit quaternion')
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def geometry(row, side, env):
    d, p = np.array(row['drone_xyz'][env]), np.array(row['package_xyz'][env])
    rd, rp = rot(row['drone_quaternion_wxyz'][env]), rot(row['package_quaternion_wxyz'][env])
    a, b = d + rd @ [0,0,-.02], p + rp @ [0,0,side/2]
    r = b-a
    length = np.linalg.norm(r)
    if length <= 0 or not np.isfinite(r).all():
        raise ValueError('Invalid measured suspension')
    return dict(residual_m=float(abs(length-.30)), floor_m=float(p[2]-side/2*sum(abs(x) for x in rp[2])),
                swing_deg=math.degrees(math.acos(float(np.clip(-r[2]/length, -1, 1)))))


def energy(row, side, env, mass):
    """Mechanical energy above the vertical equilibrium, including rod and gimbals."""
    p = np.array(row['package_xyz'][env])
    v = np.array(row['package_velocity'][env])
    w = np.array(row['package_angular_velocity'][env])
    rod_p = np.array(row['rod_xyz'][env])
    rod_v = np.array(row['rod_velocity'][env])
    rod_w = np.array(row['rod_angular_velocity'][env])
    rod_r = rot(row['rod_quaternion_wxyz'][env])
    offset = rod_r @ [0, 0, -.15]
    center = rod_p + offset
    center_v = rod_v + np.cross(rod_w, offset)
    end = rod_p + rod_r @ [0, 0, -.30]
    end_v = rod_v + np.cross(rod_w, rod_r @ [0, 0, -.30])
    inertia = np.diag([.0001*(3*.001**2+.3**2)/12]*2 + [.0001*.001**2/2])
    local_w = rod_r.T @ rod_w
    package_i = mass*side**2/6
    kinetic = .5*mass*np.dot(v,v) + .5*package_i*np.dot(w,w)
    kinetic += .5*.0001*np.dot(center_v,center_v) + .5*local_w@inertia@local_w
    kinetic += .5*.000002*np.dot(end_v,end_v)
    potential = 9.81*(mass*(p[2]-(.98-.3-side/2)) + .0001*(center[2]-.83) + .000002*(end[2]-.68))
    return float(kinetic + potential)


def read_probe(root):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    if summary.get('status') != 'PROBE_RECORDED':
        raise ValueError('Probe did not finish recording')
    if digest(root/'samples.jsonl') != summary['telemetry_sha256']:
        raise ValueError('Stale sample binding')
    rows = [json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines()]
    if len(rows) != summary['steps'] or len(rows) != summary['args']['steps']:
        raise ValueError('Incomplete telemetry')
    for i, row in enumerate(rows):
        if row['step'] != i+1 or abs(row['time_s']-(i+1)*summary['args']['dt']) > 1e-8:
            raise ValueError('Missing or mismatched physics step/time')
        if not all(np.isfinite(np.asarray(v)).all() for v in row.values()):
            raise ValueError('Nonfinite telemetry')
    return summary, rows


def verify(root, thresholds):
    root = Path(root)
    expected = {f'{kind}_{side}' for kind in ('coupled', 'anchor', 'anchor_half')
                for side in ('0.030', '0.045', '0.060')} | {f'rotor_{i}' for i in range(4)}
    actual = {p.parent.name for p in root.glob('*/summary.json')}
    if actual != expected:
        raise ValueError(f'Probe set differs from declaration: missing={sorted(expected-actual)}; extra={sorted(actual-expected)}')
    checks = []
    records = {}
    for path in sorted(root.glob('*/summary.json')):
        directory = path.parent
        summary, rows = read_probe(directory)
        side = summary['args']['side']
        metrics = [geometry(r, side, e) for r in rows for e in range(3)]
        check = dict(probe=directory.name, binding=binding(directory),
                     max_residual_m=max(r['residual_m'] for r in metrics),
                     minimum_floor_m=min(r['floor_m'] for r in metrics))
        check['constraint_pass'] = check['max_residual_m'] <= thresholds['constraint_residual_max_m']
        if directory.name.startswith('coupled_'):
            check['floor_pass'] = check['minimum_floor_m'] > thresholds['coupled_probe_floor_m']
        if directory.name.startswith('anchor_'):
            drifts, periods = [], []
            for e, ratio in enumerate((.1, .2, .3)):
                energies = [energy(r, side, e, summary['reference_mass_kg']*ratio) for r in rows]
                drifts.append(max(abs(x-energies[0]) for x in energies)/energies[0])
                crossing = [rows[i]['time_s'] for i in range(1,len(rows))
                            if rows[i-1]['package_xyz'][e][1] <= 0 < rows[i]['package_xyz'][e][1]]
                periods.append(float(np.mean(np.diff(crossing))) if len(crossing)>1 else None)
            check.update(relative_energy_drift=drifts, measured_period_s=periods,
                         energy_pass=max(drifts) <= thresholds['anchor_relative_energy_drift_limit'])
        records[directory.name] = (summary, rows)
        checks.append(check)
    comparisons = []
    for side in ('.030','.045','.060'):
        a, b = f'anchor_0{side}', f'anchor_half_0{side}'
        if a not in records or b not in records:
            continue
        rows, half = records[a][1], records[b][1]
        pos_error, angle_error = 0., 0.
        for r, h in zip(rows, half[1::2]):
            pos_error = max(pos_error, float(np.max(np.linalg.norm(np.array(r['package_xyz'])-
                                                                 np.array(h['package_xyz']), axis=-1))))
            for e in range(3):
                angle_error = max(angle_error, abs(geometry(r,float(side),e)['swing_deg']-
                                                   geometry(h,float(side),e)['swing_deg']))
        comparisons.append(dict(side_m=float(side), max_position_difference_m=pos_error,
                                max_swing_difference_deg=angle_error,
                                passed=pos_error <= thresholds['half_dt_position_max_difference_m'] and
                                angle_error <= thresholds['half_dt_swing_max_difference_deg']))
    failed = any(not c[k] for c in checks for k in c if k.endswith('_pass')) or any(not c['passed'] for c in comparisons)
    return dict(schema='payload-probe-check-v1', status='FAIL' if failed else 'DIAGNOSTICS_ONLY',
                checker_sha256=digest(__file__), physical_training_admitted=False,
                checks=checks, timestep_comparisons=comparisons,
                remaining=['Runtime collision extents', 'Selective mass/inertia/geometry reset isolation',
                           'Rotor impulse sign/COM audit', 'Compression/reaction observability audit',
                           'All-target and perturbed-state clearance', 'Small-angle analytical period comparison'])


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--thresholds', default='configs/payload_preflight.yaml')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    receipt = verify(args.root, yaml.safe_load(Path(args.thresholds).read_text()))
    receipt['thresholds_sha256'] = digest(args.thresholds)
    out.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(receipt['status'])


if __name__ == '__main__':
    main()
