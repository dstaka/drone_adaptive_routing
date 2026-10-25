"""Independent anchored two-body energy and crossing-period reconstruction.

This module screens only passive small-angle cases. It has no admission path.
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LENGTH_M = .3
GRAVITY = 9.81


def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def properties(runtime):
    result = {}
    for i, link in enumerate(runtime['links']):
        if link['name'] in ('suspension_rod', 'payload'):
            mass = float(link['mass_kg'])
            inertia = np.asarray(runtime['inertia_kg_m2'][i], dtype=float)
            if (not math.isfinite(mass) or mass <= 0 or inertia.shape != (3, 3)
                    or not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0):
                raise ValueError('Invalid actual mass or inertia')
            result[link['name']] = (mass, inertia)
    if len(result) != 2:
        raise ValueError('Both actual rigid bodies are required')
    return result


def mechanical_energy(row, runtime, side_m, anchor_z):
    """Actual COM kinetic energy plus gravity relative to lowest hanging state."""
    props = properties(runtime)
    rod_mass, rod_inertia = props['suspension_rod']
    load_mass, load_inertia = props['payload']
    rod_r = rotation(row['rod_quaternion_wxyz'])
    package_r = rotation(row['package_quaternion_wxyz'])
    offset = rod_r @ np.array([0., 0., -LENGTH_M/2])
    rod_com = np.asarray(row['rod_xyz'])+offset
    rod_omega = np.asarray(row['rod_angular_velocity'])
    rod_v = np.asarray(row['rod_velocity'])+np.cross(rod_omega, offset)
    load_omega = np.asarray(row['package_angular_velocity'])
    load_v = np.asarray(row['package_velocity'])
    kinetic = (rod_mass*np.dot(rod_v, rod_v)+load_mass*np.dot(load_v, load_v)
               + rod_omega @ (rod_r @ rod_inertia @ rod_r.T) @ rod_omega
               + load_omega @ (package_r @ load_inertia @ package_r.T) @ load_omega)/2
    potential = GRAVITY*(rod_mass*(rod_com[2]-(anchor_z-LENGTH_M/2))
                         + load_mass*(row['package_xyz'][2]-(anchor_z-LENGTH_M-side_m/2)))
    energy = float(kinetic+potential)
    if not math.isfinite(energy):
        raise ValueError('Nonfinite passive energy')
    return energy


def analytical_period(runtime, side_m):
    """Slow small-angle mode of a massive rod plus independently rotating cube.

    Generalized coordinates are absolute rod and cube tilt. M[0,1] accounts
    for the lower attachment's translation; each body's COM inertia is retained.
    """
    props = properties(runtime)
    mr, ir = props['suspension_rod']
    mp, ip = props['payload']
    a, length = side_m/2, LENGTH_M
    rod_i, cube_i = float(np.mean(np.diag(ir)[:2])), float(np.mean(np.diag(ip)[:2]))
    mass = np.array([[mr*(length/2)**2+rod_i+mp*length**2, mp*length*a],
                     [mp*length*a, mp*a*a+cube_i]])
    stiffness = np.diag([GRAVITY*(mr*length/2+mp*length), GRAVITY*mp*a])
    values = np.linalg.eigvals(np.linalg.solve(mass, stiffness))
    if not np.isreal(values).all() or values.min() <= 0:
        raise ValueError('Invalid pendulum eigenvalues')
    return float(2*math.pi/math.sqrt(values.min()))


def crossing_period(rows, side_m):
    """Median same-direction zero-crossing period of measured attachment tilt."""
    directions = []
    for row in rows:
        a = np.asarray(row['drone_xyz'])+rotation(row['drone_quaternion_wxyz']) @ [0, 0, -.02]
        b = np.asarray(row['package_xyz'])+rotation(row['package_quaternion_wxyz']) @ [0, 0, side_m/2]
        directions.append(b-a)
    directions = np.asarray(directions)
    axis = directions[0, :2]
    if np.linalg.norm(axis) < 1e-8:
        raise ValueError('No initial small-angle excitation direction')
    axis = axis/np.linalg.norm(axis)
    angles = np.arctan2(directions[:, :2] @ axis, -directions[:, 2])
    times = np.asarray([row['time_s'] for row in rows])
    crossings = []
    for i in range(1, len(rows)):
        if angles[i-1] > 0 >= angles[i]:
            fraction = angles[i-1]/(angles[i-1]-angles[i])
            crossings.append(float(times[i-1]+fraction*(times[i]-times[i-1])))
    if len(crossings) < 3:
        raise ValueError('Insufficient full measured periods')
    intervals = np.diff(crossings)
    return float(np.median(intervals)), len(crossings), float(np.max(intervals)-np.min(intervals))


def analyse_passive(rows, runtime, case, *, energy_limit=.05, period_limit=.1):
    """Reconstruct all raw samples, preserving loss as a measured quantity."""
    failures = []
    result = dict(status='QUALIFICATION_BLOCKED', physical_training_admitted=False,
                  failure_codes=failures, sample_count=len(rows),
                  energy_reference='Lowest static hanging anchor: rod COM z=a-.15; cube COM z=a-.3-side/2',
                  initial_energy_j=None, positive_energy_fraction=None, minimum_energy_fraction_of_initial=None,
                  measured_period_s=None, analytical_period_s=None, period_relative_error=None, crossings=0)

    def fail(code):
        if code not in failures:
            failures.append(code)

    if case.get('kind') != 'anchor' or case.get('duration_s', 0) < 120 or case.get('sample_dt_s') != .01:
        fail('INCOMPLETE_ANCHORED_CASE')
    expected = round(case.get('duration_s', 120)/case.get('sample_dt_s', .01))+1
    if len(rows) != expected or len(rows) < 12001:
        fail('INCOMPLETE_TRACE')
    if not rows:
        return result
    try:
        side = float(case['side_m'])
        first = rows[0]
        anchor = np.asarray(first['drone_xyz'])+rotation(first['drone_quaternion_wxyz']) @ [0, 0, -.02]
        energies = []
        for i, row in enumerate(rows):
            if row['step'] != i or not math.isclose(row['time_s'], i*.01, abs_tol=1e-8, rel_tol=0):
                fail('SAMPLE_CONTINUITY')
            actual_anchor = np.asarray(row['drone_xyz'])+rotation(row['drone_quaternion_wxyz']) @ [0, 0, -.02]
            if (np.linalg.norm(actual_anchor-anchor) > 1e-6
                    or np.linalg.norm(row['drone_velocity']) > 1e-6
                    or np.linalg.norm(row['drone_angular_velocity']) > 1e-6):
                fail('ANCHOR_MOVED')
            if np.any(np.asarray(row['rpm']) != 0):
                fail('PASSIVE_MOTOR_INPUT')
            energies.append(mechanical_energy(row, runtime, side, float(anchor[2])))
        initial = energies[0]
        result['initial_energy_j'] = initial
        if initial <= 1e-12:
            fail('NO_POSITIVE_INITIAL_EXCITATION')
        else:
            fraction = max(0., (max(energies)-initial)/initial)
            result.update(positive_energy_fraction=fraction,
                          minimum_energy_fraction_of_initial=min(energies)/initial,
                          terminal_energy_fraction_of_initial=energies[-1]/initial,
                          maximum_energy_j=max(energies), minimum_energy_j=min(energies))
            if fraction > min(.05, energy_limit):
                fail('PASSIVE_POSITIVE_ENERGY_GROWTH')
        predicted = analytical_period(runtime, side)
        measured, crossings, spread = crossing_period(rows, side)
        relative = abs(measured-predicted)/predicted
        result.update(analytical_period_s=predicted, measured_period_s=measured,
                      period_relative_error=relative, crossings=crossings, crossing_period_range_s=spread)
        if relative > min(.1, period_limit):
            fail('PASSIVE_PERIOD_MISMATCH')
    except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError):
        fail('INVALID_PASSIVE_RAW_STATE')
    if not failures:
        result['status'] = 'PASSIVE_SCREEN_PASSED'
    return result


def verify_passive_probe(probe, *, source_root=ROOT):
    """Integrate the current independent binding/state checker, never its runner flag."""
    script = ROOT/'scripts/check_payload_ball.py'
    spec = importlib.util.spec_from_file_location('passive_ball_raw_checker', script)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    bound = checker.verify_probe(probe, source_root=source_root)
    receipt = dict(schema='payload-ball-passive-independent-v1', status='QUALIFICATION_BLOCKED',
                   physical_training_admitted=False, policy_admitted=False,
                   failure_codes=[], raw_binding_check=bound, passive=None,
                   module_sha256=checker.digest(__file__),
                   interpretation_ja='固定anchor9条件の受動診断だけを再構成。運用域、powered、reset、refinementの合格や学習許可を示さない。')
    if bound['status'] != 'EXPERIMENTAL_DIAGNOSTICS_ONLY':
        receipt['failure_codes'].append('SOURCE_OR_RAW_EVIDENCE_INVALID')
        return receipt
    probe = Path(probe)
    declaration = json.loads((probe/'declaration.json').read_text())
    if declaration.get('fixture') != 'anchor' or (declaration.get('case') or {}).get('kind') != 'anchor':
        receipt['failure_codes'].append('NOT_AN_ANCHORED_SMALL_ANGLE_CASE')
        return receipt
    rows = [json.loads(line) for line in (probe/'samples.jsonl').read_text().splitlines()]
    runtime = json.loads((probe/'runtime-initial.json').read_text())['runtime']
    limits = json.loads((Path(source_root)/'configs/payload_retry_probe_manifest.json').read_text())['limits']
    result = analyse_passive(rows, runtime, declaration['case'],
                             energy_limit=limits['passive_positive_energy_fraction'],
                             period_limit=limits['period_relative_error'])
    receipt['passive'] = result
    receipt['failure_codes'] = result['failure_codes']
    if result['status'] == 'PASSIVE_SCREEN_PASSED':
        receipt['status'] = 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
    return receipt
