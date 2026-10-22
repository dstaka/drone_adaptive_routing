"""Recovery declarations and passive-only barrier; never optimizer admission.

The old 55-case declaration/checker remains immutable. A new numerical candidate
must execute all 62 conditions with its actual base/half timesteps. These CPU
screens do not establish motor, reset, batch or powered-work qualification.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .payload_passive_check import analytical_period, crossing_period, mechanical_energy, rotation
from .payload_qualification import validate_declaration
from .payload_recovery_passive_binding import physical_errors, initial_errors, wrench_errors

ROOT = Path(__file__).resolve().parents[2]
LEGACY = 'configs/payload_retry_probe_manifest.json'
SCHEMA = 'payload-recovery-domain-v1'
SOURCE_FILES = (
    'scripts/probe_payload_ball.py',
    'src/adaptive_drone_patrol/payload_ball_bank.py',
    'src/adaptive_drone_patrol/payload_ball_asset.py',
    'src/adaptive_drone_patrol/payload_contract.py',
    'src/adaptive_drone_patrol/payload_qualification.py',
    'src/adaptive_drone_patrol/payload_recovery_qualification.py',
    'src/adaptive_drone_patrol/payload_recovery_passive_binding.py',
    'src/adaptive_drone_patrol/payload_passive_check.py',
    'src/adaptive_drone_patrol/payload_probe_controller.py',
    'src/adaptive_drone_patrol/payload_substep_probe.py',
    'src/adaptive_drone_patrol/util.py',
)
ENGINE_REQUIRED = {
    'utils/mjcf.py', 'utils/geom.py', 'options/morphs.py', 'options/solvers.py',
    'engine/simulator.py', 'engine/scene.py',
    'engine/solvers/rigid/rigid_solver.py',
    'engine/solvers/rigid/abd/forward_dynamics.py',
    'engine/solvers/rigid/abd/forward_kinematics.py',
    'engine/solvers/rigid/abd/accessor.py',
    'engine/entities/rigid_entity/rigid_entity.py',
    'engine/entities/rigid_entity/rigid_link.py',
    'engine/entities/rigid_entity/inertial.py',
    'engine/entities/rigid_entity/description.py',
}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest(legacy, *, base_substeps=2):
    """Preserve inherited physical conditions, expressing actual numerical dt."""
    validate_declaration(legacy)
    if type(base_substeps) is not int or not 1 <= base_substeps <= 128:
        raise ValueError('Finite integer base_substeps in [1,128] required')
    result = copy.deepcopy(legacy)
    result.update(schema=SCHEMA, profile=f'cf2x-native-ball-recovery-s{base_substeps}-v1',
                  base_substeps=base_substeps, physical_training_admitted=False,
                  legacy_manifest_sha256=_digest(ROOT / LEGACY))
    result['settings']['rigid_dt_s'] = .01 / base_substeps
    result['settings']['integrator'] = 'approximate_implicitfast'
    result['limits']['unexplained_passive_energy_loss_fraction'] = .20
    for case in result['cases']:
        # IDs identify inherited conditions; explicit settings identify timestep.
        half = case['kind'] == 'refinement' and case['substeps'] == 4
        case['legacy_case_id'] = case['id']
        case['timestep_role'] = 'half' if half else 'base'
        case['substeps'] = base_substeps * (2 if half else 1)
    extras = [c for c in result['cases'] if c['id'].startswith('large_')
              and c['mass_ratio'] in (.1, .3)]
    extras += [next(c for c in result['cases'] if c['id'] == 'sweep_170')]
    for original in extras:
        case = copy.deepcopy(original)
        case.update(id=original['id'] + '_passive_half', passive_half_dt=True,
                    timestep_role='half', substeps=2*base_substeps)
        result['cases'].append(case)
    return result


def validate_manifest(manifest):
    """Reject omissions, edited draws/tolerances and mismatched actual dt."""
    try:
        baseline = json.loads((ROOT / LEGACY).read_text())
        expected = make_manifest(baseline, base_substeps=manifest['base_substeps'])
        if manifest != expected:
            raise ValueError('Recovery declaration differs from its complete frozen template')
    except (KeyError, TypeError) as exc:
        raise ValueError('Invalid recovery declaration') from exc
    return True


def passive_cases(manifest):
    validate_manifest(manifest)
    return [c for c in manifest['cases'] if c['kind'] in ('anchor', 'large_angle')]


def screen_passive(case, rows, runtime, force_rows):
    """Reconstruct all samples and observed zero inputs independently of status.

    Source/property binding is intentionally separate in verify_passive_probe.
    A passing synthetic screen cannot authorize any powered step or optimizer.
    """
    failures = []
    result = dict(status='PASSIVE_BLOCKED', physical_training_admitted=False,
                  failure_codes=failures, sample_count=len(rows), force_records=len(force_rows))

    def fail(code):
        if code not in failures:
            failures.append(code)

    try:
        dt, duration = case['sample_dt_s'], case['duration_s']
        count = round(duration/dt)
        if dt != .01 or case['kind'] not in ('anchor', 'large_angle'):
            fail('INVALID_PASSIVE_CONDITION')
        if len(rows) != count+1 or not rows or len(force_rows) != count:
            fail('INCOMPLETE_TRACE_OR_FORCE_OBSERVATION')
        if not rows:
            return result
        first = rows[0]
        anchor = np.asarray(first['drone_xyz']) + rotation(first['drone_quaternion_wxyz']) @ [0, 0, -.02]
        energies = []
        max_residual = 0.
        for i, row in enumerate(rows):
            if row['step'] != i or not math.isclose(row['time_s'], i*dt, abs_tol=1e-8, rel_tol=0):
                fail('SAMPLE_CONTINUITY')
            if np.any(np.asarray(row['rpm']) != 0):
                fail('PASSIVE_MOTOR_INPUT')
            for body in ('drone', 'rod', 'package'):
                for field, size in [('xyz', 3), ('quaternion_wxyz', 4), ('velocity', 3),
                                    ('angular_velocity', 3)]:
                    value = np.asarray(row[f'{body}_{field}'], dtype=float)
                    if value.shape != (size,) or not np.isfinite(value).all():
                        raise ValueError('Invalid raw vector')
                    norm = np.linalg.norm(value)
                    if field == 'quaternion_wxyz' and abs(norm-1) > 1e-4:
                        fail('INVALID_QUATERNION')
                    if field == 'velocity' and norm > 20:
                        fail('FINITE_SPEED_EXCURSION')
                    if field == 'angular_velocity' and norm > 200:
                        fail('FINITE_ANGULAR_SPEED_EXCURSION')
            actual_anchor = np.asarray(row['drone_xyz']) + rotation(row['drone_quaternion_wxyz']) @ [0, 0, -.02]
            if (np.linalg.norm(actual_anchor-anchor) > 1e-6
                    or np.linalg.norm(row['drone_velocity']) > 1e-6
                    or np.linalg.norm(row['drone_angular_velocity']) > 1e-6):
                fail('ANCHOR_MOVED')
            endpoint = np.asarray(row['package_xyz']) + rotation(row['package_quaternion_wxyz']) @ [0, 0, case['side_m']/2]
            residual = abs(np.linalg.norm(endpoint-actual_anchor)-.3)
            max_residual = max(max_residual, float(residual))
            if residual > 2e-5:
                fail('CONSTRAINT_RESIDUAL')
            if row.get('contact') is not False:
                fail('CONTACT_OR_MISSING_OBSERVATION')
            energies.append(mechanical_energy(row, runtime, case['side_m'], float(anchor[2])))
        for i, row in enumerate(force_rows):
            if row['step'] != i or not math.isclose(row['time_s'], i*dt, abs_tol=1e-8, rel_tol=0):
                fail('FORCE_CONTINUITY')
            rpm = np.asarray(row['rpm'], dtype=float)
            if rpm.shape != (4,) or not np.isfinite(rpm).all() or np.any(rpm != 0):
                fail('PASSIVE_MOTOR_INPUT')
            for field in ('negative_external_force_field', 'negative_external_torque_about_root_com_field'):
                value = np.asarray(row[field], dtype=float)
                if (value.ndim != 3 or value.shape[0] == 0 or value.shape[1:] != (1, 3)
                        or not np.isfinite(value).all() or np.any(value != 0)):
                    fail('NONZERO_OR_INVALID_EXTERNAL_WRENCH')
        initial = energies[0]
        if initial <= 1e-12:
            fail('NO_POSITIVE_INITIAL_EXCITATION')
        else:
            growth = max(0., (max(energies)-initial)/initial)
            minimum = min(energies)/initial
            result.update(initial_energy_j=initial, maximum_energy_j=max(energies),
                          positive_energy_fraction=growth, minimum_energy_fraction_of_initial=minimum,
                          terminal_energy_fraction_of_initial=energies[-1]/initial,
                          maximum_constraint_residual_m=max_residual)
            if growth > .05:
                fail('PASSIVE_POSITIVE_ENERGY_GROWTH')
            if minimum < .8:
                fail('UNEXPLAINED_PASSIVE_ENERGY_LOSS')
        if case['kind'] == 'anchor':
            if duration < 120:
                fail('SHORT_ANCHOR')
            predicted = analytical_period(runtime, case['side_m'])
            measured, crossings, _ = crossing_period(rows, case['side_m'])
            relative = abs(measured-predicted)/predicted
            result.update(measured_period_s=measured, analytical_period_s=predicted,
                          period_relative_error=relative, crossings=crossings)
            if relative > .1:
                fail('PASSIVE_PERIOD_MISMATCH')
        elif duration < 30:
            fail('SHORT_LARGE_ANGLE')
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, np.linalg.LinAlgError):
        fail('INVALID_PASSIVE_RAW_STATE')
    if not failures:
        result['status'] = 'PASSIVE_SCREEN_PASSED'
    return result


def verify_passive_probe(probe, manifest, *, source_root=ROOT, engine_root=None):
    """Bind independent screens to actual source, engine, properties and bytes."""
    validate_manifest(manifest)
    probe, source_root = Path(probe), Path(source_root)
    summary = json.loads((probe/'summary.json').read_text())
    declaration = json.loads((probe/'declaration.json').read_text())
    initial_record = json.loads((probe/'runtime-initial.json').read_text())
    case = next(c for c in passive_cases(manifest) if c['id'] == declaration['case']['id'])
    manifest_hash = _digest(source_root/'configs/payload_recovery_probe_manifest.json')
    for record in (declaration, summary, initial_record):
        if (record['case'] != case or record['settings'] != manifest['settings']
                or record['fixture'] != 'anchor' or record.get('physical_training_admitted') is not False
                or record.get('diagnostic_duration_s') is not None):
            raise ValueError('Actual case/settings/fixture differ or short diagnostic cannot qualify')
        if record['manifest_sha256'] != manifest_hash:
            raise ValueError('Manifest source differs')
        bindings = record['source_binding']
        if set(bindings) != set(SOURCE_FILES):
            raise ValueError('Complete recovery source binding required')
        if any(_digest(source_root/name) != value for name, value in bindings.items()):
            raise ValueError('Executed source differs')
        if record['backend_source_sha256'] != bindings['src/adaptive_drone_patrol/payload_ball_bank.py']:
            raise ValueError('Backend source differs')
    if engine_root is None or not summary.get('engine_source'):
        raise ValueError('Actual pinned engine source required')
    engine_root = Path(engine_root)
    expected_engine_paths = {str(p.relative_to(engine_root)) for p in engine_root.rglob('*.py')}
    if (not ENGINE_REQUIRED <= expected_engine_paths
            or set(summary['engine_source']) != expected_engine_paths
            or initial_record['engine_source'] != summary['engine_source']):
        raise ValueError('Complete actual pinned engine source binding required')
    if any(_digest(engine_root/name) != value for name, value in summary['engine_source'].items()):
        raise ValueError('Installed engine differs')
    for record in (summary, initial_record):
        if (record['original_asset_sha256'] != _digest(engine_root/'assets/urdf/drones/cf2x.urdf')
                or record['asset']['generated_asset_sha256'] != _digest(probe/'candidate.xml')):
            raise ValueError('Actual original/generated asset differs')
    for name, key in [('samples.jsonl', 'samples_sha256'), ('raw-states.jsonl', 'raw_state_sha256'),
                      ('motor-inputs.jsonl', 'motor_inputs_sha256')]:
        if _digest(probe/name) != summary[key]:
            raise ValueError('Actual raw bytes differ')
    if summary['runtime_initial_sha256'] != _digest(probe/'runtime-initial.json'):
        raise ValueError('Actual runtime bytes differ')
    runtime = initial_record['runtime']
    if runtime != summary['runtime'] or physical_errors(runtime, case):
        raise ValueError('Actual physical properties or numerical settings differ')
    rows = [json.loads(line) for line in (probe/'samples.jsonl').open()]
    force_rows = [json.loads(line) for line in (probe/'motor-inputs.jsonl').open()]
    raw = [json.loads(line) for line in (probe/'raw-states.jsonl').open()]
    if len(raw) != len(rows) or any(any(sample.get(key) != value for key, value in row.items())
                                    for row, sample in zip(raw, rows)):
        raise ValueError('Preclip raw states differ from screened samples')
    if not rows or initial_errors(rows[0], case):
        raise ValueError('Actual initial state differs from physical case')
    if any(wrench_errors(row, runtime) for row in force_rows):
        raise ValueError('Actual complete zero-wrench observations required')
    result = screen_passive(case, rows, runtime, force_rows)
    result.update(case_id=case['id'], samples_sha256=summary['samples_sha256'],
                  runtime_sha256=_digest(probe/'runtime-initial.json'))
    return result


def require_passive_barrier(manifest, probes, *, source_root=ROOT, engine_root=None):
    """Re-read 32 complete raw traces; only permit subsequent powered diagnostics."""
    cases = passive_cases(manifest)
    if set(probes) != {c['id'] for c in cases}:
        raise ValueError('All 32 passive traces are required before powered diagnostics')
    if any(not isinstance(probe, (str, Path)) for probe in probes.values()):
        raise ValueError('Raw probe directories required; receipt statuses cannot unlock barrier')
    if len({Path(probe).resolve() for probe in probes.values()}) != len(cases):
        raise ValueError('Duplicate probe directory cannot replace independent cases')
    results = {}
    for case in cases:
        probe = probes[case['id']]
        if not isinstance(probe, (str, Path)):
            raise ValueError('Raw probe directories required; receipt statuses cannot unlock barrier')
        result = verify_passive_probe(probe, manifest, source_root=source_root, engine_root=engine_root)
        if result['case_id'] != case['id']:
            raise ValueError('Probe mapping differs from actual declared case')
        results[case['id']] = result
        if result['status'] != 'PASSIVE_SCREEN_PASSED':
            raise ValueError(f'Passive condition failed: {case["id"]}')
    return dict(status='PASSIVE_BARRIER_PASSED', physical_training_admitted=False,
                powered_diagnostics_permitted=True, results=results)
