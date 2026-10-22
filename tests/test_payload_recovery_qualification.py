"""Recovery staging must reject legacy, incomplete or invalid passive evidence."""
import copy
import importlib.util
import json
import hashlib
from pathlib import Path

import pytest

from test_payload_passive_check import oscillation, runtime
from test_payload_recovery_passive_binding import fixture as measured_fixture

ROOT = Path(__file__).resolve().parents[1]


def api():
    from adaptive_drone_patrol import payload_recovery_qualification
    return payload_recovery_qualification


def manifest(substeps=8):
    old = json.loads((ROOT / 'configs/payload_retry_probe_manifest.json').read_text())
    return api().make_manifest(old, base_substeps=substeps)


def test_recovery_preserves_all55_conditions_and_adds_seven_actual_half_dt_runs():
    m = manifest(8)
    assert len(m['cases']) == 62
    assert len(api().passive_cases(m)) == 32
    assert m['settings']['rigid_dt_s'] == .00125
    half = [c for c in m['cases'] if c.get('passive_half_dt')]
    assert len(half) == 7
    assert all(c['substeps'] == 16 and c['duration_s'] == 30 for c in half)
    assert {c['initial_swing_deg'] for c in half} == {90., 170.}
    refine = [c for c in m['cases'] if c['kind'] == 'refinement']
    assert {c['substeps'] for c in refine} == {8, 16}
    assert api().validate_manifest(m)


def test_probe_accepts_explicit_recovery62_but_still_checks_legacy55():
    spec = importlib.util.spec_from_file_location('recovery_probe_test', ROOT/'scripts/probe_payload_ball.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe.validate_probe_declaration(manifest(128))
    old = json.loads((ROOT / 'configs/payload_retry_probe_manifest.json').read_text())
    assert probe.validate_probe_declaration(old)
    old['cases'].pop()
    with pytest.raises(ValueError):
        probe.validate_probe_declaration(old)


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf'), 31])
def test_small_diagnostic_duration_is_bounded_and_cannot_replace_declared_case(duration):
    spec = importlib.util.spec_from_file_location('recovery_probe_duration', ROOT/'scripts/probe_payload_ball.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    with pytest.raises(ValueError):
        probe.probe_duration({'duration_s': 30}, duration)


def test_small_diagnostic_keeps_full_qualification_declaration_unchanged():
    spec = importlib.util.spec_from_file_location('recovery_probe_duration', ROOT/'scripts/probe_payload_ball.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    case = {'duration_s': 30}
    assert probe.probe_duration(case, 3) == 3
    assert case == {'duration_s': 30}


@pytest.mark.parametrize('short', [None, 3.])
def test_powered_recovery_probe_cannot_bypass_actual_passive_barrier(short):
    spec = importlib.util.spec_from_file_location('recovery_probe_stage', ROOT/'scripts/probe_payload_ball.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    m = manifest()
    case = next(c for c in m['cases'] if c['kind'] == 'powered')
    # A short-duration switch is deliberately absent from the barrier API.
    # Both requests must independently reconstruct every passive raw trace.
    if short:
        assert probe.probe_duration(case, short) == short
    with pytest.raises(ValueError, match='32 passive'):
        probe.require_probe_stage(m, case, 'loaded')


def test_recovery_case_cannot_be_run_on_unloaded_or_original_fixture():
    spec = importlib.util.spec_from_file_location('recovery_probe_stage', ROOT/'scripts/probe_payload_ball.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    m = manifest()
    with pytest.raises(ValueError, match='fixture'):
        probe.require_probe_stage(m, m['cases'][0], 'original')


@pytest.mark.parametrize('mutation', ['omit', 'duplicate', 'short', 'loosen', 'dt', 'initial', 'reset'])
def test_changed_conditions_cannot_rescue_recovery(mutation):
    m = manifest()
    if mutation == 'omit':
        m['cases'].pop()
    elif mutation == 'duplicate':
        m['cases'][-1] = copy.deepcopy(m['cases'][0])
    elif mutation == 'short':
        m['cases'][0]['duration_s'] = 119.99
    elif mutation == 'loosen':
        m['limits']['passive_positive_energy_fraction'] = .051
    elif mutation == 'dt':
        m['cases'][0]['substeps'] *= 2
    elif mutation == 'initial':
        m['cases'][0]['initial_swing_deg'] = 4.9
    else:
        m['cases'][0]['reset_during_run'] = True
    with pytest.raises(ValueError):
        api().validate_manifest(m)


@pytest.mark.parametrize('substeps', [True, 0, -2, 1.5, float('inf')])
def test_invalid_numerical_settings_fail(substeps):
    with pytest.raises(ValueError):
        manifest(substeps)


def passive_fixture():
    case = next(c for c in manifest()['cases'] if c['id'] == 'anchor_0.045_0.20')
    rows = oscillation()
    for row in rows:
        row['contact'] = False
    forces = [dict(step=i, time_s=i*.01, rpm=[0]*4,
                   negative_external_force_field=[[[0, 0, 0]]],
                   negative_external_torque_about_root_com_field=[[[0, 0, 0]]])
              for i in range(len(rows)-1)]
    return case, rows, forces


def test_passive_barrier_needs_all32_independently_screened_traces():
    m = manifest()
    with pytest.raises(ValueError, match='32'):
        api().require_passive_barrier(m, {})
    forged = {c['id']: {'status': 'PASSIVE_SCREEN_PASSED'} for c in api().passive_cases(m)}
    with pytest.raises(ValueError):
        api().require_passive_barrier(m, forged)


def test_same_raw_directory_cannot_replace32_independent_cases(tmp_path):
    m = manifest()
    paths = {c['id']: tmp_path for c in api().passive_cases(m)}
    with pytest.raises(ValueError, match='Duplicate'):
        api().require_passive_barrier(m, paths)


@pytest.mark.parametrize('mutation', ['energy', 'quat', 'force', 'missing_force', 'speed', 'terminal'])
def test_passive_screen_rejects_abnormal_state_and_unobserved_zero_input(mutation):
    case, rows, forces = passive_fixture()
    if mutation == 'energy':
        rows[100]['package_velocity'] = [2, 0, 0]
    elif mutation == 'quat':
        rows[100]['package_quaternion_wxyz'] = [2, 0, 0, 0]
    elif mutation == 'force':
        forces[100]['negative_external_force_field'][0][0][0] = .001
    elif mutation == 'missing_force':
        forces = []
    elif mutation == 'speed':
        rows[100]['package_velocity'] = [21, 0, 0]
    else:
        rows = rows[:-1]
    result = api().screen_passive(case, rows, runtime(), forces)
    assert result['status'] == 'PASSIVE_BLOCKED'
    assert result['failure_codes']
    assert not result['physical_training_admitted']


def test_synthetic_passive_screen_reports_energy_loss_and_cannot_admit_optimizer():
    case, rows, forces = passive_fixture()
    result = api().screen_passive(case, rows, runtime(), forces)
    assert result['status'] == 'PASSIVE_SCREEN_PASSED'
    assert result['positive_energy_fraction'] < .01
    assert result['minimum_energy_fraction_of_initial'] > .98
    assert result['period_relative_error'] < .01
    assert not result['physical_training_admitted']


def disk_probe(tmp_path, mutation=None):
    """Synthetic artifact bytes exercise binding rejection, never physical PASS."""
    module, m = api(), manifest(128)
    source, engine, probe = [tmp_path/name for name in ('source', 'engine', 'probe')]
    probe.mkdir()
    for name in module.SOURCE_FILES:
        p = source/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('synthetic source fixture: ' + name)
    for name in module.ENGINE_REQUIRED:
        p = engine/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('synthetic engine fixture: ' + name)
    asset = engine/'assets/urdf/drones/cf2x.urdf'
    asset.parent.mkdir(parents=True)
    asset.write_text('<robot name="synthetic_binding_fixture"/>')
    (probe/'candidate.xml').write_text('<mujoco model="synthetic_binding_fixture"/>')
    manifest_path = source/'configs/payload_recovery_probe_manifest.json'
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(m))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    rt, _, row, wrench = measured_fixture()
    case = next(c for c in m['cases'] if c['id'] == 'large_0.060_0.10')
    row.update(step=0, time_s=0., rpm=[0]*4, contact=False)
    wrench.update(step=0, time_s=0.)
    declaration = dict(case=case, fixture='anchor', physical_training_admitted=False,
                       diagnostic_duration_s=None, settings=m['settings'],
                       manifest_sha256=digest(manifest_path),
                       source_binding={name:digest(source/name) for name in module.SOURCE_FILES},
                       backend_source_sha256=digest(source/'src/adaptive_drone_patrol/payload_ball_bank.py'))
    initial = dict(declaration, runtime=rt,
                   engine_source={str(p.relative_to(engine)):digest(p) for p in engine.rglob('*.py')},
                   original_asset_sha256=digest(asset),
                   asset={'generated_asset_sha256':digest(probe/'candidate.xml')})
    summary = copy.deepcopy(initial)
    raw = copy.deepcopy(row)
    if mutation == 'source_subset':
        summary['source_binding'].pop('scripts/probe_payload_ball.py')
    elif mutation == 'engine_subset':
        summary['engine_source'].pop('utils/geom.py')
        initial['engine_source'] = copy.deepcopy(summary['engine_source'])
    elif mutation == 'mass':
        initial['runtime']['links'][6]['mass_kg'] *= 10
        summary['runtime'] = copy.deepcopy(initial['runtime'])
    elif mutation == 'dt':
        initial['runtime']['actual_solver_settings']['substep_dt'] = '.005'
        summary['runtime'] = copy.deepcopy(initial['runtime'])
    elif mutation == 'raw':
        raw['package_velocity'] = [1, 0, 0]
    elif mutation == 'initial':
        row['package_xyz'][0] += .01
        raw = copy.deepcopy(row)
    elif mutation == 'empty_wrench':
        wrench['negative_external_force_field'] = []
    for name, data in [('samples.jsonl', row), ('raw-states.jsonl', raw), ('motor-inputs.jsonl', wrench)]:
        (probe/name).write_text(json.dumps(data)+'\n')
    (probe/'declaration.json').write_text(json.dumps(declaration))
    (probe/'runtime-initial.json').write_text(json.dumps(initial))
    summary.update(samples_sha256=digest(probe/'samples.jsonl'),
                   raw_state_sha256=digest(probe/'raw-states.jsonl'),
                   motor_inputs_sha256=digest(probe/'motor-inputs.jsonl'),
                   runtime_initial_sha256=digest(probe/'runtime-initial.json'))
    if mutation == 'runtime_bytes':
        (probe/'runtime-initial.json').write_text(json.dumps(initial)+'\n')
    (probe/'summary.json').write_text(json.dumps(summary))
    return probe, m, source, engine


@pytest.mark.parametrize('mutation', ['source_subset', 'engine_subset', 'mass', 'dt', 'raw',
                                      'initial', 'empty_wrench', 'runtime_bytes'])
def test_real_artifact_path_rejects_relabelled_or_incomplete_bindings(tmp_path, mutation):
    probe, m, source, engine = disk_probe(tmp_path, mutation)
    with pytest.raises(ValueError):
        api().verify_passive_probe(probe, m, source_root=source, engine_root=engine)


def test_valid_binding_bytes_do_not_turn_one_sample_into_complete_passive_evidence(tmp_path):
    probe, m, source, engine = disk_probe(tmp_path)
    result = api().verify_passive_probe(probe, m, source_root=source, engine_root=engine)
    assert result['status'] == 'PASSIVE_BLOCKED'
    assert 'INCOMPLETE_TRACE_OR_FORCE_OBSERVATION' in result['failure_codes']
    assert not result['physical_training_admitted']
