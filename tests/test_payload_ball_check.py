"""Independent negative evidence checks; fixtures contain no simulator success claims."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/check_payload_ball.py'
SOURCES = ('scripts/probe_payload_ball.py', 'src/adaptive_drone_patrol/payload_ball_asset.py',
           'src/adaptive_drone_patrol/payload_qualification.py', 'src/adaptive_drone_patrol/payload_contract.py')


def checker():
    assert SCRIPT.is_file(), 'independent ball checker must exist'
    spec = importlib.util.spec_from_file_location('ball_check', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, data):
    path.write_text(json.dumps(data, allow_nan=True) + '\n')


def finish(probe, rows):
    (probe / 'samples.jsonl').write_text(''.join(json.dumps(r, allow_nan=True)+'\n' for r in rows))
    summary = json.loads((probe / 'summary.json').read_text())
    summary['samples_sha256'] = hashlib.sha256((probe / 'samples.jsonl').read_bytes()).hexdigest()
    save(probe / 'summary.json', summary)


@pytest.fixture
def probe(tmp_path):
    p = tmp_path / 'probe'
    p.mkdir()
    manifest = json.loads((ROOT / 'configs/payload_retry_probe_manifest.json').read_text())
    mass = .02700047753751278 * .2
    inertia = mass * .045**2 / 6
    links = [{'name': 'base_link', 'idx': 1, 'mass_kg': .027}]
    links += [{'name': f'prop{i}_link', 'idx': i+2, 'mass_kg': 1.1920928955078125e-7} for i in range(4)]
    links += [{'name': 'suspension_rod', 'idx': 6, 'mass_kg': .0001},
              {'name': 'payload', 'idx': 7, 'mass_kg': mass}]
    runtime = dict(reference_mass_kg=.02700047753751278, links=links,
                   inertia_kg_m2=[[[inertia if i == j else 0 for j in range(3)] for i in range(3)]]*7,
                   n_qs=15, n_dofs=12,
                   joints=[dict(name='vehicle_free', n_qs=7, n_dofs=6),
                           dict(name='suspension_ball', n_qs=4, n_dofs=3),
                           dict(name='payload_ball', n_qs=4, n_dofs=3)],
                   initial_entity_pos=[0, 0, 1], initial_base_pos=[0, 0, 1],
                   initial_links_pos=[[0, 0, 1]]*5+[[0, 0, .98], [0, 0, .6575]],
                   initial_links_quat=[[1, 0, 0, 0]]*7,
                   rotor_indices=[2, 3, 4, 5], rotor_spin=[-1, 1, -1, 1],
                   kf=3.16e-10, km=7.94e-12,
                   rpm_mapping='(1+0.8*action)*14468.429183500699',
                   motor_application='same pinned solver.set_drone_rpm; rotor local +z force/torque at original link COM',
                   package_aabb=[[-.0225, -.0225, .635], [.0225, .0225, .68]],
                   damping_after=[0]*12, armature=[0]*12)
    common = dict(schema='payload-ball-probe-v2', fixture='loaded', case=None,
                  manifest_sha256=hashlib.sha256((ROOT / 'configs/payload_retry_probe_manifest.json').read_bytes()).hexdigest(),
                  source_binding={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
                  settings=manifest['settings'], physical_training_admitted=False)
    initial = dict(common, runtime=runtime)
    save(p / 'declaration.json', dict(common, status='STARTED', steps=0))
    save(p / 'runtime-initial.json', initial)
    save(p / 'summary.json', dict(initial, status='CONSTRUCTION_RECORDED', steps=100,
                                measured_duration_s=1., failures=[], operational_qualification='NOT_RUN'))
    rows=[]
    for i in range(101):
        row=dict(step=i, time_s=i*.01, motor_work_j=0., rpm=[0]*4,
                 drone_xyz=[0, 0, 1], rod_xyz=[0, 0, .98], package_xyz=[0, 0, .6575],
                 constraint_residual_m=0., package_floor_m=.635,
                 attachment_a_world=[0, 0, .98], attachment_b_world=[0, 0, .68],
                 qpos=[0]*15, dofs_velocity=[0]*12)
        for name in ('drone', 'rod', 'package'):
            row[name+'_quaternion_wxyz']=[1, 0, 0, 0]
            row[name+'_velocity']=[0, 0, 0]
            row[name+'_angular_velocity']=[0, 0, 0]
        rows.append(row)
    finish(p, rows)
    return p, rows


def test_complete_constructor_trace_never_admits_training(probe):
    p, _ = probe
    result = checker().verify_probe(p)
    assert result['status'] == 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
    assert result['physical_training_admitted'] is False
    assert result['checks']['sample_count'] == 101
    assert result['checks']['minimum_package_floor_m'] == pytest.approx(.635)


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('package_velocity', [21, 0, 0], 'FINITE_NUMERICAL_EXCURSION'),
    ('package_velocity', [float('nan'), 0, 0], 'NONFINITE_STATE'),
    ('rod_quaternion_wxyz', [2, 0, 0, 0], 'BAD_QUATERNION'),
    ('rod_xyz', [0, 0, .97], 'ATTACHMENT_CONSTRAINT'),
    ('package_floor_m', 10, 'DERIVED_METRIC_MISMATCH'),
    ('step', 99, 'SAMPLE_CONTINUITY'),
    ('time_s', .5, 'SAMPLE_CONTINUITY'),
])
def test_reconstructs_raw_state_instead_of_trusting_status(probe, field, value, reason):
    p, rows = probe
    rows[10][field] = value
    finish(p, rows)
    result = checker().verify_probe(p)
    assert result['status'] == 'QUALIFICATION_BLOCKED'
    assert reason in result['failure_codes']
    assert not result['physical_training_admitted']


def test_terminal_sample_cannot_be_omitted_even_with_updated_hash(probe):
    p, rows = probe
    finish(p, rows[:-1])
    result = checker().verify_probe(p)
    assert 'INCOMPLETE_TRACE' in result['failure_codes']


def test_sample_hash_tampering_is_rejected(probe):
    p, _ = probe
    with (p / 'samples.jsonl').open('a') as stream:
        stream.write('{}\n')
    assert 'SAMPLES_HASH_MISMATCH' in checker().verify_probe(p)['failure_codes']


@pytest.mark.parametrize('key', ['manifest_sha256', 'source_binding'])
def test_rejects_changed_manifest_or_actual_local_source(probe, key):
    p, _ = probe
    d=json.loads((p / 'declaration.json').read_text())
    if key == 'source_binding':
        d[key][SOURCES[0]]='0'*64
    else:
        d[key]='0'*64
    save(p / 'declaration.json', d)
    assert 'BINDING_MISMATCH' in checker().verify_probe(p)['failure_codes']


def test_old_pass_receipt_is_not_a_new_constructor_trace(probe):
    p, _ = probe
    save(p / 'summary.json', dict(status='PASS', physical_training_admitted=True))
    result=checker().verify_probe(p)
    assert 'OLD_OR_UNSUPPORTED_RECEIPT' in result['failure_codes']
    assert not result['physical_training_admitted']


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('rotor_indices', [1, 2, 3, 4], 'ROTOR_IDENTITY'),
    ('initial_entity_pos', [0, 0, .9], 'ROOT_SEMANTICS'),
    ('reference_mass_kg', .027, 'REFERENCE_MASS_DISCREPANCY'),
    ('package_aabb', [[0, 0, 0], [1, 1, 1]], 'PACKAGE_COLLISION_EXTENT'),
])
def test_runtime_properties_are_checked_from_actual_measurements(probe, field, value, reason):
    p, _ = probe
    data=json.loads((p / 'runtime-initial.json').read_text())
    data['runtime'][field]=value
    save(p / 'runtime-initial.json', data)
    assert reason in checker().verify_probe(p)['failure_codes']


def test_wrong_package_inertia_is_rejected(probe):
    p, _ = probe
    data=json.loads((p / 'runtime-initial.json').read_text())
    data['runtime']['inertia_kg_m2'][6][0][0]=1.
    save(p / 'runtime-initial.json', data)
    assert 'PACKAGE_INERTIA' in checker().verify_probe(p)['failure_codes']


def test_construction_failure_and_missing_traceback_are_distinct(probe):
    p, _ = probe
    data=json.loads((p / 'summary.json').read_text())
    data['status']='CONSTRUCTION_FAILED'
    save(p / 'summary.json', data)
    (p / 'samples.jsonl').unlink()
    result=checker().verify_probe(p)
    assert result['classification'] == 'CONSTRUCTION_FAILURE_WITH_MISSING_EVIDENCE'
    assert 'traceback.txt' in result['missing_evidence']
    (p / 'traceback.txt').write_text('private failure path must not be copied into receipt')
    assert checker().verify_probe(p)['classification'] == 'CONSTRUCTION_FAILURE'


def test_cli_refuses_overwrite_and_sanitizes_probe_path(probe, tmp_path):
    p, _ = probe
    out=tmp_path / 'receipt.json'
    command=[sys.executable, str(SCRIPT), '--probe', str(p), '--out', str(out)]
    result=subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert str(tmp_path) not in out.read_text()
    before=out.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert out.read_bytes() == before


def test_rotated_cube_floor_is_reconstructed_with_oriented_extent(probe):
    p, rows = probe
    # Forty-five degrees about vertical does not change the floor; about x does.
    rows[10]['package_quaternion_wxyz']=[.9238795325112867, .3826834323650898, 0, 0]
    rows[10]['package_floor_m']=.635  # stale upright floor
    finish(p, rows)
    result=checker().verify_probe(p)
    assert 'DERIVED_METRIC_MISMATCH' in result['failure_codes']
    assert result['checks']['minimum_package_floor_m'] == pytest.approx(.6256801948466054)


def test_missing_all_files_returns_sanitized_missing_evidence(tmp_path):
    result=checker().verify_probe(tmp_path)
    assert result['classification'] == 'MISSING_EVIDENCE'
    assert set(result['missing_evidence']) == {'declaration.json', 'summary.json', 'runtime-initial.json', 'samples.jsonl'}
    assert str(tmp_path.parent) not in json.dumps(result)


@pytest.mark.parametrize('value', [None, float('nan'), 'invalid'])
def test_malformed_summary_duration_fails_closed_without_crashing(probe, value):
    p, _ = probe
    summary=json.loads((p / 'summary.json').read_text())
    summary['measured_duration_s']=value
    save(p / 'summary.json', summary)
    assert 'SUMMARY_DURATION_MISMATCH' in checker().verify_probe(p)['failure_codes']


def test_malformed_receipt_object_is_classified(probe):
    p, _ = probe
    save(p / 'declaration.json', ['not an object'])
    assert 'MALFORMED_EVIDENCE' in checker().verify_probe(p)['failure_codes']


def test_consistently_shifted_rotor_indices_still_fail_original_identity(probe):
    p, _ = probe
    data=json.loads((p / 'runtime-initial.json').read_text())
    data['runtime']['rotor_indices']=[12, 13, 14, 15]
    for link in data['runtime']['links']:
        link['idx']+=10
    save(p / 'runtime-initial.json', data)
    assert 'ROTOR_IDENTITY' in checker().verify_probe(p)['failure_codes']


@pytest.mark.parametrize('field', ['qpos', 'dofs_velocity', 'rpm'])
def test_missing_raw_generalized_state_or_motor_command_is_rejected(probe, field):
    p, rows = probe
    del rows[10][field]
    finish(p, rows)
    assert 'INCOMPLETE_RAW_STATE' in checker().verify_probe(p)['failure_codes']


def test_malformed_case_is_rejected_without_crashing(probe):
    p, _ = probe
    data=json.loads((p / 'declaration.json').read_text())
    data['case']={'id': 'unknown'}
    save(p / 'declaration.json', data)
    assert 'BINDING_MISMATCH' in checker().verify_probe(p)['failure_codes']


def test_explicit_source_snapshot_uses_actual_bytes_instead_of_receipt_hashes(probe, tmp_path):
    p, _ = probe
    snapshot=tmp_path / 'snapshot'
    (snapshot / 'configs').mkdir(parents=True)
    (snapshot / 'configs/payload_retry_probe_manifest.json').write_bytes(
        (ROOT / 'configs/payload_retry_probe_manifest.json').read_bytes())
    bindings={}
    for name in SOURCES:
        dest=snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes()+b'\n# preserved execution version\n')
        bindings[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    for name in ('declaration.json', 'runtime-initial.json', 'summary.json'):
        data=json.loads((p / name).read_text())
        data['source_binding']=bindings
        save(p / name, data)
    assert 'BINDING_MISMATCH' in checker().verify_probe(p)['failure_codes']
    result=checker().verify_probe(p, source_root=snapshot)
    assert result['status'] == 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
    assert str(snapshot) not in json.dumps(result)
    (snapshot / SOURCES[0]).write_text('tampered after measurement')
    assert 'BINDING_MISMATCH' in checker().verify_probe(p, source_root=snapshot)['failure_codes']


def test_missing_source_snapshot_files_fail_closed(probe, tmp_path):
    p, _ = probe
    assert 'BINDING_MISMATCH' in checker().verify_probe(p, source_root=tmp_path)['failure_codes']


def test_cli_accepts_explicit_source_root(probe, tmp_path):
    p, _ = probe
    out=tmp_path / 'snapshot-check.json'
    result=subprocess.run([sys.executable, str(SCRIPT), '--probe', str(p), '--out', str(out),
                           '--source-root', str(ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())['physical_training_admitted'] is False


def test_archived_manifest_is_hashed_from_explicit_source_root(probe, tmp_path):
    p, _ = probe
    source=tmp_path / 'source'
    for name in (*SOURCES, 'configs/payload_retry_probe_manifest.json'):
        target=source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest=source / 'configs/payload_retry_probe_manifest.json'
    manifest.write_text(manifest.read_text()+'\n')
    assert 'BINDING_MISMATCH' in checker().verify_probe(p, source_root=source)['failure_codes']
    actual_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name in ('declaration.json', 'runtime-initial.json', 'summary.json'):
        data=json.loads((p / name).read_text())
        data['manifest_sha256']=actual_hash
        save(p / name, data)
    result=checker().verify_probe(p, source_root=source)
    assert result['manifest_sha256'] == actual_hash
    assert result['status'] == 'EXPERIMENTAL_DIAGNOSTICS_ONLY'


def test_matching_archived_hash_does_not_relax_checker_limits(probe, tmp_path):
    p, _ = probe
    source=tmp_path / 'source'
    for name in (*SOURCES, 'configs/payload_retry_probe_manifest.json'):
        target=source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest=source / 'configs/payload_retry_probe_manifest.json'
    data=json.loads(manifest.read_text())
    data['limits']['speed_max_m_s']=1e8
    save(manifest, data)
    actual_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name in ('declaration.json', 'runtime-initial.json', 'summary.json'):
        data=json.loads((p / name).read_text())
        data['manifest_sha256']=actual_hash
        save(p / name, data)
    assert 'MANIFEST_POLICY_MISMATCH' in checker().verify_probe(p, source_root=source)['failure_codes']


def anchor_v4(probe):
    """Hand fixture: free imported vehicle has epsilon rotors; fixed does not."""
    p, rows=probe
    audit=json.loads((p / 'runtime-initial.json').read_text())
    save(p / 'reference-free-runtime.json', audit)
    audit_binding=dict(path='reference-free-runtime.json',
                       sha256=hashlib.sha256((p / 'reference-free-runtime.json').read_bytes()).hexdigest())
    for name in ('declaration.json', 'runtime-initial.json', 'summary.json'):
        data=json.loads((p / name).read_text())
        data['fixture']='anchor'
        data['reference_mass_audit']=audit_binding
        if 'runtime' in data:
            runtime=data['runtime']
            runtime['fixture_original_mass_kg']=.027000000700354576
            runtime['links'][0]['mass_kg']=.027000000700354576
            for link in runtime['links'][1:5]:
                link['mass_kg']=0.
            runtime['n_qs']=8
            runtime['n_dofs']=6
            runtime['joints']=runtime['joints'][1:]
        save(p / name, data)
    for row in rows:
        row['qpos']=[0]*8
        row['dofs_velocity']=[0]*6
    finish(p, rows)
    return p


def test_v4_anchor_separates_fixed_fixture_mass_and_bound_free_reference(probe):
    p=anchor_v4(probe)
    result=checker().verify_probe(p)
    assert result['status'] == 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
    assert result['checks']['fixture_original_mass_kg'] == .027000000700354576
    assert result['checks']['reference_mass_kg'] == .02700047753751278
    assert not result['physical_training_admitted']


@pytest.mark.parametrize('mode', ['unbound', 'tampered', 'wrong_free_sum', 'fixed_audit', 'wrong_fixture_sum'])
def test_v4_anchor_rejects_unbound_or_mismatched_free_audit(probe, mode):
    p=anchor_v4(probe)
    if mode == 'unbound':
        for name in ('declaration.json', 'runtime-initial.json', 'summary.json'):
            data=json.loads((p / name).read_text())
            del data['reference_mass_audit']
            save(p / name, data)
    elif mode in ('tampered', 'wrong_free_sum', 'fixed_audit'):
        audit=json.loads((p / 'reference-free-runtime.json').read_text())
        if mode == 'fixed_audit':
            audit['fixture']='anchor'
        else:
            audit['runtime']['links'][0]['mass_kg']=.02
        save(p / 'reference-free-runtime.json', audit)
        if mode != 'tampered':
            actual=hashlib.sha256((p / 'reference-free-runtime.json').read_bytes()).hexdigest()
            for name in ('declaration.json', 'runtime-initial.json', 'summary.json'):
                data=json.loads((p / name).read_text())
                data['reference_mass_audit']['sha256']=actual
                save(p / name, data)
    else:
        data=json.loads((p / 'runtime-initial.json').read_text())
        data['runtime']['fixture_original_mass_kg']=.025
        save(p / 'runtime-initial.json', data)
    result=checker().verify_probe(p)
    assert result['status'] == 'QUALIFICATION_BLOCKED'
    assert 'REFERENCE_MASS_AUDIT_INVALID' in result['failure_codes']
