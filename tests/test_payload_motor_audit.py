"""CPU audit fixtures exercise rejection logic; none are physical measurements."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / 'scripts/check_payload_motor_audit.py'
SOURCES = ('scripts/probe_payload_ball.py', 'src/adaptive_drone_patrol/payload_ball_asset.py',
           'src/adaptive_drone_patrol/payload_qualification.py', 'src/adaptive_drone_patrol/payload_contract.py')
CONTROLLER = 'src/adaptive_drone_patrol/payload_probe_controller.py'
MANIFEST = 'configs/payload_retry_probe_manifest.json'
PLAN = 'configs/payload_retry_motor_audit.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=True) + '\n')


def checker():
    assert SCRIPT.exists(), 'independent motor checker has not been implemented'
    spec = importlib.util.spec_from_file_location('motor_checker', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def finish(probe, rows):
    sample = probe / 'samples.jsonl'
    raw = probe / 'raw-states.jsonl'
    sample.write_text(''.join(json.dumps(r, allow_nan=True)+'\n' for r in rows))
    # The producer writes pre-derived state before adding wrenches and new work.
    raw.write_text(''.join(json.dumps({k: v for k, v in r.items()
                                     if k in ('step', 'time_s', 'rpm', 'qpos', 'dofs_velocity')
                                     or k.startswith(('drone_', 'package_', 'rod_'))}, allow_nan=True)+'\n'
                           for r in rows))
    summary = json.loads((probe / 'summary.json').read_text())
    summary.update(samples_sha256=digest(sample), raw_state_sha256=digest(raw))
    save(probe / 'summary.json', summary)


@pytest.fixture
def audit(tmp_path):
    source, probes = tmp_path / 'snapshot', tmp_path / 'probes'
    for name in (*SOURCES, CONTROLLER, MANIFEST, PLAN):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    plan = json.loads((source / PLAN).read_text())
    manifest = json.loads((source / MANIFEST).read_text())
    offsets = np.array([[0., 0., 0.], *plan['rotor_com_body_m']])
    masses = [.027000000700354576, *([1.1920928955078125e-7]*4)]
    inertia = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
    for fixture in ('original', 'unloaded', 'loaded'):
        for test in ('hover', 'rotor0', 'rotor1', 'rotor2', 'rotor3'):
            p = probes / f'motor-{fixture}-{test}-v4'
            p.mkdir(parents=True)
            loaded = fixture == 'loaded'
            links = [dict(name=name, idx=i+1, mass_kg=masses[i]) for i, name in enumerate(
                ['base_link', *[f'prop{j}_link' for j in range(4)]])]
            if loaded:
                links += [dict(name='suspension_rod', idx=6, mass_kg=.0001),
                          dict(name='payload', idx=7, mass_kg=sum(masses)*.2)]
            runtime = dict(reference_mass_kg=sum(masses), fixture_original_mass_kg=sum(masses), links=links,
                           inertia_kg_m2=[inertia.tolist()]+[np.diag([1e-8]*3).tolist()]*4,
                           links_com_body_m=[[0, 0, 0]]*5,
                           initial_entity_pos=[0, 0, 1], initial_base_pos=[0, 0, 1],
                           initial_links_pos=(offsets+np.array([0, 0, 1])).tolist(),
                           initial_links_quat=[[1, 0, 0, 0]]*5,
                           rotor_indices=[2, 3, 4, 5], rotor_spin=plan['spin'], kf=plan['kf'], km=plan['km'],
                           n_qs=15 if loaded else 7, n_dofs=12 if loaded else 6,
                           rpm_mapping='(1+0.8*action)*14468.429183500699')
            if loaded:
                runtime['inertia_kg_m2'] += [inertia.tolist(), inertia.tolist()]
                runtime['links_com_body_m'] += [[0, 0, -.15], [0, 0, 0]]
                runtime['initial_links_pos'] += [[0, 0, .98], [0, 0, .6575]]
                runtime['initial_links_quat'] += [[1, 0, 0, 0]]*2
            runtime['initial_links_internal_quaternion_wxyz'] = copy.deepcopy(runtime['initial_links_quat'])
            runtime['links_inertial_quaternion_wxyz'] = [[1, 0, 0, 0] for _ in links]
            excitation = None if test == 'hover' else int(test[-1])
            common = dict(schema='payload-ball-probe-v2', fixture=fixture, case=None,
                          settings=manifest['settings'], physical_training_admitted=False,
                          manifest_sha256=digest(source / MANIFEST), motor_plan_sha256=digest(source / PLAN),
                          source_binding={name: digest(source / name) for name in SOURCES},
                          controller_source_sha256=digest(source / CONTROLLER), rotor_excitation=excitation)
            save(p / 'declaration.json', dict(common, status='STARTED', steps=0))
            measured = dict(common, original_asset_sha256=plan['original_asset_sha256'], runtime=runtime,
                            engine_source={'rigid/solver.py': 'a'*64})
            if fixture != 'original':
                (p / 'candidate.xml').write_text('<mujoco/>\n')
                measured['asset'] = dict(original_asset_sha256=plan['original_asset_sha256'],
                                         generated_asset_sha256=digest(p / 'candidate.xml'),
                                         loaded=loaded, fixed=False)
            save(p / 'runtime-initial.json', measured)
            n = 100 if excitation is None else 5
            save(p / 'summary.json', dict(measured, status='CONSTRUCTION_RECORDED', steps=n,
                                         measured_duration_s=n*.01, failures=[],
                                         policy_admitted=False, operational_qualification='NOT_RUN'))
            rows, xyz, velocity, omega = [], np.array([0., 0., 1.]), np.zeros(3), np.zeros(3)
            total = sum(link['mass_kg'] for link in links)
            previous_rpm = np.zeros(4)
            work = 0.
            for i in range(n+1):
                # A hand fixture with real nonzero reaction signs and paired differences.
                q = [1, 0, 0, 0]
                relative = offsets[1:]
                com = xyz+relative
                rotor_velocity = velocity+np.cross(omega, relative)
                forces = previous_rpm[:, None]**2*plan['kf']*np.array([0, 0, 1])
                torques = previous_rpm[:, None]**2*plan['km']*np.array(plan['spin'])[:, None]*np.array([0, 0, 1])
                power = float(np.sum(forces*rotor_velocity)+np.sum(torques*omega))
                work += power*.01 if i else 0
                row = dict(step=i, time_s=i*.01, rpm=previous_rpm.tolist(), drone_xyz=xyz.tolist(),
                           drone_quaternion_wxyz=q, drone_internal_quaternion_wxyz=q,
                           drone_velocity=velocity.tolist(),
                           drone_angular_velocity=omega.tolist(),
                           qpos=[*xyz, *q]+([1, 0, 0, 0]*2 if loaded else []),
                           dofs_velocity=[*velocity, *omega]+([0]*6 if loaded else []),
                           rotor_com_world=com.tolist(), rotor_com_velocity=rotor_velocity.tolist(),
                           rotor_angular_velocity=[omega.tolist()]*4,
                           rotor_force_world_n=forces.tolist(), rotor_torque_world_nm=torques.tolist(),
                           motor_power_w=power, motor_work_j=work, contact=False,
                           drone_floor_m=float(xyz[2]-.0125), roll_deg=0., pitch_deg=0.)
                if loaded:
                    row.update(package_xyz=(xyz+np.array([.005*i, 0, -.3425])).tolist(),
                               package_quaternion_wxyz=q, package_velocity=(velocity+[.5 if i else 0, 0, 0]).tolist(),
                               package_angular_velocity=[0, 0, 0], rod_xyz=(xyz+[0, 0, -.02]).tolist(),
                               rod_quaternion_wxyz=q, rod_velocity=velocity.tolist(), rod_angular_velocity=omega.tolist())
                rows.append(row)
                if i == n:
                    break
                if excitation is None:
                    previous_rpm = np.full(4, plan['single_rotor_rpm']*np.sqrt(total/sum(masses)))
                    continue
                previous_rpm = np.zeros(4)
                previous_rpm[excitation] = plan['single_rotor_rpm']
                f = previous_rpm[:, None]**2*plan['kf']*np.array([0, 0, 1])
                tau = np.cross(relative, f).sum(axis=0)
                tau[2] += sum(previous_rpm**2*plan['km']*np.array(plan['spin']))
                velocity += (f.sum(axis=0)/total+np.array([0, 0, -9.81]))*.01
                xyz += velocity*.01
                omega += np.linalg.solve(inertia, tau)*.01
            finish(p, rows)
    return probes, source


def test_all_fifteen_checked_without_operational_admission(audit):
    probes, source = audit
    result = checker().verify_motor_audit(probes, source)
    assert result['motor_audit_passed'], result
    assert result['status'] == 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
    assert not result['physical_training_admitted']
    assert result['operational_qualification'] == 'NOT_RUN'
    assert len(result['traces']) == 15
    assert result['traces']['motor-original-hover-v4']['samples'] == 101
    assert result['traces']['motor-loaded-rotor3-v4']['samples'] == 6
    assert result['coupling']['rotor0']['maximum_root_velocity_delta_m_s'] > 1e-8
    assert str(source) not in json.dumps(result)


def edit_rows(probes, name, mutate):
    p = probes / name
    rows = [json.loads(line) for line in (p / 'samples.jsonl').read_text().splitlines()]
    mutate(rows)
    finish(p, rows)


@pytest.mark.parametrize('mutation,code', [
    ('omit_terminal', 'INCOMPLETE_TRACE'), ('time', 'SAMPLE_CONTINUITY'),
    ('rpm', 'MOTOR_INPUT'), ('force', 'MOTOR_WRENCH'), ('torque', 'MOTOR_WRENCH'),
    ('com', 'ROTOR_COM_FRAME'), ('nonfinite', 'NONFINITE_STATE'),
])
def test_rejects_rehashed_broken_trace(audit, mutation, code):
    probes, source = audit
    def mutate(rows):
        if mutation == 'omit_terminal':
            rows.pop()
        elif mutation == 'time':
            rows[3]['time_s'] = .02
        elif mutation == 'rpm':
            rows[0]['rpm'][0] = 1.
        elif mutation == 'force':
            rows[3]['rotor_force_world_n'][0][0] = .1
        elif mutation == 'torque':
            rows[3]['rotor_torque_world_nm'][0][2] *= -1
        elif mutation == 'com':
            rows[3]['rotor_com_world'][0] = rows[3]['drone_xyz']
        else:
            rows[3]['drone_velocity'][0] = float('nan')
    edit_rows(probes, 'motor-unloaded-rotor0-v4', mutate)
    assert code in checker().verify_motor_audit(probes, source)['failure_codes']


@pytest.mark.parametrize('field,value,code', [
    ('rotor_indices', [1, 2, 3, 4], 'ROTOR_IDENTITY'),
    ('rotor_spin', [1, 1, -1, 1], 'ROTOR_IDENTITY'),
    ('kf', 1., 'MOTOR_CONSTANTS'),
    ('initial_entity_pos', [0, 0, .9], 'ROOT_SEMANTICS'),
])
def test_runtime_motor_and_root_contracts_cannot_be_relabelled(audit, field, value, code):
    probes, source = audit
    p = probes / 'motor-unloaded-hover-v4'
    for name in ('runtime-initial.json', 'summary.json'):
        data = json.loads((p / name).read_text())
        data['runtime'][field] = value
        save(p / name, data)
    assert code in checker().verify_motor_audit(probes, source)['failure_codes']


@pytest.mark.parametrize('field', ['mass_kg', 'inertia', 'com'])
def test_retained_original_vehicle_properties_are_compared(audit, field):
    probes, source = audit
    p = probes / 'motor-loaded-rotor1-v4'
    for name in ('runtime-initial.json', 'summary.json'):
        data = json.loads((p / name).read_text())
        if field == 'mass_kg':
            data['runtime']['links'][0]['mass_kg'] += .001
        elif field == 'inertia':
            data['runtime']['inertia_kg_m2'][0][0][0] *= 2
        else:
            data['runtime']['links_com_body_m'][0][0] += .01
        save(p / name, data)
    assert 'VEHICLE_PROPERTY_MISMATCH' in checker().verify_motor_audit(probes, source)['failure_codes']


def test_same_timestamp_equivalence_is_measured_not_summary_claimed(audit):
    probes, source = audit
    def change(rows):
        rows[3]['drone_velocity'][0] += .01
        for v in rows[3]['rotor_com_velocity']:
            v[0] += .01
    edit_rows(probes, 'motor-unloaded-rotor2-v4', change)
    assert 'UNLOADED_EQUIVALENCE' in checker().verify_motor_audit(probes, source)['failure_codes']


def test_actual_moment_response_is_required_beyond_reported_wrench(audit):
    probes, source = audit
    def remove_roll_reaction(rows):
        for row in rows:
            row['drone_angular_velocity'][0] = 0.
            for w in row['rotor_angular_velocity']:
                w[0] = 0.
            v = np.array(row['drone_velocity'])
            offsets = np.array(row['rotor_com_world'])-row['drone_xyz']
            row['rotor_com_velocity'] = (v+np.cross(row['drone_angular_velocity'], offsets)).tolist()
    for fixture in ('original', 'unloaded'):
        edit_rows(probes, f'motor-{fixture}-rotor0-v4', remove_roll_reaction)
    assert 'MOTOR_REACTION' in checker().verify_motor_audit(probes, source)['failure_codes']


def test_identical_loaded_motion_does_not_prove_coupling(audit):
    probes, source = audit
    unloaded = [json.loads(line) for line in (probes/'motor-unloaded-rotor0-v4/samples.jsonl').read_text().splitlines()]
    def change(rows):
        for loaded, original in zip(rows, unloaded):
            for key in original:
                if key not in ('qpos', 'dofs_velocity'):
                    loaded[key] = copy.deepcopy(original[key])
            loaded['qpos'][:7] = original['qpos']
    edit_rows(probes, 'motor-loaded-rotor0-v4', change)
    assert 'COUPLING_UNRESOLVED' in checker().verify_motor_audit(probes, source)['failure_codes']


@pytest.mark.parametrize('mutation', ['missing_probe', 'samples_hash', 'raw_hash', 'raw_mismatch',
                                      'source', 'controller', 'plan_binding', 'asset'])
def test_missing_or_tampered_evidence_blocks_audit(audit, mutation):
    probes, source = audit
    p = probes / 'motor-loaded-hover-v4'
    if mutation == 'missing_probe':
        shutil.rmtree(p)
    elif mutation in ('samples_hash', 'raw_hash'):
        name = 'samples.jsonl' if mutation == 'samples_hash' else 'raw-states.jsonl'
        with (p/name).open('a') as stream:
            stream.write('{}\n')
    elif mutation == 'raw_mismatch':
        raw = p/'raw-states.jsonl'
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        rows[3]['drone_xyz'][0] = .5
        raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        summary = json.loads((p/'summary.json').read_text())
        summary['raw_state_sha256'] = digest(raw)
        save(p/'summary.json', summary)
    elif mutation in ('source', 'controller'):
        name = SOURCES[0] if mutation == 'source' else CONTROLLER
        (source/name).write_text('changed after measured execution\n')
    elif mutation == 'asset':
        (p/'candidate.xml').write_text('<tampered/>\n')
    else:
        data = json.loads((p/'declaration.json').read_text())
        data['motor_plan_sha256'] = '0'*64
        save(p/'declaration.json', data)
    result = checker().verify_motor_audit(probes, source)
    assert not result['motor_audit_passed']
    assert result['status'] == 'QUALIFICATION_BLOCKED'
    assert result['failure_codes']
    assert not result['physical_training_admitted']


def test_cli_writes_sanitized_receipt_and_refuses_overwrite(audit, tmp_path):
    probes, source = audit
    out = tmp_path/'audit.json'
    command = [sys.executable, str(SCRIPT), '--probes', str(probes), '--source-root', str(source), '--out', str(out)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert str(tmp_path) not in out.read_text()
    before = out.read_bytes()
    assert subprocess.run(command, capture_output=True, timeout=30).returncode != 0
    assert out.read_bytes() == before


@pytest.mark.parametrize('field', ['time_s', 'motor_power_w', 'drone_floor_m'])
def test_nonfinite_scalars_write_a_blocking_receipt(audit, tmp_path, field):
    probes, source = audit
    def change(rows):
        rows[3][field] = float('nan')
    edit_rows(probes, 'motor-loaded-hover-v4', change)
    out = tmp_path/'nonfinite-receipt.json'
    result = subprocess.run([sys.executable, str(SCRIPT), '--probes', str(probes),
                             '--source-root', str(source), '--out', str(out)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stderr
    receipt = json.loads(out.read_text())
    assert 'NONFINITE_STATE' in receipt['failure_codes']
    assert not receipt['motor_audit_passed']


def test_snapshot_missing_files_blocks_instead_of_using_current_sources(audit, tmp_path):
    probes, _ = audit
    result = checker().verify_motor_audit(probes, tmp_path/'missing-snapshot')
    assert not result['motor_audit_passed']
    assert result['failure_codes'] == ['SOURCE_OR_DECLARATION_INVALID']


def test_changed_plan_cannot_relax_an_equivalence_ceiling(audit):
    probes, source = audit
    plan = json.loads((source/PLAN).read_text())
    plan['unloaded_equivalence_position_max_m'] = 1.
    save(source/PLAN, plan)
    result = checker().verify_motor_audit(probes, source)
    assert result['failure_codes'] == ['SOURCE_OR_DECLARATION_INVALID']


def test_opposite_quaternion_sign_is_same_measured_orientation(audit):
    probes, source = audit
    def change(rows):
        for row in rows:
            row['drone_quaternion_wxyz'] = [-x for x in row['drone_quaternion_wxyz']]
            row['qpos'][3:7] = row['drone_quaternion_wxyz']
    edit_rows(probes, 'motor-unloaded-rotor3-v4', change)
    assert checker().verify_motor_audit(probes, source)['motor_audit_passed']


@pytest.mark.parametrize('frame', ['internal', 'inertial'])
def test_rotated_runtime_basis_preserves_same_authored_inertia(audit, frame):
    # A ninety-degree y basis rotation swaps x/z tensor coordinates, not physics.
    probes, source = audit
    q = [2**-.5, 0, 2**-.5, 0]
    for test in ('hover', 'rotor0', 'rotor1', 'rotor2', 'rotor3'):
        p = probes/f'motor-loaded-{test}-v4'
        for name in ('runtime-initial.json', 'summary.json'):
            data = json.loads((p/name).read_text())
            runtime = data['runtime']
            runtime['inertia_kg_m2'][0] = np.diag([2.17e-5, 1.4e-5, 1.4e-5]).tolist()
            key = ('initial_links_internal_quaternion_wxyz' if frame == 'internal'
                   else 'links_inertial_quaternion_wxyz')
            runtime[key][0] = q
            runtime['root_aligned'] = False
            save(p/name, data)
        if frame == 'internal':
            def change(rows):
                for row in rows:
                    row['drone_internal_quaternion_wxyz'] = q
                    row['qpos'][3:7] = q
            edit_rows(probes, p.name, change)
    result = checker().verify_motor_audit(probes, source)
    assert result['motor_audit_passed'], result['failure_codes']
    tensor = result['traces']['motor-loaded-hover-v4']['vehicle_inertia_authored_kg_m2']['base_link']
    assert np.allclose(tensor, np.diag([1.4e-5, 1.4e-5, 2.17e-5]), atol=1e-18, rtol=0)


def test_wrong_physical_axis_fails_even_when_inertia_eigenvalues_match(audit):
    # Sorting eigenvalues would incorrectly accept a changed yaw/roll response.
    probes, source = audit
    p = probes/'motor-unloaded-hover-v4'
    for name in ('runtime-initial.json', 'summary.json'):
        data = json.loads((p/name).read_text())
        data['runtime']['inertia_kg_m2'][0] = np.diag([2.17e-5, 1.4e-5, 1.4e-5]).tolist()
        save(p/name, data)
    assert 'VEHICLE_PROPERTY_MISMATCH' in checker().verify_motor_audit(probes, source)['failure_codes']


@pytest.mark.parametrize('field', ['initial_links_internal_quaternion_wxyz', 'links_inertial_quaternion_wxyz'])
def test_old_runtime_without_inertia_frames_stays_blocked(audit, field):
    probes, source = audit
    p = probes/'motor-original-hover-v4'
    for name in ('runtime-initial.json', 'summary.json'):
        data = json.loads((p/name).read_text())
        del data['runtime'][field]
        save(p/name, data)
    result = checker().verify_motor_audit(probes, source)
    assert not result['motor_audit_passed']
    assert 'INERTIAL_FRAME_UNVERIFIED' in result['failure_codes']


@pytest.mark.parametrize('mode', ['valid', 'tampered', 'missing', 'one_trace_missing'])
def test_backend_source_hash_is_recomputed_when_present(audit, mode):
    probes, source = audit
    relative = 'src/adaptive_drone_patrol/payload_ball_bank.py'
    backend = source/relative
    backend.write_text('# controlled snapshot backend\n')
    expected = digest(backend)
    for p in probes.iterdir():
        for name in ('declaration.json', 'runtime-initial.json', 'summary.json'):
            data = json.loads((p/name).read_text())
            data['backend_source_sha256'] = expected
            save(p/name, data)
    if mode == 'tampered':
        backend.write_text('# changed backend\n')
    elif mode == 'missing':
        backend.unlink()
    elif mode == 'one_trace_missing':
        p = probes/'motor-loaded-hover-v4'
        for name in ('declaration.json', 'runtime-initial.json', 'summary.json'):
            data = json.loads((p/name).read_text())
            del data['backend_source_sha256']
            save(p/name, data)
    result = checker().verify_motor_audit(probes, source)
    if mode == 'valid':
        assert result['motor_audit_passed']
        assert result['backend_source_sha256'] == expected
    else:
        assert not result['motor_audit_passed']
        assert 'BACKEND_BINDING_MISMATCH' in result['failure_codes']


@pytest.mark.parametrize('zero_inertia', [False, True])
def test_missing_inertial_orientation_only_irrelevant_for_exact_zero_tensor(audit, zero_inertia):
    probes, source = audit
    for p in probes.iterdir():
        for name in ('runtime-initial.json', 'summary.json'):
            data = json.loads((p/name).read_text())
            runtime = data['runtime']
            runtime['links_inertial_quaternion_wxyz'][1] = None
            if zero_inertia:
                runtime['inertia_kg_m2'][1] = [[0, 0, 0]]*3
            save(p/name, data)
    result = checker().verify_motor_audit(probes, source)
    if zero_inertia:
        assert result['motor_audit_passed']
        assert 'prop0_link' in result['traces']['motor-original-hover-v4']['zero_inertia_orientation_irrelevant']
    else:
        assert 'INERTIAL_FRAME_UNVERIFIED' in result['failure_codes']
        assert not result['motor_audit_passed']


def test_actual_solver_wrench_fields_are_checked():
    """The raw solver fields, including each arm moment, are independently tested."""
    module = checker()
    plan = json.loads((ROOT / PLAN).read_text())
    rpm = np.array([plan['single_rotor_rpm'], 0, 0, 0])
    xyz = np.array([0., 0., 1.])
    com = xyz + np.array(plan['rotor_com_body_m'])
    force = rpm[:, None]**2*plan['kf']*np.array([0, 0, 1])
    torque = np.cross(com-xyz, force)+rpm[:, None]**2*plan['km']*np.array(plan['spin'])[:, None]*[0, 0, 1]
    f = np.zeros((8, 1, 3))
    t = f.copy()
    root = f.copy()
    f[2:6, 0] = -force
    t[2:6, 0] = -torque
    root[2:6, 0] = xyz
    row = dict(step=0, time_s=0., drone_quaternion_wxyz=[1, 0, 0, 0], rotor_com_world=com.tolist())
    record = dict(step=0, time_s=0., rpm=rpm.tolist(), rotor_indices=[2, 3, 4, 5],
                  negative_external_force_field=f.tolist(),
                  negative_external_torque_about_root_com_field=t.tolist(), root_com_world_field=root.tolist())
    assert module.check_solver_wrench(record, row, plan) == []
    damaged = copy.deepcopy(record)
    damaged['negative_external_torque_about_root_com_field'][2][0][0] = 0.
    assert 'ACTUAL_SOLVER_WRENCH_MISMATCH' in module.check_solver_wrench(damaged, row, plan)
    damaged = copy.deepcopy(record)
    damaged['negative_external_force_field'][6][0][2] = -.01
    assert 'DIRECT_NONROTOR_WRENCH' in module.check_solver_wrench(damaged, row, plan)
    damaged = copy.deepcopy(record)
    damaged['step'] = 1
    assert 'MOTOR_INPUT_CONTINUITY' in module.check_solver_wrench(damaged, row, plan)
