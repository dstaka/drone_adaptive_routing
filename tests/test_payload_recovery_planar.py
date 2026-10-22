"""Independent planar mechanics and fail-closed diagnostics; no engine import."""
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from adaptive_drone_patrol.payload_recovery_planar import PlanarModel, diagnose


def model():
    return PlanarModel(.0001, .0054, 7.50025e-7, 1.8225e-6, .045)


def record(substeps=2):
    return dict(fixture='anchor', case=dict(side_m=.045, sample_dt_s=.01, duration_s=.02,
                substeps=substeps, initial_swing_deg=30., initial_direction_rad=0., reset_during_run=False),
                settings=dict(control_dt_s=.01, gravity_m_s2=[0., 0., -9.81], rigid_dt_s=.01/substeps),
                runtime=dict(links=[dict(name='suspension_rod', mass_kg=.0001),
                                   dict(name='payload', mass_kg=.0054)],
                inertia_kg_m2=[np.diag([7.50025e-7, 7.50025e-7, 5e-11]).tolist(),
                               np.diag([1.8225e-6]*3).tolist()],
                links_com_body_m=[[0., 0., -.15], [0., 0., 0.]],
                links_inertial_quaternion_wxyz=[[1., 0., 0., 0.]]*2,
                links_frame_offset_quaternion_wxyz=[[1., 0., 0., 0.]]*2,
                damping_after=[0., 0., 0., 0., 0., 0.], armature=[0.]*6,
                actual_solver_settings=dict(dt='.01', substep_dt=str(.01/substeps),
                                            _substep_dt=str(.01/substeps))))


def row(state, i):
    t, p, u, v = state
    a = .0225
    return dict(step=i, time_s=i*.01, rpm=[0.]*4, contact=False,
                drone_xyz=[0., 0., 1.], drone_quaternion_wxyz=[1., 0., 0., 0.],
                drone_velocity=[0.]*3, drone_angular_velocity=[0.]*3,
                rod_xyz=[0., 0., .98], rod_velocity=[0.]*3,
                rod_quaternion_wxyz=[math.cos(t/2), 0., -math.sin(t/2), 0.],
                rod_angular_velocity=[0., -u, 0.],
                package_xyz=[.3*math.sin(t)+a*math.sin(p), 0., .98-.3*math.cos(t)-a*math.cos(p)],
                package_velocity=[.3*math.cos(t)*u+a*math.cos(p)*v, 0.,
                                  .3*math.sin(t)*u+a*math.sin(p)*v],
                package_quaternion_wxyz=[math.cos(p/2), 0., -math.sin(p/2), 0.],
                package_angular_velocity=[0., -v, 0.])


def probe(tmp_path):
    p = tmp_path/'probe'
    p.mkdir()
    states = [np.array([math.pi/6, math.pi/6, 0., 0.])]
    for _ in range(2):
        states.append(model().step(model().step(states[-1], .005), .005))
    (p/'runtime-initial.json').write_text(json.dumps(record()))
    (p/'samples.jsonl').write_text(''.join(json.dumps(row(s, i))+'\n' for i, s in enumerate(states)))
    return p


def test_conservative_equations_have_zero_instantaneous_energy_derivative():
    m = model()
    s = np.array([1.2, -.4, 2., -3.])
    direction = np.r_[s[2:], m.acceleration(s)]
    derivative = (m.energy(s+1e-7*direction)-m.energy(s-1e-7*direction))/2e-7
    assert derivative == pytest.approx(0., abs=1e-9)
    assert m.energy(np.zeros(4)) == 0.
    assert np.allclose(m.acceleration(np.zeros(4)), [0., 0.])


def test_general_cube_side_and_velocity_first_initial_step_are_physical():
    m = PlanarModel(.0001, .0027, 7.50025e-7, 1.62e-6, .06)
    s = np.array([math.pi/2, math.pi/2, 0., 0.])
    assert np.allclose(m.acceleration(s), [-33.1895, 2.9370], atol=.001)
    result = m.step(s, .005)
    assert np.allclose(result[:2], s[:2]+.005*result[2:])


def test_restart_each_observation_reproduces_local_map_and_energy_increment(tmp_path):
    result = diagnose(probe(tmp_path), substeps=2)
    assert result['physical_training_admitted'] is False
    assert result['status'] == 'CPU_PLANAR_DIAGNOSTIC_ONLY'
    assert result['sample_count'] == 3
    assert result['local_replay']['maximum_state_error'] < 1e-12
    assert result['local_replay']['maximum_energy_increment_error_j'] < 1e-15
    assert result['local_replay']['observed_total_energy_change_j'] == pytest.approx(
        result['local_replay']['predicted_summed_local_energy_change_j'], abs=1e-15)
    assert len(result['mechanical_energy_j']) == 3
    assert result['kinematic_closure']['maximum_cube_position_error_m'] < 1e-12
    assert len(result['inputs']) == 2
    assert result['source_sha256']


@pytest.mark.parametrize('mutation', ['nonplanar_q', 'nonplanar_v', 'moving_anchor', 'rpm', 'contact',
                                    'time', 'nonfinite', 'initial', 'inertia', 'com', 'frame', 'damping'])
def test_invalid_scope_or_measured_inputs_are_rejected_before_replay(tmp_path, mutation):
    p = probe(tmp_path)
    info = json.loads((p/'runtime-initial.json').read_text())
    rows = [json.loads(line) for line in (p/'samples.jsonl').read_text().splitlines()]
    if mutation == 'nonplanar_q':
        rows[1]['rod_quaternion_wxyz'] = [math.cos(.1), math.sin(.1), 0., 0.]
    elif mutation == 'nonplanar_v':
        rows[1]['package_velocity'][1] = .01
    elif mutation == 'moving_anchor':
        rows[1]['drone_xyz'][0] = .01
    elif mutation == 'rpm':
        rows[1]['rpm'][0] = 1.
    elif mutation == 'contact':
        rows[1]['contact'] = True
    elif mutation == 'time':
        rows[1]['time_s'] = .0101
    elif mutation == 'nonfinite':
        rows[1]['package_quaternion_wxyz'][0] = math.nan
    elif mutation == 'initial':
        info['case']['initial_swing_deg'] = 90.
    elif mutation == 'inertia':
        info['runtime']['inertia_kg_m2'][0][1][1] = -1.
    elif mutation == 'com':
        info['runtime']['links_com_body_m'][0][0] = .01
    elif mutation == 'frame':
        info['runtime']['links_inertial_quaternion_wxyz'][0] = [0., 1., 0., 0.]
    else:
        info['runtime']['damping_after'][0] = .1
    (p/'runtime-initial.json').write_text(json.dumps(info))
    (p/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):
        diagnose(p, substeps=2)


@pytest.mark.parametrize('bad', [True, 2., 0, -1, '2', None])
def test_substep_count_is_explicit_positive_integer(tmp_path, bad):
    with pytest.raises(ValueError):
        diagnose(probe(tmp_path), substeps=bad)


def test_declared_and_actual_dt_must_match_explicit_replay_substeps(tmp_path):
    p = probe(tmp_path)
    with pytest.raises(ValueError):
        diagnose(p, substeps=4)
    data = json.loads((p/'runtime-initial.json').read_text())
    data['runtime']['actual_solver_settings']['substep_dt'] = '.0025'
    (p/'runtime-initial.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        diagnose(p, substeps=2)


@pytest.mark.parametrize('state', [[0., 0., 0.], [0., math.nan, 0., 0.], [True]*4, ['0']*4])
def test_model_rejects_invalid_shapes_or_nonfinite_states(state):
    with pytest.raises(ValueError):
        model().energy(state)


def test_cli_writes_unique_receipt_and_refuses_to_overwrite(tmp_path):
    p = probe(tmp_path)
    out = tmp_path/'receipt.json'
    command = [sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/diagnose_payload_planar.py'),
               '--probe', str(p), '--out', str(out), '--substeps', '2']
    assert subprocess.run(command, capture_output=True).returncode == 0
    first = out.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert out.read_bytes() == first


@pytest.mark.parametrize('field,value', [('motor_power_w', .001), ('motor_work_j', .001),
    ('rotor_force_world_n', [[0., 0., .001]]*4), ('rotor_torque_world_nm', [[.001, 0., 0.]]*4)])
def test_recorded_nonzero_force_or_work_inputs_are_rejected(tmp_path, field, value):
    p = probe(tmp_path)
    rows = [json.loads(line) for line in (p/'samples.jsonl').read_text().splitlines()]
    rows[1][field] = value
    (p/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):
        diagnose(p, substeps=2)


def test_fractional_terminal_interval_cannot_be_silently_rounded(tmp_path):
    p = probe(tmp_path)
    data = json.loads((p/'runtime-initial.json').read_text())
    data['case']['duration_s'] = .024
    (p/'runtime-initial.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        diagnose(p, substeps=2)


@pytest.mark.parametrize('bad', [True, '0.1', math.nan, math.inf, -1., 0., 10**1000])
def test_mechanical_parameters_reject_coercion_nonfinite_and_unrepresentable_values(bad):
    with pytest.raises(ValueError):
        PlanarModel(bad, .0054, 7.50025e-7, 1.8225e-6, .045)


def test_nonfinite_derived_mechanics_are_rejected():
    with pytest.raises(ValueError):
        PlanarModel(1e308, 1e308, 7.50025e-7, 1.8225e-6, .045)
    with pytest.raises(ValueError):
        model().energy([1., 1., 1e308, 1e308])
