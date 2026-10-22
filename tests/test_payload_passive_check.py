"""Physical invariants and negative passive evidence tests, independent of Genesis."""
import math
from pathlib import Path

import pytest


def api():
    path=Path(__file__).resolve().parents[1] / 'src/adaptive_drone_patrol/payload_passive_check.py'
    assert path.is_file(), 'independent passive checker must exist'
    from adaptive_drone_patrol import payload_passive_check
    return payload_passive_check


def runtime():
    return dict(links=[dict(name='suspension_rod', mass_kg=.0001), dict(name='payload', mass_kg=.0054)],
                inertia_kg_m2=[[[7.50025e-7, 0, 0], [0, 7.50025e-7, 0], [0, 0, 5e-11]],
                               [[1.8225e-6, 0, 0], [0, 1.8225e-6, 0], [0, 0, 1.8225e-6]]])


def static_row():
    return dict(step=0, time_s=0., drone_xyz=[0, 0, 1], drone_quaternion_wxyz=[1, 0, 0, 0],
                drone_velocity=[0, 0, 0], drone_angular_velocity=[0, 0, 0],
                rod_xyz=[0, 0, .98], rod_quaternion_wxyz=[1, 0, 0, 0],
                rod_velocity=[0, 0, 0], rod_angular_velocity=[0, 0, 0],
                package_xyz=[0, 0, .6575], package_quaternion_wxyz=[1, 0, 0, 0],
                package_velocity=[0, 0, 0], package_angular_velocity=[0, 0, 0], rpm=[0]*4)


def oscillation(period=1.1393076709700611):
    # Hand-derived small-angle slow eigenvector ratio of the two-body pendulum.
    omega=2*math.pi/period
    ratio=1.0525263294130076
    rows=[]
    for i in range(12001):
        t=i*.01
        angle=.04*math.cos(omega*t)
        rate=-.04*omega*math.sin(omega*t)
        phi=ratio*angle
        phidot=ratio*rate
        row=static_row()
        row.update(step=i, time_s=t)
        row['rod_quaternion_wxyz']=[math.cos(angle/2), math.sin(angle/2), 0, 0]
        row['rod_angular_velocity']=[rate, 0, 0]
        row['package_quaternion_wxyz']=[math.cos(phi/2), math.sin(phi/2), 0, 0]
        row['package_angular_velocity']=[phidot, 0, 0]
        row['package_xyz']=[0, .3*math.sin(angle)+.0225*math.sin(phi),
                            .98-.3*math.cos(angle)-.0225*math.cos(phi)]
        row['package_velocity']=[0, .3*math.cos(angle)*rate+.0225*math.cos(phi)*phidot,
                                 .3*math.sin(angle)*rate+.0225*math.sin(phi)*phidot]
        rows.append(row)
    return rows


def case():
    return dict(kind='anchor', side_m=.045, mass_ratio=.2, duration_s=120., sample_dt_s=.01)


def test_lowest_static_hanging_energy_is_zero():
    assert api().mechanical_energy(static_row(), runtime(), .045, .98) == pytest.approx(0, abs=1e-17)


def test_rod_com_velocity_uses_omega_cross_offset_and_cube_rotates_independently():
    row=static_row()
    row['rod_angular_velocity']=[1, 0, 0]
    row['package_velocity']=[1, 2, 3]
    row['package_angular_velocity']=[0, 2, 0]
    assert api().mechanical_energy(row, runtime(), .045, .98) == pytest.approx(.0378051450125)


def test_elevating_anchor_does_not_change_reference_excitation_energy():
    row=static_row()
    for name in ('drone_xyz', 'rod_xyz', 'package_xyz'):
        row[name][2]+=2
    assert api().mechanical_energy(row, runtime(), .045, 2.98) == pytest.approx(0, abs=1e-16)


def test_two_body_period_includes_actual_rod_and_cube_inertia():
    assert api().analytical_period(runtime(), .045) == pytest.approx(1.1393076709700611, rel=1e-12)


def test_complete_passive_samples_have_crossing_period_and_no_training_admission():
    result=api().analyse_passive(oscillation(), runtime(), case())
    assert result['status'] == 'PASSIVE_SCREEN_PASSED'
    assert not result['physical_training_admitted']
    assert result['crossings'] > 90
    assert result['measured_period_s'] == pytest.approx(1.1393076709700611, abs=.001)
    assert result['positive_energy_fraction'] < .01


def test_positive_energy_growth_is_rejected():
    rows=oscillation()
    rows[100]['package_velocity']=[2, 0, 0]
    result=api().analyse_passive(rows, runtime(), case())
    assert 'PASSIVE_POSITIVE_ENERGY_GROWTH' in result['failure_codes']


def test_wrong_period_is_rejected_even_when_energy_never_increases():
    result=api().analyse_passive(oscillation(period=1.5), runtime(), case())
    assert 'PASSIVE_PERIOD_MISMATCH' in result['failure_codes']


def test_energy_dissipation_is_recorded_without_fail_condition():
    rows=oscillation()
    # Preserve the observed oscillation trajectory but reduce measured kinetic
    # energy, demonstrating that loss itself is not a checker rejection.
    for row in rows:
        for name in ('rod_angular_velocity', 'package_velocity', 'package_angular_velocity'):
            row[name]=[v*.8 for v in row[name]]
    result=api().analyse_passive(rows, runtime(), case())
    assert 'PASSIVE_POSITIVE_ENERGY_GROWTH' not in result['failure_codes']
    assert result['minimum_energy_fraction_of_initial'] < .8


@pytest.mark.parametrize('mode', ['terminal', 'initial', 'timing', 'nonfinite', 'anchor_motion'])
def test_incomplete_nonfinite_or_moving_anchor_evidence_fails(mode):
    rows=oscillation()
    if mode == 'terminal':
        rows=rows[:-1]
    elif mode == 'initial':
        rows=rows[1:]
    elif mode == 'timing':
        rows[100]['time_s']+=.001
    elif mode == 'nonfinite':
        rows[100]['package_velocity']=[float('nan'), 0, 0]
    else:
        rows[100]['drone_xyz'][2]+=.01
    assert api().analyse_passive(rows, runtime(), case())['status'] == 'QUALIFICATION_BLOCKED'


def test_missing_sources_and_probe_files_never_produce_pass(tmp_path):
    result=api().verify_passive_probe(tmp_path, source_root=tmp_path)
    assert result['status'] == 'QUALIFICATION_BLOCKED'
    assert not result['physical_training_admitted']
    assert 'SOURCE_OR_RAW_EVIDENCE_INVALID' in result['failure_codes']


def test_passive_cli_writes_missing_evidence_receipt_and_refuses_overwrite(tmp_path):
    import os
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    output=tmp_path / 'result.json'
    command=[sys.executable, str(root / 'scripts/check_payload_ball_passive.py'),
             '--probe', str(tmp_path / 'absent'), '--source-root', str(tmp_path), '--out', str(output)]
    environment=dict(os.environ, PYTHONPATH=str(root / 'src'))
    completed=subprocess.run(command, env=environment, capture_output=True, text=True)
    assert completed.returncode == 2, completed.stderr
    before=output.read_bytes()
    assert str(tmp_path).encode() not in before
    assert subprocess.run(command, env=environment, capture_output=True).returncode != 0
    assert output.read_bytes() == before
