"""Reconstruct actual RK stage motor power, signs and integration weights."""

import copy
import numpy as np
import pytest


def stages():
    rpm = np.array([15000.0] * 4)
    force = 3.16e-10 * rpm**2
    torque = 7.94e-12 * rpm**2 * np.array([-1, 1, -1, 1])
    points = np.array(
        [[0.028, -0.028, 1.0], [-0.028, -0.028, 1.0], [-0.028, 0.028, 1.0], [0.028, 0.028, 1.0]]
    )
    qvel = [0.0, 0.0, 0.2] + [0.0] * 9
    generalized = [0.0, 0.0, sum(force)] + [0.0] * 9
    result = []
    for t in [0.0, 0.005, 0.005, 0.01]:
        result.append(
            dict(
                time_s=t,
                qpos=[0.0, 0.0, 1.0] + [1.0, 0.0, 0.0, 0.0] * 3,
                qvel=qvel,
                rpm=rpm.tolist(),
                rotor_com_world=points.tolist(),
                rotor_force_world_n=np.c_[np.zeros(4), np.zeros(4), force].tolist(),
                rotor_torque_world_nm=np.c_[np.zeros(4), np.zeros(4), torque].tolist(),
                motor_generalized_force=generalized,
                motor_power_w=sum(force) * 0.2,
                dissipative_power_w=0.0,
            )
        )
    return result


def test_signed_stage_work_matches_known_constant_power():
    from adaptive_drone_patrol.payload_solver_work import reconstruct_interval_work

    rows = stages()
    result = reconstruct_interval_work(rows, substeps=1, start_time_s=0.0, total_mass_kg=0.0325)
    assert result["motor_work_j"] == pytest.approx(
        0.01 * sum(np.array(rows[0]["rotor_force_world_n"])[:, 2]) * 0.2
    )
    assert result["dissipative_work_j"] == 0


@pytest.mark.parametrize("mutation", ["power", "sign", "point", "time", "missing", "hidden_force", "damping"])
def test_incomplete_or_self_reported_work_cannot_pass(mutation):
    from adaptive_drone_patrol.payload_solver_work import reconstruct_interval_work

    rows = copy.deepcopy(stages())
    if mutation == "power":
        rows[1]["motor_power_w"] += 0.1
    elif mutation == "sign":
        rows[1]["rotor_torque_world_nm"][0][2] *= -1
    elif mutation == "point":
        rows[1]["rotor_com_world"][0][0] += 0.01
    elif mutation == "time":
        rows[1]["time_s"] += 0.001
    elif mutation == "missing":
        rows.pop()
    elif mutation == "hidden_force":
        rows[1]["motor_generalized_force"][8] = 0.1
    else:
        rows[1]["dissipative_power_w"] = 0.1
    with pytest.raises(ValueError):
        reconstruct_interval_work(rows, substeps=1, start_time_s=0.0, total_mass_kg=0.0325)


def test_vectorized_stage_reconstruction_matches_scalar_and_detects_corruption():
    from adaptive_drone_patrol.payload_solver_work import pack_stages, reconstruct_interval_work
    from adaptive_drone_patrol.payload_solver_work_check import reconstruct_batch_work

    rows = stages()
    array = pack_stages(rows)[None]
    expected = reconstruct_interval_work(rows, substeps=1, start_time_s=0.0, total_mass_kg=0.0325)
    result = reconstruct_batch_work(array, substeps=1, first_step=0, total_mass_kg=0.0325)
    assert result["motor_work_j"][0] == pytest.approx(expected["motor_work_j"])
    array[0, 1, 0] += 0.001
    with pytest.raises(ValueError):
        reconstruct_batch_work(array, substeps=1, first_step=0, total_mass_kg=0.0325)


def test_physical_damping_uses_authored_urdf_mass_not_solver_regularization():
    from adaptive_drone_patrol.payload_solver_work import pack_stages, reconstruct_interval_work
    from adaptive_drone_patrol.payload_solver_work_check import reconstruct_batch_work

    total = 0.02700047753751278 + 0.0001 + 0.02700047753751278 * 0.2
    authored = 0.027 + 0.0001 + 0.02700047753751278 * 0.2
    rows = stages()
    for row in rows:
        row["qvel"][3] = 100.0
        row["dissipative_power_w"] = -authored * 1e-5 * 100.0**2
    result = reconstruct_interval_work(rows, substeps=1, start_time_s=0, total_mass_kg=total)
    batch = reconstruct_batch_work(pack_stages(rows)[None], substeps=1, first_step=0, total_mass_kg=total)
    assert batch["dissipative_work_j"][0] == pytest.approx(result["dissipative_work_j"])
    assert result["dissipative_work_j"] == pytest.approx(-authored * 1e-3)
    for row in rows:
        row["dissipative_power_w"] = -total * 1e-5 * 100.0**2
    with pytest.raises(ValueError, match="damping work"):
        reconstruct_interval_work(rows, substeps=1, start_time_s=0, total_mass_kg=total)
    with pytest.raises(ValueError, match="Dissipative"):
        reconstruct_batch_work(pack_stages(rows)[None], substeps=1, first_step=0, total_mass_kg=total)
