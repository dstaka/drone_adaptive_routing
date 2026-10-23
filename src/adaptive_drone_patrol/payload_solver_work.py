"""Independent RK4 stage wrench/work reconstruction for the actual rotor model.

MuJoCo free-joint translations are world-frame and angular velocities are in the
root body frame. Motor inputs act only on the original vehicle; passive joints
receive no direct generalized motor force. Work is positive into the system.
"""

import math
import numpy as np

from .payload_contract import rotation
from .payload_passive_check import mechanical_energy

ROTOR_COM = np.array([[0.028, -0.028, 0.0], [-0.028, -0.028, 0.0], [-0.028, 0.028, 0.0], [0.028, 0.028, 0.0]])
SPIN = np.array([-1.0, 1.0, -1.0, 1.0])
FIELD_WIDTHS = [
    ("time_s", 1),
    ("qpos", 15),
    ("qvel", 12),
    ("motor_generalized_force", 12),
    ("motor_power_w", 1),
    ("dissipative_power_w", 1),
    ("rotor_force_world_n", 12),
    ("rotor_torque_world_nm", 12),
    ("rotor_com_world", 12),
    ("rpm", 4),
]
WIDTH = sum(width for _, width in FIELD_WIDTHS)


def pack_stages(rows):
    """Losslessly store observed float64 stage fields without JSON expansion."""
    return np.array(
        [
            np.concatenate([np.asarray(row[key], dtype=float).reshape(-1) for key, _ in FIELD_WIDTHS])
            for row in rows
        ],
        dtype=np.float64,
    )


def unpack_stages(array):
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] != WIDTH:
        raise ValueError("Incomplete stage field array")
    rows = []
    for vector in array:
        row = {}
        offset = 0
        for key, width in FIELD_WIDTHS:
            value = vector[offset : offset + width]
            offset += width
            if key in ("rotor_force_world_n", "rotor_torque_world_nm", "rotor_com_world"):
                value = value.reshape(4, 3)
            row[key] = float(value[0]) if width == 1 else value.tolist()
        rows.append(row)
    return rows


def reconstruct_interval_work(rows, *, substeps, start_time_s, total_mass_kg):
    """Require all actual stages, reconstruct rotor mechanics and signed work."""
    if type(substeps) is not int or substeps <= 0 or len(rows) != 4 * substeps:
        raise ValueError("Every four RK stages per actual substep are required")
    if not math.isfinite(total_mass_kg) or total_mass_kg <= 0:
        raise ValueError("Actual total mass required")
    h = 0.01 / substeps
    motor = 0.0
    dissipative = 0.0
    maximum_power = 0.0
    for i, row in enumerate(rows):
        fields = {key: np.asarray(row[key], dtype=float) for key, _ in FIELD_WIDTHS}
        if not all(np.isfinite(v).all() for v in fields.values()):
            raise ValueError("Nonfinite actual stage")
        if any(fields[key].size != width for key, width in FIELD_WIDTHS):
            raise ValueError("Actual stage field shape")
        qpos, qvel, rpm = fields["qpos"], fields["qvel"], fields["rpm"]
        if np.any(rpm < 0) or np.any(rpm > 1.8 * 14468.429183500699 + 1e-8):
            raise ValueError("Original motor authority exceeded")
        expected_t = start_time_s + (i // 4 + [0, 0.5, 0.5, 1][i % 4]) * h
        if not math.isclose(float(fields["time_s"]), expected_t, abs_tol=1e-7, rel_tol=0):
            raise ValueError("RK stage time/order differs")
        R = rotation(qpos[3:7])
        axis = R[:, 2]
        force = rpm[:, None] ** 2 * 3.16e-10 * axis
        torque = rpm[:, None] ** 2 * 7.94e-12 * SPIN[:, None] * axis
        point = qpos[:3] + ROTOR_COM @ R.T
        for key, expected in [
            ("rotor_force_world_n", force),
            ("rotor_torque_world_nm", torque),
            ("rotor_com_world", point),
        ]:
            if fields[key].shape != expected.shape or not np.allclose(
                fields[key], expected, rtol=1e-10, atol=1e-10
            ):
                raise ValueError("Actual original rotor COM wrench differs")
        generalized = np.zeros(12)
        generalized[:3] = force.sum(axis=0)
        generalized[3:6] = R.T @ (torque + np.cross(point - qpos[:3], force)).sum(axis=0)
        if not np.allclose(fields["motor_generalized_force"], generalized, rtol=1e-10, atol=1e-10):
            raise ValueError("Unobserved or direct load force")
        power = float(generalized @ qvel)
        loss = -(total_mass_kg - 0.02700047753751278 + 0.027) * 1e-5 * float(qvel[3:6] @ qvel[3:6])
        if not math.isclose(float(fields["motor_power_w"]), power, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError("Motor power sign/frame differs")
        if not math.isclose(float(fields["dissipative_power_w"]), loss, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError("Declared physical damping work differs")
        weight = h * [1, 2, 2, 1][i % 4] / 6
        motor += weight * power
        dissipative += weight * loss
        maximum_power = max(maximum_power, abs(power))
    return dict(
        motor_work_j=motor, dissipative_work_j=dissipative, maximum_absolute_stage_power_w=maximum_power
    )


def total_mechanical_energy(row, runtime, side_m):
    """Reconstruct every finite body COM's kinetic and gravitational energy."""
    energy = mechanical_energy(row, runtime, side_m, 0.0)
    d = np.asarray(row["drone_xyz"])
    v = np.asarray(row["drone_velocity"])
    omega = np.asarray(row["drone_angular_velocity"])
    R = rotation(row["drone_quaternion_wxyz"])
    for i in range(5):
        mass = runtime["links"][i]["mass_kg"]
        arm = R @ np.asarray(runtime["links_com_body_m"][i])
        com = d + arm
        velocity = v + np.cross(omega, arm)
        inertia = R @ np.asarray(runtime["inertia_kg_m2"][i]) @ R.T
        energy += (
            0.5 * mass * float(velocity @ velocity)
            + 0.5 * float(omega @ inertia @ omega)
            + mass * 9.81 * com[2]
        )
    if not math.isfinite(energy):
        raise ValueError("Nonfinite independently reconstructed mechanical energy")
    return float(energy)
