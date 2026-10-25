"""Privileged deterministic probe control through four retained rotor wrenches.

This is diagnostic excitation only, never a learned or acceptance controller.
It reads vehicle pose/velocity and total mass, not payload swing. No damping,
pose correction or direct payload actuation is applied by this controller.
"""

import numpy as np
from .payload_contract import rotation

HOVER_RPM = 14468.429183500699
KF = 3.16e-10
KM = 7.94e-12
ROTOR_XY = np.array([[0.028, -0.028], [-0.028, -0.028], [-0.028, 0.028], [0.028, 0.028]])
SPIN = np.array([-1, 1, -1, 1])
ALLOCATION = np.array([[KF] * 4, ROTOR_XY[:, 1] * KF, -ROTOR_XY[:, 0] * KF, KM * SPIN])


def wrench_from_rpm(rpm):
    """Return local collective thrust and roll/pitch/yaw torque at rotor COMs."""
    rpm = np.asarray(rpm, dtype=float)
    if rpm.shape != (4,) or not np.isfinite(rpm).all() or np.any(rpm < 0):
        raise ValueError("Expected four finite nonnegative rotor RPMs")
    return ALLOCATION @ (rpm * rpm)


def diagnostic_rpm(position, quaternion, velocity, angular_velocity, target, total_mass):
    """Geometric vehicle PD; all gains and bounds are fixed before probes."""
    parts = [np.asarray(v, dtype=float) for v in (position, quaternion, velocity, angular_velocity, target)]
    if [p.shape for p in parts] != [(3,), (4,), (3,), (3,), (3,)] or not all(
        np.isfinite(p).all() for p in parts
    ):
        raise ValueError("Nonfinite or invalid diagnostic input")
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Expected physical system mass")
    position, quaternion, velocity, angular_velocity, target = parts
    R = rotation(quaternion)
    acceleration = 1.2 * (target - position) - 2.0 * velocity + np.array([0, 0, 9.81])
    acceleration[0:2] = np.clip(acceleration[0:2], -5.0, 5.0)
    acceleration[2] = np.clip(acceleration[2], 4.0, 15.0)
    z = acceleration / np.linalg.norm(acceleration)
    y = np.cross(z, [1, 0, 0])
    y /= np.linalg.norm(y)
    desired = np.column_stack([np.cross(y, z), y, z])
    skew = 0.5 * (desired.T @ R - R.T @ desired)
    attitude = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    omega = R.T @ angular_velocity
    inertia = np.array([1.4e-5, 1.4e-5, 2.17e-5])
    torque = -100 * inertia * attitude - 20 * inertia * omega + np.cross(omega, inertia * omega)
    thrust = max(0.0, total_mass * float(acceleration @ R[:, 2]))
    squared = np.linalg.solve(ALLOCATION, np.r_[thrust, torque])
    return np.sqrt(np.clip(squared, (0.2 * HOVER_RPM) ** 2, (1.8 * HOVER_RPM) ** 2))
