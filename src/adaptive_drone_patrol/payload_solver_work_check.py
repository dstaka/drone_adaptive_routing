"""Independent array reconstruction of saved native RK stages; never simulation."""

import numpy as np
from .payload_solver_work import FIELD_WIDTHS, WIDTH, ROTOR_COM, SPIN


def reconstruct_batch_work(array, *, substeps, first_step, total_mass_kg):
    """Validate every stage field and integrate positive-into-system work."""
    a = np.asarray(array, dtype=float)
    if (
        type(substeps) is not int
        or substeps <= 0
        or a.ndim != 3
        or a.shape[1:] != (4 * substeps, WIDTH)
        or not np.isfinite(a).all()
        or not np.isfinite(total_mass_kg)
        or total_mass_kg <= 0
    ):
        raise ValueError("Complete finite native stage batch required")
    fields = {}
    offset = 0
    for key, width in FIELD_WIDTHS:
        fields[key] = a[..., offset : offset + width]
        offset += width
    q = fields["qpos"]
    v = fields["qvel"]
    rpm = fields["rpm"]
    w, x, y, z = np.moveaxis(q[..., 3:7], -1, 0)
    if np.max(abs(w * w + x * x + y * y + z * z - 1)) > 1e-4:
        raise ValueError("Invalid native quaternion")
    R = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*a.shape[:2], 3, 3)
    if np.any(rpm < 0) or np.any(rpm > 1.8 * 14468.429183500699 + 1e-8):
        raise ValueError("Original motor authority exceeded")
    force = rpm[..., None] ** 2 * 3.16e-10 * R[..., None, :, 2]
    torque = rpm[..., None] ** 2 * 7.94e-12 * SPIN[:, None] * R[..., None, :, 2]
    point = q[..., :3][..., None, :] + np.einsum("...ij,kj->...ki", R, ROTOR_COM)
    for key, expected in [
        ("rotor_force_world_n", force),
        ("rotor_torque_world_nm", torque),
        ("rotor_com_world", point),
    ]:
        if not np.allclose(fields[key].reshape(expected.shape), expected, rtol=1e-10, atol=1e-10):
            raise ValueError("Original motor stage wrench/frame differs")
    moment = np.sum(torque + np.cross(point - q[..., :3][..., None, :], force), axis=-2)
    generalized = np.concatenate(
        [force.sum(axis=-2), np.einsum("...ji,...j->...i", R, moment), np.zeros((*a.shape[:2], 6))], axis=-1
    )
    if not np.allclose(fields["motor_generalized_force"], generalized, rtol=1e-10, atol=1e-10):
        raise ValueError("Unobserved/direct load force")
    power = np.sum(generalized * v, axis=-1)
    loss = -(total_mass_kg - 0.02700047753751278 + 0.027) * 1e-5 * np.sum(v[..., 3:6] ** 2, axis=-1)
    if not np.allclose(fields["motor_power_w"][..., 0], power, rtol=1e-10, atol=1e-10):
        raise ValueError("Motor stage power differs")
    if not np.allclose(fields["dissipative_power_w"][..., 0], loss, rtol=1e-10, atol=1e-10):
        raise ValueError("Dissipative stage power differs")
    h = 0.01 / substeps
    times = (np.arange(a.shape[0]) + first_step)[:, None] * 0.01 + (
        np.arange(4 * substeps) // 4 + np.tile([0, 0.5, 0.5, 1], substeps)
    ) * h
    if not np.allclose(fields["time_s"][..., 0], times, atol=1e-7, rtol=0):
        raise ValueError("Stage time/order differs")
    weights = np.tile([1.0, 2.0, 2.0, 1.0], substeps) * h / 6
    return dict(
        motor_work_j=power @ weights,
        dissipative_work_j=loss @ weights,
        endpoint_absolute_work_bound_j=0.01 * np.maximum(abs(power[:, 0]), abs(power[:, -1])),
        rpm=rpm[:, 0],
        rpm_all_stages=rpm,
    )
