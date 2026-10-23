"""Independent RK4 planar reference and conservative signed-work balance.

Work is cumulative from sample zero, positive into the mechanical system.
Dissipative work must be observed independently; a measured loss is not work.
This module cannot grant physical qualification or optimizer admission.
"""

import math
import numpy as np


def integrate_reference(model, initial, *, duration_s, dt_s):
    """Integrate the measured Lagrangian model with classical fourth-order RK."""
    for value in (duration_s, dt_s):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Positive finite duration and timestep required")
    steps = round(duration_s / dt_s)
    if steps < 1 or not math.isclose(steps * dt_s, duration_s, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Duration must contain complete integration intervals")
    state = np.asarray(initial, dtype=float).copy()
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("Finite planar initial state required")
    result = np.empty((steps + 1, 4))
    result[0] = state

    def derivative(s):
        return np.r_[s[2:], model.acceleration(s)]

    for i in range(steps):
        k1 = derivative(state)
        k2 = derivative(state + dt_s * k1 / 2)
        k3 = derivative(state + dt_s * k2 / 2)
        k4 = derivative(state + dt_s * k3)
        state = state + dt_s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite reference trajectory")
        result[i + 1] = state
    return result


def energy_balance(energy_j, *, applied_work_j, dissipative_work_j):
    """Check both positive growth and absolute unexplained drift against E(0)."""
    energy, applied, dissipative = [
        np.asarray(x, dtype=float) for x in (energy_j, applied_work_j, dissipative_work_j)
    ]
    if (
        energy.ndim != 1
        or len(energy) < 2
        or applied.shape != energy.shape
        or dissipative.shape != energy.shape
        or not all(np.isfinite(x).all() for x in (energy, applied, dissipative))
        or energy[0] <= 1e-12
        or applied[0] != 0
        or dissipative[0] != 0
        or np.any(dissipative > 0)
    ):
        raise ValueError("Finite aligned energies, signed cumulative work and positive excitation required")
    residual = energy - energy[0] - applied - dissipative
    absolute = float(np.max(np.abs(residual)) / energy[0])
    positive = float(max(0.0, np.max(energy - energy[0]) / energy[0]))
    return dict(
        schema="payload-solver-energy-balance-v1",
        initial_excitation_j=float(energy[0]),
        maximum_absolute_unexplained_fraction=absolute,
        positive_growth_fraction=positive,
        terminal_unexplained_fraction=float(residual[-1] / energy[0]),
        passed=bool(absolute <= 0.05 and positive <= 0.05),
        physical_training_admitted=False,
        work_convention="Cumulative work into the system is positive; reference is E(0).",
    )
