"""Conservative numerical reference must converge and reject unexplained loss."""

import numpy as np
import pytest
from adaptive_drone_patrol.payload_recovery_planar import PlanarModel


def test_reference_fourth_order_convergence_and_energy():
    from adaptive_drone_patrol.payload_solver_reference import integrate_reference

    m = PlanarModel(0.0001, 0.0054, 7.50025e-7, 1.8225e-6, 0.045)
    initial = [1.5, 1.5, 0.0, 0.0]
    trajectories = [integrate_reference(m, initial, duration_s=0.2, dt_s=h) for h in [0.002, 0.001, 0.0005]]
    errors = [np.linalg.norm(x[-1] - trajectories[-1][-1]) for x in trajectories[:-1]]
    assert errors[0] > 10 * errors[1]
    assert max(abs(m.energy(trajectories[-1]) - m.energy(initial))) / m.energy(initial) < 1e-6


def test_static_reference_is_exact_and_initial_data_unchanged():
    from adaptive_drone_patrol.payload_solver_reference import integrate_reference

    m = PlanarModel(0.0001, 0.0054, 7.50025e-7, 1.8225e-6, 0.045)
    initial = np.zeros(4)
    assert np.array_equal(integrate_reference(m, initial, duration_s=0.1, dt_s=0.001), np.zeros((101, 4)))
    assert np.array_equal(initial, np.zeros(4))


@pytest.mark.parametrize(
    "energies,work,expected",
    [
        ([1.0, 1.13], [0.0, 0.0], False),
        ([1.0, 0.87], [0.0, 0.0], False),
        ([1.0, 0.87], [0.0, -0.13], True),
        ([1.0, 1.04], [0.0, 0.0], True),
        ([1.0, 1.13], [0.0, -0.13], False),
    ],
)
def test_absolute_balance_rejects_loss_and_wrong_work_sign(energies, work, expected):
    from adaptive_drone_patrol.payload_solver_reference import energy_balance

    result = energy_balance(energies, applied_work_j=work, dissipative_work_j=[0.0, 0.0])
    assert result["passed"] is expected


@pytest.mark.parametrize(
    "dt,duration", [(0.0, 1.0), (0.003, 0.01), (float("nan"), 1.0), (0.01, -1.0), (True, 1.0)]
)
def test_reference_rejects_invalid_or_nonintegral_intervals(dt, duration):
    from adaptive_drone_patrol.payload_solver_reference import integrate_reference

    with pytest.raises(ValueError):
        integrate_reference(
            PlanarModel(0.0001, 0.0054, 7.50025e-7, 1.8225e-6, 0.045), [0.0] * 4, duration_s=duration, dt_s=dt
        )
