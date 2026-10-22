"""Analytical motor-only diagnostic control tests; not flight qualification."""

import numpy as np
import pytest


def test_hover_allocation_retains_rpm_mapping_and_weight():
    from adaptive_drone_patrol.payload_probe_controller import diagnostic_rpm, wrench_from_rpm

    rpm = diagnostic_rpm([0, 0, 1], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], 0.0325)
    wrench = wrench_from_rpm(rpm)
    assert wrench[0] == pytest.approx(0.0325 * 9.81)
    assert np.allclose(wrench[1:], [0, 0, 0], atol=1e-12)
    assert np.all(rpm >= 0.2 * 14468.429183500699)
    assert np.all(rpm <= 1.8 * 14468.429183500699)


@pytest.mark.parametrize(
    "rotor,signs", [(0, [-1, -1, -1]), (1, [-1, 1, 1]), (2, [1, 1, -1]), (3, [1, -1, 1])]
)
def test_rotor_com_application_has_all_three_moments(rotor, signs):
    from adaptive_drone_patrol.payload_probe_controller import wrench_from_rpm

    rpm = np.zeros(4)
    rpm[rotor] = 10000
    wrench = wrench_from_rpm(rpm)
    assert wrench[0] == pytest.approx(3.16e-10 * 1e8)
    assert np.sign(wrench[1:]).tolist() == signs


def test_diagnostic_control_is_bounded_and_rejects_invalid_input():
    from adaptive_drone_patrol.payload_probe_controller import diagnostic_rpm

    args = [[0, 0, 1], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0], [2.7, 1.8, 3], 0.0352]
    rpm = diagnostic_rpm(*args)
    assert np.isfinite(rpm).all() and max(rpm) <= 1.8 * 14468.429183500699
    args[0] = [float("nan"), 0, 1]
    with pytest.raises(ValueError):
        diagnostic_rpm(*args)
