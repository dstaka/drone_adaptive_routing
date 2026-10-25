import numpy as np
import pytest

from adaptive_drone_patrol.payload_hybrid_controller import Controller, measured_state, SENSOR_FIELDS


def sensors():
    s = {}
    for body in ('drone', 'rod', 'package'):
        s[body+'_quaternion_wxyz'] = [1., 0., 0., 0.]
        s[body+'_angular_velocity'] = [0., 0., 0.]
    s['drone_xyz'] = [0., 0., 1.]
    s['drone_velocity'] = [0., 0., 0.]
    return s


def test_measurement_boundary_ignores_privileged_parameters():
    s = sensors()
    a = measured_state(s, [0, 0, 1])
    s.update(mass_ratio=.3, side_m=.06, qpos=[999], qvel=[999], seed=1)
    assert np.array_equal(measured_state(s, [0, 0, 1]), a)
    assert len(a) == 20
    assert not any(x in SENSOR_FIELDS for x in ('mass_ratio', 'side_m', 'qpos', 'seed'))


def test_zero_correction_equivalence_bounds_once_per_control_tick_and_reset():
    c = Controller(np.zeros((4, 20)), np.full(4, .12))
    s = sensors()
    a = c.action(s, [0, 0, 1], tick=0)
    c.reset()
    b = c.action(s, [0, 0, 1], tick=0, correction=np.zeros(3))
    assert a == pytest.approx(b)
    with pytest.raises(ValueError, match='tick'):
        c.action(s, [0, 0, 1], tick=0)
    with pytest.raises(ValueError):
        c.action(s, [0, 0, 1], tick=1, correction=[.6, 0, 0])
    assert np.max(np.abs(b)) <= 1
    c.reset()
    assert c.tick is None
    assert c.integral == 0


def test_payload_rotation_is_observed_and_feedback_can_be_disabled():
    s = sensors()
    s['package_angular_velocity'] = [1., 0., 0.]
    assert np.linalg.norm(measured_state(s, [0, 0, 1])) > 0
    k = np.ones((4, 20))*.01
    on = Controller(k, np.zeros(4))
    off = Controller(k, np.zeros(4), vibration_feedback=False)
    assert not np.allclose(on.action(s, [0,0,1], tick=0), off.action(s, [0,0,1], tick=0))
