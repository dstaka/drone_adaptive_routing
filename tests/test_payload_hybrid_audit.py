import copy

import numpy as np
import pytest

from adaptive_drone_patrol.payload_hybrid_audit import independent_metrics, interval_update
from adaptive_drone_patrol.payload_contract import Payload, payload_metrics


def raw():
    result = dict(contact=False,drone_xyz=[0.,0.,1.],package_xyz=[0.,0.,.6575])
    for body in ('drone','package'):
        result[body+'_quaternion_wxyz'] = [1.,0.,0.,0.]
        result[body+'_velocity'] = [0.,0.,0.]
        result[body+'_angular_velocity'] = [0.,0.,0.]
    return result


def test_independent_attachment_offsets_full_spin_and_oriented_floor():
    s = raw()
    s['package_angular_velocity'] = [0.,0.,.6]
    m = independent_metrics(s,[0,0,1],.045)
    expected = payload_metrics(s,[0,0,1],Payload(.02700047753751278,.2,.045))
    for key in ('drone_error_m','package_error_m','package_speed_m_s','swing_rate_deg_s',
                'constraint_residual_m','package_spin_deg_s','package_floor_m'):
        assert m[key] == pytest.approx(expected[key],abs=1e-9)
    assert m['package_spin_deg_s'] == pytest.approx(np.degrees(.6))
    assert m['drone_floor_m'] == pytest.approx(.9875)
    bad = copy.deepcopy(s)
    bad['package_xyz'] = [0,0,.5]
    assert independent_metrics(bad,[0,0,1],.045)['constraint_residual_m'] > .1


def test_independent_interval_count_requires_both_endpoints_and_resets():
    state = (False,0)
    for i in range(100):
        state = interval_update(state,True)
        assert state[1]<100
    state = interval_update(state,True)
    assert state[1]==100
    assert interval_update(state,False)==(False,0)
