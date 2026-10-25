import math

import pytest

from adaptive_drone_patrol.payload_contract import SharedSettling, settled
from adaptive_drone_patrol.payload_hybrid_contract import (
    VIDEO_PROFILE, STRICT_PROFILE, MissionClock, acceptance_counts,
)


def metrics():
    return dict(drone_error_m=.2, drone_speed_m_s=.3, package_error_m=.2,
                package_speed_m_s=.15, swing_angle_deg=5., swing_rate_deg_s=15.,
                package_spin_deg_s=30., package_floor_m=.11, drone_floor_m=.11,
                drone_roll_deg=60., drone_pitch_deg=-60., constraint_residual_m=2e-5)


def test_profiles_are_explicit_canonical_and_legacy_stays_strict():
    m = metrics()
    assert VIDEO_PROFILE.id == 'payload-settling-video-v1'
    assert VIDEO_PROFILE.valid(m)
    assert not STRICT_PROFILE.valid(m)
    assert not settled(m)
    assert len(VIDEO_PROFILE.sha256) == 64
    assert VIDEO_PROFILE.sha256 != STRICT_PROFILE.sha256
    for key, bound in VIDEO_PROFILE.limits:
        assert not VIDEO_PROFILE.valid(dict(m, **{key: math.nextafter(bound, math.inf)}))
        assert not VIDEO_PROFILE.valid(dict(m, **{key: math.nan}))
    assert not VIDEO_PROFILE.valid(m, contact=True)
    assert not VIDEO_PROFILE.valid(dict(m, drone_floor_m=.1))


def test_shared_intervals_reset_and_shadow_cannot_trigger_live_completion():
    live, shadow = SharedSettling(), SharedSettling()
    m = metrics()
    for i in range(101):
        complete = live.update(i*.01, VIDEO_PROFILE.valid(m), 'A')
        strict = shadow.update(i*.01, STRICT_PROFILE.valid(m), 'A')
        assert complete == (i == 100)
        assert not strict
    assert not live.update(1.01, True, 'B')
    assert live.dwell_s == 0
    assert not live.update(1.02, False, 'B')


def test_no_aggregate_deadline_and_only_real_target_change_resets_leg():
    clock = MissionClock(reset_wall_s=20.)
    for i in range(11):
        clock.command(i*20., str(i), [float(i), 0., 1.])
        assert clock.check(i*20.+19., 21.+i, complete=True) == 'COMPLETE'
    assert clock.check(220., 40.) == 'CONTINUE'
    assert not clock.command(225., '10', [10., 0., 1.])
    assert clock.leg_start_s == 200.
    assert clock.check(230., 45., complete=True) == 'COMPLETE'
    assert clock.check(230., 45.) == 'LEG_TIMEOUT'
    assert clock.check(230., 45., complete=True, unsafe=True) == 'SAFETY_FAILURE'
    assert clock.check(230., 620., complete=True) == 'WALL_TIMEOUT'
    with pytest.raises(ValueError):
        clock.check(230., 619.)


def test_branch_specific_denominators():
    assert acceptance_counts('controller') == dict(validation=12, transitions=120, basic=9,
        endpoints=18, heldout=100, missions=127, methods=1)
    assert acceptance_counts('nn')['missions'] == 254
    with pytest.raises(ValueError):
        acceptance_counts('legacy')
