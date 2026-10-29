"""Continuous reference contracts; these do not qualify physical flight."""
import json

import numpy as np
import pytest

from adaptive_drone_patrol.dual_coordination_reference import (
    ContinuousReference, KinematicState, ReferenceLimits, ReferenceInfeasible,
    nominal_duration,
)


def actual(reference, time, offset=(0., 0., 0.)):
    p, v, _ = reference.sample(time)
    return KinematicState(p + offset, v)


def test_stationary_reference_holds_and_nominal_duration_matches_existing_curve():
    ref = ContinuousReference((1., 2., 3.), start_time=4.)
    for t in (4., 5., 100.):
        p, v, a = ref.sample(t)
        np.testing.assert_array_equal(p, [1., 2., 3.])
        np.testing.assert_array_equal(v, np.zeros(3))
        np.testing.assert_array_equal(a, np.zeros(3))
    assert nominal_duration([0, 0, 0], [0, 0, 0]) == 3.
    assert nominal_duration([0, 0, 0], [10, 0, 0]) == pytest.approx(15.625)


def test_retarget_preserves_midflight_position_velocity_and_acceleration():
    ref = ContinuousReference((0., 0., 1.))
    flight = ref.retarget((5., 1., 2.), 0., actual_state=actual(ref, 0.))
    t = flight.duration * .37
    before = flight.sample(t)
    after = flight.retarget((-2., 4., 1.5), t, actual_state=actual(flight, t))
    for expected, observed in zip(before, after.sample(t)):
        np.testing.assert_allclose(observed, expected, atol=2e-13)
    np.testing.assert_allclose(after.sample(after.end_time)[0], [-2., 4., 1.5])
    for derivative in after.sample(after.end_time)[1:]:
        np.testing.assert_array_equal(derivative, np.zeros(3))
    assert after.bounds().speed <= 1.2
    assert after.bounds().acceleration <= .8
    assert after.bounds().jerk <= 2.


def test_bounds_cover_interior_extrema_not_just_endpoint_derivatives():
    ref = ContinuousReference((0., 0., 0.)).retarget(
        (10., 0., 0.), 0., actual_state=KinematicState((0, 0, 0), (0, 0, 0)))
    # Rest-to-rest quintic has exact maxima 1.875 d/T, 10sqrt(3)d/(3T²), 60d/T³.
    exact = [18.75/ref.duration, (100*np.sqrt(3)/3)/ref.duration**2, 600/ref.duration**3]
    certified = ref.bounds()
    for upper, truth in zip((certified.speed, certified.acceleration, certified.jerk), exact):
        assert upper >= truth * (1-1e-13)
        assert upper <= truth * 1.005


def test_brake_is_continuous_stops_ahead_and_reports_actual_tracking_error():
    ref = ContinuousReference((0., 0., 1.))
    flight = ref.retarget((8., 0., 1.), 0., actual_state=actual(ref, 0.))
    t = flight.duration/2
    p, v, a = flight.sample(t)
    brake = flight.brake(t, actual_state=actual(flight, t, (.12, 0, 0)))
    for expected, observed in zip((p, v, a), brake.sample(t)):
        np.testing.assert_allclose(observed, expected, atol=2e-13)
    assert brake.target[0] > p[0]
    assert brake.tracking_error.position == pytest.approx(.12)
    assert brake.tracking_error.velocity == pytest.approx(0.)
    np.testing.assert_array_equal(brake.sample(brake.end_time + 10)[1:], np.zeros((2, 3)))
    assert brake.bounds().speed <= 1.2
    assert brake.bounds().acceleration <= .8
    assert brake.bounds().jerk <= 2.


def test_snapshot_roundtrip_is_exact_and_does_not_alias_input():
    initial = [0., 0., 1.]
    ref = ContinuousReference(initial)
    initial[0] = 9
    ref = ref.retarget([3., 2., 1.], 0., actual_state=actual(ref, 0.))
    ref = ref.brake(.7, actual_state=actual(ref, .7))
    snapshot = ref.to_dict()
    restored = ContinuousReference.from_dict(json.loads(json.dumps(snapshot)))
    snapshot['control_points'][0][0] = 999
    for t in np.linspace(ref.start_time, ref.end_time + 1, 15):
        np.testing.assert_array_equal(restored.sample(t), ref.sample(t))
    with pytest.raises((AttributeError, TypeError)):
        ref.duration = 99


@pytest.mark.parametrize('bad', [(1., 2.), (1., float('nan'), 2.), (float('inf'), 0, 0)])
def test_rejects_invalid_position_and_actual_state(bad):
    with pytest.raises(ValueError):
        ContinuousReference(bad)
    with pytest.raises(ValueError):
        KinematicState((0, 0, 0), bad)


def test_finite_search_fails_closed_without_mutating_current_reference():
    ref = ContinuousReference((0., 0., 1.), limits=ReferenceLimits(max_duration=.1, max_attempts=2))
    with pytest.raises(ReferenceInfeasible):
        ref.retarget((100., 0., 1.), 0., actual_state=actual(ref, 0.))
    np.testing.assert_array_equal(ref.sample(10)[0], [0, 0, 1])
    with pytest.raises(ValueError):
        ReferenceLimits(max_attempts=0)
    with pytest.raises(ValueError):
        ref.sample(float('nan'))
    with pytest.raises(ValueError):
        ref.retarget((1, 0, 0), -1., actual_state=actual(ref, 0.))


def test_stationary_braking_holds_without_drift():
    ref = ContinuousReference((1., 2., 3.))
    brake = ref.brake(4., actual_state=actual(ref, 4.))
    np.testing.assert_array_equal(brake.sample(100), ([1, 2, 3], [0, 0, 0], [0, 0, 0]))


@pytest.mark.parametrize('field,value', [('position', None), ('velocity', None)])
def test_actual_state_requires_position_and_velocity(field, value):
    state = dict(position=(0, 0, 0), velocity=(0, 0, 0))
    state[field] = value
    with pytest.raises(ValueError):
        KinematicState(**state)


def test_one_attempt_budget_rejects_an_infeasible_retarget():
    ref = ContinuousReference((0, 0, 1), limits=ReferenceLimits(max_attempts=1))
    ref = ref.retarget((8, 0, 1), 0, actual_state=actual(ref, 0))
    t = ref.duration/2
    with pytest.raises(ReferenceInfeasible):
        ref.retarget(ref.sample(t)[0], t, actual_state=actual(ref, t))


@pytest.mark.parametrize('error', [-1., float('nan'), float('inf')])
def test_restore_rejects_invalid_tracking_error(error):
    ref = ContinuousReference((0, 0, 1))
    snapshot = ref.brake(0, actual_state=actual(ref, 0)).to_dict()
    snapshot['tracking_error']['position'] = error
    with pytest.raises(ValueError):
        ContinuousReference.from_dict(snapshot)


def test_restore_rejects_motion_outside_certified_limits():
    ref = ContinuousReference((0, 0, 1))
    snapshot = ref.retarget((4, 0, 1), 0, actual_state=actual(ref, 0)).to_dict()
    snapshot['duration'] = .001
    with pytest.raises(ValueError):
        ContinuousReference.from_dict(snapshot)


def test_moving_brake_serializes_native_json_scalars_and_supervisor_horizon():
    from adaptive_drone_patrol.dual_coordination_safety import ReferenceSupervisor
    start = (0.,0.,1.)
    reference = ContinuousReference(start).retarget((4.,0.,1.),0.,
        actual_state=KinematicState(start,(0.,0.,0.)))
    hold = ContinuousReference((0.,5.,1.))
    now = .85
    states = [KinematicState(r.sample(now)[0],r.sample(now)[1]) for r in (reference,hold)]
    brake = reference.brake(now,actual_state=states[0])
    assert type(brake.to_dict()['duration']) is float
    assert type(brake.to_dict()['start_time']) is float
    decision = ReferenceSupervisor().decide((reference,hold),states,now,priority=1)
    assert type(decision.horizon_s) is float


def test_batched_positions_exactly_match_scalar_samples_across_moving_and_hold_segments():
    hold = ContinuousReference((-1.,2.,3.))
    moving = hold.retarget((4.,-2.,1.),0.,actual_state=actual(hold,0.))
    brake = moving.brake(1.3,actual_state=actual(moving,1.3))
    for ref in (hold,moving,brake):
        times = np.linspace(ref.start_time,ref.end_time+2.,79)
        expected = np.array([ref.sample(t)[0] for t in times])
        np.testing.assert_array_equal(ref.positions(times),expected)
        positions = ref.positions(times)
        positions[0] = 999.
        np.testing.assert_array_equal(ref.positions(times),expected)
        assert ref.positions([]).shape == (0,3)
        for invalid in ([ref.start_time-.01],[float('nan')],[[ref.start_time]]):
            with pytest.raises(ValueError):
                ref.positions(invalid)


def test_brake_near_nominal_speed_peak_uses_tighter_conservative_certificate():
    # Exact development pilot002 failure at18.53s: coarse depth6 upper1.2000238
    # hid an admissible stop (depth10 upper1.1999715), below the same1.2m/s cap.
    import numpy as np
    from adaptive_drone_patrol.dual_coordination_reference import ContinuousReference,KinematicState
    reference=ContinuousReference([-2.2574846772991513,1.7335248129420815,2.152548596622853],start_time=15.2)
    state=KinematicState(reference.target,[0,0,0])
    reference=reference.retarget([-1.720730862092656,-1.677042794662317,4.688765039965419],15.2,actual_state=state)
    p,v,a=reference.sample(18.53)
    brake=reference.brake(18.53,actual_state=KinematicState(p,v,a))
    assert brake.bounds().speed<=1.2
    assert brake.bounds().acceleration<=.8
    assert brake.bounds().jerk<=2.
    for old,new in zip((p,v,a),brake.sample(18.53)):
        np.testing.assert_allclose(old,new,rtol=0,atol=1e-12)
    restored=ContinuousReference.from_dict(brake.to_dict())
    assert restored.bounds()==brake.bounds()


def test_brake_search_can_shorten_duration_when_longer_stops_exceed_speed_cap():
    import math
    import numpy as np
    from adaptive_drone_patrol.dual_coordination_reference import ContinuousReference,KinematicState
    # Exact next failure from pilot003. Increasing T from2.9984s worsened
    # overshoot; a shorter2.4987s member of the same C2 stop family is admissible.
    start=[1.1703085519575023,1.6670503818462155,4.843751974793301]
    reference=ContinuousReference(start,start_time=17.4).retarget(
        [-.3282430937268912,1.1881130985360875,2.2223179401144963],17.4,
        actual_state=KinematicState(start,[0,0,0]))
    p,v,a=reference.sample(19.75)
    initial=max(.2,2*np.linalg.norm(v)/.8,math.sqrt(6*np.linalg.norm(v)/2),4*np.linalg.norm(a)/2)
    brake=reference.brake(19.75,actual_state=KinematicState(p,v,a))
    assert brake.duration<initial
    assert brake.bounds().speed<=1.2 and brake.bounds().acceleration<=.8 and brake.bounds().jerk<=2.
    for old,new in zip((p,v,a),brake.sample(19.75)):
        np.testing.assert_allclose(old,new,rtol=0,atol=1e-12)
    np.testing.assert_allclose(brake.target,p+v*brake.duration/2+a*brake.duration**2/12,rtol=0,atol=1e-12)
