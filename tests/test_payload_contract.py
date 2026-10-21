"""CPU logic fixtures are not payload flight evidence."""

import math
import numpy as np
import pytest
from adaptive_drone_patrol.payload_contract import (
    Payload, PayloadSampler, SharedSettling, payload_metrics, settled,
    actor_observation, OBSERVATION_CONTRACT, require_checkpoint_contract,
)


def sample():
    return dict(drone_xyz=[0, 0, 1], drone_quaternion_wxyz=[1, 0, 0, 0],
                drone_velocity=[0, 0, 0], drone_angular_velocity=[0, 0, 0],
                package_xyz=[0, 0, .6575], package_quaternion_wxyz=[1, 0, 0, 0],
                package_velocity=[0, 0, 0], package_angular_velocity=[0, 0, 0],
                attachment_a_world=[0, 0, .98], attachment_b_world=[0, 0, .68],
                attachment_a_velocity=[0, 0, 0], attachment_b_velocity=[0, 0, 0])


def test_payload_inertia_and_invalid_parameters():
    p = Payload(.028, .2, .045)
    assert p.mass == pytest.approx(.0056)
    assert p.inertia == pytest.approx([.0056 * .045**2 / 6] * 3)
    for ratio in [.09, .31, math.nan, True]:
        with pytest.raises(ValueError):
            Payload(.028, ratio, .045)
    for side in [0, .04, math.inf, True]:
        with pytest.raises(ValueError):
            Payload(.028, .2, side)


def test_episode_resampling_is_independent_and_selective():
    a, b = PayloadSampler(.028, 7, 'training'), PayloadSampler(.028, 7, 'training')
    first = a.reset([0, 1])
    assert first == b.reset([0, 1])
    assert a.reset([0])[0] != first[0]
    assert a.current[1] == first[1]
    assert a.reset([1])[1] == b.reset([1])[1]
    c = PayloadSampler(.028, 7, 'validation')
    assert c.reset([0])[0] != first[0]
    fixed = PayloadSampler(.028, 7, 'training', randomized=False)
    assert fixed.reset([0])[0] == fixed.reset([0])[0] == Payload(.028, .2, .045)
    with pytest.raises(ValueError):
        a.reset([0, 0])


def test_size_mass_distribution_is_full_and_episode_varying():
    a = PayloadSampler(.028, 7, 'training')
    draws = [a.reset([0])[0] for _ in range(6000)]
    for side in [.03, .045, .06]:
        group = [p.mass_ratio for p in draws if p.side_m == side]
        assert 1800 < len(group) < 2200
        assert .19 < np.mean(group) < .21
        assert min(group) < .101 and max(group) > .299


def test_nominal_metrics_and_attachment_angular_velocity():
    p = Payload(.028, .2, .045)
    m = payload_metrics(sample(), [0, 0, 1], p)
    assert m['constraint_residual_m'] < 1e-12
    assert m['package_floor_m'] == pytest.approx(.635)
    assert settled(m)
    s = sample()
    s['drone_angular_velocity'] = [1, 0, 0]
    m = payload_metrics(s, [0, 0, 1], p)
    assert m['swing_rate_deg_s'] == pytest.approx(math.degrees(.02 / .3))


def test_turning_point_spin_contact_and_invalid_geometry_do_not_settle():
    p = Payload(.028, .2, .045)
    s = sample()
    s['package_xyz'][0] = .04
    assert not settled(payload_metrics(s, [0, 0, 1], p))
    s = sample()
    s['package_angular_velocity'] = [0, 0, math.radians(11)]
    assert not settled(payload_metrics(s, [0, 0, 1], p))
    s = sample()
    s['package_xyz'][2] = .11
    q = math.sqrt(.5)
    s['package_quaternion_wxyz'] = [q, q, 0, 0]
    assert not settled(payload_metrics(s, [0, 0, 1], p))
    s = sample()
    s['package_xyz'][0] = math.nan
    with pytest.raises(ValueError):
        payload_metrics(s, [0, 0, 1], p)


def test_oriented_cube_lowest_point_is_not_center_or_unrotated_extent():
    s = sample()
    s['package_quaternion_wxyz'] = [math.cos(math.pi/8), math.sin(math.pi/8), 0, 0]
    m = payload_metrics(s, [0, 0, 1], Payload(.028, .2, .045))
    assert m['package_floor_m'] == pytest.approx(.6575 - .045 * math.sqrt(2) / 2)


def test_shared_dwell_needs_100_intervals_and_resets_on_interruption():
    d = SharedSettling(.01)
    assert not d.update(0, True)
    for i in range(1, 100):
        assert not d.update(i*.01, True)
    assert d.update(1, True)
    assert not d.update(1.01, False)
    assert not d.update(1.02, True)
    for i in range(103, 202):
        assert not d.update(i*.01, True)
    assert d.update(2.02, True)
    assert not d.update(2.03, True, target_id='B')
    with pytest.raises(ValueError):
        d.update(2.03, True)
    # A missing sample must not count unobserved time as settled.
    with pytest.raises(ValueError):
        d.update(2.06, True)


def test_versioned_observation_excludes_hidden_parameters():
    s = sample()
    old = np.zeros(17)
    obs = actor_observation(old, s)
    assert obs.shape == (29,)
    s.update(mass=10, side_m=5, seed=80, geometry_id=4)
    assert np.array_equal(obs, actor_observation(old, s))
    assert OBSERVATION_CONTRACT['size'] == 29
    require_checkpoint_contract(OBSERVATION_CONTRACT, OBSERVATION_CONTRACT)
    with pytest.raises(ValueError):
        require_checkpoint_contract({'size': 29}, OBSERVATION_CONTRACT)
    other = dict(OBSERVATION_CONTRACT, clip=[-2, 2])
    with pytest.raises(ValueError):
        require_checkpoint_contract(other, OBSERVATION_CONTRACT)


def test_training_envelope_guard_and_numeric_excursion_are_separate():
    from adaptive_drone_patrol.payload_contract import training_outside_envelope, numerical_excursion
    s=sample()
    assert not training_outside_envelope(s,[0,0,1])
    s['drone_xyz']=[6.01,0,1]
    assert training_outside_envelope(s,[0,0,1])
    assert not numerical_excursion(s,1.)
    s['package_velocity']=[20000,0,0]
    assert numerical_excursion(s,1.)


def test_known_numerical_failure_blocks_old_short_probe_receipt():
    from adaptive_drone_patrol.payload_contract import require_physical_admission
    import pytest
    receipt={'status':'PASS','physical_training_admitted':True,'thresholds_sha256':'old'}
    with pytest.raises(ValueError,match='operating-domain'):
        require_physical_admission(receipt,{'operating_domain_admission':'blocked_numerical_instability'},'old')
    with pytest.raises(ValueError,match='thresholds'):
        require_physical_admission(receipt,{'operating_domain_admission':'admitted'},'new')
