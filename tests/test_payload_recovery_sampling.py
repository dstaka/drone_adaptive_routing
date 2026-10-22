"""CPU planning checks cannot qualify a physical batch or admit optimization."""

from dataclasses import replace
import math

import numpy as np
import pytest

from adaptive_drone_patrol.payload_recovery_sampling import (
    SamplePlan, StageExposure, audit_rollout, forecast_pair,
)


def stages(total=300000):
    return (
        StageExposure('hover', total // 2, 1000, 1, 1000, ('hover',)),
        StageExposure('motion', total // 2, 3000, 2, 3000,
                      ('short_transfer', 'long_transfer', 'reversal', 'vertical_xyz', 'repeated_targets')),
    )


def plan(**kwargs):
    fields = dict(n_envs=8, rollout_steps=100, total_transitions=300000,
                  ppo_epochs=5, minibatches=4, stages=stages())
    fields.update(kwargs)
    return SamplePlan(**fields)


def forecast(**kwargs):
    fields = dict(evidence_enabled_transitions_per_s=1000., optimizer_step_s=.01,
                  per_method_setup_s=10., remaining_s=2000.,
                  downstream_seconds=dict(validation=100., acceptance=500., recording=300.,
                                          verification=50., reserve=25.))
    fields.update(kwargs)
    return forecast_pair(plan(), plan(), **fields)


def test_total_environment_samples_are_separate_from_vector_and_optimizer_steps():
    p = plan()
    assert p.rollouts == 375
    assert p.transitions_per_rollout == 800
    assert p.vector_steps == 37500
    assert p.per_environment_steps == 37500
    assert p.optimizer_steps == 7500
    assert p.per_environment_control_seconds == 375.
    assert p.seed == 7
    assert p.training_admitted is False


def test_plan_never_silently_rounds_default_budget_for_large_batch():
    with pytest.raises(ValueError, match='divisib'):
        plan(n_envs=32)
    declared = stages(307200)
    declared = (declared[0], replace(declared[1], min_completed_sequences_per_env=1))
    p = plan(n_envs=32, total_transitions=307200, stages=declared)
    assert p.rollouts == 96
    assert p.vector_steps == 9600
    assert p.total_transitions == 307200


@pytest.mark.parametrize('field', ['n_envs', 'rollout_steps', 'total_transitions', 'ppo_epochs', 'minibatches'])
@pytest.mark.parametrize('bad', [True, False, 8., -1, 0, float('nan'), float('inf'), '8', None])
def test_plan_rejects_ambiguous_or_nonpositive_integer_counts(field, bad):
    with pytest.raises(ValueError):
        plan(**{field: bad})


def test_minibatches_need_exact_full_rollout_partition():
    with pytest.raises(ValueError, match='minibatch'):
        plan(minibatches=3)


def test_all_stage_transitions_and_per_environment_exposure_must_be_explicit():
    with pytest.raises(ValueError, match='stage'):
        plan(stages=())
    with pytest.raises(ValueError, match='stage'):
        plan(stages=(stages()[0],))
    with pytest.raises(ValueError, match='continuous'):
        plan(stages=(replace(stages()[0], min_continuous_steps_per_env=20000), stages()[1]))
    with pytest.raises(ValueError, match='sequence'):
        plan(stages=(stages()[0], replace(stages()[1], min_completed_sequences_per_env=7)))
    with pytest.raises(ValueError, match='stage'):
        plan(stages=(replace(stages()[0], total_transitions=149999),
                     replace(stages()[1], total_transitions=150001)))
    with pytest.raises(ValueError, match='coverage'):
        plan(stages=(stages()[0], replace(stages()[1], required_motion_coverage=('hover',))))


@pytest.mark.parametrize('field', ['total_transitions', 'min_continuous_steps_per_env',
                                 'min_completed_sequences_per_env', 'sequence_steps'])
@pytest.mark.parametrize('bad', [True, 1., 0, -2, math.nan, '3'])
def test_stage_requirements_reject_coerced_counts(field, bad):
    with pytest.raises(ValueError):
        replace(stages()[0], **{field: bad})


def test_rollout_padding_does_not_inflate_generated_or_optimizer_samples():
    valid = np.ones((100, 10), dtype=bool)
    result = audit_rollout(plan(), valid, [True] * 8 + [False] * 2)
    assert result.generated_valid == 800
    assert result.generated_invalid == 0
    assert result.padded_slots == 200
    assert result.optimizer_transitions == 800
    assert result.quarantined is False
    assert result.stop_required is False


def test_one_invalid_active_transition_quarantines_whole_rollout_before_ppo():
    valid = np.ones((100, 10), dtype=bool)
    valid[13, 4] = False
    valid[:, 9] = False
    result = audit_rollout(plan(), valid, [True] * 8 + [False] * 2)
    assert result.generated_valid == 799
    assert result.generated_invalid == 1
    assert result.padded_slots == 200
    assert result.optimizer_transitions == 0
    assert result.quarantined is True
    assert result.stop_required is True


@pytest.mark.parametrize('valid,active', [
    (np.ones((99, 8), dtype=bool), [True] * 8),
    (np.ones((100, 8), dtype=bool), [True] * 7 + [False]),
    (np.ones((100, 8), dtype=float), [True] * 8),
    (np.ones((100, 8), dtype=bool), [1] * 8),
    (np.ones((100, 8, 1), dtype=bool), [True] * 8),
    ([[True] * 8] * 100, [True] * 9),
])
def test_rollout_rejects_partial_or_ambiguous_slot_masks(valid, active):
    with pytest.raises(ValueError):
        audit_rollout(plan(), valid, active)


def test_forecast_includes_two_matched_methods_and_every_downstream_reserve():
    f = forecast()
    assert f.per_method_training_s == 385.
    assert f.pair_training_s == 770.
    assert f.downstream_s == 975.
    assert f.required_s == 1745.
    assert f.remaining_after_s == 255.
    assert f.fits is True
    assert f.training_admitted is False
    assert f.evidence_kind == 'cpu-planning-estimate'
    assert forecast(remaining_s=1000.).fits is False


def test_forecast_rejects_method_specific_plan_or_missing_phase():
    with pytest.raises(ValueError, match='identical'):
        forecast_pair(plan(), plan(ppo_epochs=4),
                      evidence_enabled_transitions_per_s=1000., optimizer_step_s=.01,
                      per_method_setup_s=10., remaining_s=2000., downstream_seconds={})
    with pytest.raises(ValueError, match='downstream'):
        forecast(downstream_seconds=dict(validation=1., acceptance=1., recording=1., verification=1.))


@pytest.mark.parametrize('field', ['evidence_enabled_transitions_per_s', 'optimizer_step_s',
                                 'per_method_setup_s', 'remaining_s'])
@pytest.mark.parametrize('bad', [True, '1', None, -1., math.nan, math.inf])
def test_forecast_rejects_nonfinite_or_coerced_wall_time_inputs(field, bad):
    with pytest.raises(ValueError):
        forecast(**{field: bad})


@pytest.mark.parametrize('bad', [0., -1., math.nan, math.inf, True, '5'])
def test_forecast_requires_finite_positive_downstream_reserves(bad):
    with pytest.raises(ValueError):
        forecast(downstream_seconds=dict(validation=100., acceptance=bad, recording=300.,
                                        verification=50., reserve=25.))


def test_episode_draws_depend_on_logical_identity_not_worker_or_completion_order():
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeLoads

    sampler = EpisodeLoads(.02700047753751278)
    serial = {episode: sampler.draw(episode, randomized=True) for episode in range(100)}
    shuffled = {episode: sampler.draw(episode, randomized=True) for episode in reversed(range(100))}
    assert serial == shuffled
    assert len({draw.mass_stream_sha256 for draw in serial.values()}) == 100
    assert len({draw.size_stream_sha256 for draw in serial.values()}) == 100
    assert all(draw.mass_stream_sha256 != draw.size_stream_sha256 for draw in serial.values())
    assert all(draw.seed == 7 and draw.logical_episode_id == key for key, draw in serial.items())
    assert all(draw.payload.side_m == .045 and draw.payload.mass_ratio == .2
               for draw in (sampler.draw(i, randomized=False) for i in range(20)))


def test_size_mass_streams_cover_full_domain_independently():
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeLoads

    draws = [EpisodeLoads(.02700047753751278).draw(i, randomized=True) for i in range(6000)]
    for side in (.030, .045, .060):
        ratios = [draw.payload.mass_ratio for draw in draws if draw.payload.side_m == side]
        assert 1800 < len(ratios) < 2200
        assert min(ratios) < .101 and max(ratios) > .299
        assert .196 < np.mean(ratios) < .204
    assert len({draw.payload.mass_ratio for draw in draws}) == 6000
    assert abs(np.corrcoef([draw.payload.side_m for draw in draws],
                          [draw.payload.mass_ratio for draw in draws])[0, 1]) < .05


@pytest.mark.parametrize('bad', [True, -1, 1., '7', None, math.inf, math.nan])
def test_logical_episode_identity_cannot_be_coerced(bad):
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeLoads

    with pytest.raises(ValueError):
        EpisodeLoads(.027).draw(bad, randomized=True)


@pytest.mark.parametrize('bad', [True, 0., -1., '0.027', None, math.inf, math.nan])
def test_episode_reference_mass_must_be_positive_finite_real(bad):
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeLoads

    with pytest.raises(ValueError):
        EpisodeLoads(bad)


def test_sampling_rejects_other_final_seeds_and_nonboolean_method_flags():
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeLoads

    for seed in (True, 7., 8, -1, '7'):
        with pytest.raises(ValueError):
            EpisodeLoads(.027, seed=seed)
    for randomized in (1, 0, 'fixed', None, np.bool_(True)):
        with pytest.raises(ValueError):
            EpisodeLoads(.027).draw(1, randomized=randomized)


def test_thirty_selective_resets_preserve_ongoing_episode_cursor_across_rollouts():
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeSchedule

    schedule = EpisodeSchedule(.027)
    schedule.reset({0: 0, 1: 1}, randomized=True)
    schedule.advance([0, 1], intervals=100)
    ongoing = schedule.snapshot()[1]
    for identity in range(2, 32):
        schedule.reset({0: identity}, randomized=True)
        assert schedule.snapshot()[0].control_steps == 0
        assert schedule.snapshot()[1] == ongoing
        schedule.advance([0], intervals=100)
    schedule.advance([0, 1], intervals=100)
    after = schedule.snapshot()
    assert after[1].control_steps == 200
    assert after[1].draw == ongoing.draw
    assert after[0].control_steps == 200
    assert after[0].draw.logical_episode_id == 31
    snapshot = schedule.snapshot()
    snapshot.clear()
    assert len(schedule.snapshot()) == 2


def test_scheduler_rejects_identity_reuse_and_validates_before_partial_mutation():
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeSchedule

    schedule = EpisodeSchedule(.027)
    schedule.reset({0: 7, 1: 8}, randomized=True)
    before = schedule.snapshot()
    for assignment in ({0: 8}, {0: 9, 1: 9}, {0: 9, -1: 10}, {0: 9, 2: True}, {}):
        with pytest.raises(ValueError):
            schedule.reset(assignment, randomized=True)
        assert schedule.snapshot() == before
    schedule.reset({0: 9}, randomized=True)
    with pytest.raises(ValueError):
        schedule.reset({0: 7}, randomized=True)
    current = schedule.snapshot()
    for ids, intervals in (([0, 2], 1), ([0, 0], 1), ([True], 1), ([0], 1.), ([0], -1), ([], 1)):
        with pytest.raises(ValueError):
            schedule.advance(ids, intervals=intervals)
        assert schedule.snapshot() == current


def test_sampling_metadata_is_audit_only_and_cannot_add_hidden_actor_inputs():
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeLoads
    from adaptive_drone_patrol.payload_retry_contract import actor_observation

    sample = dict(drone_xyz=[0., 0., 1.], drone_quaternion_wxyz=[1., 0., 0., 0.],
                  drone_velocity=[0., 0., 0.], drone_angular_velocity=[0., 0., 0.],
                  package_xyz=[0., 0., .66], package_velocity=[0., 0., 0.],
                  attachment_a_world=[0., 0., .98], attachment_b_world=[0., 0., .68],
                  attachment_a_velocity=[0., 0., 0.], attachment_b_velocity=[0., 0., 0.])
    original = actor_observation(sample, [0., 0., 1.], [0.] * 4)
    sampler = EpisodeLoads(.027)
    for episode in range(30):
        draw = sampler.draw(episode, randomized=True)
        record = draw.audit_record()
        assert 'actor_observation' not in record and 'actor_fields' not in record
        assert record['payload']['side_m'] in (.03, .045, .06)
        assert record['payload']['mass_ratio'] == draw.payload.mass_ratio
        sample.update(record)
        assert np.array_equal(actor_observation(sample, [0., 0., 1.], [0.] * 4), original)


def test_accounting_records_exact_minibatch_size_and_invalid_physical_slot_coordinates():
    assert plan().minibatch_size == 200
    valid = np.ones((100, 10), dtype=bool)
    active = [True, False, True, True, True, True, True, True, False, True]
    valid[0, 2] = False
    valid[99, 9] = False
    result = audit_rollout(plan(), valid, active)
    assert result.invalid_slot_coordinates == ((0, 2), (99, 9))
    assert result.optimizer_transitions == 0
    assert result.generated_invalid == 2


class CoercionTrap:
    def __float__(self):
        raise AssertionError('Untrusted input was coerced')

    def __int__(self):
        raise AssertionError('Untrusted input was coerced')


def test_numeric_boundaries_reject_custom_coercion_and_overflow_without_executing_it():
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeLoads

    with pytest.raises(ValueError):
        plan(n_envs=CoercionTrap())
    with pytest.raises(ValueError):
        forecast(optimizer_step_s=CoercionTrap())
    with pytest.raises(ValueError):
        EpisodeLoads(CoercionTrap())
    with pytest.raises(ValueError):
        forecast(evidence_enabled_transitions_per_s=5e-324)
    with pytest.raises(ValueError):
        forecast(downstream_seconds=dict(validation=1e308, acceptance=1e308, recording=1.,
                                        verification=1., reserve=1.))


def test_draw_stream_configuration_cannot_change_after_construction():
    from dataclasses import FrozenInstanceError
    from adaptive_drone_patrol.payload_recovery_sampling import EpisodeLoads

    loads = EpisodeLoads(.027)
    with pytest.raises(FrozenInstanceError):
        loads.seed = 8


@pytest.mark.parametrize('field', ['evidence_enabled_transitions_per_s', 'optimizer_step_s',
                                 'per_method_setup_s', 'remaining_s'])
def test_forecast_needs_positive_observed_costs_for_every_input(field):
    with pytest.raises(ValueError):
        forecast(**{field: 0.})
