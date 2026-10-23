"""Declared method and seed must reproduce every consumed reset draw."""

import copy

import numpy as np
import pytest

from adaptive_drone_patrol.payload_cases import initial_state
from adaptive_drone_patrol.payload_contract import PayloadSampler


def reset_rows(method, seed, count):
    mass = .02700047753751278
    sampler = PayloadSampler(mass, seed, 'training', randomized=method == 'randomized')
    pose = np.random.default_rng(np.random.SeedSequence([seed, 710, 101]))
    for episode in range(1, count + 1):
        payload = sampler.reset([0])[0]
        yield dict(episode=episode, draw_index=sampler.draw_indices[0],
                   payload=vars(payload), initial=initial_state(pose, payload.side_m))


@pytest.mark.parametrize('method', ['randomized', 'fixed'])
def test_reset_draws_replay_in_order_for_both_methods(method):
    from adaptive_drone_patrol.payload_solver_exposure import TrainingResetReplay

    replay = TrainingResetReplay(method, 71)
    for row in reset_rows(method, 71, 30):
        assert replay.verify(row)


@pytest.mark.parametrize('change', ['method', 'seed', 'draw_index', 'mass', 'pose', 'order'])
def test_self_consistent_but_wrong_method_seed_or_reset_is_rejected(change):
    from adaptive_drone_patrol.payload_solver_exposure import TrainingResetReplay

    rows = list(reset_rows('fixed' if change == 'method' else 'randomized',
                           72 if change == 'seed' else 71, 2))
    row = copy.deepcopy(rows[0])
    if change == 'draw_index':
        row['draw_index'] += 1
    elif change == 'mass':
        row['payload']['mass_ratio'] = .2
    elif change == 'pose':
        row['initial']['swing_angle_deg'] += .01
    elif change == 'order':
        row = rows[1]
    with pytest.raises(ValueError, match='reset'):
        TrainingResetReplay('randomized', 71).verify(row)
