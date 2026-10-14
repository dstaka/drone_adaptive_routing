import copy
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from adaptive_drone_patrol.scenario import Scenario, generate_scenario
from adaptive_drone_patrol.objective import leg_cost, RoutingState
from adaptive_drone_patrol.routing_env import RoutingEnv
from adaptive_drone_patrol.baselines import exact_static, tour_cost


def fixture(event=None, return_to_base=False):
    return Scenario('fixture', 'validation', 10001,
                    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 0]],
                    [1, 5, 1], np.ones((4, 4)).tolist(), [0, 0, 0],
                    event=event, return_to_base=return_to_base)


def test_hand_computed_priority_completion_objective():
    s = fixture()
    a = leg_cost(s, 0, 1, 0)
    b = leg_cost(s, 1, 2, a['completion_time_s'])
    assert a['cost'] + b['cost'] == pytest.approx(7 + 6 * math.sqrt(2))
    b = leg_cost(s, 0, 2, 0)
    a = leg_cost(s, 2, 1, b['completion_time_s'])
    assert a['cost'] + b['cost'] == pytest.approx(7 + 2 * math.sqrt(2))


def test_invalid_actions_never_advance_and_terminal_mask():
    e = RoutingEnv(fixture())
    obs, info = e.reset(seed=4)
    assert obs.dtype == np.float32 and e.observation_space.contains(obs)
    rewards = [e.step(0)[1]]
    before = (e.state.elapsed, e.state.cost, e.state.current)
    for i in range(3):
        _, r, term, trunc, info = e.step(0)
        assert (e.state.elapsed, e.state.cost, e.state.current) == before
        assert r == 0 and info['invalid_action']
        assert trunc == (i == 2)
    e.reset(seed=4)
    rewards = [e.step(i)[1] for i in [0, 1, 2]]
    assert e.state.cost == pytest.approx(-sum(rewards))
    assert not e.action_masks().any()
    with pytest.raises(RuntimeError, match='terminal'):
        e.step(0)


def test_event_is_after_departure_accounting_and_future_is_hidden():
    event = {'kind': 'priority', 'after': 1, 'checkpoint': 1, 'value': 5}
    s = fixture(event)
    no_event = fixture()
    e = RoutingEnv(s)
    e.reset()
    n = RoutingEnv(no_event)
    n.reset()
    assert np.array_equal(e.state.observation(), n.state.observation())
    _, r, *_ = e.step(0)
    assert r == -2
    assert e.state.scenario.priorities[1] == 5
    assert s.priorities == [1, 5, 1]
    assert e.state.events[0]['boundary'] == 1


def test_static_oracle_lower_bound_and_single_return():
    for seed in range(20000, 20010):
        s = generate_scenario(seed, 'static')
        order, best = exact_static(s)
        assert sorted(order) == [0, 1, 2]
        for route in [[0, 1, 2], [2, 1, 0], [1, 0, 2]]:
            assert best <= tour_cost(s, route) + 1e-9
    e = RoutingEnv(fixture(return_to_base=True))
    e.reset()
    reward = sum(e.step(i)[1] for i in [0, 1, 2])
    assert e.state.cost == pytest.approx(-reward)
    assert e.state.return_accounted
    assert e.state.cost - tour_cost(fixture(), [0, 1, 2]) == pytest.approx(math.sqrt(8))


def test_seeded_generation_validation_and_checker():
    assert generate_scenario(30001, 'dynamic').to_dict() == generate_scenario(30001, 'dynamic').to_dict()
    s = fixture().to_dict()
    for key, value in [('priorities', [1, np.nan, 3]), ('nominal_speed_mps', 0),
                       ('coords', [[0, 0, 0]] * 4)]:
        bad = copy.deepcopy(s)
        bad[key] = value
        with pytest.raises(ValueError):
            Scenario.from_dict(bad)
    check_env(RoutingEnv(fixture()), skip_render_check=True)


def test_departure_snapshot_is_immutable():
    state = RoutingState(fixture())
    snap = state.departure(0)
    state.scenario.priorities[0] = 5
    assert snap['priority'] == 1
    state.complete(0, snap)
    assert state.cost == 2


def test_training_events_sample_only_at_boundary_from_observed_eligible_state():
    s = generate_scenario(73, 'training', 'priority')
    assert s.event['sample_at_boundary']
    a, b = RoutingState(s), RoutingState(s)
    assert np.array_equal(a.observation(), b.observation())
    for state in [a, b]:
        state.complete(0, state.departure(0))
        state.apply_event()
    assert a.events == b.events
    assert a.events[0]['checkpoint'] in [1, 2]
    assert 1 <= a.events[0]['value'] <= 5
