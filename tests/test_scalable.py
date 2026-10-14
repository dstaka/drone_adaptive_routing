"""Behavioral regressions for variable-size analytical routing and exact costs."""

from itertools import permutations
import copy
import numpy as np
import pytest
from adaptive_drone_patrol import baselines, objective
from adaptive_drone_patrol.scenario import Scenario, generate_scenario
from adaptive_drone_patrol.routing_env import RoutingEnv


def independent_cost(s, order, current=0, elapsed=0):
    travel = penalty = 0.
    for a in order:
        d = float(np.linalg.norm(np.array(s.coords[current]) - s.coords[a + 1]))
        elapsed += d / s.nominal_speed_mps + s.service_duration_s[a]
        travel += d * s.multipliers[current][a + 1]
        penalty += s.priorities[a] * elapsed
        current = a + 1
    if s.return_to_base and order:
        travel += np.linalg.norm(np.array(s.coords[current]) - s.coords[0]) * s.multipliers[current][0]
    return s.alpha * travel + s.beta * penalty


@pytest.mark.parametrize('n', [3, 10, 12])
def test_variable_episode_reward_mask_and_no_terminal_query(n):
    s = generate_scenario(20001, 'static', num_checkpoints=n)
    env = RoutingEnv(s)
    obs, _ = env.reset()
    assert obs.shape == (objective.schema_for(s)['size'],)
    rewards = []
    for a in range(n):
        rewards.append(env.step(a)[1])
        assert env.state.visited.sum() == a + 1
        assert not env.action_masks()[a]
    assert env.state.cost == pytest.approx(-sum(rewards))
    assert env.state.cost == pytest.approx(independent_cost(s, list(range(n))))
    with pytest.raises(RuntimeError, match='terminal'):
        env.step(0)


def test_new_schema_ids_events_and_legacy_normalization():
    s = generate_scenario(30001, 'dynamic', num_checkpoints=10)
    assert s.event['checkpoint'] == 'C'
    assert s.to_dict()['schema_version'] == 2
    for field, value in [('checkpoint_ids', ['A'] * 10), ('schema_version', 9),
                         ('event', {'kind': 'priority', 'after': 1, 'checkpoint': 2, 'value': 5})]:
        data = s.to_dict()
        data[field] = value
        with pytest.raises(ValueError):
            Scenario.from_dict(data)
    old = generate_scenario(20001, 'static').to_dict()
    old.pop('schema_version')
    old.pop('checkpoint_ids')
    assert Scenario.from_dict(old).checkpoint_ids == ['A', 'B', 'C']
    large = s.to_dict()
    large.pop('schema_version')
    with pytest.raises(ValueError):
        Scenario.from_dict(large)
    old['schema_version'] = None
    with pytest.raises(ValueError):
        Scenario.from_dict(old)


def test_future_events_hidden_boundary_snapshot_and_skip():
    s = generate_scenario(30001, 'dynamic', num_checkpoints=10)
    no = copy.deepcopy(s)
    no.event = None
    st, other = objective.RoutingState(s), objective.RoutingState(no)
    assert np.array_equal(st.observation(), other.observation())
    snap = st.departure(2)
    st.complete(2, snap)
    before = st.cost
    assert st.apply_event()['skipped']
    assert st.cost == before and st.legs[0] == snap
    assert st.apply_event() is None
    assert s.priorities == no.priorities


@pytest.mark.parametrize('n', range(3, 8))
@pytest.mark.parametrize('prefix', [False, True])
def test_exact_subset_dp_matches_independent_full_enumeration(n, prefix):
    s = generate_scenario(20000 + n, 'static', num_checkpoints=n)
    st = objective.RoutingState(s)
    if prefix:
        st.complete(0, st.departure(0))
    eligible = np.flatnonzero(st.mask()).tolist()
    result = baselines.exact_remaining(st)
    costs = [(independent_cost(s, order, st.current, st.elapsed), order)
             for order in permutations(eligible)]
    assert result['status'] == 'OPTIMAL'
    assert result['cost'] == pytest.approx(min(costs)[0])
    for a in eligible:
        forced = baselines.exact_remaining(st, first=a)
        assert forced['order'][0] == a
        assert forced['cost'] == pytest.approx(min(c for c, o in costs if o[0] == a))


def test_two_point_nearer_lower_priority_exact_fixture():
    s = Scenario('tradeoff', 'validation', 10001,
                 [[0, 0, 1], [.5, 0, 1], [2, 0, 1]], [1, 5],
                 np.ones((3, 3)).tolist(), [.5, .5],
                 schema_version=2, checkpoint_ids=['L', 'H'])
    st = objective.RoutingState(s)
    low, high = [baselines.exact_remaining(st, first=a) for a in [0, 1]]
    assert low['cost'] == pytest.approx(18.)
    assert low['travel_cost'] == pytest.approx(2.)
    assert low['completion_penalty'] == pytest.approx(16.)
    assert high['cost'] == pytest.approx(20.5)
    assert high['travel_cost'] == pytest.approx(3.5)
    assert high['completion_penalty'] == pytest.approx(17.)


@pytest.mark.parametrize('action', [1.5, True, '1', -1, 12])
def test_invalid_action_types_never_create_a_visit(action):
    env = RoutingEnv(generate_scenario(20001, 'static', num_checkpoints=10))
    env.reset()
    _, reward, _, _, info = env.step(action)
    assert env.state.visited.sum() == 0 and reward == 0 and info['invalid_action']


def test_exact_return_ties_and_nonzero_prefix_elapsed():
    s = Scenario('ties', 'validation', 10001,
                 [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]],
                 [2, 2, 2], np.ones((4, 4)).tolist(), [.5] * 3, return_to_base=True)
    st = objective.RoutingState(s)
    for first in [0, 1, 2]:
        result = baselines.exact_remaining(st, first=first)
        assert result['cost'] == pytest.approx(min(independent_cost(s, o) for o in permutations(range(3)) if o[0] == first))
    st.complete(0, st.departure(0))
    st.elapsed = 17.25
    result = baselines.exact_remaining(st)
    assert result['cost'] == pytest.approx(min(independent_cost(s, o, st.current, 17.25) for o in permutations([1, 2])))
