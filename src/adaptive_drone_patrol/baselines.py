"""Fixed, greedy and exact static reference planners with shared accounting."""

from itertools import permutations
import numpy as np
from .objective import RoutingState


def tour_cost(scenario, order):
    state = RoutingState(scenario)
    for a in order:
        state.complete(a, state.departure(a))
    return state.cost


def exact_static(scenario):
    if scenario.event:
        raise ValueError('Exact enumeration is a static reference only')
    candidates = [(tour_cost(scenario, order), order) for order in permutations(range(3))]
    cost, order = min(candidates)
    return list(order), cost


def choose_baseline(state, name):
    eligible = np.flatnonzero(state.mask()).tolist()
    if not eligible:
        raise RuntimeError('No terminal policy query is allowed')
    if name == 'fixed':
        return eligible[0]
    if name == 'greedy':
        s = state.scenario
        return min(eligible, key=lambda a: (np.linalg.norm(np.asarray(s.coords[state.current]) - s.coords[a + 1])
                                           * s.multipliers[state.current][a + 1] / s.priorities[a], a))
    raise ValueError(f'Unknown baseline {name}')
