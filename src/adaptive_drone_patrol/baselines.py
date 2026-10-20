"""Shared-objective baselines and exact subset dynamic programming."""

from functools import lru_cache
import time
import numpy as np
from .objective import RoutingState


def remaining_cost(state, order):
    """Replay a complete suffix without events, excluding incurred prefix cost."""
    st = state.copy()
    prefix_cost, prefix_penalty = st.cost, st.penalty
    prefix_travel = sum(leg['travel_cost'] for leg in st.legs)
    for a in order:
        st.complete(a, st.departure(a))
    if not st.terminal:
        raise ValueError('A remaining order must visit every eligible destination')
    travel = sum(leg['travel_cost'] for leg in st.legs) - prefix_travel
    if st.return_accounted and not state.return_accounted:
        s = st.scenario
        travel += float(np.linalg.norm(np.array(s.coords[st.current]) - s.coords[0])) * s.multipliers[st.current][0]
    result = {'order': list(order), 'cost': st.cost - prefix_cost, 'travel_cost': travel,
              'completion_penalty': st.penalty - prefix_penalty}
    if st.scenario.schema_version == 3:
        result.update(weighted_travel_cost=st.scenario.alpha * travel,
                      weighted_completion_penalty=st.scenario.beta * result['completion_penalty'])
    return result


def tour_cost(scenario, order):
    return remaining_cost(RoutingState(scenario), order)['cost']


def exact_remaining(state, first=None):
    """Prove the frozen-condition optimum in O(N^2 2^N), including absolute time.

    Every time increment delays all priorities still unvisited. Charging that
    increment times their sum makes subset/current sufficient; elapsed time is
    a common additive term. Directed logical costs never change travel time.
    """
    start = time.perf_counter()
    s = state.scenario
    eligible = np.flatnonzero(state.mask()).tolist()
    if first is not None and first not in eligible:
        raise ValueError('Forced first action is ineligible')
    if len(eligible) > 16:
        raise ValueError('Exact subset reference supports at most 16 remaining destinations')
    coords = np.asarray(s.coords)
    d = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    priorities = np.asarray(s.priorities)
    m = len(eligible)
    weights = [0.] * (1 << m)
    for mask in range(1, 1 << m):
        bit = mask & -mask
        weights[mask] = weights[mask ^ bit] + priorities[eligible[bit.bit_length() - 1]]

    @lru_cache(None)
    def solve(mask, current):
        if mask == 0:
            return (s.alpha * d[current, 0] * s.multipliers[current][0] if s.return_to_base else 0.), ()
        best = (float('inf'), ())
        for k, a in enumerate(eligible):
            if not mask & (1 << k):
                continue
            suffix, route = solve(mask ^ (1 << k), a + 1)
            increment = d[current, a + 1] / s.nominal_speed_mps + s.service_duration_s[a]
            cost = s.alpha * d[current, a + 1] * s.multipliers[current][a + 1] + s.beta * weights[mask] * increment + suffix
            best = min(best, (float(cost), (a,) + route))
        return best

    mask = (1 << m) - 1
    if not eligible:
        result = {'order': [], 'cost': 0., 'travel_cost': 0., 'completion_penalty': 0.}
    else:
        if first is None:
            _, route = solve(mask, state.current)
        else:
            k = eligible.index(first)
            _, suffix = solve(mask ^ (1 << k), first + 1)
            route = (first,) + suffix
        result = remaining_cost(state, route)
    return {**result, 'status': 'OPTIMAL', 'proven': True, 'solver': 'weighted-completion-subset-dp-v1',
            'elapsed_s': time.perf_counter() - start, 'states': solve.cache_info().currsize}


def exact_static(scenario):
    if scenario.event:
        raise ValueError('Exact static reference requires no events')
    result = exact_remaining(RoutingState(scenario))
    return result['order'], result['cost']


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
