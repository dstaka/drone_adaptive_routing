"""Frozen-condition analytical replay; these records never represent flight evidence."""

import hashlib
import json
import numpy as np
from .objective import RoutingState
from .scenario import Scenario, LEGACY_PROFILE, INTEGER_PROFILE, draw_integer_priorities
from .baselines import exact_remaining, remaining_cost


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def snapshot(state):
    data = state.scenario.to_dict()
    data['event'] = None  # Future event schedules are outside frozen-condition analysis.
    return {'scenario': data, 'current': int(state.current), 'visited': state.visited.tolist(),
            'elapsed': state.elapsed}


def restore(value):
    state = RoutingState(Scenario.from_dict(value['scenario']))
    n = state.scenario.num_checkpoints
    if (type(value['current']) is not int or value['current'] not in range(n + 1) or
            len(value['visited']) != n or any(type(v) is not bool for v in value['visited']) or
            not np.isfinite(value['elapsed']) or value['elapsed'] < 0):
        raise ValueError('Invalid boundary snapshot')
    state.current = value['current']
    state.visited = np.asarray(value['visited'], dtype=bool)
    state.elapsed = value['elapsed']
    if (state.current == 0 and state.visited.any()) or (state.current > 0 and not state.visited[state.current - 1]):
        raise ValueError('Boundary current node is inconsistent with visits')
    return state


def rollout(state, chooser):
    st, order = state.copy(), []
    st.scenario.event = None
    while not st.terminal:
        a = int(chooser(st))
        st.complete(a, st.departure(a))
        order.append(a)
    return order


def event_comparison(state, chooser):
    """Compare independent policy suffixes, repriced at the same post-event boundary."""
    before, after = state.copy(), state.copy()
    no_order = rollout(before, chooser)
    event = after.apply_event()
    new_order = rollout(after, chooser)
    retained, replanned = remaining_cost(after, no_order), remaining_cost(after, new_order)
    return {'boundary_id': f'{state.scenario.scenario_id}:boundary-{int(state.visited.sum())}',
            'boundary_hash_before': content_hash(snapshot(before)),
            'boundary_hash_after': content_hash(snapshot(after)), 'event': event,
            'applicable': bool(event and not event['skipped']),
            'no_event_order': no_order, 'post_event_order': new_order,
            'no_event_first': no_order[0] if no_order else None,
            'post_event_first': new_order[0] if new_order else None,
            'retained_order_post_event': retained, 'replanned_order_post_event': replanned,
            'improvement': retained['cost'] - replanned['cost'],
            'scope': 'Complete hypothetical suffixes under currently observed conditions; no further events.'}


def tolerance(cost):
    return max(1e-6, 1e-8 * abs(cost))


def optimality_label(cost, exact):
    if exact.get('status') != 'OPTIMAL' or not exact.get('proven'):
        return 'UNCONFIRMED'
    return 'OPTIMAL' if abs(cost - exact['cost']) <= tolerance(exact['cost']) else 'SUBOPTIMAL'


def explain_decision(state, chooser, policy_hash, run_id, decision_id):
    if state.terminal:
        raise ValueError('No terminal explanation query is allowed')
    frozen = snapshot(state)
    order = rollout(state, chooser)
    policy = remaining_cost(state, order)
    exact = exact_remaining(state)
    forced = [dict(action=int(a), **exact_remaining(state, first=int(a))) for a in np.flatnonzero(state.mask())]
    s, low = state.scenario, order[0]
    distances = np.linalg.norm(np.asarray(s.coords)[1:] - s.coords[state.current], axis=1)
    eligible = np.flatnonzero(state.mask()).tolist()
    maximum = max(s.priorities[a] for a in eligible)
    chosen_q = next(q for q in forced if q['action'] == low)
    high = next((a for a in sorted(eligible, key=lambda a: s.checkpoint_ids[a])
                 if s.priorities[a] == maximum and s.priorities[low] < maximum and
                 (s.schema_version != 3 or maximum - s.priorities[low] >= 10) and
                 distances[low] < distances[a] and
                 next(q for q in forced if q['action'] == a)['cost'] > exact['cost'] + tolerance(exact['cost'])), None)
    qualifies = (len(eligible) >= 3 and high is not None and
                 abs(chosen_q['cost'] - exact['cost']) <= tolerance(exact['cost']))
    return {'schema_version': 2, 'evidence_kind': 'analytical', 'run_id': run_id,
            'decision_id': decision_id, 'boundary_id': f'{run_id}:{decision_id}',
            'snapshot': frozen, 'boundary_hash': content_hash(frozen), 'policy_hash': policy_hash,
            'scope': 'Frozen currently observed conditions; remaining objective excludes incurred prefix cost; no future events or physical optimality claim.',
            'policy_action': low, 'policy_rollout': policy, 'exact': exact, 'forced_first': forced,
            'equality_tolerance': tolerance(exact['cost']), 'policy_gap': policy['cost'] - exact['cost'],
            'policy_optimality': optimality_label(policy['cost'], exact),
            'candidate_priorities_distances': [{'action': a, 'id': s.checkpoint_ids[a],
                  'priority': s.priorities[a], 'distance_m': float(distances[a])} for a in eligible],
            'selected_L': s.checkpoint_ids[low] if qualifies else None,
            'selected_H': s.checkpoint_ids[high] if qualifies else None,
            'nearer_lower_priority_gate': qualifies,
            'maximum_priority_ids': sorted(s.checkpoint_ids[a] for a in eligible if s.priorities[a] == maximum)}


def verify_explanation(value, policy_hash, run_id, decision_id):
    """Independently replay every cost and re-solve every claimed optimum."""
    def finite_tree(item):
        if isinstance(item, dict):
            return all(finite_tree(v) for v in item.values())
        if isinstance(item, list):
            return all(finite_tree(v) for v in item)
        return not isinstance(item, (float, int)) or np.isfinite(item)
    if not finite_tree(value):
        raise ValueError('Explanation numbers must be finite')
    if (value['evidence_kind'] != 'analytical' or value['schema_version'] != 2 or
            value['policy_hash'] != policy_hash or value['run_id'] != run_id or
            value['decision_id'] != decision_id or value['boundary_id'] != f'{run_id}:{decision_id}' or
            value['boundary_hash'] != content_hash(value['snapshot'])):
        raise ValueError('Explanation provenance mismatch')
    st = restore(value['snapshot'])
    exact = exact_remaining(st)
    # Derive the declared tolerance from the recorded J*: removing a prefix
    # accumulator can change a recomputed suffix by floating-point roundoff.
    tol = tolerance(value['exact']['cost'])
    if value['equality_tolerance'] != tol:
        raise ValueError('Explanation tolerance mismatch')
    if len(value['forced_first']) != int(st.mask().sum()):
        raise ValueError('Missing forced-first alternatives')
    seen = set()
    for record in [value['policy_rollout'], value['exact'], *value['forced_first']]:
        independent = remaining_cost(st, record['order'])
        components = ['cost', 'travel_cost', 'completion_penalty']
        if st.scenario.schema_version == 3:
            components += ['weighted_travel_cost', 'weighted_completion_penalty']
        if any(key not in record or abs(record[key] - independent[key]) > tol for key in components):
            raise ValueError('Explanation component cost mismatch')
        if record is value['policy_rollout']:
            continue
        first = record.get('action')
        reference = exact_remaining(st, first=first) if first is not None else exact
        if first is not None:
            if first in seen or record['order'][0] != first:
                raise ValueError('Invalid forced-first coverage')
            seen.add(first)
        if record['status'] != 'OPTIMAL' or not record['proven'] or abs(record['cost'] - reference['cost']) > tol:
            raise ValueError('Unproved or incorrect optimum')
    if value['policy_rollout']['order'][0] != value['policy_action']:
        raise ValueError('Policy action/rollout mismatch')
    if abs(value['policy_gap'] - (value['policy_rollout']['cost'] - exact['cost'])) > tol:
        raise ValueError('Policy gap mismatch')
    if value['policy_optimality'] != optimality_label(value['policy_rollout']['cost'], value['exact']):
        raise ValueError('Unsupported optimality label')
    s = st.scenario
    distances = np.linalg.norm(np.asarray(s.coords)[1:] - s.coords[st.current], axis=1)
    eligible = np.flatnonzero(st.mask()).tolist()
    candidates = [{'action': a, 'id': s.checkpoint_ids[a], 'priority': s.priorities[a],
                   'distance_m': float(distances[a])} for a in eligible]
    if value['candidate_priorities_distances'] != candidates:
        raise ValueError('Candidate annotation mismatch')
    low = value['policy_action']
    maximum = max(s.priorities[a] for a in eligible)
    chosen_cost = next(q['cost'] for q in value['forced_first'] if q['action'] == low)
    if s.schema_version == 3 and value.get('maximum_priority_ids') != sorted(
            s.checkpoint_ids[a] for a in eligible if s.priorities[a] == maximum):
        raise ValueError('Maximum-priority tie metadata mismatch')
    high = next((a for a in sorted(eligible, key=lambda a: s.checkpoint_ids[a])
                 if s.priorities[a] == maximum and s.priorities[low] < maximum and
                 (s.schema_version != 3 or maximum - s.priorities[low] >= 10) and
                 distances[low] < distances[a] and
                 next(q['cost'] for q in value['forced_first'] if q['action'] == a) > exact['cost'] + tol), None)
    qualifies = bool(len(eligible) >= 3 and high is not None and abs(chosen_cost - exact['cost']) <= tol)
    if (value['nearer_lower_priority_gate'] != qualifies or
            value['selected_L'] != (s.checkpoint_ids[low] if qualifies else None) or
            value['selected_H'] != (s.checkpoint_ids[high] if qualifies else None)):
        raise ValueError('Unsupported nearer/lower-priority explanation')
    return True


def demo_fixture(seed, priority_profile=LEGACY_PROFILE):
    if priority_profile not in (LEGACY_PROFILE, INTEGER_PROFILE):
        raise ValueError('Unsupported priority profile')
    integer = priority_profile == INTEGER_PROFILE
    rng = np.random.default_rng(seed)
    points = [[x, y, 1.] for y in [-.6, 0., .6] for x in [-.9, -.3, .3, .9]
              if not (y == 0 and abs(x) == .3)]
    return Scenario('normal', 'demo', seed, [[0., 0., 1.]] + points,
                    draw_integer_priorities(seed, 10) if integer else rng.uniform(2., 3., 10).tolist(),
                    np.ones((11, 11)).tolist(), [.5] * 10, beta=.05 if integer else 1.,
                    schema_version=3 if integer else 2, checkpoint_ids=list('ABCDEFGHIJ'))
