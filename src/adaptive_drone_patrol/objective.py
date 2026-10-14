"""Shared departure-snapshot costs, boundary events and observation encoding."""

import copy
import numpy as np
from .scenario import Scenario

SCHEMA = {'id': 'routing-v1-65', 'dtype': 'float32', 'size': 65,
          'fields': {'current_onehot': [0, 4], 'visited': [4, 7], 'coords': [7, 19],
                     'distance_matrix': [19, 35], 'priorities': [35, 38],
                     'edge_costs': [38, 54], 'service': [54, 57], 'elapsed': [57, 58],
                     'completed': [58, 59], 'constants': [59, 65]},
          'normalization': {'coordinates_m': 2., 'distance_m': 4., 'priority': 5.,
                            'edge_cost': 20., 'service_s': 1., 'time_s': 30.,
                            'count': 3., 'alpha': 1., 'beta': 1., 'speed_mps': 1.}}



def schema_for(scenario):
    """Generate offsets and a fixed per-model normalization/identity contract."""
    n = scenario.num_checkpoints
    widths = [('current_onehot', n + 1), ('visited', n), ('coords', 3 * (n + 1)),
              ('distance_matrix', (n + 1) ** 2), ('priorities', n),
              ('edge_costs', (n + 1) ** 2), ('service', n), ('elapsed', 1),
              ('completed', 1), ('constants', 6)]
    fields, offset = {}, 0
    for name, width in widths:
        fields[name] = [offset, offset + width]
        offset += width
    return {'id': f'routing-v2-n{n}-{offset}', 'dtype': 'float32', 'size': offset,
            'num_checkpoints': n, 'checkpoint_ids': scenario.checkpoint_ids.copy(),
            'fields': fields, 'normalization': {**SCHEMA['normalization'], 'count': float(n),
                'time_s': max(30., 8. * n)},
            'objective': {'alpha': scenario.alpha, 'beta': scenario.beta,
                          'nominal_speed_mps': scenario.nominal_speed_mps,
                          'return_to_base': scenario.return_to_base}}


def leg_cost(scenario, current, target, elapsed):
    distance = float(np.linalg.norm(np.asarray(scenario.coords[current]) - scenario.coords[target]))
    priority = float(scenario.priorities[target - 1])
    multiplier = float(scenario.multipliers[current][target])
    completion = elapsed + distance / scenario.nominal_speed_mps + scenario.service_duration_s[target - 1]
    travel = distance * multiplier
    penalty = priority * completion
    return {'from': current, 'to': target, 'distance_m': distance, 'priority': priority,
            'multiplier': multiplier, 'completion_time_s': completion,
            'travel_cost': travel, 'completion_penalty': penalty,
            'cost': scenario.alpha * travel + scenario.beta * penalty}


class RoutingState:
    def __init__(self, scenario):
        self.scenario = Scenario.from_dict(scenario.to_dict())
        self.current = 0
        self.visited = np.zeros(scenario.num_checkpoints, dtype=bool)
        self.elapsed = self.cost = self.penalty = self.distance = 0.
        self.invalid_count = 0
        self.events = []
        self.legs = []
        self.return_accounted = False

    def copy(self):
        return copy.deepcopy(self)

    @property
    def terminal(self):
        return bool(self.visited.all())

    def mask(self):
        return ~self.visited.copy()

    def departure(self, action):
        if self.terminal or type(action) not in (int, np.int64, np.int32) or action not in range(self.scenario.num_checkpoints) or self.visited[action]:
            raise ValueError('Ineligible action')
        return leg_cost(self.scenario, self.current, action + 1, self.elapsed)

    def complete(self, action, snapshot):
        if self.terminal or action not in range(self.scenario.num_checkpoints) or self.visited[action] or snapshot['from'] != self.current or snapshot['to'] != action + 1:
            raise ValueError('Departure snapshot does not match the current leg')
        self.visited[action] = True
        self.current = action + 1
        self.elapsed = snapshot['completion_time_s']
        self.cost += snapshot['cost']
        self.penalty += snapshot['completion_penalty']
        self.distance += snapshot['distance_m']
        self.legs.append(copy.deepcopy(snapshot))
        return_cost = 0.
        if self.terminal and self.scenario.return_to_base:
            d = float(np.linalg.norm(np.asarray(self.scenario.coords[self.current]) - self.scenario.coords[0]))
            return_cost = self.scenario.alpha * d * self.scenario.multipliers[self.current][0]
            self.cost += return_cost
            self.distance += d
            self.elapsed += d / self.scenario.nominal_speed_mps
            self.return_accounted = True
        return snapshot['cost'] + return_cost

    def apply_event(self):
        event = self.scenario.event
        if not event or int(self.visited.sum()) != event['after'] or self.events:
            return None
        event = copy.deepcopy(event)
        if event.get('sample_at_boundary'):
            rng = np.random.default_rng(self.scenario.seed + 100000 + self.current)
            eligible = np.flatnonzero(self.mask())
            target = int(rng.choice(eligible))
            event['value'] = int(rng.integers(1, 6))
            if event['kind'] == 'priority':
                event['checkpoint'] = self.scenario.checkpoint_ids[target]
            else:
                event['edges'] = [[(['S'] + self.scenario.checkpoint_ids)[self.current], self.scenario.checkpoint_ids[target]]]
        record = copy.deepcopy(event)
        record.update(boundary=int(self.visited.sum()), skipped=False,
                      pre_state=self.observation().tolist())
        if event['kind'] == 'priority':
            j = self.scenario.node_indices[event['checkpoint']] - 1
            record['checkpoint_id'] = event['checkpoint']
            record['checkpoint'] = j
            record['old'] = self.scenario.priorities[j]
            if self.visited[j]:
                record['skipped'] = True
            else:
                self.scenario.priorities[j] = event['value']
        else:
            edges = [[self.scenario.node_indices[i], self.scenario.node_indices[j]] for i, j in event['edges']]
            record['old'] = [self.scenario.multipliers[i][j] for i, j in edges]
            for i, j in edges:
                self.scenario.multipliers[i][j] = event['value']
        record['post_state'] = self.observation().tolist()
        self.events.append(record)
        return record

    def observation(self, legacy=False):
        s = self.scenario
        coords = np.asarray(s.coords)
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        n = s.num_checkpoints
        time_scale = 30. if legacy else schema_for(s)['normalization']['time_s']
        return np.concatenate([np.eye(n + 1)[self.current], self.visited, coords.ravel() / 2,
                               distances.ravel() / 4, np.asarray(s.priorities) / 5,
                               (distances * np.asarray(s.multipliers)).ravel() / 20,
                               s.service_duration_s, [self.elapsed / time_scale, self.visited.sum() / n],
                               [s.alpha, s.beta, s.nominal_speed_mps, float(s.return_to_base),
                                float(np.asarray(s.service_duration_s).sum()), 1.]]).astype(np.float32)
