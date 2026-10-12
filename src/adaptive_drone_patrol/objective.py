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
        self.visited = np.zeros(3, dtype=bool)
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
        if self.terminal or action not in range(3) or self.visited[action]:
            raise ValueError('Ineligible action')
        return leg_cost(self.scenario, self.current, action + 1, self.elapsed)

    def complete(self, action, snapshot):
        if self.visited[action] or snapshot['from'] != self.current or snapshot['to'] != action + 1:
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
                event['checkpoint'] = target
            else:
                event['edges'] = [[self.current, target + 1]]
        record = copy.deepcopy(event)
        record.update(boundary=int(self.visited.sum()), skipped=False,
                      pre_state=self.observation().tolist())
        if event['kind'] == 'priority':
            j = event['checkpoint']
            record['old'] = self.scenario.priorities[j]
            if self.visited[j]:
                record['skipped'] = True
            else:
                self.scenario.priorities[j] = event['value']
        else:
            record['old'] = [self.scenario.multipliers[i][j] for i, j in event['edges']]
            for i, j in event['edges']:
                self.scenario.multipliers[i][j] = event['value']
        record['post_state'] = self.observation().tolist()
        self.events.append(record)
        return record

    def observation(self):
        s = self.scenario
        coords = np.asarray(s.coords)
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        return np.concatenate([np.eye(4)[self.current], self.visited, coords.ravel() / 2,
                               distances.ravel() / 4, np.asarray(s.priorities) / 5,
                               (distances * np.asarray(s.multipliers)).ravel() / 20,
                               s.service_duration_s, [self.elapsed / 30, self.visited.sum() / 3],
                               [s.alpha, s.beta, s.nominal_speed_mps, float(s.return_to_base),
                                float(np.asarray(s.service_duration_s).sum()), 1.]]).astype(np.float32)
