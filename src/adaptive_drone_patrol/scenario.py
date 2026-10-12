"""Validated three-checkpoint scenarios and disjoint seeded input generation."""

from dataclasses import asdict, dataclass
import copy
import numpy as np

IDS = ('A', 'B', 'C')


@dataclass
class Scenario:
    scenario_id: str
    split: str
    seed: int
    coords: list
    priorities: list
    multipliers: list
    service_duration_s: list
    alpha: float = 1.
    beta: float = 1.
    nominal_speed_mps: float = 1.
    return_to_base: bool = False
    event: dict | None = None

    def __post_init__(self):
        for name, shape in [('coords', (4, 3)), ('priorities', (3,)),
                            ('multipliers', (4, 4)), ('service_duration_s', (3,))]:
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'{name} must be finite with shape {shape}')
            setattr(self, name, a.tolist())
        if len({tuple(p) for p in self.coords}) != 4:
            raise ValueError('Positions must be distinct')
        if not all(1 <= p <= 5 for p in self.priorities):
            raise ValueError('Priorities must lie in [1,5]')
        if not np.all((np.asarray(self.multipliers) >= 1) & (np.asarray(self.multipliers) <= 5)):
            raise ValueError('Multipliers must lie in [1,5]')
        if not all(np.isfinite(x) and x >= 0 for x in [self.alpha, self.beta, *self.service_duration_s]):
            raise ValueError('Objective coefficients and service must be finite and nonnegative')
        if not np.isfinite(self.nominal_speed_mps) or self.nominal_speed_mps <= 0:
            raise ValueError('nominal_speed_mps must be positive and finite')
        if not isinstance(self.scenario_id, str) or not self.scenario_id or type(self.seed) is not int:
            raise ValueError('Scenario ID and integer seed are required')
        if type(self.return_to_base) is not bool:
            raise ValueError('return_to_base must be boolean')
        if self.event is not None:
            e = self.event
            if not isinstance(e, dict) or e.get('kind') not in ('priority', 'cost') or type(e.get('after')) is not int or e['after'] != 1:
                raise ValueError('Events must be priority/cost updates after first completion')
            if type(e.get('value')) not in (int, float) or not np.isfinite(e['value']) or not 1 <= e['value'] <= 5:
                raise ValueError('Event value must lie in [1,5]')
            if type(e.get('sample_at_boundary', False)) is not bool:
                raise ValueError('sample_at_boundary must be boolean')
            if e['kind'] == 'priority' and (type(e.get('checkpoint')) is not int or e['checkpoint'] not in range(3)):
                raise ValueError('Event checkpoint must be 0, 1 or 2')
            if e['kind'] == 'cost':
                if not isinstance(e.get('edges'), list) or not e['edges'] or any(not isinstance(edge, (list, tuple)) or len(edge) != 2 or any(type(n) is not int or n not in range(4) for n in edge)
                                           or edge[0] == edge[1] for edge in e['edges']):
                    raise ValueError('Cost events require valid directed edges')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**copy.deepcopy(value))

    def validate_physical(self, arrival_radius=.2):
        if not np.isfinite(arrival_radius) or arrival_radius <= 0:
            raise ValueError('Arrival radius must be positive and finite')
        points = np.asarray(self.coords)
        if any(np.linalg.norm(points[i] - points[j]) < 2 * arrival_radius + .10 - 1e-9
               for i in range(1, 4) for j in range(i + 1, 4)):
            raise ValueError('Checkpoint centers violate minimum physical separation')
        if any(np.linalg.norm(points[0] - p) <= arrival_radius for p in points[1:]):
            raise ValueError('Initial pose lies within checkpoint arrival radius')


def generate_scenario(seed, split, kind=None):
    ranges = {'training': (0, 10000), 'validation': (10000, 20000),
              'static': (20000, 20100), 'dynamic': (30000, 30050), 'demo': (40000, 40100)}
    if split not in ranges or not ranges[split][0] <= seed < ranges[split][1]:
        raise ValueError(f'Seed {seed} is outside split {split}')
    rng = np.random.default_rng(seed)
    coords = np.column_stack([rng.uniform(-1, 1, (4, 2)), np.ones(4)])
    coords[0] = [0, 0, 1]
    priorities = rng.integers(1, 6, 3)
    multipliers = rng.uniform(1, 5, (4, 4))
    np.fill_diagonal(multipliers, 1)
    if kind is None:
        kind = ('priority' if seed % 2 else 'cost') if split == 'dynamic' else 'static'
    event = None
    if kind == 'priority':
        event = {'kind': 'priority', 'after': 1, 'checkpoint': int(rng.integers(3)), 'value': 5}
    elif kind == 'cost':
        target = int(rng.integers(1, 4))
        event = {'kind': 'cost', 'after': 1, 'edges': [[i, target] for i in range(4) if i != target],
                 'value': 5}
    elif kind != 'static':
        raise ValueError(f'Unknown scenario kind {kind}')
    if event and split == 'training':
        event['sample_at_boundary'] = True
    return Scenario(f'{split}-{seed}', split, int(seed), coords.tolist(), priorities.tolist(),
                    multipliers.tolist(), [.5] * 3, event=event)
