"""Versioned, validated routing scenarios with explicit legacy normalization."""

from dataclasses import asdict, dataclass
import copy
import numpy as np

IDS = ('A', 'B', 'C')  # Historical physical runner contract.
LEGACY_PROFILE = 'legacy-5-v1'
INTEGER_PROFILE = 'integer-100-v1'
PRIORITY_GENERATOR = 'numpy-PCG64-SeedSequence(seed,100)-integers-1-101-v1'


def draw_integer_priorities(seed, count):
    """Draw independently with replacement on a geometry-independent stream."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 100]))).integers(
        1, 101, count).tolist()


def checkpoint_ids(n):
    if type(n) is not int or not 1 <= n <= 26:
        raise ValueError('num_checkpoints must be an integer in [1,26]')
    return [chr(65 + i) for i in range(n)]


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
    schema_version: int | None = None
    checkpoint_ids: list | None = None

    def __post_init__(self):
        legacy = self.schema_version is None
        if legacy:
            if self.checkpoint_ids is not None or len(self.priorities) != 3:
                raise ValueError('Unversioned scenarios require exactly three checkpoints and no IDs')
            self.schema_version, self.checkpoint_ids = 2, list(IDS)
        if type(self.schema_version) is not int or self.schema_version not in (2, 3):
            raise ValueError('Unsupported scenario schema_version; expected 2 or 3')
        integer = self.schema_version == 3
        if integer and (not isinstance(self.priorities, list) or
                        any(type(p) is not int or not 1 <= p <= 100 for p in self.priorities)):
            raise ValueError('Integer priorities require integer tokens in [1,100]')
        ids = self.checkpoint_ids
        if (not isinstance(ids, list) or not ids or
                any(not isinstance(i, str) or not i.strip() or i != i.strip() or i == 'S' for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('checkpoint_ids must be unique nonempty strings different from S')
        self.checkpoint_ids = list(ids)
        n = len(ids)
        for name, shape in [('coords', (n + 1, 3)), ('priorities', (n,)),
                            ('multipliers', (n + 1, n + 1)), ('service_duration_s', (n,))]:
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'{name} must be finite with shape {shape}')
            if not (integer and name == 'priorities'):
                setattr(self, name, a.tolist())
        if len({tuple(p) for p in self.coords}) != n + 1:
            raise ValueError('Positions must be distinct')
        if not integer and not all(1 <= p <= 5 for p in self.priorities):
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
        if integer and (self.alpha != 1 or self.beta != .05 or self.nominal_speed_mps != 1
                        or self.return_to_base or any(t != .5 for t in self.service_duration_s)):
            raise ValueError('Integer profile requires alpha=1, beta=0.05, speed=1, service=0.5 and no return')
        if self.event is not None:
            e = copy.deepcopy(self.event)
            if (not isinstance(e, dict) or e.get('kind') not in ('priority', 'cost') or
                    type(e.get('after')) is not int or e['after'] != 1):
                raise ValueError('Events must be priority/cost updates after first completion')
            if integer and e['kind'] == 'priority':
                if type(e.get('value')) is not int or not 1 <= e['value'] <= 100:
                    raise ValueError('Priority event requires an integer token in [1,100]')
            elif type(e.get('value')) not in (int, float) or not np.isfinite(e['value']) or not 1 <= e['value'] <= 5:
                raise ValueError('Event value must lie in [1,5]')
            if type(e.get('sample_at_boundary', False)) is not bool:
                raise ValueError('sample_at_boundary must be boolean')
            if e['kind'] == 'priority':
                j = e.get('checkpoint')
                if legacy and type(j) is int and j in range(3):
                    j = ids[j]
                if not isinstance(j, str) or j not in ids:
                    raise ValueError('Event checkpoint must reference a checkpoint ID')
                e['checkpoint'] = j
            else:
                edges = e.get('edges')
                if not isinstance(edges, list) or not edges:
                    raise ValueError('Cost events require directed node-ID pairs')
                normalized = []
                for edge in edges:
                    if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                        raise ValueError('Invalid directed edge')
                    if legacy:
                        if any(type(i) is not int or i not in range(4) for i in edge):
                            raise ValueError('Legacy edge indices must be integers in [0,3]')
                        edge = [(['S'] + ids)[i] for i in edge]
                    if any(not isinstance(i, str) or i not in self.node_indices for i in edge) or edge[0] == edge[1]:
                        raise ValueError('Invalid directed node-ID pair')
                    normalized.append(list(edge))
                e['edges'] = normalized
            self.event = e

    @property
    def priority_profile(self):
        return INTEGER_PROFILE if self.schema_version == 3 else LEGACY_PROFILE

    @property
    def priority_max(self):
        return 100 if self.schema_version == 3 else 5

    @property
    def num_checkpoints(self):
        return len(self.checkpoint_ids)

    @property
    def node_indices(self):
        return {name: i for i, name in enumerate(['S'] + self.checkpoint_ids)}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if 'schema_version' in value and value['schema_version'] is None:
            raise ValueError('Explicit null schema_version is ambiguous; omit it only for legacy three-point input')
        return cls(**copy.deepcopy(value))

    def validate_physical(self, arrival_radius=.2):
        if not np.isfinite(arrival_radius) or arrival_radius <= 0:
            raise ValueError('Arrival radius must be positive and finite')
        points = np.asarray(self.coords)
        if any(np.linalg.norm(points[i] - points[j]) < 2 * arrival_radius + .10 - 1e-9
               for i in range(1, len(points)) for j in range(i + 1, len(points))):
            raise ValueError('Checkpoint centers violate minimum physical separation')
        if any(np.linalg.norm(points[0] - p) <= arrival_radius for p in points[1:]):
            raise ValueError('Initial pose lies within checkpoint arrival radius')


def generate_scenario(seed, split, kind=None, num_checkpoints=3, priority_profile=LEGACY_PROFILE):
    if priority_profile not in (LEGACY_PROFILE, INTEGER_PROFILE):
        raise ValueError('Unsupported priority profile')
    integer = priority_profile == INTEGER_PROFILE
    ranges = {'training': (0, 10000), 'validation': (10000, 20000),
              'static': (20000, 20100), 'dynamic': (30000, 30050), 'demo': (40000, 40100)}
    if split not in ranges or not ranges[split][0] <= seed < ranges[split][1]:
        raise ValueError(f'Seed {seed} is outside split {split}')
    ids = checkpoint_ids(num_checkpoints)
    n = len(ids)
    rng = np.random.default_rng(seed)
    coords = np.column_stack([rng.uniform(-1, 1, (n + 1, 2)), np.ones(n + 1)])
    coords[0] = [0, 0, 1]
    priorities = rng.integers(1, 6, n)
    multipliers = rng.uniform(1, 5, (n + 1, n + 1))
    np.fill_diagonal(multipliers, 1)
    if integer:
        priorities = draw_integer_priorities(seed, n)
    if kind is None:
        kind = ('priority' if seed % 2 else 'cost') if split == 'dynamic' else 'static'
    event = None
    if kind == 'priority':
        event = {'kind': 'priority', 'after': 1, 'checkpoint': 'C' if n >= 3 else ids[-1],
                 'value': 100 if integer else 5}
    elif kind == 'cost':
        target = ids[int(rng.integers(n))]
        event = {'kind': 'cost', 'after': 1, 'edges': [[i, target] for i in ['S'] + ids if i != target],
                 'value': 5}
    elif kind != 'static':
        raise ValueError(f'Unknown scenario kind {kind}')
    if event and split == 'training':
        event['sample_at_boundary'] = True
    return Scenario(f'{split}-{seed}', split, int(seed), coords.tolist(),
                    priorities if integer else priorities.tolist(), multipliers.tolist(), [.5] * n,
                    beta=.05 if integer else 1., event=event, schema_version=3 if integer else 2,
                    checkpoint_ids=ids)
