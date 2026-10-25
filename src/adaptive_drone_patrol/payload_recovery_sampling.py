"""Exact CPU sample plans and conservative accounting, without flight admission.

These declarations prove arithmetic feasibility only. They neither measure GPU
throughput nor prove effective geometry isolation, actual curriculum coverage,
state continuity, physical qualification or a supported masked PPO rollout.
"""

from dataclasses import dataclass, field
import math
import hashlib
import re
from collections.abc import Mapping

import numpy as np

from .payload_contract import Payload, SIZES_M


_MOTIONS = frozenset({'hover', 'short_transfer', 'long_transfer', 'reversal',
                      'vertical_xyz', 'repeated_targets'})
_DOWNSTREAM = frozenset({'validation', 'acceptance', 'recording', 'verification', 'reserve'})


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _seconds(value, name, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f'{name} must be a finite real number')
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


@dataclass(frozen=True)
class StageExposure:
    """Human-declared minimum exposure, not evidence of actual achievement.

    Steps are continuous 0.01-second control intervals within one episode.
    A completed sequence cannot straddle an episode reset. Requirements have no
    defaults: the caller must declare every stage's intended motion coverage.
    """

    name: str
    total_transitions: int
    min_continuous_steps_per_env: int
    min_completed_sequences_per_env: int
    sequence_steps: int
    required_motion_coverage: tuple[str, ...]

    def __post_init__(self):
        if type(self.name) is not str or not re.fullmatch(r'[a-z][a-z0-9_]*', self.name):
            raise ValueError('Invalid stage name')
        for name in ('total_transitions', 'min_continuous_steps_per_env',
                     'min_completed_sequences_per_env', 'sequence_steps'):
            _integer(getattr(self, name), name)
        coverage = self.required_motion_coverage
        if (type(coverage) is not tuple or not coverage or
                any(type(x) is not str or x not in _MOTIONS for x in coverage) or
                len(set(coverage)) != len(coverage)):
            raise ValueError('Each stage needs explicit unique declared motion coverage')


@dataclass(frozen=True)
class SamplePlan:
    """One method's full-rollout plan; partial rollouts are deliberately unsupported.

    Stage boundaries count total transitions. Rollout boundaries carry no reset
    semantics: physical episode/history continuity remains a runtime obligation.
    """

    n_envs: int
    rollout_steps: int
    total_transitions: int
    ppo_epochs: int
    minibatches: int
    stages: tuple[StageExposure, ...]
    seed: int = 7
    training_admitted: bool = field(default=False, init=False)

    def __post_init__(self):
        for name in ('n_envs', 'rollout_steps', 'total_transitions', 'ppo_epochs', 'minibatches'):
            _integer(getattr(self, name), name)
        if type(self.seed) is not int or self.seed != 7:
            raise ValueError('The common recovery plan requires seed7')
        if self.total_transitions % self.transitions_per_rollout:
            raise ValueError('S must be divisible by N*T; no silent rounding or partial rollout')
        if self.transitions_per_rollout % self.minibatches:
            raise ValueError('A full PPO rollout must divide exactly into minibatches')
        if (type(self.stages) is not tuple or not self.stages or
                any(type(stage) is not StageExposure for stage in self.stages)):
            raise ValueError('Explicit stage exposure declarations are required')
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError('Duplicate stage names')
        if sum(stage.total_transitions for stage in self.stages) != self.total_transitions:
            raise ValueError('All stage transitions must sum to S')
        covered = set()
        for stage in self.stages:
            if stage.total_transitions % self.n_envs:
                raise ValueError('Each stage must divide evenly across active environments')
            available = stage.total_transitions // self.n_envs
            if available < stage.min_continuous_steps_per_env:
                raise ValueError(f'{stage.name}: insufficient per-environment continuous exposure')
            if available < stage.sequence_steps * stage.min_completed_sequences_per_env:
                raise ValueError(f'{stage.name}: insufficient per-environment completed sequence exposure')
            covered.update(stage.required_motion_coverage)
        if covered != _MOTIONS:
            raise ValueError('Declared stage coverage must include every required motion')
        _seconds(self.per_environment_control_seconds, 'per-environment control seconds', positive=True)

    @property
    def transitions_per_rollout(self):
        return self.n_envs * self.rollout_steps

    @property
    def rollouts(self):
        return self.total_transitions // self.transitions_per_rollout

    @property
    def vector_steps(self):
        return self.rollout_steps * self.rollouts

    @property
    def per_environment_steps(self):
        return self.vector_steps

    @property
    def per_environment_control_seconds(self):
        try:
            return self.per_environment_steps * .01
        except OverflowError as exc:
            raise ValueError('Per-environment control duration must be finite') from exc

    @property
    def minibatch_size(self):
        return self.transitions_per_rollout // self.minibatches

    @property
    def optimizer_steps(self):
        return self.rollouts * self.ppo_epochs * self.minibatches


@dataclass(frozen=True)
class RolloutAccounting:
    generated_valid: int
    generated_invalid: int
    padded_slots: int
    optimizer_transitions: int
    quarantined: bool
    stop_required: bool
    invalid_slot_coordinates: tuple[tuple[int, int], ...]


def audit_rollout(plan, valid_mask, active_slots):
    """Audit one complete rollout; one invalid active sample quarantines it all.

    Padding is outside N, never a generated environment transition. This function
    does not implement masked PPO: its caller must independently prove bucket
    isolation and mask application. Preserve raw evidence before any reset.
    """
    if type(plan) is not SamplePlan:
        raise ValueError('Expected a recovery SamplePlan')
    valid, active = np.asarray(valid_mask), np.asarray(active_slots)
    if (valid.dtype != np.bool_ or active.dtype != np.bool_ or valid.ndim != 2 or
            active.ndim != 1 or valid.shape != (plan.rollout_steps, active.size) or
            int(active.sum()) != plan.n_envs):
        raise ValueError('Expected full-rollout strict boolean masks with exactly N active slots')
    selected = valid[:, active]
    good = int(selected.sum())
    bad = selected.size - good
    quarantined = bad > 0
    invalid = tuple((int(t), int(slot)) for t, slot in np.argwhere((~valid) & active[None, :]))
    return RolloutAccounting(good, bad, int(valid.size - selected.size),
                             0 if quarantined else good, quarantined, quarantined, invalid)


@dataclass(frozen=True)
class BudgetForecast:
    per_method_training_s: float
    pair_training_s: float
    downstream_s: float
    required_s: float
    remaining_after_s: float
    fits: bool
    evidence_kind: str = field(default='cpu-planning-estimate', init=False)
    training_admitted: bool = field(default=False, init=False)


def forecast_pair(randomized_plan, fixed_plan, *, evidence_enabled_transitions_per_s,
                  optimizer_step_s, per_method_setup_s, downstream_seconds, remaining_s):
    """Forecast both identical plans, keeping all downstream reserves explicit.

    Rate must describe the complete evidence-enabled simulation/inference/reset/
    logging path. Optimizer cost is additional, per actual minibatch update.
    Caller must retain the real timing evidence separately; a fitting estimate
    never admits learning. No GPU utilization discount or N multiplier applies.
    """
    if (type(randomized_plan) is not SamplePlan or type(fixed_plan) is not SamplePlan or
            randomized_plan != fixed_plan):
        raise ValueError('Both methods require an identical frozen sample/PPO/exposure plan')
    rate = _seconds(evidence_enabled_transitions_per_s, 'transition rate', positive=True)
    update = _seconds(optimizer_step_s, 'optimizer step seconds', positive=True)
    setup = _seconds(per_method_setup_s, 'per-method setup seconds', positive=True)
    remaining = _seconds(remaining_s, 'remaining seconds', positive=True)
    if not isinstance(downstream_seconds, Mapping) or set(downstream_seconds) != _DOWNSTREAM:
        raise ValueError('Every required downstream phase/reserve must be explicit')
    downstream = sum(_seconds(value, f'downstream {key}', positive=True)
                     for key, value in downstream_seconds.items())
    try:
        per_method = (randomized_plan.total_transitions / rate +
                      randomized_plan.optimizer_steps * update + setup)
        pair = 2 * per_method
        required = pair + downstream
    except OverflowError as exc:
        raise ValueError('Forecast overflow; finite phase/job limits required') from exc
    if not all(math.isfinite(x) for x in (per_method, pair, downstream, required)):
        raise ValueError('Forecast overflow; finite phase/job limits required')
    return BudgetForecast(per_method, pair, downstream, required, remaining - required, required <= remaining)


@dataclass(frozen=True)
class EpisodeDraw:
    """Private load/scheduling metadata. Never append this record to actor input."""

    logical_episode_id: int
    seed: int
    payload: Payload
    size_stream_sha256: str
    mass_stream_sha256: str

    def audit_record(self):
        return dict(schema='payload-recovery-episode-load-pcg64-v1',
                    logical_episode_id=self.logical_episode_id, seed=self.seed,
                    size_stream_sha256=self.size_stream_sha256,
                    mass_stream_sha256=self.mass_stream_sha256,
                    payload=dict(reference_mass_kg=self.payload.reference_mass_kg,
                                 mass_ratio=self.payload.mass_ratio, side_m=self.payload.side_m,
                                 mass_kg=self.payload.mass, inertia_kg_m2=self.payload.inertia))


@dataclass(frozen=True)
class EpisodeLoads:
    """Stateless seed7 load draws keyed by a stable logical episode identity.

    The scheduler must allocate identities independently of worker/completion
    order, e.g. from a frozen episode list. No worker, batch or stage enters this
    stream key. Separate SHA256 domains seed explicit NumPy PCG64 size and mass
    generators. Replaying an identity reproduces the same draw and never consumes
    another episode's state. This helper does not sample evaluation manifests or
    prove that any physical setter applied the draw.
    """

    reference_mass_kg: float
    seed: int = 7

    def __post_init__(self):
        _seconds(self.reference_mass_kg, 'reference mass', positive=True)
        if type(self.seed) is not int or self.seed != 7:
            raise ValueError('The recovery episode stream requires seed7')

    def draw(self, logical_episode_id, *, randomized):
        _integer(logical_episode_id, 'logical episode identity', minimum=0)
        if type(randomized) is not bool:
            raise ValueError('Randomization must be an explicit boolean method flag')
        streams = {}
        for name in ('size', 'mass'):
            key = f'payload-recovery-episode-load-pcg64-v1\0{self.seed}\0{logical_episode_id}\0{name}'
            streams[name] = hashlib.sha256(key.encode('ascii')).hexdigest()
        if randomized:
            size_rng = np.random.Generator(np.random.PCG64(int(streams['size'], 16)))
            mass_rng = np.random.Generator(np.random.PCG64(int(streams['mass'], 16)))
            side = SIZES_M[int(size_rng.integers(0, 3))]
            ratio = float(mass_rng.uniform(.1, .3))
        else:
            side, ratio = .045, .2
        return EpisodeDraw(logical_episode_id, self.seed,
                           Payload(self.reference_mass_kg, ratio, side),
                           streams['size'], streams['mass'])


@dataclass(frozen=True)
class EpisodeProgress:
    draw: EpisodeDraw
    control_steps: int = 0


class EpisodeSchedule:
    """CPU identity/cursor ledger, without any backend/body/action state.

    Reset is explicit and selective. Advancing across a rollout boundary never
    assigns another identity or changes load. Actual state/history/reset-force
    isolation needs separate backend evidence; these counters cannot provide it.
    """

    def __init__(self, reference_mass_kg, seed=7):
        self._loads = EpisodeLoads(reference_mass_kg, seed)
        self._current = {}
        self._used_ids = set()

    def reset(self, assignments, *, randomized):
        if type(assignments) is not dict or not assignments:
            raise ValueError('Explicit environment-to-logical-episode assignments required')
        if type(randomized) is not bool:
            raise ValueError('Randomization must be an explicit boolean method flag')
        for env_id, episode_id in assignments.items():
            _integer(env_id, 'environment identity', minimum=0)
            _integer(episode_id, 'logical episode identity', minimum=0)
        identities = list(assignments.values())
        if len(set(identities)) != len(identities) or self._used_ids.intersection(identities):
            raise ValueError('Logical episode identities must be unique and never reused on reset')
        updates = {env: EpisodeProgress(self._loads.draw(identity, randomized=randomized))
                   for env, identity in assignments.items()}
        self._current.update(updates)
        self._used_ids.update(identities)

    def advance(self, env_ids, *, intervals=1):
        _integer(intervals, 'continuous control intervals')
        if type(env_ids) not in (tuple, list) or not env_ids:
            raise ValueError('Explicit nonempty unique environment identities required')
        for env_id in env_ids:
            _integer(env_id, 'environment identity', minimum=0)
            if env_id not in self._current:
                raise ValueError('Cannot advance an unassigned environment')
        if len(set(env_ids)) != len(env_ids):
            raise ValueError('Cannot advance the same environment twice')
        for env_id in env_ids:
            current = self._current[env_id]
            self._current[env_id] = EpisodeProgress(current.draw, current.control_steps + intervals)

    def snapshot(self):
        return dict(self._current)
