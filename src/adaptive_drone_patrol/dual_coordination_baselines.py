"""Causal B-static and B-dynamic selectors from the accepted specification §7.

Selectors are pure: they never commit mission state, reset clocks, or provide
safety clearance. Dynamic conflict evidence must come from the caller's swept
reference/envelope predictor; there is deliberately no conflict-free default.

ETA interpretation: continuing an unchanged, matching commanded reference uses
its remaining duration. A new target (or a reference currently stopping short
of the owned goal) uses common rest-to-rest ``nominal_duration(current, goal)``.
Add the remaining service dwell only for the same owning drone/target; every new
assignment starts a full second. This avoids restarting a nominal full leg on
every policy tick. These causal estimates are not physical completion proofs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dual_coordination_reference import ContinuousReference, ReferenceLimits, nominal_duration
from .dual_coordination_schema import WAIT, decode_action, encode_action


def _coordinates(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Finite positions with the declared shape are required')
    return tuple(tuple(float(x) for x in row) for row in array)


@dataclass(frozen=True)
class ConflictPrediction:
    pair_conflict: bool = False
    unavailable_hover_conflict: tuple[bool, bool] = (False, False)

    def __post_init__(self):
        flags = tuple(self.unavailable_hover_conflict)
        if type(self.pair_conflict) is not bool or len(flags) != 2 or any(type(x) is not bool for x in flags):
            raise ValueError('Conflict evidence requires explicit boolean flags')
        object.__setattr__(self, 'unavailable_hover_conflict', flags)


@dataclass(frozen=True)
class ConflictRequest:
    """Only observed inputs; predictor constructs the requested swept paths.

    ``targets`` are 0..9 or WAIT. ``unavailable_hover_conflict[d]`` in the
    result means drone d's requested path conflicts with the *other* unavailable
    drone's hover. Existing references preserve current p/v/a for path prediction;
    current positions expose measured tracking displacement. No event schedule,
    true hidden masses, or nominal optimum is provided.
    """
    targets: tuple[int, int]
    references: tuple[ContinuousReference, ContinuousReference]
    current_positions: tuple[tuple[float, float, float], ...]
    goals: tuple[tuple[float, float, float], ...]
    time: float
    unavailable_remaining: tuple[float, float]


@dataclass(frozen=True)
class BaselineDecision:
    action: int
    targets: tuple[int, int]
    priority: int
    unimpeded_etas: tuple[float | None, float | None] = (None, None)
    adjusted_etas: tuple[float | None, float | None] = (None, None)
    ownership_changes: int = 0
    objective: float | None = None
    priority_weighted_eta: float | None = None


def _candidates(mission):
    observed = mission.observed_state()
    choices = [decode_action(int(index)) for index in np.flatnonzero(mission.legal_mask())]
    if not choices:
        raise ValueError('No legal action exists for this mission state')
    return observed, choices


def _ownership_changes(mission, targets):
    new_owners = {goal: drone for drone, goal in enumerate(targets) if goal != WAIT}
    # This matches mission.apply: initial assignments and DONE goals are not
    # ownership changes; release and transfer each count the affected goal once.
    return sum(goal.status != 'DONE' and goal.owner is not None
               and new_owners.get(index) != goal.owner for index, goal in enumerate(mission.goals))


@dataclass(frozen=True, init=False)
class StaticBaseline:
    starts: tuple[tuple[float, float, float], ...]
    goals: tuple[tuple[float, float, float], ...]
    partition: tuple[int, ...]

    def __init__(self, starts, goals):
        starts, goals = _coordinates(starts, (2, 3)), _coordinates(goals, (10, 3))
        distances = np.linalg.norm(np.asarray(goals)[:, None]-np.asarray(starts), axis=2)
        object.__setattr__(self, 'starts', starts)
        object.__setattr__(self, 'goals', goals)
        object.__setattr__(self, 'partition', tuple(int(d) for d in np.argmin(distances, axis=1)))

    @property
    def empty_partitions(self):
        return tuple(drone not in self.partition for drone in range(2))

    def select(self, mission, current_positions):
        positions = _coordinates(current_positions, (2, 3))
        _, choices = _candidates(mission)
        choices = [choice for choice in choices if all(
            target == WAIT or self.partition[target] == drone for drone, target in enumerate(choice[:2]))]
        if not choices:
            raise ValueError('Structural locks are incompatible with the fixed static partition')
        # Disjoint eligibility sets make per-drone nearest selection independent.
        # Maximizing work first permits WAIT only when its partition cannot work.
        def key(choice):
            distances = tuple(float(np.linalg.norm(np.subtract(self.goals[target], positions[d])))
                              if target != WAIT else float('inf') for d, target in enumerate(choice[:2]))
            return (-sum(t != WAIT for t in choice[:2]), *distances, *choice)
        target1, target2, priority = min(choices, key=key)
        return BaselineDecision(encode_action(target1, target2, priority), (target1, target2), priority,
                                ownership_changes=_ownership_changes(mission, (target1, target2)))


@dataclass(frozen=True, init=False)
class DynamicBaseline:
    goals: tuple[tuple[float, float, float], ...]
    limits: ReferenceLimits

    def __init__(self, goals, *, limits=ReferenceLimits()):
        object.__setattr__(self, 'goals', _coordinates(goals, (10, 3)))
        object.__setattr__(self, 'limits', limits)

    def select(self, mission, current_positions, *, references, conflict_predictor):
        positions = _coordinates(current_positions, (2, 3))
        references = tuple(references)
        if len(references) != 2 or not all(isinstance(r, ContinuousReference) for r in references):
            raise ValueError('Both current continuous references are required')
        if not callable(conflict_predictor):
            raise ValueError('An explicit swept-reference conflict predictor is required')
        observed, choices = _candidates(mission)
        time = observed['world_time']
        if any(reference.limits != self.limits or reference.start_time > time for reference in references):
            raise ValueError('References must use the common limits and already exist at the current time')
        remaining = tuple(ticks/100 for ticks in observed['remaining_unavailable_ticks'])
        work = max(sum(target != WAIT for target in choice[:2]) for choice in choices)
        choices = [choice for choice in choices if sum(target != WAIT for target in choice[:2]) == work]
        if work == 0:
            choice = min(choices)
            return BaselineDecision(encode_action(*choice), choice[:2], choice[2])
        pairs = sorted(set(choice[:2] for choice in choices))
        decisions = []
        for targets in pairs:
            etas = []
            for drone, target in enumerate(targets):
                if target == WAIT:
                    etas.append(None)
                    continue
                reference = references[drone]
                continuing = mission.targets[drone] == target
                transfer = (max(0., reference.end_time-time)
                            if continuing and reference.target == self.goals[target]
                            else nominal_duration(positions[drone], self.goals[target], self.limits))
                dwell = mission.goals[target].dwell_ticks/100 if continuing else 0.
                etas.append(transfer+max(0., 1.-dwell))
            active = [d for d, eta in enumerate(etas) if eta is not None]
            desired_priority = min(active, key=lambda d: (etas[d], d))+1
            legal_priorities = [p for t1, t2, p in choices if (t1, t2) == targets]
            priority = desired_priority if desired_priority in legal_priorities else min(legal_priorities)
            request = ConflictRequest(targets, references, positions, self.goals, time, remaining)
            prediction = conflict_predictor(request)
            if not isinstance(prediction, ConflictPrediction):
                raise ValueError('Conflict predictor must return explicit ConflictPrediction evidence')
            adjusted = etas.copy()
            if prediction.pair_conflict and len(active) == 2:
                adjusted[2-priority] += etas[priority-1]+.20
            for drone in active:
                other = 1-drone
                if prediction.unavailable_hover_conflict[drone] and not observed['available'][other]:
                    adjusted[drone] += remaining[other]
            changes = _ownership_changes(mission, targets)
            objective = max(adjusted[d] for d in active)+changes
            weighted = sum(mission.priorities[targets[d]]/100*adjusted[d] for d in active)
            decisions.append(BaselineDecision(encode_action(*targets, priority), targets, priority,
                                             tuple(etas), tuple(adjusted), changes, objective, weighted))
        return min(decisions, key=lambda d: (d.objective, d.priority_weighted_eta, *d.targets, d.priority))
