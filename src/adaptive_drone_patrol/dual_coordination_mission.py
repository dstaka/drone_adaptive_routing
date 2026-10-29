"""Pure causal mission semantics; this module does not simulate or certify flight.

Time is integer external 100 Hz ticks. Call ``apply(action, 0)`` after reset,
then ``advance`` exactly once per audited physics tick, and ``apply`` at most
once on each 20-tick decision boundary. Pass only events that trigger on that
advance; no future schedule is stored. Completion precedes events, which precede
the boundary's policy action. Boolean settling inputs are endpoint predicates. A dwell interval counts only
when both consecutive endpoints pass. Optional ``seed_settling`` supplies the
reset endpoint after the initial action. Inputs must come from the actual
physical predicates; ``settling_predicate`` implements the inspection thresholds.

Drone indices in state are 0/1; event and action priority IDs are public 1/2.
WAIT is 10. Both active targets permit both priorities; one active target makes
its drone the canonical priority, and both WAIT canonically use priority 1.
Every receipt has additive reward components. Sum advance and apply receipts,
or difference cumulative ``reward_components``, without counting both methods.
Snapshots include all mission state, but physical/controller/RNG state must be
saved separately by the environment for a complete resumable checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import json

import numpy as np

from .dual_coordination_schema import WAIT, decode_action

TICKS_PER_SECOND = 100
DECISION_TICKS = 20
SERVICE_TICKS = 100
COMMITMENT_TICKS = 100
LEG_DEADLINE_TICKS = 3000
TEAM_DEADLINE_TICKS = 30000
REWARD_KEYS = ('time', 'priority', 'supervisor', 'ownership', 'completion', 'failure')


def settling_predicate(*, drone_position_error, package_position_error, drone_speed,
                       package_speed, swing_angle_deg, swing_rate_deg_s, box_angular_speed_deg_s):
    """Inclusive payload-settling-video-v1 thresholds; inputs are nonnegative norms."""
    values = (drone_position_error, package_position_error, drone_speed, package_speed,
              swing_angle_deg, swing_rate_deg_s, box_angular_speed_deg_s)
    limits = (.20, .20, .30, .15, 5., 15., 30.)
    return all(math.isfinite(v) and 0 <= v <= limit for v, limit in zip(values, limits))


@dataclass
class GoalState:
    status: str = 'UNASSIGNED'
    owner: int | None = None
    first_owner: int | None = None
    dwell_ticks: int = 0
    completed_tick: int | None = None


class MissionState:
    """Atomic ownership and clocks with no reference, geometry, or policy heuristic.

    ``held`` accounts supervisor-held drone time. ``safety_interrupted`` breaks
    the service lock/dwell while retaining ownership. Ordinary failed settling
    resets dwell but retains a service lock. ``final_settling`` is evaluated at
    each drone's *current* hover reference; its dwell resets on target changes.
    ``no_transfer`` reserves a goal for its first owner even while that drone
    commands WAIT or becomes unavailable; it never resets that drone's leg.
    """

    def __init__(self, priorities=None, *, no_transfer=False):
        priorities = [1]*10 if priorities is None else list(priorities)
        if len(priorities) != 10 or any(type(p) is not int or not 1 <= p <= 100 for p in priorities):
            raise ValueError('Ten priorities in 1..100 required')
        if type(no_transfer) is not bool:
            raise ValueError('no_transfer must be boolean')
        self.no_transfer = no_transfer
        self.tick = 0
        self.goals = [GoalState() for _ in range(10)]
        self.targets = (WAIT, WAIT)
        self.priority = 1
        self.priorities = priorities
        self.available = [True, True]
        self.unavailable_until = [None, None]
        self.commitment_start_ticks = [None, None]
        self.leg_start_ticks = [None, None]
        self.final_dwell_ticks = [0, 0]
        self._previous_settling = [False, False]
        self._previous_final_settling = [False, False]
        self.completed_counts = [0, 0]
        self.safety_held = [False, False]
        self.last_action = None
        self.last_action_tick = None
        self.terminal_reason = None
        self.team_completion_tick = None
        self.last_goal_completion_tick = None
        self.event_receipts = []
        self.action_receipts = []
        self.reward_components = {key: 0. for key in REWARD_KEYS}
        self._event_override_tick = None

    def _receipt(self, components=None, **fields):
        reward = {key: 0. for key in REWARD_KEYS}
        reward.update(components or {})
        for key, value in reward.items():
            self.reward_components[key] += value
        return dict(tick=self.tick, reward_components=reward, reward=sum(reward.values()), **fields)

    def _legal(self, targets, priority):
        if self.terminal_reason is not None:
            return False
        if targets[0] == targets[1] != WAIT:
            return False
        active = [d for d, target in enumerate(targets) if target != WAIT]
        canonical = active[0]+1 if len(active) == 1 else 1
        if len(active) < 2 and priority != canonical:
            return False
        for drone, (old, new) in enumerate(zip(self.targets, targets)):
            if not self.available[drone] and new != WAIT:
                return False
            if new != WAIT:
                goal = self.goals[new]
                if goal.status == 'DONE':
                    return False
                if self.no_transfer and goal.first_owner not in (None, drone):
                    return False
            if old != WAIT and old != new:
                goal = self.goals[old]
                if goal.status == 'SERVICING':
                    return False
                start = self.commitment_start_ticks[drone]
                if (start is not None and self.tick-start < COMMITMENT_TICKS
                        and self._event_override_tick != self.tick):
                    return False
        return True

    def legal_mask(self):
        """Structural alternatives only, never filtered by distance or conflict."""
        return np.fromiter((self._legal(decoded[:2], decoded[2])
                            for decoded in map(decode_action, range(242))), dtype=bool, count=242)

    def apply(self, action, tick):
        """Commit one legal joint action atomically; invalid actions mutate nothing."""
        target1, target2, priority = decode_action(action)
        targets = (target1, target2)
        if (type(tick) is not int or tick != self.tick or tick % DECISION_TICKS
                or self.last_action_tick == tick or not self._legal(targets, priority)):
            raise ValueError('Action is illegal or not a fresh current decision boundary')
        previous = [goal.owner for goal in self.goals]
        for drone, old in enumerate(self.targets):
            if old != WAIT and old != targets[drone]:
                goal = self.goals[old]
                goal.dwell_ticks = 0
                goal.status = 'ASSIGNED' if self.no_transfer else 'UNASSIGNED'
                if not self.no_transfer:
                    goal.owner = None
        for drone, target in enumerate(targets):
            if target != self.targets[drone]:
                self.final_dwell_ticks[drone] = 0
                self._previous_settling[drone] = False
                self._previous_final_settling[drone] = False
                self.commitment_start_ticks[drone] = tick if target != WAIT else None
            if target != WAIT:
                goal = self.goals[target]
                goal.owner = drone
                if goal.first_owner is None:
                    goal.first_owner = drone
                if goal.status != 'SERVICING':
                    goal.status = 'ASSIGNED'
                if self.leg_start_ticks[drone] is None:
                    self.leg_start_ticks[drone] = tick
        changes = sum(old is not None and old != goal.owner
                      for old, goal in zip(previous, self.goals))
        self.targets = targets
        self.priority = priority
        self.last_action = action
        self.last_action_tick = tick
        self._event_override_tick = None
        for receipt in self.event_receipts:
            if receipt['observed_tick'] is not None and receipt['action_tick'] is None:
                receipt['action_tick'] = tick
        receipt = self._receipt({'ownership': -.05*changes}, action=action,
                                targets=list(targets), priority=priority,
                                owners=[goal.owner for goal in self.goals], ownership_changes=changes)
        self.action_receipts.append(deepcopy(receipt))
        return receipt

    def seed_settling(self, *, settling=(False, False), final_settling=(False, False)):
        """Seed audited reset endpoint after apply(action,0); no dwell is credited."""
        if self.tick != 0:
            raise ValueError('Only the reset endpoint may be seeded')
        settling = self._flags(settling, 'settling')
        final_settling = self._flags(final_settling, 'final_settling')
        self._previous_settling = [bool(valid) and target != WAIT
                                  for valid,target in zip(settling,self.targets)]
        self._previous_final_settling = list(map(bool,final_settling))
        for drone, valid in enumerate(self._previous_settling):
            if valid:
                self.goals[self.targets[drone]].status = 'SERVICING'

    @staticmethod
    def _flags(values, name):
        values = tuple(values)
        if len(values) != 2 or any(type(value) not in (bool, np.bool_) for value in values):
            raise ValueError(f'{name} requires two booleans')
        return values

    def _validate_event(self, event, tick):
        if event is None:
            return
        if not isinstance(event, dict) or event.get('tick') != tick:
            raise ValueError('Only an event triggering on this tick may be supplied')
        kind = event.get('kind')
        if kind == 'priority':
            valid = (set(event) == {'kind','tick','goal','priority'}
                     and type(event['goal']) is int and 0 <= event['goal'] < 10
                     and event['priority'] == 100)
        elif kind == 'availability-delay':
            valid = (set(event) == {'kind','tick','drone_id','duration_ticks'}
                     and type(event['drone_id']) is int and event['drone_id'] in (1,2)
                     and type(event['duration_ticks']) is int and event['duration_ticks'] == 800)
        else:
            valid = False
        if not valid:
            raise ValueError('Malformed observed event')

    def advance(self, tick, *, settling=(False, False), final_settling=(False, False),
                held=(False, False), safety_interrupted=(False, False), event=None, failure_reason=None):
        """Audit one elapsed 0.01 s interval, complete service, then trigger events.

        At the end of each 20th tick, triggered events become policy-observable.
        Their priorities affect reward from the *following* interval. External
        failures take precedence over successful completion. A completion exactly
        at a deadline satisfies that deadline, provided all success gates hold.
        """
        if type(tick) is not int or tick != self.tick+1 or self.terminal_reason is not None:
            raise ValueError('Advance requires the next live 100 Hz tick')
        settling = self._flags(settling, 'settling')
        final_settling = self._flags(final_settling, 'final_settling')
        held = self._flags(held, 'held')
        safety_interrupted = self._flags(safety_interrupted, 'safety_interrupted')
        self._validate_event(event, tick)
        if failure_reason is not None and (not isinstance(failure_reason,str) or not failure_reason):
            raise ValueError('Failure reason must be a nonempty string')
        components = {'time': -.01,
                      'priority': -.02*.01*sum(p/100 for p,g in zip(self.priorities,self.goals)
                                               if g.status != 'DONE'),
                      'supervisor': -.10*.01*sum(held)}
        self.tick = tick
        self.safety_held = list(map(bool,held))
        completed = []
        targets = list(self.targets)
        for drone, target in enumerate(targets):
            final_valid = bool(final_settling[drone]) and not safety_interrupted[drone]
            if final_valid and self._previous_final_settling[drone]:
                self.final_dwell_ticks[drone] = min(SERVICE_TICKS,self.final_dwell_ticks[drone]+1)
            else:
                self.final_dwell_ticks[drone] = 0
            self._previous_final_settling[drone] = bool(final_valid)
            previous_valid = self._previous_settling[drone]
            self._previous_settling[drone] = False
            if target == WAIT:
                continue
            goal = self.goals[target]
            if safety_interrupted[drone]:
                goal.status = 'ASSIGNED'
                goal.dwell_ticks = 0
            elif settling[drone] and self.available[drone] and failure_reason is None:
                goal.status = 'SERVICING'
                goal.dwell_ticks = goal.dwell_ticks+1 if previous_valid else 0
                self._previous_settling[drone] = True
                if goal.dwell_ticks == SERVICE_TICKS:
                    goal.status = 'DONE'
                    goal.completed_tick = tick
                    completed.append(target)
                    self.completed_counts[drone] += 1
                    self.leg_start_ticks[drone] = None
                    self.commitment_start_ticks[drone] = None
                    targets[drone] = WAIT
            else:
                goal.dwell_ticks = 0
        self.targets = tuple(targets)
        if completed:
            self.last_goal_completion_tick = tick
        components['completion'] = 5*len(completed)
        for drone, until in enumerate(self.unavailable_until):
            if until is not None and tick >= until:
                self.available[drone] = True
                self.unavailable_until[drone] = None
        event_changes = 0
        if event is not None:
            receipt = dict(event=deepcopy(event), trigger_tick=tick, observed_tick=None,
                           action_tick=None, no_op=False)
            if event['kind'] == 'priority':
                goal = self.goals[event['goal']]
                receipt['no_op'] = goal.status == 'DONE' or self.priorities[event['goal']] == 100
            else:
                drone = event['drone_id']-1
                self.available[drone] = False
                self.unavailable_until[drone] = tick+event['duration_ticks']
                target = self.targets[drone]
                if target != WAIT:
                    goal = self.goals[target]
                    goal.dwell_ticks = 0
                    goal.status = 'ASSIGNED' if self.no_transfer else 'UNASSIGNED'
                    if not self.no_transfer:
                        goal.owner = None
                        event_changes = 1
                targets = list(self.targets)
                targets[drone] = WAIT
                self.targets = tuple(targets)
                self.commitment_start_ticks[drone] = None
                self.final_dwell_ticks[drone] = 0
                self._previous_settling[drone] = False
                self._previous_final_settling[drone] = False
            self.event_receipts.append(receipt)
        components['ownership'] = -.05*event_changes
        if tick % DECISION_TICKS == 0:
            for receipt in self.event_receipts:
                if receipt['observed_tick'] is None:
                    receipt['observed_tick'] = tick
                    observed_event = receipt['event']
                    if observed_event['kind'] == 'priority' and not receipt['no_op']:
                        self.priorities[observed_event['goal']] = observed_event['priority']
                    self._event_override_tick = tick
        reason = failure_reason
        # Completion above ends only the completing drone's leg. Another drone's
        # expired persistent leg remains a failure even when all goals are DONE.
        if reason is None:
            expired = next((d for d,start in enumerate(self.leg_start_ticks)
                            if start is not None and tick-start >= LEG_DEADLINE_TICKS),None)
            if expired is not None:
                reason = f'leg_timeout:{expired+1}'
        if (reason is None and tick <= TEAM_DEADLINE_TICKS
                and all(g.status == 'DONE' for g in self.goals) and min(self.final_dwell_ticks) >= 100):
            reason = 'success'
            self.team_completion_tick = tick
        if reason is None and tick >= TEAM_DEADLINE_TICKS:
            reason = 'team_timeout'
        if reason is not None:
            self.terminal_reason = reason
            components['failure'] = 0 if reason == 'success' else -1000
        return self._receipt(components, newly_completed=completed, ownership_changes=event_changes,
                             terminal_reason=reason)

    def observed_state(self):
        """Causal policy inputs at a decision boundary, with no future event fields."""
        if self.tick % DECISION_TICKS:
            raise ValueError('Policy observation is only exposed at 5 Hz boundaries')
        return deepcopy(dict(tick=self.tick, world_time=self.tick/100, targets=list(self.targets),
                             priority=self.priority, priorities=self.priorities,
                             goals=[asdict(goal) for goal in self.goals], available=self.available,
                             remaining_unavailable_ticks=[max(0,until-self.tick) if until else 0
                                                          for until in self.unavailable_until],
                             commitment_ages=[self.tick-start if start is not None else None
                                              for start in self.commitment_start_ticks],
                             leg_ages=[self.tick-start if start is not None else None
                                       for start in self.leg_start_ticks],
                             final_dwell_ticks=self.final_dwell_ticks, last_action=self.last_action,
                             safety_held=self.safety_held,
                             events=[r for r in self.event_receipts if r['observed_tick'] is not None]))

    def to_dict(self):
        """JSON-safe, complete semantic checkpoint; this is not a policy observation."""
        state = deepcopy(vars(self))
        state['goals'] = [asdict(goal) for goal in self.goals]
        state['targets'] = list(self.targets)
        return dict(schema='dual-mission-state-v1', state=state)

    @classmethod
    def from_dict(cls, snapshot):
        """Restore the exact semantic state, including receipts and pending events."""
        if not isinstance(snapshot,dict) or set(snapshot) != {'schema','state'} or snapshot['schema'] != 'dual-mission-state-v1':
            raise ValueError('Exact dual-mission-state-v1 snapshot required')
        try:
            # Checkpoints must be portable finite JSON, not arbitrary Python state.
            json.dumps(snapshot, allow_nan=False)
            state = deepcopy(snapshot['state'])
            mission = cls(state['priorities'], no_transfer=state['no_transfer'])
            if set(state) != set(vars(mission)) or len(state['goals']) != 10:
                raise ValueError('Incomplete mission snapshot')
            tick = state['tick']
            if type(tick) is not int or tick < 0:
                raise ValueError('Invalid checkpoint tick')
            for key in ('available','unavailable_until','commitment_start_ticks','leg_start_ticks',
                        'final_dwell_ticks','completed_counts','safety_held','targets',
                        '_previous_settling','_previous_final_settling'):
                if not isinstance(state[key],list) or len(state[key]) != 2:
                    raise ValueError('Two-drone checkpoint field required')
            for key in ('available','safety_held','_previous_settling','_previous_final_settling'):
                if any(type(value) is not bool for value in state[key]):
                    raise ValueError('Boolean checkpoint flags required')
            for key in ('commitment_start_ticks','leg_start_ticks'):
                if any(value is not None and (type(value) is not int or not 0 <= value <= tick)
                       for value in state[key]):
                    raise ValueError('Invalid checkpoint clock')
            for key in ('final_dwell_ticks','completed_counts'):
                limit = 100 if key == 'final_dwell_ticks' else 10
                if any(type(value) is not int or not 0 <= value <= limit for value in state[key]):
                    raise ValueError('Invalid checkpoint counter')
            targets = tuple(state['targets'])
            if (any(type(target) is not int or not 0 <= target <= WAIT for target in targets)
                    or targets[0] == targets[1] != WAIT):
                raise ValueError('Invalid checkpoint targets')
            goals = [GoalState(**goal) for goal in state['goals']]
            counts = [0,0]
            for goal_id, goal in enumerate(goals):
                if (goal.status not in ('UNASSIGNED','ASSIGNED','SERVICING','DONE')
                        or goal.owner not in (None,0,1) or goal.first_owner not in (None,0,1)
                        or type(goal.dwell_ticks) is not int or not 0 <= goal.dwell_ticks <= 100):
                    raise ValueError('Invalid checkpoint goal')
                if (goal.status == 'UNASSIGNED') != (goal.owner is None):
                    raise ValueError('Checkpoint owner/status mismatch')
                if goal.status == 'DONE':
                    if (type(goal.completed_tick) is not int or not 0 <= goal.completed_tick <= tick
                            or goal.dwell_ticks != 100 or goal_id in targets):
                        raise ValueError('Invalid checkpoint completion')
                    counts[goal.owner] += 1
                elif goal.completed_tick is not None:
                    raise ValueError('Unfinished goal has completion time')
                if goal.status == 'SERVICING' and targets[goal.owner] != goal_id:
                    raise ValueError('Service owner is not targeting the goal')
            for drone,target in enumerate(targets):
                if target != WAIT:
                    if goals[target].owner != drone or not state['available'][drone]:
                        raise ValueError('Target owner or availability mismatch')
                    commitment = state['commitment_start_ticks'][drone]
                    leg = state['leg_start_ticks'][drone]
                    if commitment is None or leg is None or leg > commitment:
                        raise ValueError('Active assignment requires its original leg and commitment clocks')
            if counts != state['completed_counts'] or set(state['reward_components']) != set(REWARD_KEYS):
                raise ValueError('Checkpoint counters mismatch')
            state['goals'] = goals
            state['targets'] = targets
            mission.__dict__.update(state)
            return mission
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError('Malformed mission snapshot') from error
