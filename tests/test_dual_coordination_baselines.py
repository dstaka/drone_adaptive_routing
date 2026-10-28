"""Executable formula tests for the two prospective causal baselines."""
import numpy as np
import pytest

from adaptive_drone_patrol.dual_coordination_baselines import (
    StaticBaseline, DynamicBaseline, ConflictPrediction,
)
from adaptive_drone_patrol.dual_coordination_mission import MissionState, GoalState
from adaptive_drone_patrol.dual_coordination_reference import ContinuousReference, KinematicState, ReferenceLimits
from adaptive_drone_patrol.dual_coordination_schema import WAIT, encode_action


STARTS = ((0., 0., 1.), (20., 0., 1.))
GOALS = ((0., 0., 1.), (20., 0., 1.)) + ((10., 0., 1.),)*8


def mission_with_remaining(*remaining):
    mission = MissionState()
    for goal in range(10):
        if goal not in remaining:
            mission.goals[goal] = GoalState('DONE', 0, 0, 100, 0)
    return mission


def refs(positions=STARTS):
    return tuple(ContinuousReference(p) for p in positions)


def clear(request):
    return ConflictPrediction()


def test_static_partition_is_fixed_and_ties_go_to_drone_one():
    baseline = StaticBaseline(STARTS, GOALS)
    assert baseline.partition == (0, 1) + (0,)*8
    mission = mission_with_remaining(0, 1)
    # Even after the physical positions swap, eligibility does not transfer.
    decision = baseline.select(mission, STARTS[::-1])
    assert decision.action == encode_action(0, 1, 1)
    assert mission.targets == (WAIT, WAIT)


def test_static_greedily_uses_current_distance_and_reports_empty_partition():
    goals = ((1., 0., 1.), (4., 0., 1.)) + ((2., 0., 1.),)*8
    baseline = StaticBaseline(STARTS, goals)
    decision = baseline.select(mission_with_remaining(0, 1), ((3.8, 0., 1.), STARTS[1]))
    assert decision.action == encode_action(1, WAIT, 1)
    assert baseline.empty_partitions == (False, True)


def test_static_keeps_structurally_locked_target_then_can_choose_nearer_one():
    goals = ((1., 0., 1.), (4., 0., 1.)) + ((2., 0., 1.),)*8
    mission = mission_with_remaining(0, 1)
    mission.apply(encode_action(0, WAIT, 1), 0)
    baseline = StaticBaseline(STARTS, goals)
    positions = ((3.9, 0., 1.), STARTS[1])
    assert baseline.select(mission, positions).targets == (0, WAIT)
    for tick in range(1, 101):
        mission.advance(tick)
    assert baseline.select(mission, positions).targets == (1, WAIT)


def test_dynamic_work_conservation_assigns_two_goals_before_scoring_cost():
    mission = mission_with_remaining(0, 1)
    decision = DynamicBaseline(GOALS).select(mission, STARTS, references=refs(), conflict_predictor=clear)
    assert decision.action == encode_action(0, 1, 1)
    assert decision.unimpeded_etas == pytest.approx((4., 4.))
    assert decision.objective == pytest.approx(4.)
    assert mission.targets == (WAIT, WAIT)


def test_dynamic_one_remaining_goal_never_assigns_both_wait():
    mission = mission_with_remaining(1)
    decision = DynamicBaseline(GOALS).select(mission, STARTS, references=refs(), conflict_predictor=clear)
    assert decision.action == encode_action(WAIT, 1, 2)
    assert decision.unimpeded_etas == (None, 4.)


def test_forced_all_wait_has_no_fictitious_zero_completion_eta():
    mission = mission_with_remaining(0, 1)
    mission.available = [False, False]
    mission.unavailable_until = [800, 800]
    decision = DynamicBaseline(GOALS).select(mission, STARTS, references=refs(), conflict_predictor=clear)
    assert decision.action == encode_action(WAIT, WAIT, 1)
    assert decision.unimpeded_etas == (None, None)
    assert decision.objective is None


def test_pair_conflict_charges_yielding_drone_priority_eta_plus_latency():
    def crossing(request):
        return ConflictPrediction(pair_conflict=True)
    mission = mission_with_remaining(0, 1)
    decision = DynamicBaseline(GOALS).select(mission, STARTS, references=refs(), conflict_predictor=crossing)
    assert decision.priority == 1
    assert decision.adjusted_etas == pytest.approx((4., 8.2))
    assert decision.objective == pytest.approx(8.2)


def test_priority_uses_lower_unimpeded_eta_not_fixed_drone_one():
    mission = mission_with_remaining(0, 1)
    positions = ((-10., 0., 1.), STARTS[1])
    decision = DynamicBaseline(GOALS).select(mission, positions, references=refs(positions), conflict_predictor=clear)
    assert decision.targets == (0, 1)
    assert decision.priority == 2
    assert decision.unimpeded_etas == pytest.approx((16.625, 4.))


def test_unavailable_hover_delay_uses_only_observed_remaining_time():
    mission = mission_with_remaining(0)
    mission.available = [True, False]
    mission.unavailable_until = [None, 800]
    def hover(request):
        assert request.unavailable_remaining == (0., 8.)
        assert request.time == 0.
        assert request.references == refs()
        assert request.current_positions == STARTS
        return ConflictPrediction(unavailable_hover_conflict=(True, False))
    decision = DynamicBaseline(GOALS).select(mission, STARTS, references=refs(), conflict_predictor=hover)
    assert decision.action == encode_action(0, WAIT, 1)
    assert decision.adjusted_etas == (12., None)


def test_continuing_target_uses_existing_remaining_reference_and_service_dwell():
    mission = mission_with_remaining(0, 1)
    mission.apply(encode_action(0, 1, 1), 0)
    flight = ContinuousReference((-10., 0., 1.))
    flight = flight.retarget(GOALS[0], 0., actual_state=KinematicState((-10., 0., 1.), (0, 0, 0)))
    for tick in range(1, 21):
        mission.advance(tick)
    mission.goals[1].status = 'SERVICING'
    mission.goals[1].dwell_ticks = 40
    positions = (tuple(flight.sample(.2)[0]), STARTS[1])
    decision = DynamicBaseline(GOALS).select(mission, positions,
        references=(flight, ContinuousReference(STARTS[1])), conflict_predictor=clear)
    assert decision.targets == (0, 1)
    assert decision.unimpeded_etas == pytest.approx((flight.end_time-.2+1., .6))


def test_changed_owner_charge_counts_release_and_transfer_once_each():
    mission = mission_with_remaining(0, 1)
    mission.apply(encode_action(1, 0, 1), 0)
    for tick in range(1, 101):
        mission.advance(tick)
    # Both old commands are far from their owners; the cheap new pair swaps them.
    decision = DynamicBaseline(GOALS).select(mission, STARTS, references=refs(), conflict_predictor=clear)
    assert decision.targets == (0, 1)
    assert decision.ownership_changes == 2
    assert decision.objective == pytest.approx(6.)


def test_structural_service_lock_is_respected_by_dynamic_selector():
    mission = mission_with_remaining(0, 1)
    mission.apply(encode_action(1, 0, 1), 0)
    mission.seed_settling(settling=(True, True))
    decision = DynamicBaseline(GOALS).select(mission, STARTS, references=refs(), conflict_predictor=clear)
    assert decision.targets == (1, 0)
    assert mission.legal_mask()[decision.action]


def test_terminal_state_and_invalid_conflict_evidence_fail_closed():
    baseline = DynamicBaseline(GOALS)
    mission = mission_with_remaining(0, 1)
    mission.terminal_reason = 'safety_failure'
    with pytest.raises(ValueError):
        baseline.select(mission, STARTS, references=refs(), conflict_predictor=clear)
    mission.terminal_reason = None
    with pytest.raises(ValueError):
        baseline.select(mission, STARTS, references=refs(), conflict_predictor=lambda request: None)
    with pytest.raises(ValueError):
        ConflictPrediction(pair_conflict=float('nan'))
    with pytest.raises(ValueError):
        baseline.select(mission, ((np.nan, 0, 1), STARTS[1]), references=refs(), conflict_predictor=clear)


def test_dynamic_ties_use_weighted_eta_then_goal_ids():
    goals = ((-10., 0., 1.), (10., 0., 1.)) + GOALS[2:]
    mission = mission_with_remaining(0, 1)
    mission.available = [True, False]
    mission.unavailable_until = [None, 800]
    baseline = DynamicBaseline(goals)
    mission.priorities[:2] = [100, 1]
    decision = baseline.select(mission, STARTS, references=refs(), conflict_predictor=clear)
    assert decision.targets == (1, WAIT)
    mission.priorities[:2] = [1, 1]
    decision = baseline.select(mission, STARTS, references=refs(), conflict_predictor=clear)
    assert decision.targets == (0, WAIT)


def test_dynamic_rejects_mismatched_limits_and_future_reference():
    mission = mission_with_remaining(0, 1)
    baseline = DynamicBaseline(GOALS)
    wrong = ContinuousReference(STARTS[0], limits=ReferenceLimits(max_speed=5.))
    with pytest.raises(ValueError):
        baseline.select(mission, STARTS, references=(wrong, refs()[1]), conflict_predictor=clear)
    future = ContinuousReference(STARTS[0], start_time=1.)
    with pytest.raises(ValueError):
        baseline.select(mission, STARTS, references=(future, refs()[1]), conflict_predictor=clear)
