"""Behavioral contracts for the causal 100 Hz mission state machine."""
import numpy as np
import pytest

from adaptive_drone_patrol.dual_coordination_mission import MissionState, settling_predicate
from adaptive_drone_patrol.dual_coordination_schema import WAIT, encode_action


def advance_to(mission, tick, **kwargs):
    result = None
    while mission.tick < tick:
        result = mission.advance(mission.tick + 1, **kwargs)
    return result


def test_atomic_swap_and_wait_takeover_count_each_changed_goal_once():
    mission = MissionState()
    mission.apply(encode_action(0, 1, 1), 0)
    advance_to(mission, 100)
    receipt = mission.apply(encode_action(1, 0, 2), 100)
    assert mission.targets == (1, 0)
    assert [g.owner for g in mission.goals[:2]] == [1, 0]
    assert receipt['ownership_changes'] == 2
    assert receipt['reward_components']['ownership'] == pytest.approx(-.10)
    advance_to(mission, 200)
    receipt = mission.apply(encode_action(WAIT, 1, 2), 200)
    assert [g.owner for g in mission.goals[:2]] == [None, 1]
    assert receipt['ownership_changes'] == 2
    assert mission.leg_start_ticks == [0, 0]


def test_mask_commitment_boundary_and_canonical_priority():
    mission = MissionState()
    mask = mission.legal_mask()
    assert mask.shape == (242,) and mask.dtype == np.bool_
    assert not mask[encode_action(0, 0, 1)]
    assert mask[encode_action(WAIT, WAIT, 1)]
    assert not mask[encode_action(WAIT, WAIT, 2)]
    assert mask[encode_action(WAIT, 0, 2)]
    assert not mask[encode_action(WAIT, 0, 1)]
    mission.apply(encode_action(0, 1, 1), 0)
    advance_to(mission, 99)
    assert not mission.legal_mask()[encode_action(1, 0, 1)]
    mission.advance(100)
    assert mission.legal_mask()[encode_action(1, 0, 1)]
    assert mission.legal_mask()[encode_action(0, 1, 2)]


def test_invalid_action_is_atomic_and_extra_policy_decisions_are_rejected():
    mission = MissionState()
    with pytest.raises(ValueError):
        mission.apply(encode_action(0, 0, 1), 0)
    assert mission.targets == (WAIT, WAIT)
    assert all(g.owner is None for g in mission.goals)
    mission.apply(encode_action(0, 1, 1), 0)
    with pytest.raises(ValueError):
        mission.apply(encode_action(0, 1, 1), 0)
    mission.advance(1)
    with pytest.raises(ValueError):
        mission.apply(encode_action(0, 1, 1), 1)
    with pytest.raises(ValueError):
        mission.advance(3)


def test_continuous_dwell_resets_but_service_lock_requires_safety_interruption():
    mission = MissionState()
    mission.apply(encode_action(0, WAIT, 1), 0)
    advance_to(mission, 99, settling=(True, False))
    assert mission.goals[0].dwell_ticks == 98
    mission.advance(100, settling=(False, False))
    assert mission.goals[0].dwell_ticks == 0
    assert mission.goals[0].status == 'SERVICING'
    assert not mission.legal_mask()[encode_action(WAIT, 0, 2)]
    mission.advance(101, safety_interrupted=(True, False))
    assert mission.goals[0].status == 'ASSIGNED'
    advance_to(mission, 120)
    assert mission.legal_mask()[encode_action(WAIT, 0, 2)]
    mission.apply(encode_action(WAIT, 0, 2), 120)
    advance_to(mission, 219, settling=(False, True))
    assert mission.goals[0].status == 'SERVICING'
    mission.advance(220, settling=(False, True))
    result = mission.advance(221, settling=(False, True))
    assert result['newly_completed'] == [0]
    assert mission.goals[0].completed_tick == 221
    assert mission.goals[0].owner == 1
    assert mission.leg_start_ticks == [0, None]
    assert result['reward_components']['completion'] == 5
    result = advance_to(mission, 240, settling=(False, True))
    assert result['newly_completed'] == []
    assert mission.reward_components['completion'] == 5
    assert not mission.legal_mask()[encode_action(0, WAIT, 1)]


def test_supervisor_hold_keeps_ownership_and_clock_but_resets_dwell():
    mission = MissionState()
    mission.apply(encode_action(0, WAIT, 1), 0)
    advance_to(mission, 50, settling=(True, False))
    result = mission.advance(51, settling=(True, False), held=(True, False),
                             safety_interrupted=(True, False))
    assert mission.goals[0].owner == 0
    assert mission.targets == (0, WAIT)
    assert mission.goals[0].dwell_ticks == 0
    assert mission.leg_start_ticks == [0, None]
    assert result['reward_components']['supervisor'] == pytest.approx(-.001)


def test_release_reassignment_and_delay_never_reset_leg_timeout():
    mission = MissionState()
    mission.apply(encode_action(0, WAIT, 1), 0)
    advance_to(mission, 100)
    mission.apply(encode_action(WAIT, 0, 2), 100)
    advance_to(mission, 200)
    mission.apply(encode_action(2, 0, 1), 200)
    advance_to(mission, 650)
    mission.advance(651, event={'kind':'availability-delay','tick':651,'drone_id':1,'duration_ticks':800})
    assert mission.targets[0] == WAIT
    assert mission.goals[2].owner is None
    assert mission.leg_start_ticks == [0, 100]
    advance_to(mission, 1451)
    assert mission.available == [True, True]
    assert mission.targets[0] == WAIT
    result = advance_to(mission, 3000)
    assert mission.terminal_reason == 'leg_timeout:1'
    assert result['reward_components']['failure'] == -1000
    with pytest.raises(ValueError):
        mission.advance(3001)


def test_all_wait_hits_team_timeout_and_failure_penalty_once():
    mission = MissionState()
    mission.apply(encode_action(WAIT, WAIT, 1), 0)
    advance_to(mission, 30000)
    assert mission.terminal_reason == 'team_timeout'
    assert mission.reward_components['failure'] == -1000
    assert mission.reward_components['time'] == pytest.approx(-300)
    assert not mission.legal_mask().any()


def test_completion_precedes_event_and_event_is_observed_only_at_policy_boundary():
    mission = MissionState(priorities=[50]*10)
    mission.apply(encode_action(0, WAIT, 1), 0)
    advance_to(mission, 549)
    advance_to(mission, 649, settling=(True, False))
    result = mission.advance(650, settling=(True, False),
        event={'kind':'priority','tick':650,'goal':0,'priority':100})
    assert result['newly_completed'] == [0]
    assert mission.event_receipts[0]['no_op'] is True
    assert mission.event_receipts[0]['observed_tick'] is None
    with pytest.raises(ValueError):
        mission.observed_state()
    advance_to(mission, 660)
    observed = mission.observed_state()
    assert observed['events'][0]['observed_tick'] == 660
    assert observed['events'][0]['trigger_tick'] == 650
    assert 'event' not in observed
    mission.apply(encode_action(1, WAIT, 1), 660)
    assert mission.event_receipts[0]['action_tick'] == 660


def test_priority_event_can_break_commitment_at_next_boundary_and_reward_is_causal():
    mission = MissionState(priorities=[1]*10)
    mission.apply(encode_action(0, 1, 1), 0)
    advance_to(mission, 600)
    mission.apply(encode_action(2, 3, 1), 600)
    advance_to(mission, 610)
    event={'kind':'priority','tick':611,'goal':9,'priority':100}
    result = mission.advance(611, event=event)
    assert result['reward_components']['priority'] == pytest.approx(-.00002)
    advance_to(mission, 620)
    assert mission.observed_state()['priorities'][9] == 100
    assert mission.legal_mask()[encode_action(9, 3, 1)]
    mission.apply(encode_action(9, 3, 1), 620)
    result = mission.advance(621)
    assert result['reward_components']['priority'] == pytest.approx(-.000218)
    advance_to(mission, 640)
    assert not mission.legal_mask()[encode_action(2, 3, 1)]


def test_no_transfer_retains_first_owner_through_wait_and_forced_delay():
    mission = MissionState(no_transfer=True)
    mission.apply(encode_action(0, 1, 1), 0)
    advance_to(mission, 100)
    receipt = mission.apply(encode_action(WAIT, 1, 2), 100)
    assert receipt['ownership_changes'] == 0
    assert mission.goals[0].owner == 0
    assert not mission.legal_mask()[encode_action(2, 0, 1)]
    advance_to(mission, 650)
    mission.advance(651,event={'kind':'availability-delay','tick':651,'drone_id':2,'duration_ticks':800})
    assert mission.goals[1].owner == 1
    assert mission.targets == (WAIT, WAIT)
    assert not mission.legal_mask()[encode_action(1, WAIT, 1)]
    assert not mission.legal_mask()[encode_action(WAIT, 2, 2)]
    advance_to(mission, 1460)
    assert mission.legal_mask()[encode_action(0, 1, 1)]


def test_all_goals_and_final_hover_complete_once_with_overlapping_final_dwell():
    mission = MissionState()
    for pair in range(5):
        tick = pair*120
        mission.apply(encode_action(pair*2,pair*2+1,1),tick)
        advance_to(mission,tick+(101 if pair == 4 else 120),settling=(True,True),final_settling=(True,True))
    assert mission.terminal_reason == 'success'
    assert mission.team_completion_tick == 581
    assert mission.last_goal_completion_tick == 581
    assert mission.reward_components['completion'] == 50
    assert mission.reward_components['failure'] == 0
    assert mission.completed_counts == [5,5]


def test_final_hover_interruption_delays_success_after_last_goal():
    mission = MissionState()
    for pair in range(5):
        mission.apply(encode_action(pair*2,pair*2+1,1),pair*120)
        advance_to(mission,pair*120+(101 if pair == 4 else 120),settling=(True,True),final_settling=(False,True))
    assert mission.last_goal_completion_tick == 581
    assert mission.terminal_reason is None
    advance_to(mission,682,final_settling=(True,True))
    assert mission.team_completion_tick == 682


def test_settling_predicate_checks_all_thresholds_including_full_box_rotation():
    values = dict(drone_position_error=.2,package_position_error=.2,drone_speed=.3,
                  package_speed=.15,swing_angle_deg=5.,swing_rate_deg_s=15.,box_angular_speed_deg_s=30.)
    assert settling_predicate(**values)
    for key in values:
        invalid = dict(values,**{key:values[key]+.0001})
        assert not settling_predicate(**invalid), key
        assert not settling_predicate(**dict(values,**{key:float('nan')})), key


def test_terminal_physical_failure_prevents_success_and_records_penalty():
    mission = MissionState()
    result = mission.advance(1,failure_reason='forbidden_contact')
    assert mission.terminal_reason == 'forbidden_contact'
    assert result['reward_components']['failure'] == -1000
    assert result['reward'] == pytest.approx(-1000.01002)


def test_snapshot_resume_preserves_partial_dwell_events_commitments_and_rewards():
    import json
    mission = MissionState(priorities=[50]*10)
    mission.apply(encode_action(0,1,1),0)
    advance_to(mission,610)
    mission.advance(611,event={'kind':'priority','tick':611,'goal':9,'priority':100})
    advance_to(mission,619,settling=(True,False))
    resumed = MissionState.from_dict(json.loads(json.dumps(mission.to_dict())))
    assert resumed.to_dict() == mission.to_dict()
    for state in (mission,resumed):
        advance_to(state,620,settling=(True,False))
        state.apply(encode_action(0,2,2),620)
        advance_to(state,712,settling=(True,False))
    assert resumed.to_dict() == mission.to_dict()
    assert resumed.goals[0].status == 'DONE'


def test_priority_only_change_does_not_extend_commitment():
    mission = MissionState()
    mission.apply(encode_action(0,1,1),0)
    advance_to(mission,80)
    mission.apply(encode_action(0,1,2),80)
    advance_to(mission,100)
    assert mission.legal_mask()[encode_action(2,3,1)]


def test_malformed_snapshot_fails_closed_without_constructing_impossible_ownership():
    import copy
    state = MissionState()
    state.apply(encode_action(0,1,1),0)
    valid = state.to_dict()
    invalid = []
    snapshot = copy.deepcopy(valid)
    del snapshot['state']['tick']
    invalid.append(snapshot)
    snapshot = copy.deepcopy(valid)
    snapshot['state']['goals'][0]['owner'] = 2
    invalid.append(snapshot)
    snapshot = copy.deepcopy(valid)
    snapshot['state']['targets'] = [0,0]
    invalid.append(snapshot)
    snapshot = copy.deepcopy(valid)
    snapshot['state']['reward_components']['time'] = float('nan')
    invalid.append(snapshot)
    snapshot = copy.deepcopy(valid)
    snapshot['state']['tick'] = -1
    invalid.append(snapshot)
    for snapshot in invalid:
        with pytest.raises(ValueError):
            MissionState.from_dict(snapshot)


def test_event_boundary_completion_then_forced_release_preserves_credit_and_clock():
    mission = MissionState()
    mission.apply(encode_action(0,1,1),0)
    advance_to(mission,499)
    advance_to(mission,599,settling=(True,False))
    result = mission.advance(600,settling=(True,False),
        event={'kind':'availability-delay','tick':600,'drone_id':1,'duration_ticks':800})
    assert result['newly_completed'] == [0]
    assert result['ownership_changes'] == 0
    assert mission.goals[0].owner == 0
    assert mission.leg_start_ticks[0] is None
    assert mission.event_receipts[0]['observed_tick'] == 600
    mission.apply(encode_action(WAIT,1,2),600)
    assert mission.event_receipts[0]['action_tick'] == 600


def test_malformed_advance_and_event_leave_state_unchanged():
    mission = MissionState()
    before = mission.to_dict()
    with pytest.raises(ValueError):
        mission.advance(1,settling=(1,0))
    with pytest.raises(ValueError):
        mission.advance(1,event={'kind':'priority','tick':2,'goal':0,'priority':100})
    with pytest.raises(ValueError):
        mission.advance(1,event={'kind':'availability-delay','tick':1,'drone_id':1,'duration_ticks':1})
    assert mission.to_dict() == before


def test_endpoint_samples_require_100_intervals_not_100_samples():
    mission = MissionState()
    mission.apply(encode_action(0,WAIT,1),0)
    advance_to(mission,100,settling=(True,False),final_settling=(True,False))
    assert mission.goals[0].status == 'SERVICING'
    assert mission.goals[0].dwell_ticks == 99
    assert mission.final_dwell_ticks == [99,0]
    mission.advance(101,settling=(True,False),final_settling=(True,False))
    assert mission.goals[0].status == 'DONE'
    assert mission.final_dwell_ticks == [100,0]


def test_initial_endpoint_seeding_and_safety_interruption_reset_continuity():
    mission = MissionState()
    mission.apply(encode_action(0,WAIT,1),0)
    mission.seed_settling(settling=(True,False),final_settling=(True,False))
    advance_to(mission,99,settling=(True,False),final_settling=(True,False))
    assert mission.goals[0].dwell_ticks == 99
    mission.advance(100,settling=(True,False),final_settling=(True,False),safety_interrupted=(True,False))
    mission.advance(101,settling=(True,False),final_settling=(True,False))
    assert mission.goals[0].dwell_ticks == 0
    assert mission.final_dwell_ticks[0] == 0
    with pytest.raises(ValueError):
        mission.seed_settling(settling=(True,False))


def test_all_goal_success_cannot_override_another_drones_expired_persistent_leg():
    mission = MissionState()
    mission.apply(encode_action(0,WAIT,1),0)
    advance_to(mission,100)
    for goal in range(9):
        start = 100+120*goal
        mission.apply(encode_action(WAIT,goal,2),start)
        advance_to(mission,start+101,settling=(False,True))
        advance_to(mission,start+120)
    mission.apply(encode_action(WAIT,9,2),1180)
    advance_to(mission,2899)
    advance_to(mission,3000,settling=(False,True),final_settling=(True,True))
    assert all(goal.status == 'DONE' for goal in mission.goals)
    assert mission.terminal_reason == 'leg_timeout:1'
    assert mission.team_completion_tick is None
    assert mission.reward_components['completion'] == 50
    assert mission.reward_components['failure'] == -1000


def test_own_inspection_completion_at_leg_deadline_ends_its_clock_first():
    mission = MissionState()
    mission.apply(encode_action(0,WAIT,1),0)
    advance_to(mission,2899)
    advance_to(mission,3000,settling=(True,False))
    assert mission.goals[0].status == 'DONE'
    assert mission.leg_start_ticks[0] is None
    assert mission.terminal_reason is None


@pytest.mark.parametrize('clock',['commitment_start_ticks','leg_start_ticks'])
def test_snapshot_active_assignment_cannot_omit_its_clock(clock):
    mission = MissionState()
    mission.apply(encode_action(0,1,1),0)
    snapshot = mission.to_dict()
    snapshot['state'][clock][0] = None
    with pytest.raises(ValueError):
        MissionState.from_dict(snapshot)
