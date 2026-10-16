import copy
import pytest
from adaptive_drone_patrol.video_delivery import validate_options, optimality_captions, validate_binding


def test_delivery_options_never_retime_to_duration():
    assert validate_options("full", 30, None, 1) == 30
    assert validate_options("submission", 30, 100, 1) == 30
    for args in [
        ("full", 30, 120, 1),
        ("submission", 30, 120, 2),
        ("submission", 24, 120, 1),
        ("submission", 30, 59, 1),
    ]:
        with pytest.raises(ValueError):
            validate_options(*args)


def test_optimal_action_is_distinct_from_policy_suffix_and_ties():
    value = {
        "exact": {"status": "OPTIMAL", "proven": True, "cost": 10},
        "policy_action": 0,
        "forced_first": [
            {"action": 0, "cost": 10, "status": "OPTIMAL", "proven": True},
            {"action": 1, "cost": 10, "status": "OPTIMAL", "proven": True},
        ],
        "policy_rollout": {"cost": 12},
        "equality_tolerance": 1e-6,
    }
    captions = optimality_captions(value)
    assert "最適な次の一手" in captions[0]
    assert "差 2.000" in captions[1]
    assert "20.00%" in captions[1]
    value["policy_rollout"]["cost"] = 10
    assert "全体が最適" in optimality_captions(value)[1]
    value["exact"]["proven"] = False
    assert optimality_captions(value) == ["最適性未確認", "最適性未確認"]


def test_stale_or_oracle_identity_is_rejected():
    d = {
        "run_id": "r",
        "decision_id": "d",
        "policy_hash": "policy",
        "selected_action": 0,
        "boundary_hash": "h",
        "boundary_snapshot": {"current": 0},
        "simulation_step": 4,
        "event_seq": 3,
        "physical_pose": [0, 0, 1],
        "evidence_kind": "physical",
    }
    e = {
        "run_id": "r",
        "decision_id": "d",
        "policy_hash": "policy",
        "policy_action": 0,
        "boundary_hash": "h",
        "snapshot": {"current": 0},
        "evidence_kind": "physical",
        "physical_reference": {
            k: d[k] for k in ["simulation_step", "event_seq", "physical_pose", "boundary_hash"]
        },
    }
    validate_binding(e, d)
    for field, new in [("policy_hash", "oracle"), ("snapshot", {"current": 1}), ("run_id", "other")]:
        changed = copy.deepcopy(e)
        changed[field] = new
        with pytest.raises(ValueError):
            validate_binding(changed, d)


def test_unproved_forced_first_cost_cannot_claim_optimal_action():
    value = {
        "exact": {"status": "OPTIMAL", "proven": True, "cost": 10},
        "policy_action": 0,
        "forced_first": [{"action": 0, "cost": 10, "status": "INCOMPLETE", "proven": False}],
        "policy_rollout": {"cost": 10},
        "equality_tolerance": 1e-6,
    }
    assert optimality_captions(value)[0] == "最適性未確認"


def test_goal_switch_distance_uses_current_target_and_actual_pose():
    from adaptive_drone_patrol.video_render import current_target_distance

    scenario = {"checkpoint_ids": ["B", "C"], "coords": [[0, 0, 1], [0, 0, 1], [1, 0, 1]]}
    # Same-step telemetry still describes the completed B leg; live target is C.
    assert current_target_distance(scenario, [0.1, 0, 1], "C") == pytest.approx(0.9)
    assert current_target_distance(scenario, [0.1, 0, 1], "B") == pytest.approx(0.1)
    assert current_target_distance(scenario, [0.1, 0, 1], None) is None
