"""Behavioral boundaries for new schemas and sealed deterministic splits."""

import copy
import numpy as np
import pytest


def test_joint_action_bijection_rejects_boolean_and_outside_domain():
    from adaptive_drone_patrol.dual_coordination_schema import encode_action, decode_action

    assert decode_action(0) == (0, 0, 1)
    assert decode_action(241) == (10, 10, 2)
    assert len({decode_action(i) for i in range(242)}) == 242
    for i in range(242):
        assert encode_action(*decode_action(i)) == i
    for bad in [True, -1, 242, 1.5]:
        with pytest.raises(ValueError):
            decode_action(bad)


def test_keyed_fields_cases_splits_and_observed_events_are_independent():
    from adaptive_drone_patrol.dual_coordination_cases import generate_case, keyed_rng, observed_event

    a = generate_case(15, 3, "priority")
    b = generate_case(15, 3, "normal")
    assert a["goals"] == b["goals"] and a["loads"] == b["loads"]
    assert a == generate_case(15, 3, "priority")
    assert a["goals"] != generate_case(20, 3, "priority")["goals"]
    assert not np.array_equal(keyed_rng(15, 3, "load1").random(10), keyed_rng(15, 3, "load2").random(10))
    assert observed_event(a, 0) is None
    assert observed_event(a, a["event"]["tick"] - 1) is None
    assert observed_event(a, a["event"]["tick"])["kind"] == "priority"
    assert np.linalg.norm(np.array(a["starts"][0]) - a["starts"][1]) > 9
    points = np.asarray(a["goals"])
    assert np.min([np.linalg.norm(points[i] - points[j]) for i in range(10) for j in range(i)]) >= 0.5


def test_heldout_is_refused_without_verified_freeze_and_forged_dict_is_insufficient():
    from adaptive_drone_patrol.dual_coordination_cases import generate_case

    for split in (30, 40):
        for fake in (None, {"schema": "dual-freeze-v1", "passed": True}):
            with pytest.raises(ValueError, match="freeze"):
                generate_case(split, 0, "normal", freeze=fake)


def test_mixed_schema_and_malformed_physics_rejected():
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    from adaptive_drone_patrol.dual_coordination_schema import validate_case

    case = generate_case(15, 0, "normal")
    validate_case(case)
    for path, bad in [
        ("schema", "single-case-v1"),
        ("starts", [[0, 0, 1]]),
        ("goals", [[0, 0, float("nan")]] * 10),
        ("priorities", [101] * 10),
    ]:
        damaged = copy.deepcopy(case)
        damaged[path] = bad
        with pytest.raises(ValueError):
            validate_case(damaged)
    damaged = copy.deepcopy(case)
    damaged["loads"][0]["mass_ratio"] = 0.31
    with pytest.raises(ValueError):
        validate_case(damaged)


def test_generator_rejection_does_not_shift_other_fields_or_case_ids():
    from adaptive_drone_patrol.dual_coordination_cases import generate_case

    a = generate_case(15, 9, "normal")
    b = generate_case(15, 9, "normal", geometry_attempt_start=1)
    assert a["case_id"] == b["case_id"]
    assert a["goals"] != b["goals"]
    assert a["loads"] == b["loads"] and a["starts"] == b["starts"]
    with pytest.raises(ValueError, match="attempt"):
        generate_case(15, 9, "normal", geometry_attempt_start=10000)
