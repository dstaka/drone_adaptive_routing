"""Strict new-profile contracts. No adaptation of single-drone evidence."""

from __future__ import annotations
import numpy as np

ROOT_SEED = 10272701
WAIT = 10
SCENARIOS = ("normal", "priority", "availability-delay")
SPLITS = {10: "train", 15: "development", 20: "validation", 30: "routing-heldout", 40: "physical-heldout"}
SCHEMAS = ("dual-case-v1", "dual-policy-v1", "dual-raw-v1", "dual-freeze-v1", "dual-media-v1")


def encode_action(target1, target2, priority):
    if any(type(v) is not int for v in (target1, target2, priority)) or not (
        0 <= target1 <= WAIT and 0 <= target2 <= WAIT and priority in (1, 2)
    ):
        raise ValueError("Action requires target IDs 0..10 and priority 1/2")
    return (target1 * 11 + target2) * 2 + priority - 1


def decode_action(index):
    if type(index) is not int or not 0 <= index < 242:
        raise ValueError("Joint action index must be an integer in [0,242)")
    return index // 22, (index // 2) % 11, index % 2 + 1


def _array(value, shape):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Finite numeric array required") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid shape or nonfinite state")
    return result


def validate_case(case):
    required = {
        "schema",
        "case_id",
        "root_seed",
        "split",
        "index",
        "scenario",
        "goals",
        "starts",
        "loads",
        "initials",
        "priorities",
        "event",
        "generator",
        "geometry_attempt",
    }
    if not isinstance(case, dict) or set(case) != required or case["schema"] != "dual-case-v1":
        raise ValueError("Exact dual-case-v1 schema required")
    if case["root_seed"] != ROOT_SEED or case["split"] not in SPLITS or case["scenario"] not in SCENARIOS:
        raise ValueError("Undeclared stream/scenario")
    if type(case["index"]) is not int or case["index"] < 0:
        raise ValueError("Nonnegative case index required")
    _array(case["goals"], (10, 3))
    _array(case["starts"], (2, 3))
    if len(case["priorities"]) != 10 or any(
        type(p) is not int or not 1 <= p <= 100 for p in case["priorities"]
    ):
        raise ValueError("Ten integer priorities 1..100 required")
    if len(case["loads"]) != 2 or len(case["initials"]) != 2:
        raise ValueError("Two independent physical assemblies required")
    for load, initial in zip(case["loads"], case["initials"]):
        if set(load) != {"side_m", "mass_ratio"} or load["side_m"] not in (0.03, 0.045, 0.06):
            raise ValueError("Declared physical box required")
        if isinstance(load["mass_ratio"], bool) or not 0.10 <= load["mass_ratio"] <= 0.30:
            raise ValueError("Physical mass ratio outside frozen domain")
        if set(initial) != {"position", "quaternion_wxyz", "swing_deg", "direction_rad"}:
            raise ValueError("Explicit initial pose required")
        _array(initial["position"], (3,))
        q = _array(initial["quaternion_wxyz"], (4,))
        if abs(np.linalg.norm(q) - 1) > 1e-10:
            raise ValueError("Unit quaternion required")
        if not 0 <= initial["swing_deg"] <= 10 or not np.isfinite(initial["direction_rad"]):
            raise ValueError("Invalid initial displacement")
    event = case["event"]
    if case["scenario"] == "normal":
        if event is not None:
            raise ValueError("Normal case cannot contain an event")
    elif (
        not isinstance(event, dict)
        or event.get("kind") != case["scenario"]
        or type(event.get("tick")) is not int
    ):
        raise ValueError("Explicit observed-event contract required")
    elif not 600 <= event["tick"] <= 1200:
        raise ValueError("Event outside declared interval")
    elif case["scenario"] == "priority":
        if (
            set(event) != {"kind", "tick", "goal", "priority"}
            or type(event["goal"]) is not int
            or not 0 <= event["goal"] < 10
            or event["priority"] != 100
        ):
            raise ValueError("Fixed priority event target required")
    elif (
        set(event) != {"kind", "tick", "drone_id", "duration_ticks"}
        or event["drone_id"] not in (1, 2)
        or event["duration_ticks"] != 800
    ):
        raise ValueError("Eight-second observed delay required")
    return case
