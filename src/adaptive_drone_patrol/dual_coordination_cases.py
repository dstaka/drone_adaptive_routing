"""Keyed PCG64 streams, independent fields; held-out never opened implicitly."""

from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .dual_coordination_schema import ROOT_SEED, SPLITS, SCENARIOS, validate_case

FIELDS = {
    "geometry": 1,
    "starts": 2,
    "load1": 3,
    "load2": 4,
    "initial1": 5,
    "initial2": 6,
    "priorities": 7,
    "event_time": 8,
    "event_target": 9,
    "training_sampling": 10,
}
GENERATOR = "dual-keyed-pcg64-v1"


def keyed_rng(split, index, field, attempt=0):
    if (
        split not in SPLITS
        or field not in FIELDS
        or any(type(v) is not int or v < 0 for v in (index, attempt))
    ):
        raise ValueError("Declared split, field and nonnegative keys required")
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([ROOT_SEED, split, index, FIELDS[field], attempt]))
    )


def declaration():
    return dict(
        generator=GENERATOR,
        root_seed=ROOT_SEED,
        splits=SPLITS,
        fields=FIELDS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits={"train": 10000, "validation": 300, "routing-heldout": 300, "physical-heldout": 127},
        tie_order=["integrate/audit", "dwell completion", "event", "5Hz policy"],
        physical_random_pair_extra=[0.03, 0.03],
        heldout_generated=False,
    )


def _quaternion(roll, pitch):
    cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return [cr * cp, sr * cp, cr * sp, -sr * sp]


def generate_case(split, index, scenario, *, freeze=None, geometry_attempt_start=0):
    if split not in SPLITS or scenario not in SCENARIOS:
        raise ValueError("Undeclared split/scenario")
    if split in (30, 40):
        # Capability is implemented by the campaign module only after selection.
        # A dictionary/boolean cannot authorize data exposure.
        if (
            freeze is None
            or type(freeze).__name__ != "VerifiedJointFreeze"
            or type(freeze).__module__ != "adaptive_drone_patrol.dual_coordination_campaign"
        ):
            raise ValueError("Verified joint freeze required before held-out generation")
        freeze.require_current()
    if type(geometry_attempt_start) is not int or not 0 <= geometry_attempt_start < 10000:
        raise ValueError("Geometry attempt limit exhausted/invalid")
    for attempt in range(geometry_attempt_start, 10000):
        rng = keyed_rng(split, index, "geometry", attempt)
        goals = rng.uniform([-2.7, -2.7, 2], [2.7, 2.7, 5], size=(10, 3))
        if all(np.linalg.norm(goals[i] - goals[j]) >= 0.5 for i in range(10) for j in range(i)):
            break
    else:
        raise ValueError("Geometry rejection attempt limit exhausted")
    rng = keyed_rng(split, index, "starts")
    orientation = int(rng.choice([-1, 1]))
    starts = np.array([[-3.5, -3.5 * orientation, 1], [3.5, 3.5 * orientation, 1.0]])
    if rng.integers(2):
        starts = starts[::-1].copy()
    starts[:, :2] += rng.uniform(-0.20, 0.20, size=(2, 2))
    loads, initials = [], []
    for d in range(2):
        rng = keyed_rng(split, index, f"load{d + 1}")
        loads.append(
            dict(side_m=float(rng.choice([0.03, 0.045, 0.06])), mass_ratio=float(rng.uniform(0.10, 0.30)))
        )
        rng = keyed_rng(split, index, f"initial{d + 1}")
        roll, pitch = np.deg2rad(rng.uniform(-2, 2, 2))
        pos = starts[d].copy()
        pos[:2] += rng.uniform(-0.02, 0.02, 2)
        initials.append(
            dict(
                position=pos.tolist(),
                quaternion_wxyz=_quaternion(roll, pitch),
                swing_deg=float(rng.uniform(0, 10)),
                direction_rad=float(rng.uniform(0, 2 * np.pi)),
            )
        )
    # Conservative initial bounding spheres: radius .50 m covers a full assembly
    # at every declared load/pose. Starts are far apart and above the .10 floor.
    if (
        np.linalg.norm(np.array(initials[0]["position"]) - initials[1]["position"]) <= 1.2
        or min(p["position"][2] for p in initials) - 0.50 <= 0.10
    ):
        raise ValueError("Unsafe initial assembly envelope; do not resample case ID")
    event = None
    if scenario != "normal":
        tick = int(np.ceil(keyed_rng(split, index, "event_time").uniform(6, 12) * 100))
        target_rng = keyed_rng(split, index, "event_target")
        event = dict(kind=scenario, tick=tick)
        event.update(
            dict(goal=int(target_rng.integers(10)), priority=100)
            if scenario == "priority"
            else dict(drone_id=int(target_rng.integers(1, 3)), duration_ticks=800)
        )
    return validate_case(
        dict(
            schema="dual-case-v1",
            case_id=f"{split}-{index:05d}-{scenario}",
            root_seed=ROOT_SEED,
            split=split,
            index=index,
            scenario=scenario,
            goals=goals.tolist(),
            starts=starts.tolist(),
            loads=loads,
            initials=initials,
            priorities=keyed_rng(split, index, "priorities").integers(1, 101, 10).tolist(),
            event=event,
            generator=GENERATOR,
            geometry_attempt=attempt,
        )
    )


def observed_event(case, tick):
    event = case["event"]
    return copy.deepcopy(event) if event is not None and tick >= event["tick"] else None


def canonical_case(layout_path, scenario="normal", *, index=0):
    case = generate_case(15, index, scenario)
    case["case_id"] = f"development-canonical-{scenario}"
    case["goals"] = json.loads(Path(layout_path).read_text())["coords"][1:]
    case["starts"] = [[-3.5, -3.5, 1.0], [3.5, 3.5, 1.0]]
    case["initials"] = [
        dict(position=p.copy(), quaternion_wxyz=[1.0, 0, 0, 0], swing_deg=0.0, direction_rad=0.0)
        for p in case["starts"]
    ]
    case["loads"] = [dict(side_m=0.045, mass_ratio=0.2) for _ in range(2)]
    return validate_case(case)
