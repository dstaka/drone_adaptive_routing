"""Pre-learning manifests and fixed development challenges; no held-out exposure."""

from __future__ import annotations
import copy
import json
from pathlib import Path
from .dual_coordination_cases import generate_case, canonical_case, declaration
from .dual_coordination_progress import atomic_json
from .dual_coordination_schema import validate_case
from .util import sha256


def development_challenges(layout_path):
    canonical = canonical_case(layout_path)
    priority = canonical_case(layout_path, "priority")
    priority["event"] = {"kind": "priority", "tick": 800, "goal": 3, "priority": 100}
    priority["case_id"] = "development-video-priority-transfer"
    crossing = copy.deepcopy(canonical)
    crossing["case_id"] = "development-video-crossing-yield"
    crossing["starts"] = [[-2.0, 0.0, 3.0], [2.0, 0.0, 3.0]]
    for initial, position in zip(crossing["initials"], crossing["starts"]):
        initial["position"] = position.copy()
    crossing["goals"][0] = [2.0, 0.0, 3.0]
    crossing["goals"][1] = [-2.0, 0.0, 3.0]
    delay = canonical_case(layout_path, "availability-delay")
    delay["case_id"] = "development-delayed-owner-takeover"
    delay["event"] = {"kind": "availability-delay", "tick": 600, "drone_id": 1, "duration_ticks": 800}
    delay["goals"][0] = [0.0, 0.0, 5.0]
    delay["goals"][1] = [0.7, 0.0, 5.0]
    records = [
        dict(
            case=canonical,
            expected_eligible_actions="all structural alternatives at reset",
            purpose="canonical normal video",
        ),
        dict(
            case=priority,
            expected_eligible_actions="observed goal D priority100 at8s; reassignment remains learned, never guaranteed",
            purpose="priority/transfer video",
        ),
        dict(
            case=crossing,
            expected_eligible_actions="A/B simultaneous shortest paths cross; WAIT or either priority remains available",
            purpose="crossing/yield video",
        ),
        dict(
            case=delay,
            expected_eligible_actions="at6s drone1 WAIT; unfinished ownership transferable to drone2",
            purpose="delayed-owner transfer",
        ),
    ]
    variants = {
        "perpendicular-crossing": ([[-2, 0, 3], [0, -2, 3]], [[2, 0, 3], [0, 2, 3]]),
        "altitude-crossing": ([[0, 0, 2], [2, 0, 4]], [[0, 0, 5], [-2, 0, 3]]),
        "box-link-clearance": ([[-1, 0, 3], [1, 0, 3.3]], [[1, 0, 3], [-1, 0, 3.3]]),
    }
    for name, (starts, goals) in variants.items():
        case = copy.deepcopy(crossing)
        case["case_id"] = "development-" + name
        case["starts"] = starts
        for i in range(2):
            case["initials"][i]["position"] = starts[i].copy()
            case["goals"][i] = goals[i]
        records.append(
            dict(
                case=case,
                expected_eligible_actions="A/B, WAIT and legal priority variants; no conflict-based mask",
                purpose=name,
            )
        )
    for name in ("mid-flight-transfer", "simultaneous-completion", "repeated-wait", "opposing-priority"):
        case = copy.deepcopy(crossing)
        case["case_id"] = "development-" + name
        records.append(
            dict(
                case=case,
                expected_eligible_actions="scripted semantic challenge; not a learned success claim",
                purpose=name,
            )
        )
    for record in records:
        validate_case(record["case"])
    return records


def declare_campaign(output, *, layout_path, training_count=10000):
    if type(training_count) is not int or not 1 <= training_count <= 10000:
        raise ValueError("Training declaration size1..10000 required; production requires10000")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "generator.json", declaration())
    with (root / "training.jsonl").open("w") as stream:
        for i in range(training_count):
            # Stage event mixtures use the same keyed geometry/load case index.
            row = {
                scenario: generate_case(10, i, scenario)
                for scenario in ("normal", "priority", "availability-delay")
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (root / "validation.jsonl").open("w") as stream:
        for i in range(300):
            scenario = ("normal", "priority", "availability-delay")[i // 100]
            stream.write(json.dumps(generate_case(20, i, scenario), separators=(",", ":")) + "\n")
    challenges = development_challenges(layout_path)
    atomic_json(root / "development-challenges.json", challenges)
    result = dict(
        schema="dual-campaign-declaration-v1",
        training_cases=training_count,
        validation_cases=300,
        heldout_generated=False,
        production_ready=False,
        full_training_manifest=training_count == 10000,
        production_prerequisites=[
            "native joint qualification",
            "stopping/tracking qualification",
            "baseline formula freeze",
            "CPU/GPU measurements and fidelity",
            "storage feasibility",
            "optimizer freeze",
        ],
        video_case_ids=[r["case"]["case_id"] for r in challenges[:3]],
        source_generator=declaration(),
        files={
            p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(root.iterdir())
            if p.is_file()
        },
    )
    atomic_json(root / "declaration.json", result)
    return result
