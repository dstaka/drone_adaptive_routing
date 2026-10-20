"""Independent standard-library replay of the declared expanded flight probes."""

import hashlib
import json
import math
from pathlib import Path

TARGETS = [
    ([-2.7, -1.8, 1.0], "corner_departure"),
    ([2.7, 1.8, 1.0], "long_horizontal"),
    ([2.7, 1.8, 3.0], "ascent"),
    ([2.7, 1.8, 1.0], "descent"),
    ([-2.7, -1.8, 3.0], "combined"),
    ([2.7, 1.8, 1.0], "reversal_combined"),
    ([0.0, 0.0, 3.0], "high_return"),
    ([0.0, 0.0, 1.0], "low_return"),
    ([0.0, 0.0, 1.0], "repeated_hover"),
]
SUITE_SHA256 = "10b50f2b109242617dc7fbac67506a3ecb1c47f111e3333c5416db81cba9835a"
SOURCE_PATHS = [
    "src/adaptive_drone_patrol/flight_agent.py",
    "src/adaptive_drone_patrol/target_adapter.py",
    "src/adaptive_drone_patrol/scene.py",
    "src/adaptive_drone_patrol/patrol_runner.py",
    "src/adaptive_drone_patrol/transition_check.py",
    "scripts/probe_dispersed_flight.py",
    "third_party/genesis_drone/hover_env.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_sources(root="."):
    return {p: digest(Path(root) / p) for p in SOURCE_PATHS}


def evidence_binding(root):
    root = Path(root)
    return {name + "_sha256": digest(root / name) for name in ["summary.json", "samples.jsonl"]}


def finite_vector(value, size):
    return (
        isinstance(value, list)
        and len(value) == size
        and all(type(x) in (int, float) and math.isfinite(x) for x in value)
    )


def verify(root, source_root=".", suite="configs/dispersed_transitions.yaml"):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    samples = [json.loads(x) for x in (root / "samples.jsonl").read_text().splitlines()]
    if (
        summary.get("status") != "SUCCEEDED"
        or len(summary.get("legs", [])) != 9
        or len(samples) != summary.get("steps")
    ):
        raise ValueError("Incomplete transition probe; original failure must be retained")
    if (
        summary.get("dt") != 0.01
        or summary.get("evidence_kind") != "physical"
        or summary.get("execution_seed") not in [11, 12, 31, 32, 33]
    ):
        raise ValueError("Unexpected timestep/evidence/split seed")
    if summary.get("suite_sha256") != SUITE_SHA256 or digest(suite) != SUITE_SHA256:
        raise ValueError("Probe does not bind the declared transition suite")
    sources = execution_sources(source_root)
    if (
        summary.get("execution_sources") != sources
        or summary.get("hover_source_sha256") != sources[SOURCE_PATHS[-1]]
    ):
        raise ValueError("Probe execution source differs from the frozen runtime")
    saturated = 0
    durations = []
    previous_end = 0
    previous_pose = [0.0, 0.0, 1.0]
    for i, (leg, (target, kind)) in enumerate(zip(summary["legs"], TARGETS)):
        if (
            leg.get("index") != i
            or leg.get("kind") != kind
            or leg.get("target") != target
            or leg.get("start_step") != previous_end
            or not finite_vector(leg.get("start_xyz"), 3)
            or math.dist(leg["start_xyz"], previous_pose) > 1e-6
        ):
            raise ValueError("Declared target sequence or prior endpoint continuity mismatch")
        end = leg.get("end_step")
        if type(end) is not int or end <= previous_end or end > len(samples):
            raise ValueError("Invalid transition coverage")
        own = samples[previous_end:end]
        if len(own) < 50:
            raise ValueError("Incomplete dwell coverage")
        for offset, row in enumerate(own, previous_end + 1):
            if (
                row.get("step") != offset
                or row.get("target_index") != i
                or row.get("target") != target
                or not all(
                    finite_vector(row.get(k), n)
                    for k, n in [
                        ("xyz", 3),
                        ("velocity", 3),
                        ("actions", 4),
                        ("quaternion_wxyz", 4),
                        ("effective_target", 3),
                    ]
                )
                or row.get("done") is not False
                or row.get("reset") is not False
            ):
                raise ValueError("Invalid physical sample/target/index/termination")
            w, x, y, z = row["quaternion_wxyz"]
            norm = math.sqrt(w * w + x * x + y * y + z * z)
            if abs(norm - 1) > 1e-4:
                raise ValueError("Invalid quaternion norm")
            w, x, y, z = [v / norm for v in [w, x, y, z]]
            tilt = math.degrees(
                max(
                    abs(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
                    abs(math.asin(max(-1, min(1, 2 * (w * y - z * x))))),
                )
            )
            recorded = row.get("tilt_deg")
            if (
                type(recorded) not in (int, float)
                or not math.isfinite(recorded)
                or abs(recorded - tilt) > 1e-5
            ):
                raise ValueError("Recorded tilt differs from physical quaternion")
            if row["xyz"][2] < 0.1 or tilt > 60:
                raise ValueError("Safety violation")
            if math.dist(row["effective_target"], target) > 1e-6:
                raise ValueError("Effective world target mismatch")
            obs = row.get("relative_observation_absmax")
            expected = min(1, max(abs(a - b) for a, b in zip(target, row["xyz"])) / 3)
            if type(obs) not in (int, float) or not math.isfinite(obs) or abs(expected - obs) > 1e-6:
                raise ValueError("Observation coordinate semantics mismatch")
            predicted = [a + v * 0.01 for a, v in zip(previous_pose, row["velocity"])]
            if math.dist(predicted, row["xyz"]) > 0.02:
                raise ValueError("Position jump inconsistent with physical velocity")
            previous_pose = row["xyz"]
            saturated += expected >= 1
        for row in own[-50:]:
            if math.dist(row["xyz"], target) > 0.2 or math.sqrt(sum(v * v for v in row["velocity"])) > 0.3:
                raise ValueError("Missing continuous 0.50 s actual arrival dwell")
        duration = len(own) * 0.01
        if duration > 30 or abs(duration - leg["duration_s"]) > 1e-9:
            raise ValueError("Finite transition timeout mismatch")
        durations.append(duration)
        previous_end = end
    if previous_end != len(samples):
        raise ValueError("Unmapped physical samples")
    return {
        "status": "PASS",
        "execution_seed": summary["execution_seed"],
        "samples": len(samples),
        "legs_verified": 9,
        "saturated_samples": saturated,
        "leg_durations_s": durations,
        "maximum_leg_s": max(durations),
        "min_altitude_m": min(r["xyz"][2] for r in samples),
        "max_tilt_deg": max(r["tilt_deg"] for r in samples),
        "suite_sha256": SUITE_SHA256,
        "execution_sources": sources,
        **evidence_binding(root),
        "continuity_tolerance_m": 0.02,
        "method": "Independent standard-library declared-target/source/hash/quaternion/sample/observation/safety/dwell replay",
    }
