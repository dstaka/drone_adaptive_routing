#!/usr/bin/env python
"""Bind fixed acceptance runs to the predeclared source, model, geometry and camera."""

import argparse
import hashlib
import json
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_bound_runs(frozen, layout, records):
    pairs = {(r["summary"]["scenario_id"], r["summary"]["execution_seed"]) for r in records}
    expected = {(kind, seed) for kind in ["normal", "emergency", "cost_shift"] for seed in [1, 2, 3]}
    if len(records) != 9 or pairs != expected:
        raise ValueError("Fixed nine-run denominator mismatch")
    if (
        layout["coordinate_sha256"] != frozen["coordinate_sha256"]
        or canonical_hash(frozen["scene"]) != frozen["scene_sha256"]
    ):
        raise ValueError("Reviewed layout/frozen camera identity mismatch")
    models = frozen["model_hashes"]
    for r in records:
        summary, scenario, scene = r["summary"], r["scenario"], r["scene"].copy()
        for actual, key in [
            ("flight_checkpoint_hash", "flight_checkpoint_hash"),
            ("flight_config_hash", "flight_cfg_hash"),
            ("routing_model_hash", "routing_model_hash"),
            ("hover_source_hash", "hover_source_hash"),
        ]:
            if summary[actual] != models[key]:
                raise ValueError("Frozen model/config/source changed")
        if (
            summary["execution_source_hash"] != frozen["source_hash"]
            or summary["arrival_settings"] != frozen["limits"]
            or summary["flight_acceptance_config_hash"] != frozen["flight_acceptance_config_sha256"]
            or summary["scenario_hash"] != frozen["scenario_hashes"][summary["scenario_id"]]
        ):
            raise ValueError("Frozen execution inputs changed")
        if (
            summary["status"] != "SUCCEEDED"
            or summary["completed_count"] != 10
            or scenario["schema_version"] != 3
            or scenario["priorities"] != layout["priorities"]
            or any(type(p) is not int for p in scenario["priorities"])
            or scenario["coords"] != [[0.0, 0.0, 1.0]] + [m["world_xyz"] for m in frozen["scene"]["markers"]]
            or any(v != 1 for row in scenario["multipliers"] for v in row)
        ):
            raise ValueError("Actual scenario differs from the reviewed shared initial state")
        event = {
            "normal": None,
            "emergency": {"kind": "priority", "after": 1, "checkpoint": "C", "value": 100},
            "cost_shift": {"kind": "cost", "after": 1, "edges": [layout["declared_cost_edge"]], "value": 5},
        }[summary["scenario_id"]]
        if scenario["event"] != event:
            raise ValueError("Declared event changed")
        actual = scene.pop("verified_runtime_positions", None)
        if actual is not None and actual != [m["world_xyz"] for m in scene["markers"]]:
            # Runtime float32 positions are checked numerically by physical validation.
            if any(
                abs(a - b) > 1e-6
                for p, q in zip(actual, [m["world_xyz"] for m in scene["markers"]])
                for a, b in zip(p, q)
            ):
                raise ValueError("Runtime marker positions changed")
        if (
            canonical_hash(scene) != frozen["scene_sha256"]
            or summary["scene_manifest_hash"] != frozen["scene_sha256"]
        ):
            raise ValueError("Frozen camera/marker/height badge contract changed")
    return {
        "status": "PASS",
        "runs_bound": 9,
        "inspections_reported": 90,
        "coordinate_sha256": layout["coordinate_sha256"],
        "source_hash": frozen["source_hash"],
        "scene_sha256": frozen["scene_sha256"],
        "model_hashes": models,
        "scope": "Frozen input bindings only; independent physical and media checks are required separately.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--freeze", required=True)
    p.add_argument("--freeze-sha256", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    raw = Path(a.freeze).read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.freeze_sha256:
        raise ValueError("Freeze changed after acceptance declaration")
    records = [
        {
            "summary": json.loads(path.read_text()),
            "scenario": json.loads((path.parent / "scenario.json").read_text()),
            "scene": json.loads((path.parent / "scene_manifest.json").read_text()),
        }
        for path in sorted(Path(a.root).glob("*/mission_summary.json"))
    ]
    result = verify_bound_runs(json.loads(raw), json.loads(Path(a.layout).read_text()), records)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
