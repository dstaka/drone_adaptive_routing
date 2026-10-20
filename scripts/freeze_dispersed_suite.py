#!/usr/bin/env python
"""Freeze one reviewed layout/model/camera/limit set before any acceptance trial."""

import argparse
import json
import pickle
from pathlib import Path
from adaptive_drone_patrol.dispersed_layout import acceptance_prerequisites, coordinate_hash
from adaptive_drone_patrol.util import load_config, json_write, sha256
from adaptive_drone_patrol.scene import scene_manifest, validate_scene
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.analytical import content_hash
from adaptive_drone_patrol.transition_check import verify as verify_transition


def validated_probe_receipt(root):
    root = Path(root)
    receipt = json.loads((root / "independent-check.json").read_text())
    fresh = verify_transition(root)
    if receipt != fresh:
        raise ValueError("Stale or unbound independent transition receipt")
    return fresh


def validate_flight_contract(configs):
    env, obs, _, commands, _ = configs
    expected = {
        "num_actions": 4,
        "base_init_pos": [0.0, 0.0, 1.0],
        "termination_if_close_to_ground": 0.1,
        "termination_if_roll_greater_than": 60.0,
        "termination_if_pitch_greater_than": 60.0,
        "termination_if_x_greater_than": 6.0,
        "termination_if_y_greater_than": 4.0,
        "termination_if_z_greater_than": 3.0,
    }
    if any(env.get(k) != v for k, v in expected.items()) or obs.get("obs_scales", {}).get("rel_pos") != 1 / 3:
        raise ValueError("Actual flight safety/observation config differs from adopted contract")
    if any(
        commands.get(k) != v
        for k, v in {
            "pos_x_range": [-2.7, 2.7],
            "pos_y_range": [-1.8, 1.8],
            "pos_z_range": [1.0, 3.0],
        }.items()
    ):
        raise ValueError("Actual trained target envelope differs from expanded domain")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout", required=True)
    p.add_argument("--execution", required=True)
    p.add_argument("--probe-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    layout = json.loads(Path(a.layout).read_text())
    cfg = load_config(a.execution)
    roots = sorted(Path(a.probe_root).glob("*/summary.json"))
    probes = [json.loads(path.read_text()) for path in roots]
    gate = acceptance_prerequisites(layout, probes, cfg["flight_checkpoint_hash"], cfg["flight_cfg_hash"])
    if not gate["passed"]:
        raise ValueError(gate["rejection_reasons"])
    for path, probe in zip(roots, probes):
        receipt = validated_probe_receipt(path.parent)
        if (
            receipt["status"] != "PASS"
            or receipt["execution_seed"] != probe["execution_seed"]
            or receipt["legs_verified"] != 9
        ):
            raise ValueError("Missing independent transition replay")
        if probe["suite_sha256"] != sha256("configs/dispersed_transitions.yaml"):
            raise ValueError("Transition suite differs from frozen declaration")
    for path, key in [
        ("flight_checkpoint", "flight_checkpoint_hash"),
        ("flight_cfg", "flight_cfg_hash"),
        ("routing_model", "routing_model_hash"),
        ("hover_source", "hover_source_hash"),
    ]:
        if sha256(cfg[path]) != cfg[key]:
            raise ValueError("Artifact hash mismatch: " + path)
    with Path(cfg["flight_cfg"]).open("rb") as stream:
        validate_flight_contract(pickle.load(stream))
    limits = load_config(cfg["flight_eval_config"])
    required = {
        "arrival_radius_m": 0.2,
        "arrival_speed_mps": 0.3,
        "arrival_hold_s": 0.2,
        "inspection_dwell_s": 0.5,
        "floor_m": 0.1,
        "max_tilt_deg": 60.0,
        "max_leg_s": 30.0,
        "max_mission_s": 120.0,
    }
    if any(limits[k] != v for k, v in required.items()):
        raise ValueError("Acceptance thresholds changed")
    scenarios = [Scenario.from_dict(load_config(path)) for path in cfg["scenario_files"]]
    if [s.scenario_id for s in scenarios] != ["normal", "emergency", "cost_shift"]:
        raise ValueError("Missing paired scenarios")
    for s in scenarios:
        if coordinate_hash(s.coords) != layout["coordinate_sha256"] or s.priorities != layout["priorities"]:
            raise ValueError("Scenario differs from operator-reviewed geometry/priority vector")
    if scenarios[1].event != {"kind": "priority", "after": 1, "checkpoint": "C", "value": 100}:
        raise ValueError("Emergency contract changed")
    if scenarios[2].event != {
        "kind": "cost",
        "after": 1,
        "edges": [layout["declared_cost_edge"]],
        "value": 5,
    }:
        raise ValueError("Declared directed edge changed")
    scene = scene_manifest(scenarios[0])
    validate_scene(scene)
    root = Path.cwd().resolve()
    sources = {
        str(path.relative_to(root)): sha256(path)
        for folder in ["src", "scripts"]
        for path in sorted((root / folder).rglob("*"))
        if path.is_file() and path.suffix in [".py", ".sh"] and "__pycache__" not in str(path)
    }
    json_write(
        out,
        {
            "schema_version": 1,
            "status": "FROZEN",
            "base_revision": "92a2a4d87cb9e3f6768e698f71e98a33e18c45db",
            "coordinate_sha256": layout["coordinate_sha256"],
            "operator_spatial_review": layout["operator_review"],
            "model_hashes": {
                key: cfg[key]
                for key in [
                    "flight_checkpoint_hash",
                    "flight_cfg_hash",
                    "routing_model_hash",
                    "hover_source_hash",
                ]
            },
            "scene": scene,
            "scene_sha256": content_hash(scene),
            "limits": limits,
            "flight_acceptance_config_sha256": sha256(cfg["flight_eval_config"]),
            "scenario_hashes": {
                s.scenario_id: sha256(path) for s, path in zip(scenarios, cfg["scenario_files"])
            },
            "cost_edge": layout["declared_cost_edge"],
            "cost_change": [1, 5],
            "fixed_execution_seeds": [1, 2, 3],
            "missions": 9,
            "inspections": 90,
            "source_hash": content_hash(sources),
            "sources": sources,
            "transition_summaries": [
                {"seed": v["execution_seed"], "summary_sha256": sha256(path)}
                for path, v in zip(roots, probes)
            ],
        },
    )
    print(
        "Frozen reviewed geometry, compatible model pair, camera, finite timeouts, and fixed 9-run denominator"
    )


if __name__ == "__main__":
    main()
