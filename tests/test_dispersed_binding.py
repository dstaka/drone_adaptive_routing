"""Frozen-input binding examples; these are not physical-flight evidence."""

import copy
import importlib.util
from pathlib import Path
import pytest
from adaptive_drone_patrol.analytical import demo_fixture, content_hash
from adaptive_drone_patrol.dispersed_layout import layout_candidate, coordinate_hash
from adaptive_drone_patrol.scene import scene_manifest


def test_binding_rejects_wrong_geometry_source_camera_and_model():
    spec = importlib.util.spec_from_file_location(
        "binding", Path(__file__).parents[1] / "scripts/verify_dispersed_binding.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verify = getattr(module, "verify_bound_runs", None)
    assert verify is not None
    s = demo_fixture(40029, "integer-100-v1")
    s.coords = layout_candidate(51000, 3353)
    scene = scene_manifest(s)
    layout = {
        "coordinate_sha256": coordinate_hash(s.coords),
        "priorities": s.priorities,
        "declared_cost_edge": ["F", "A"],
    }
    hashes = {
        "flight_checkpoint_hash": "motor",
        "flight_cfg_hash": "cfg",
        "routing_model_hash": "route",
        "hover_source_hash": "hover",
    }
    frozen = {
        "coordinate_sha256": layout["coordinate_sha256"],
        "scene": scene,
        "scene_sha256": content_hash(scene),
        "source_hash": "source",
        "model_hashes": hashes,
        "limits": {"max_leg_s": 30},
        "scenario_hashes": {k: k for k in ["normal", "emergency", "cost_shift"]},
        "flight_acceptance_config_sha256": "limits",
    }
    records = []
    for kind in ["normal", "emergency", "cost_shift"]:
        for seed in [1, 2, 3]:
            scenario = s.to_dict()
            scenario["scenario_id"] = kind
            scenario["event"] = (
                None
                if kind == "normal"
                else {"kind": "priority", "after": 1, "checkpoint": "C", "value": 100}
                if kind == "emergency"
                else {"kind": "cost", "after": 1, "edges": [["F", "A"]], "value": 5}
            )
            summary = {
                "scenario_id": kind,
                "execution_seed": seed,
                "flight_checkpoint_hash": "motor",
                "flight_config_hash": "cfg",
                "routing_model_hash": "route",
                "hover_source_hash": "hover",
                "execution_source_hash": "source",
                "scenario_hash": kind,
                "scene_manifest_hash": content_hash(scene),
                "arrival_settings": frozen["limits"],
                "flight_acceptance_config_hash": "limits",
                "status": "SUCCEEDED",
                "completed_count": 10,
            }
            records.append({"summary": summary, "scenario": scenario, "scene": copy.deepcopy(scene)})
    assert verify(frozen, layout, records)["status"] == "PASS"
    for key in ["flight_checkpoint_hash", "execution_source_hash", "scene_manifest_hash"]:
        bad = copy.deepcopy(records)
        bad[0]["summary"][key] = "changed"
        with pytest.raises(ValueError):
            verify(frozen, layout, bad)
    bad = copy.deepcopy(records)
    bad[0]["scenario"]["coords"][1][2] += 0.01
    with pytest.raises(ValueError):
        verify(frozen, layout, bad)
