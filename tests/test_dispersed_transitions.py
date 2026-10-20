"""Transition evidence must describe the declared physical task and source bytes."""

import importlib.util
import json
from pathlib import Path
import pytest


def load_checker():
    path = Path(__file__).parents[1] / "scripts/verify_dispersed_transitions.py"
    spec = importlib.util.spec_from_file_location("dispersed_transition_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_freezer():
    path = Path(__file__).parents[1] / "scripts/freeze_dispersed_suite.py"
    spec = importlib.util.spec_from_file_location("dispersed_freezer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stationary_probe(root):
    root.mkdir()
    legs = []
    samples = []
    for i in range(9):
        legs.append(
            {
                "index": i,
                "kind": "stationary",
                "target": [0, 0, 1],
                "start_xyz": [0, 0, 1],
                "start_step": i * 50,
                "end_step": (i + 1) * 50,
                "duration_s": 0.5,
            }
        )
        for j in range(50):
            samples.append(
                {
                    "step": i * 50 + j + 1,
                    "target_index": i,
                    "target": [0, 0, 1],
                    "xyz": [0, 0, 1],
                    "velocity": [0, 0, 0],
                    "quaternion_wxyz": [1, 0, 0, 0],
                    "effective_target": [0, 0, 1],
                    "actions": [0] * 4,
                    "tilt_deg": 0,
                    "relative_observation_absmax": 0,
                    "done": False,
                    "reset": False,
                }
            )
    (root / "summary.json").write_text(
        json.dumps({"status": "SUCCEEDED", "execution_seed": 11, "dt": 0.01, "legs": legs, "steps": 450})
    )
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    return root


def test_stationary_hovers_cannot_certify_declared_domain_transitions(tmp_path):
    checker = load_checker()
    root = stationary_probe(tmp_path / "fake")
    with pytest.raises(ValueError):
        checker.verify(root)


def test_transition_receipt_binds_exact_evidence_and_execution_source(tmp_path):
    checker = load_checker()
    binding = getattr(checker, "evidence_binding", None)
    assert binding is not None
    root = stationary_probe(tmp_path / "fake")
    before = binding(root)
    with (root / "samples.jsonl").open("a") as f:
        f.write("{}\n")
    assert binding(root) != before


def declared_probe(root):
    import math
    from adaptive_drone_patrol.transition_check import TARGETS, SUITE_SHA256, execution_sources

    root.mkdir()
    samples = []
    legs = []
    pose = [0.0, 0.0, 1.0]
    for i, (target, kind) in enumerate(TARGETS):
        start = len(samples)
        before = pose.copy()
        count = max(1, math.ceil(math.dist(pose, target) / 0.01))
        velocity = [(b - a) / (count * 0.01) for a, b in zip(pose, target)]
        for j in range(count + 50):
            xyz = [a + (b - a) * min(j + 1, count) / count for a, b in zip(before, target)]
            samples.append(
                {
                    "step": len(samples) + 1,
                    "target_index": i,
                    "target": target,
                    "xyz": xyz,
                    "velocity": velocity if j < count else [0.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "effective_target": target,
                    "actions": [0.0] * 4,
                    "tilt_deg": 0.0,
                    "relative_observation_absmax": min(1, max(abs(a - b) for a, b in zip(target, xyz)) / 3),
                    "done": False,
                    "reset": False,
                }
            )
        pose = target.copy()
        legs.append(
            {
                "index": i,
                "kind": kind,
                "target": target,
                "start_xyz": before,
                "start_step": start,
                "end_step": len(samples),
                "duration_s": (len(samples) - start) * 0.01,
            }
        )
    sources = execution_sources()
    summary = {
        "status": "SUCCEEDED",
        "execution_seed": 11,
        "dt": 0.01,
        "evidence_kind": "physical",
        "legs": legs,
        "steps": len(samples),
        "suite_sha256": SUITE_SHA256,
        "execution_sources": sources,
        "hover_source_sha256": sources["third_party/genesis_drone/hover_env.py"],
    }
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    return root


def test_freeze_recomputes_receipts_and_rejects_stale_metadata(tmp_path):
    checker = load_checker()
    freezer = load_freezer()
    root = declared_probe(tmp_path / "declared")
    receipt = checker.verify(root)
    (root / "independent-check.json").write_text(json.dumps(receipt))
    validate = getattr(freezer, "validated_probe_receipt", None)
    assert validate is not None
    assert validate(root) == receipt
    summary = json.loads((root / "summary.json").read_text())
    summary["extra_note"] = "changed after verification"
    (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        validate(root)


def test_freeze_inspects_actual_flight_config_contract():
    freezer = load_freezer()
    inspect = getattr(freezer, "validate_flight_contract", None)
    assert inspect is not None
    cfg = [
        {
            "num_actions": 4,
            "base_init_pos": [0.0, 0.0, 1.0],
            "termination_if_close_to_ground": 0.1,
            "termination_if_roll_greater_than": 60.0,
            "termination_if_pitch_greater_than": 60.0,
            "termination_if_x_greater_than": 6.0,
            "termination_if_y_greater_than": 4.0,
            "termination_if_z_greater_than": 3.0,
        },
        {"obs_scales": {"rel_pos": 1 / 3}},
        {},
        {"pos_x_range": [-2.7, 2.7], "pos_y_range": [-1.8, 1.8], "pos_z_range": [1.0, 3.0]},
        {},
    ]
    inspect(cfg)
    cfg[1]["obs_scales"]["rel_pos"] = 1 / 6
    with pytest.raises(ValueError):
        inspect(cfg)
