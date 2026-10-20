"""Acceptance cannot freeze without approvals and compatible physical probes."""

from adaptive_drone_patrol import dispersed_layout


def test_acceptance_gate_keeps_missing_review_and_failed_probes_incomplete():
    approved = {"operator_review": {"status": "APPROVED"}, "coordinate_sha256": "layout"}
    probes = [
        {
            "status": "SUCCEEDED",
            "execution_seed": seed,
            "legs": [{}] * 9,
            "checkpoint_sha256": "model",
            "config_sha256": "config",
            "suite_sha256": "suite",
            "evidence_kind": "physical",
        }
        for seed in [11, 12, 31, 32, 33]
    ]
    gate = getattr(dispersed_layout, "acceptance_prerequisites", None)
    assert gate is not None
    assert gate(approved, probes, "model", "config")["passed"]
    assert not gate({"operator_review": "PENDING"}, probes, "model", "config")["passed"]
    probes[0]["status"] = "FAILED"
    assert not gate(approved, probes, "model", "config")["passed"]
    probes[0]["status"] = "SUCCEEDED"
    probes[0]["checkpoint_sha256"] = "other"
    assert not gate(approved, probes, "model", "config")["passed"]
    probes[0]["checkpoint_sha256"] = "model"
    assert not gate(approved, probes[:-1], "model", "config")["passed"]
