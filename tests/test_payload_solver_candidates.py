"""Versioned solver diagnostics preserve physical cases and powered barriers."""

import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def solver_manifest(**kwargs):
    from adaptive_drone_patrol.payload_solver_candidates import make_manifest

    return make_manifest(
        json.loads((ROOT / "configs/payload_retry_probe_manifest.json").read_text()), **kwargs
    )


def probe():
    spec = importlib.util.spec_from_file_location("solver_probe", ROOT / "scripts/probe_payload_ball.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_solver_manifest_preserves_domain_and_declares_precision_integrator_and_balance():
    from adaptive_drone_patrol.payload_solver_candidates import validate_manifest

    m = solver_manifest(base_substeps=16, precision="64", integrator="implicitfast")
    assert len(m["cases"]) == 62
    assert m["settings"]["precision"] == "64"
    assert m["settings"]["integrator"] == "implicitfast"
    assert m["limits"]["absolute_unexplained_passive_energy_fraction"] == 0.05
    assert validate_manifest(m)
    assert probe().validate_probe_declaration(m)


@pytest.mark.parametrize("mutation", ["case", "energy", "precision", "integrator"])
def test_mutated_candidate_rejected(mutation):
    from adaptive_drone_patrol.payload_solver_candidates import validate_manifest

    m = solver_manifest()
    if mutation == "case":
        m["cases"].pop()
    elif mutation == "energy":
        m["limits"]["absolute_unexplained_passive_energy_fraction"] = 0.06
    elif mutation == "precision":
        m["settings"]["precision"] = "16"
    else:
        m["settings"]["integrator"] = "midpoint"
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_solver_powered_path_cannot_bypass_unimplemented_qualification():
    m = solver_manifest()
    powered = next(c for c in m["cases"] if c["kind"] == "powered")
    with pytest.raises(ValueError, match="32 passive"):
        probe().require_probe_stage(m, powered, "loaded")


def test_fixture_integrator_rejected_before_importing_genesis(tmp_path):
    from adaptive_drone_patrol.payload_ball_bank import build_ball_fixture
    from adaptive_drone_patrol.payload_contract import Payload

    with pytest.raises(ValueError, match="integrator"):
        build_ball_fixture(tmp_path, Payload(0.02700047753751278, 0.2, 0.045), integrator="invented")


def test_cpu_diagnostic_backend_is_explicit_and_rejects_other_platforms():
    m = solver_manifest(execution_backend="cpu")
    assert m["settings"]["execution_backend"] == "cpu"
    with pytest.raises(ValueError):
        solver_manifest(execution_backend="unrecorded")
