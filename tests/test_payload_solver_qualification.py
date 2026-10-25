"""Independent solver checks must reject incomplete, weak or stale evidence."""

import numpy as np
import pytest
from test_payload_recovery_planar import record, row


def manifest():
    from adaptive_drone_patrol.payload_solver_qualification import make_manifest

    return make_manifest(base_substeps=16)


def test_new_manifest_preserves_62_full_conditions_and_passive32():
    from adaptive_drone_patrol.payload_solver_qualification import validate_manifest, passive_cases

    m = manifest()
    assert len(m["cases"]) == 62
    assert len(passive_cases(m)) == 32
    assert m["settings"]["backend"] == "mujoco"
    assert m["settings"]["integrator"] == "RK4"
    assert validate_manifest(m)


@pytest.mark.parametrize("field", ["cases", "limits", "settings"])
def test_modified_domain_or_limit_is_rejected(field):
    from adaptive_drone_patrol.payload_solver_qualification import validate_manifest

    m = manifest()
    if field == "cases":
        m["cases"].pop()
    elif field == "limits":
        m["limits"]["absolute_unexplained_passive_energy_fraction"] = 0.051
    else:
        m["settings"]["precision"] = "32"
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_empty_and_duplicate_raw_cannot_pass_passive_gate():
    from adaptive_drone_patrol.payload_solver_qualification import screen_passive

    m = manifest()
    case = next(c for c in m["cases"] if c["id"] == "sweep_170")
    r = record()["runtime"]
    assert not screen_passive(case, [], r)["passed"]
    rows = [row(np.array([0.1, 0.1, 0.0, 0.0]), 0)] * 3001
    assert not screen_passive(case, rows, r)["passed"]


def test_new_gate_rejects_absolute_13_percent_energy_loss_even_when_positive_growth_is_zero():
    from adaptive_drone_patrol.payload_solver_qualification import screen_passive

    m = manifest()
    case = next(c for c in m["cases"] if c["id"] == "sweep_170")
    r = record()["runtime"]
    # Coherent planar positions and velocities with an artificially shrinking
    # excitation still fail the independent conservative energy balance.
    rows = []
    for i in range(3001):
        x = row(np.array([0.1 * (1 - 0.2 * i / 3000), 0.1 * (1 - 0.2 * i / 3000), 0.0, 0.0]), i)
        x["actual_applied_generalized_force"] = [0.0] * 6
        x["actual_passive_generalized_force"] = [0.0] * 6
        rows.append(x)
    result = screen_passive(case, rows, r)
    assert not result["passed"]
    assert "ABSOLUTE_PASSIVE_ENERGY_BALANCE" in result["failure_codes"]


def test_optimizer_admission_rejects_status_only_or_missing_raw(tmp_path):
    from adaptive_drone_patrol.payload_solver_qualification import require_qualification

    with pytest.raises(ValueError):
        require_qualification({"status": "PASS", "physical_training_admitted": True}, {}, tmp_path)


def test_missing_complete_domain_cannot_issue_training_capability(tmp_path):
    from adaptive_drone_patrol.payload_solver_qualification import issue_qualification

    with pytest.raises(ValueError):
        issue_qualification({"passive": tmp_path})
