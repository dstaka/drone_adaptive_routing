"""Production inference requires complete raw admission and a bound checkpoint."""

import pytest


def test_production_admission_rejects_a_status_label():
    from adaptive_drone_patrol.payload_solver_flight import SolverAdmission

    with pytest.raises(ValueError):
        SolverAdmission.from_receipt({"status": "PASS"}, {})


def test_admission_cannot_be_constructed_from_unverified_metadata():
    from adaptive_drone_patrol.payload_solver_flight import SolverAdmission

    with pytest.raises(ValueError):
        SolverAdmission({"status": "PASS"}, object())
