"""Motor/reset admission rejects labels without actual raw and native model data."""

import pytest


def test_status_only_motor_receipt_is_not_an_audit(tmp_path):
    from adaptive_drone_patrol.payload_solver_audits import verify_motor_audit

    with pytest.raises(ValueError):
        verify_motor_audit(tmp_path)


def test_status_only_reset_receipt_is_not_an_audit(tmp_path):
    from adaptive_drone_patrol.payload_solver_audits import verify_reset_audit

    with pytest.raises(ValueError):
        verify_reset_audit(tmp_path)
