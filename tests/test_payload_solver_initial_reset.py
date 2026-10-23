"""Initial-pose reset evidence requires actual raw, not a declared count."""

import pytest


def test_missing_actual_initial_reset_evidence_cannot_qualify(tmp_path):
    from adaptive_drone_patrol.payload_solver_initial_reset import verify_initial_resets

    with pytest.raises(ValueError):
        verify_initial_resets(tmp_path)
