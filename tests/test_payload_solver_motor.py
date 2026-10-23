"""Independent original URDF importer versus generated same-backend vehicle."""

import os
from pathlib import Path
import pytest

pytest.importorskip("mujoco")


def test_independent_original_import_matches_unloaded_motor_mechanics(tmp_path):
    from adaptive_drone_patrol.payload_solver_motor import original_fixture
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture, HOVER_RPM
    from adaptive_drone_patrol.payload_contract import Payload

    original = os.environ.get("PAYLOAD_SOLVER_CF2X")
    if not original:
        pytest.skip("Actual source asset required")
    a = original_fixture(Path(original), tmp_path / "original", substeps=16)
    b = MuJoCoFixture(
        Path(original),
        tmp_path / "unloaded",
        Payload(0.02700047753751278, 0.2, 0.045),
        loaded=False,
        substeps=16,
    )
    assert a.runtime()["free_drone_mass_kg"] == pytest.approx(0.02700047753751278, rel=1e-12)
    for _ in range(5):
        a.step_rpm([HOVER_RPM, 0.0, 0.0, 0.0])
        b.step_rpm([HOVER_RPM, 0.0, 0.0, 0.0])
        for field in ["xyz", "quaternion_wxyz", "velocity", "angular_velocity"]:
            assert a.sample()["drone_" + field] == pytest.approx(b.sample()["drone_" + field], abs=1e-10)
