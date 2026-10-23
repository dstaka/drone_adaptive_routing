"""Actual runtime ordering; uses the same MuJoCo source as qualification."""

import json
import os
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("mujoco")


@pytest.fixture
def fixture(tmp_path):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload

    original = os.environ.get("PAYLOAD_SOLVER_CF2X")
    if not original:
        pytest.skip("Actual source asset required")
    return MuJoCoFixture(
        Path(original), tmp_path / "physics", Payload(0.02700047753751278, 0.2, 0.045), substeps=16
    )


def test_runtime_current_action_and_retarget_preserve_state_history(tmp_path, fixture):
    from adaptive_drone_patrol.payload_solver_runtime import SolverRuntime

    core = SolverRuntime(fixture, tmp_path / "runtime")
    core.reset_case()
    core.step([0.1, -0.2, 2.0, -2.0])
    assert core.observation[13:17] == pytest.approx([0.1, -0.2, 1.0, -1.0])
    assert fixture.rpm == pytest.approx((1 + 0.8 * np.array([0.1, -0.2, 1.0, -1.0])) * 14468.429183500699)
    before = fixture.raw_state()
    core.set_target([0.15, 0.0, 1.0], target_id="A")
    assert fixture.raw_state() == before
    assert core.observation[13:17] == pytest.approx([0.1, -0.2, 1.0, -1.0])
    assert core.step_count == 1
    core.close()


def test_runtime_exception_retains_actual_partial_state_and_preceding_window(tmp_path, fixture):
    from adaptive_drone_patrol.payload_solver_runtime import SolverRuntime

    core = SolverRuntime(fixture, tmp_path / "runtime")
    core.reset_case()
    actual_step = fixture.step_rpm

    def failing_step(*args, **kwargs):
        actual_step(*args, **kwargs)
        fixture.data.qvel[0] = float("nan")
        raise RuntimeError("Injected partial backend failure")

    fixture.step_rpm = failing_step
    with pytest.raises(RuntimeError, match="quarantined"):
        core.step([0.0] * 4)
    abort = json.loads((core.output / "abort.json").read_text())
    assert np.isnan(abort["actual"]["qvel"][0])
    assert abort["actual"]["time_s"] > 0
    assert abort["preceding"]
    with pytest.raises(RuntimeError):
        core.step([0.0] * 4)
    core.close()


def test_runtime_dwell_requires_100_full_intervals(tmp_path, fixture):
    from adaptive_drone_patrol.payload_solver_runtime import SolverRuntime

    core = SolverRuntime(fixture, tmp_path / "runtime")
    core.reset_case()
    action = (np.sqrt((0.02700047753751278 + 0.0001 + fixture.payload.mass) / 0.02700047753751278) - 1) / 0.8
    results = [core.step([action] * 4) for _ in range(100)]
    assert results[-1]["complete"]
    assert not any(r["complete"] for r in results[:-1])
    assert results[-1]["vehicle_collision_aabbs"]["vehicle_collision"][0][2] < results[-1]["drone_xyz"][2]
    core.close()


def test_initial_reset_keeps_world_cube_orientation_and_declared_drone_roll(tmp_path, fixture):
    from adaptive_drone_patrol.payload_solver_runtime import SolverRuntime
    from adaptive_drone_patrol.payload_cases import initial_state

    initial = initial_state(np.random.default_rng(81080), fixture.payload.side_m)
    core = SolverRuntime(fixture, tmp_path / "runtime")
    core.reset_case(initial=initial)
    actual = fixture.raw_state()
    for body in ["drone", "package"]:
        assert actual[body + "_xyz"] == pytest.approx(initial[body + "_xyz"], abs=1e-10)
        assert actual[body + "_quaternion_wxyz"] == pytest.approx(
            initial[body + "_quaternion_wxyz"], abs=1e-10
        )
    core.close()
