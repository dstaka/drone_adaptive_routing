"""Actual N1 learning environment checks; no optimizer is constructed."""

import os
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")


def test_actual_training_steps_and_retarget_keep_bodies_history_and_valid_counts(tmp_path):
    from adaptive_drone_patrol.payload_solver_training import SolverTrainingEnv

    original = os.environ.get("PAYLOAD_SOLVER_CF2X")
    if not original:
        pytest.skip("Actual pinned original asset required")
    env = SolverTrainingEnv(tmp_path / "env", Path(original), randomized=True, seed=83001, pilot=True)
    try:
        assert env.get_observations()["policy"].shape == (1, 29)
        payload = env.payload
        action = (np.sqrt((0.02700047753751278 + 0.0001 + payload.mass) / 0.02700047753751278) - 1) / 0.8
        for _ in range(20):
            _, reward, done, _ = env.step(torch.full((1, 4), float(action)))
            assert torch.isfinite(reward).all() and not done.item()
        assert env.counters()["valid"] == env.counters()["generated"] == 20
        assert env.counters()["invalid"] == 0
        actual, history = env.core.fixture.raw_state(), env.core.last_actions.copy()
        env.core.set_target([0.15, 0.0, 1.0], target_id="actual-continuity")
        assert env.core.fixture.raw_state() == actual
        assert np.array_equal(env.core.last_actions, history)
        assert env.episode_count == 1 and env.core.step_count == 20
        assert env.get_observations()["policy"].shape == (1, 29)
        assert len(env.initial["drone_xyz"]) == 3
    finally:
        env.close()
