"""Actual GPU integration checks; these never run as CPU substitutes."""
import os
import numpy as np
import pytest
from adaptive_drone_patrol.util import load_config

pytestmark = pytest.mark.gpu


def test_real_target_refresh_continuity_no_resample_and_pre_reset_capture():
    from adaptive_drone_patrol.flight_agent import FlightSession
    cfg = load_config(os.environ.get('PATROL_EXECUTION_CONFIG', 'configs/execution.yaml'))
    f = FlightSession(cfg, seed=11, camera=False)
    before = f.env.drone.get_pos().clone()
    actions_before = f.env.last_actions.clone()
    f.set_target([.6, 0, 1])
    assert np.array_equal(before.cpu().numpy(), f.env.drone.get_pos().cpu().numpy())
    assert np.array_equal(actions_before.cpu().numpy(), f.env.last_actions.cpu().numpy())
    assert np.allclose(f.adapter.current_observation()['policy'][0, :3].cpu().numpy(), [.2, 0, 0])
    f.env._resample_commands([0])
    assert np.allclose(f.env.commands[0].cpu().numpy(), [.6, 0, 1])
    for _ in range(60):
        sample = f.advance()
        assert not sample['done']
    assert np.linalg.norm(np.asarray(sample['xyz']) - [0, 0, 1]) > .01
    f.env.env_cfg['termination_if_close_to_ground'] = 2.
    failed = f.advance()
    assert failed['done'] and failed['reset']
    assert np.allclose(failed['xyz'], f.env.base_pos[0].cpu().numpy())
    assert np.linalg.norm(np.asarray(failed['xyz']) - [0, 0, 1]) > .01
