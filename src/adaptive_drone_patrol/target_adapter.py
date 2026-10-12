"""External world-frame goals for the inspected Genesis 1.4.3 HoverEnv."""

import importlib.util
from pathlib import Path
import numpy as np
from .util import sha256


def validate_target(world_xyz, bounds):
    a = np.asarray(world_xyz, dtype=float)
    b = np.asarray(bounds, dtype=float)
    if a.shape != (3,) or b.shape != (3, 2) or not np.isfinite(a).all():
        raise ValueError('Target must be a finite world-frame xyz vector')
    if np.any(a < b[:, 0] - 1e-7) or np.any(a > b[:, 1] + 1e-7):
        raise ValueError('Target is outside the frozen training envelope')
    return a


def load_hover_class(source, expected_hash=None, external=True):
    source = Path(source)
    if expected_hash and sha256(source) != expected_hash:
        raise ValueError('Pinned HoverEnv source hash mismatch')
    text = source.read_text()
    if text.count('GUI=True,') != 1:
        raise ValueError('Expected exactly one notebook camera GUI setting')
    spec = importlib.util.spec_from_file_location('pinned_hover_env', source)
    module = importlib.util.module_from_spec(spec)
    exec(compile(text.replace('GUI=True,', 'GUI=False,', 1), str(source), 'exec'), module.__dict__)
    base = module.HoverEnv
    if not external:
        return base

    class ExternalHoverEnv(base):
        """Suppress evaluation resampling/reset while preserving upstream physics."""
        external_target = False
        initializing = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initializing = False
            self.reset_detected = False

        def _resample_commands(self, envs_idx):
            if not self.external_target:
                super()._resample_commands(envs_idx)

        def reset_idx(self, envs_idx):
            if self.initializing:
                return super().reset_idx(envs_idx)
            if len(envs_idx):
                self.reset_detected = True
            # Leave pre-reset pose, velocity, done and cause intact for the runner.

    return ExternalHoverEnv


class TargetAdapter:
    def __init__(self, env):
        self.env = env
        self.bounds = [env.command_cfg[k] for k in ('pos_x_range', 'pos_y_range', 'pos_z_range')]
        self.target = None

    def set_target(self, world_xyz):
        import torch
        a = validate_target(world_xyz, self.bounds)
        e = self.env
        command = torch.as_tensor(a, dtype=e.commands.dtype, device=e.commands.device)
        e.commands[:] = command
        e.external_target = True
        e.rel_pos = e.commands - e.base_pos
        e.last_rel_pos = e.commands - e.last_base_pos
        e._update_observation()
        if e.target is not None:
            e.target.set_pos(e.commands, zero_velocity=True)
        self.target = a.copy()
        self.assert_target()

    def assert_target(self):
        e = self.env
        actual = e.commands[0].detach().cpu().numpy()
        if self.target is None or not np.allclose(actual, self.target, atol=1e-6):
            raise RuntimeError('Effective target mismatch')
        relative = e.get_observations()['policy'][0, :3].detach().cpu().numpy()
        expected = np.clip((self.target - e.base_pos[0].detach().cpu().numpy()) * e.obs_scales['rel_pos'], -1, 1)
        if not np.allclose(relative, expected, atol=1e-6):
            raise RuntimeError('Stale target-relative observation')

    def current_observation(self):
        self.assert_target()
        return self.env.get_observations()

    def step(self, actions):
        self.assert_target()
        if not actions.isfinite().all():
            raise RuntimeError('Nonfinite motor action')
        result = self.env.step(actions)
        self.assert_target()
        return result
