"""Load only explicit, provenance-checked operator-owned PPO artifacts."""

import pickle
from pathlib import Path
from .target_adapter import load_hover_class, TargetAdapter
from .util import sha256


def load_flight(config, external=True, camera=True, seed=1):
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    for key, hash_key in [('flight_checkpoint', 'flight_checkpoint_hash'), ('flight_cfg', 'flight_cfg_hash')]:
        if not config.get(key) or not config.get(hash_key):
            raise ValueError(f'Provide explicit {key} and {hash_key} in private execution config')
        if sha256(config[key]) != config[hash_key]:
            raise ValueError(f'Artifact hash mismatch: {key}')
    source = config['hover_source']
    cls = load_hover_class(source, config['hover_source_hash'], external)
    gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=seed)
    with Path(config['flight_cfg']).open('rb') as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg['reward_scales'] = {}
    env_cfg['visualize_target'] = camera
    env_cfg['visualize_camera'] = camera
    if external:
        env_cfg['episode_length_s'] = 150
    env = cls(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
              command_cfg=command_cfg, show_viewer=False)
    runner = OnPolicyRunner(env, train_cfg, str(Path(config['flight_checkpoint']).parent), device=gs.device)
    runner.load(config['flight_checkpoint'])
    policy = runner.get_inference_policy(device=gs.device)
    if env.get_observations()['policy'].shape != (1, 17) or env.num_actions != 4:
        raise RuntimeError('Notebook observation/action architecture mismatch')
    return env, TargetAdapter(env), policy


class FlightSession:
    """Single-environment learned motor inference and immutable physical samples."""
    def __init__(self, config, seed=1, camera=True):
        self.env, self.adapter, self.policy = load_flight(config, seed=seed, camera=camera)
        self.dt = self.env.dt
        if camera:
            self.cam = self.env.cam

    def set_target(self, target):
        import numpy as np
        before = self.env.drone.get_pos().clone()
        self.adapter.set_target(target)
        after = self.env.drone.get_pos().clone()
        if not np.array_equal(before.cpu().numpy(), after.cpu().numpy()):
            raise RuntimeError('Target switch modified physical position')

    def advance(self):
        import torch
        with torch.no_grad():
            actions = self.policy(self.adapter.current_observation())
            _, _, done, _ = self.adapter.step(actions)
        e = self.env
        return {'xyz': e.drone.get_pos()[0].detach().cpu().tolist(),
                'velocity': e.drone.get_vel()[0].detach().cpu().tolist(),
                'quaternion_wxyz': e.drone.get_quat()[0].detach().cpu().tolist(),
                'effective_target': e.commands[0].detach().cpu().tolist(),
                'actions': actions[0].detach().cpu().tolist(),
                'tilt_deg': float(e.base_euler[0, :2].abs().max().item()),
                'done': bool(done[0].item()), 'reset': bool(e.reset_detected)}
