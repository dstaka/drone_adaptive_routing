"""Load only explicit, provenance-checked operator-owned PPO artifacts."""

import pickle
from pathlib import Path
from .target_adapter import load_hover_class, TargetAdapter
from .util import sha256
from .patrol_runner import quaternion_tilt_degrees


def load_flight(config, external=True, camera=True, seed=1):
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    for key, hash_key in [('flight_checkpoint', 'flight_checkpoint_hash'), ('flight_cfg', 'flight_cfg_hash')]:
        if not config.get(key) or not config.get(hash_key):
            raise ValueError(f'Provide explicit {key} and {hash_key} in private execution config')
        if sha256(config[key]) != config[hash_key]:
            raise ValueError(f'Artifact hash mismatch: {key}')
    source = config['hover_source']
    manifest = config.get('scene_manifest')
    cls = load_hover_class(source, config['hover_source_hash'], external, scene=manifest)
    with Path(config['flight_cfg']).open('rb') as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg['reward_scales'] = {}
    from .target_adapter import validate_target
    if manifest:
        from .scene import validate_scene
        validate_scene(manifest)
        bounds = [command_cfg[k] for k in ('pos_x_range', 'pos_y_range', 'pos_z_range')]
        for marker in manifest['markers']:
            validate_target(marker['world_xyz'], bounds)
    gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=seed)
    env_cfg['visualize_target'] = camera and manifest is None
    env_cfg['visualize_camera'] = camera and manifest is None
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
        self.scene_manifest = config.get('scene_manifest')
        if self.scene_manifest:
            self.verify_markers()
        if camera:
            self.cam = self.env.cam

    def set_target(self, target):
        import numpy as np
        if self.scene_manifest:
            self.verify_markers()
        before = [x.clone() for x in (self.env.drone.get_pos(), self.env.drone.get_vel(), self.env.last_actions)]
        self.adapter.set_target(target)
        after = [self.env.drone.get_pos(), self.env.drone.get_vel(), self.env.last_actions]
        if any(not np.array_equal(a.cpu().numpy(), b.cpu().numpy()) for a,b in zip(before, after)):
            raise RuntimeError('Target switch modified physical state or motor history')

    def verify_markers(self):
        import numpy as np
        actual = [m.get_pos().detach().cpu().numpy().reshape(-1,3)[0].tolist()
                  for m in self.env.mission_markers]
        expected = [m['world_xyz'] for m in self.scene_manifest['markers']]
        if not np.allclose(actual, expected, atol=1e-6):
            raise RuntimeError('Persistent scene landmark moved or differs from scenario')
        self.rendered_marker_positions = actual

    def advance(self):
        import torch
        with torch.no_grad():
            actions = self.policy(self.adapter.current_observation())
            _, _, done, _ = self.adapter.step(actions)
        return self.sample(actions[0].detach().cpu().tolist(), bool(done[0].item()))

    def sample(self, actions=None, done=False):
        e = self.env
        quaternion = e.drone.get_quat()[0].detach().cpu().tolist()
        return {'xyz': e.drone.get_pos()[0].detach().cpu().tolist(),
                'velocity': e.drone.get_vel()[0].detach().cpu().tolist(),
                'quaternion_wxyz': quaternion,
                'effective_target': e.commands[0].detach().cpu().tolist(),
                'actions': actions if actions is not None else [0.] * 4,
                'relative_observation_absmax': float(e.get_observations()['policy'][0, :3].abs().max().item()),
                'tilt_deg': quaternion_tilt_degrees(quaternion),
                'done': done, 'reset': bool(e.reset_detected)}
