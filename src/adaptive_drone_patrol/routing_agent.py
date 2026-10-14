"""CPU MaskablePPO training, strict checkpoint contracts and validation selection."""

import json
from pathlib import Path
import time
import math
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from .routing_env import RoutingEnv
from .objective import SCHEMA, schema_for
from .scenario import generate_scenario
from .util import sha256, unique_dir, json_write


class RoutingAgent:
    def __init__(self, path, expected_hash=None):
        self.path = Path(path)
        self.hash = sha256(self.path)
        if expected_hash and self.hash != expected_hash:
            raise ValueError('Routing checkpoint hash mismatch')
        self.schema = json.loads(self.path.with_suffix('.schema.json').read_text())
        self.legacy = self.schema == SCHEMA
        self.num_checkpoints = 3 if self.legacy else self.schema.get('num_checkpoints')
        if not self.legacy:
            contract = generate_scenario(0, 'training', num_checkpoints=self.num_checkpoints)
            if self.schema != schema_for(contract):
                raise ValueError('Routing schema mismatch; retrain with the required contract')
        torch.set_num_threads(1)
        self.model = MaskablePPO.load(self.path, device='cpu')
        if (self.model.action_space.n != self.num_checkpoints or
                self.model.observation_space.shape != (self.schema['size'],)):
            raise ValueError('Checkpoint dimensions mismatch; retrain for this destination count')

    def validate_scenario(self, scenario):
        if (scenario.num_checkpoints != self.num_checkpoints or
                (not self.legacy and schema_for(scenario) != self.schema) or
                (self.legacy and (scenario.checkpoint_ids != ['A', 'B', 'C'] or
                    scenario.alpha != 1. or scenario.beta != 1. or scenario.nominal_speed_mps != 1. or
                    scenario.return_to_base))):
            raise ValueError('Scenario/model contract mismatch; retrain for this N, ID order and objective')

    def predict_features(self, features, mask):
        obs, masks = np.asarray(features, dtype=np.float32), np.asarray(mask)
        if masks.dtype != np.dtype(bool):
            raise ValueError('Invalid mask values; actual booleans are required')
        if obs.shape != (self.schema['size'],) or not np.isfinite(obs).all() or masks.shape != (self.num_checkpoints,):
            raise ValueError('Invalid observation or mask shape; retrain for the required dimensions')
        if not masks.any():
            raise RuntimeError('No terminal policy query is allowed')
        start, end = self.schema['fields']['visited']
        visited = obs[start:end]
        if not np.isin(visited, [0., 1.]).all() or not np.array_equal(masks, visited == 0):
            raise ValueError('Observation visited flags and action mask disagree')
        start, _ = self.schema['fields']['constants']
        expected = [1., 1., 1., 0.] if self.legacy else [self.schema['objective'][key] for key in
                    ['alpha', 'beta', 'nominal_speed_mps', 'return_to_base']]
        if not np.allclose(obs[start:start + 4], expected, rtol=1e-6, atol=1e-7):
            raise ValueError('Observation objective constants disagree with the frozen model')
        action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)
        if action not in range(self.num_checkpoints) or not masks[action]:
            raise RuntimeError('Model produced an invalid action')
        return action

    def predict(self, state):
        self.validate_scenario(state.scenario)
        return self.predict_features(state.observation(legacy=self.legacy), state.mask())



def training_env(num_checkpoints, event_mixture=False, reward_scale=1.):
    """Scale only optimizer rewards; raw RoutingEnv accounting stays unchanged."""
    import gymnasium as gym
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError('Optimizer reward scale must be positive and finite')
    env = RoutingEnv(num_checkpoints=num_checkpoints, event_mixture=event_mixture)
    return gym.wrappers.TransformReward(env, lambda reward: reward * reward_scale)


def train_one(seed, output, static_steps=50000, event_steps=50000, n_epochs=10,
              n_steps=256, batch_size=64, learning_rate=3e-4, num_checkpoints=10,
              max_actual_steps=200000, net_arch=None, policy_arch='mlp',
              warmstart_cases=0, warmstart_epochs=0, teacher_directory=None, reward_scale=1., **kwargs):
    actual = sum(math.ceil(s / n_steps) * n_steps for s in [static_steps, event_steps]) if n_steps > 0 else 0
    if n_steps <= 0 or static_steps <= 0 or event_steps <= 0 or actual > max_actual_steps:
        raise ValueError('Both curriculum stages required; rounded actual steps exceed declared cap')
    output = Path(output)
    if output.with_suffix('.zip').exists() or output.with_suffix('.schema.json').exists():
        raise FileExistsError('Refusing to overwrite an existing checkpoint')
    directory = unique_dir(output.parent / (output.name + '_run'))
    torch.set_num_threads(1)
    env = training_env(num_checkpoints, reward_scale=reward_scale)
    architecture = net_arch or [128, 128]
    if policy_arch not in ('mlp', 'candidate'):
        raise ValueError('Unknown policy_arch')
    from .routing_policy import CandidatePolicy
    policy = CandidatePolicy if policy_arch == 'candidate' else 'MlpPolicy'
    model = MaskablePPO(policy, env, seed=seed, device='cpu', gamma=1.,
                        learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                        n_epochs=n_epochs, policy_kwargs={'net_arch': architecture}, verbose=0)
    model.set_logger(configure(str(directory), ['csv']))
    start = time.monotonic()
    initialization = None
    if warmstart_cases:
        if policy_arch != 'candidate' or warmstart_epochs <= 0:
            raise ValueError('Teacher initialization requires candidate policy and positive epochs')
        from .routing_teacher import prepare_teacher, warmstart
        teacher = Path(teacher_directory) if teacher_directory else prepare_teacher(directory / 'teacher', num_checkpoints, warmstart_cases)
        initialization = warmstart(model, teacher, warmstart_epochs, seed, warmstart_cases)
        if teacher_directory is None:
            import shutil
            shutil.copyfile(teacher / 'teacher.jsonl.gz', directory / 'teacher.jsonl.gz')
    model.learn(static_steps)
    static_actual = model.num_timesteps
    model.set_env(training_env(num_checkpoints, event_mixture=True, reward_scale=reward_scale))
    model.learn(event_steps, reset_num_timesteps=False)
    model.save(output)
    path = output.with_suffix('.zip')
    contract = schema_for(generate_scenario(0, 'training', num_checkpoints=num_checkpoints))
    json_write(path.with_suffix('.schema.json'), contract)
    json_write(directory / 'training.json', {'seed': seed, 'num_checkpoints': num_checkpoints,
               'static_steps_actual': static_actual, 'total_steps_actual': model.num_timesteps,
               'requested_static': static_steps, 'requested_events': event_steps,
               'event_mixture': [.5, .25, .25], 'elapsed_s': time.monotonic() - start,
               'sha256': sha256(path), 'gamma': 1., 'n_steps': n_steps, 'batch_size': batch_size,
               'n_epochs': n_epochs, 'learning_rate': learning_rate, 'net_arch': architecture,
               'policy_arch': policy_arch, 'optimizer_reward_scale': reward_scale, 'candidate_architecture': {
                   'relations': [4, 32, 32], 'nodes': [44, 64, 64], 'score': [128, 64, 1],
                   'critic': [64, 64, 1]} if policy_arch == 'candidate' else None,
               'warmstart': initialization, 'device': 'cpu'})
    return path


def validation(agent, episodes=100):
    from .benchmark import evaluate_scenario
    costs, changes, event_count, scenarios = [], 0, 0, []
    for index in range(episodes):
        kind = ['static', 'priority', 'cost'][index % 3]
        s = generate_scenario(10000 + index, 'validation', kind, agent.num_checkpoints)
        scenarios.append(s.to_dict())
        result = evaluate_scenario(s, agent.predict)
        costs.append(result['analytical_cost'])
        changes += result['decision_changes']
        event_count += result['event_opportunities']
    return {'mean_cost': float(np.mean(costs)), 'invalid_actions': 0, 'decision_changes': changes,
            'event_cases': event_count, 'episodes': episodes, 'scenarios': scenarios}



def _train_seed(args):
    seed, out, config = args
    path = train_one(seed, out / f'seed_{seed}', **config)
    scores = validation(RoutingAgent(path), config.get('validation_episodes', 100))
    json_write(out / f'validation_{seed}.json', scores.pop('scenarios'))
    return {'seed': seed, 'path': str(path), 'sha256': sha256(path), 'validation': scores}


def train(config, seeds, out):
    out = unique_dir(out)
    config = dict(config)
    config.setdefault('num_checkpoints', 10)
    json_write(out / 'config.json', config)
    models = []
    if config.get('warmstart_cases'):
        from .routing_teacher import prepare_teacher
        teacher = prepare_teacher(out / 'teacher', config['num_checkpoints'], config['warmstart_cases'])
        config['teacher_directory'] = str(teacher)
    arguments = [(seed, out, config) for seed in seeds]
    workers = config.get('parallel_seeds', 1)
    if type(workers) is not int or workers < 1 or workers > 3:
        raise ValueError('parallel_seeds must be an integer in [1,3]')
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for item in pool.map(_train_seed, arguments):
                models.append(item)
                json_write(out / 'models.partial.json', {'models': models})
                print(json.dumps(item), flush=True)
    else:
        for args in arguments:
            item = _train_seed(args)
            models.append(item)
            json_write(out / 'models.partial.json', {'models': models})
            print(json.dumps(item), flush=True)
    selected = min(models, key=lambda m: (m['validation']['invalid_actions'],
                    m['validation']['mean_cost'], -m['validation']['decision_changes'], m['seed']))
    result = {'schema': RoutingAgent(selected['path']).schema, 'models': models, 'selected': selected,
              'selection_rationale': 'Validation invalid actions, mean shared-objective cost, event changes, then seed; frozen before held-out evaluation.'}
    json_write(out / 'models.json', result)
    return result


def freeze_validation_models(manifests, out):
    """Freeze one validation winner per seed, before opening any held-out case."""
    from datetime import datetime, timezone
    trials, contract = [], None
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        data = json.loads(manifest.read_text())
        for item in data['models']:
            item = dict(item)
            path = Path(item['path'])
            if not path.is_absolute():
                path = manifest.parent / path.name
            item['path'] = str(path.resolve())
            if sha256(path) != item['sha256']:
                raise ValueError('Frozen candidate checkpoint hash mismatch')
            schema = json.loads(path.with_suffix('.schema.json').read_text())
            if contract is not None and schema != contract:
                raise ValueError('Candidate model contracts disagree')
            contract = schema
            score = item['validation']
            if not math.isfinite(score['mean_cost']) or score['invalid_actions'] < 0:
                raise ValueError('Invalid validation evidence')
            item['source_manifest'] = str(manifest)
            trials.append(item)
    if contract is None or contract.get('num_checkpoints') != 10:
        raise ValueError('Delivery requires frozen ten-action models')
    def key(item):
        return (item['validation']['invalid_actions'], item['validation']['mean_cost'],
                -item['validation']['decision_changes'], item['seed'], item['path'])
    winners = []
    for seed in [42, 43, 44]:
        candidates = [item for item in trials if item['seed'] == seed]
        if not candidates:
            raise ValueError(f'Missing mandatory training seed {seed}')
        winners.append(min(candidates, key=key))
    result = {'schema': contract, 'models': winners, 'selected': min(winners, key=key),
              'all_trials': trials, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'heldout_opened': False,
              'selection_rationale': 'One winner per training seed and global winner by validation invalid count, mean objective cost, event changes, seed and path. Frozen before held-out evaluation; all trials retained.'}
    directory = unique_dir(out)
    json_write(directory / 'models.json', result)
    return result
