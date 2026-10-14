"""CPU MaskablePPO training, explicit masked inference and validation selection."""

import json
from pathlib import Path
import time
import math
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from .routing_env import RoutingEnv
from .objective import SCHEMA, RoutingState
from .scenario import generate_scenario
from .util import sha256, unique_dir, json_write


class RoutingAgent:
    def __init__(self, path, expected_hash=None):
        self.path = Path(path)
        self.hash = sha256(self.path)
        if expected_hash and self.hash != expected_hash:
            raise ValueError('Routing checkpoint hash mismatch')
        schema = json.loads(self.path.with_suffix('.schema.json').read_text())
        if schema != SCHEMA:
            raise ValueError('Routing observation schema mismatch')
        torch.set_num_threads(1)
        self.model = MaskablePPO.load(self.path, device='cpu')

    def predict_features(self, features, mask):
        import numpy as np
        obs, masks = np.asarray(features, dtype=np.float32), np.asarray(mask, dtype=bool)
        if obs.shape != (SCHEMA['size'],) or not np.isfinite(obs).all() or masks.shape != (3,):
            raise ValueError('Invalid observation or mask shape')
        if not masks.any():
            raise RuntimeError('No terminal policy query is allowed')
        action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)
        if action not in range(3) or not masks[action]:
            raise RuntimeError('Model produced an invalid action')
        return action

    def predict(self, state):
        return self.predict_features(state.observation(), state.mask())


def train_one(seed, output, static_steps=50000, event_steps=50000, n_epochs=10,
              n_steps=256, batch_size=64, learning_rate=3e-4, **kwargs):
    if n_steps <= 0 or static_steps <= 0 or event_steps <= 0 or sum(math.ceil(s / n_steps) * n_steps for s in [static_steps, event_steps]) > 200000:
        raise ValueError('Both curriculum stages required; at most 200000 actual rounded steps')
    output = Path(output)
    directory = unique_dir(output.parent / (output.name + '_run'))
    torch.set_num_threads(1)
    env = RoutingEnv()
    model = MaskablePPO('MlpPolicy', env, seed=seed, device='cpu', gamma=1.,
                        learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                        n_epochs=n_epochs, policy_kwargs={'net_arch': [64, 64]}, verbose=0)
    model.set_logger(configure(str(directory), ['csv']))
    start = time.monotonic()
    model.learn(static_steps)
    static_actual = model.num_timesteps
    model.set_env(RoutingEnv(event_mixture=True))
    model.learn(event_steps, reset_num_timesteps=False)
    model.save(output)
    path = output.with_suffix('.zip')
    json_write(path.with_suffix('.schema.json'), SCHEMA)
    json_write(directory / 'training.json', {'seed': seed, 'static_steps_actual': static_actual,
               'total_steps_actual': model.num_timesteps, 'requested_static': static_steps,
               'requested_events': event_steps, 'event_mixture': [.5, .25, .25],
               'elapsed_s': time.monotonic() - start, 'sha256': sha256(path), 'gamma': 1.,
               'n_steps': n_steps, 'batch_size': batch_size, 'n_epochs': n_epochs,
               'learning_rate': learning_rate, 'net_arch': [64, 64], 'device': 'cpu'})
    return path


def validation(agent, episodes=100):
    costs, invalid, changes, event_count = [], 0, 0, 0
    for index in range(episodes):
        kind = ['static', 'priority', 'cost'][index % 3]
        state = RoutingState(generate_scenario(10000 + index, 'validation', kind))
        while not state.terminal:
            a = agent.predict(state)
            state.complete(a, state.departure(a))
            if state.visited.sum() == 1 and state.scenario.event:
                counterfactual = agent.predict(state)
                state.apply_event()
                changes += agent.predict(state) != counterfactual
                event_count += 1
        costs.append(state.cost)
        invalid += state.invalid_count
    return {'mean_cost': sum(costs) / len(costs), 'invalid_actions': invalid,
            'decision_changes': changes, 'event_cases': event_count, 'episodes': episodes}


def train(config, seeds, out):
    out = unique_dir(out)
    models = []
    for seed in seeds:
        path = train_one(seed, out / f'seed_{seed}', **config)
        item = {'seed': seed, 'path': str(path), 'sha256': sha256(path),
                'validation': validation(RoutingAgent(path), config.get('validation_episodes', 100))}
        models.append(item)
        print(json.dumps(item), flush=True)
    selected = min(models, key=lambda m: (m['validation']['invalid_actions'],
                    -m['validation']['decision_changes'], m['validation']['mean_cost'], m['seed']))
    result = {'schema': SCHEMA, 'models': models, 'selected': selected,
              'selection_rationale': 'Validation invalid actions, event changes, then mean objective cost; no held-out results used.'}
    json_write(out / 'models.json', result)
    return result
