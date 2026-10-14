"""Optional training-only exact-label warm start; never imported by inference."""

import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
from .analytical import snapshot
from .baselines import exact_remaining
from .objective import RoutingState
from .scenario import generate_scenario
from .util import sha256, json_write


def prepare_teacher(directory, n, cases):
    """Persist all inputs and labels from training seeds only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if type(cases) is not int or cases <= 0:
        raise ValueError('Teacher cases must be positive')
    start = time.monotonic()
    rng = np.random.default_rng(8128)
    observations, masks, labels = [], [], []
    with gzip.open(directory / 'teacher.jsonl.gz', 'wt') as stream:
        for index in range(cases):
            kind = ['static', 'priority', 'cost'][index % 3]
            state = RoutingState(generate_scenario(index % 10000, 'training', kind, n))
            prefix = rng.permutation(n)[:int(rng.integers(n))]
            for a in prefix:
                state.complete(int(a), state.departure(int(a)))
                state.apply_event()
            exact = exact_remaining(state)
            observations.append(state.observation())
            masks.append(state.mask())
            labels.append(exact['order'][0])
            stream.write(json.dumps({'snapshot': snapshot(state), 'exact': exact}) + '\n')
    np.savez_compressed(directory / 'teacher.npz', observations=np.asarray(observations),
                        masks=np.asarray(masks), labels=np.asarray(labels))
    record = {'cases': cases, 'num_checkpoints': n, 'split': 'training', 'scenario_seeds': [0, min(cases - 1, 9999)],
              'sha256': sha256(directory / 'teacher.npz'), 'manifest_sha256': sha256(directory / 'teacher.jsonl.gz'),
              'elapsed_s': time.monotonic() - start, 'purpose': 'Supervised initialization before static and event-inclusive MaskablePPO; no teacher at inference.'}
    json_write(directory / 'teacher.json', record)
    return directory


def warmstart(model, directory, epochs, seed, expected_cases):
    start = time.monotonic()
    directory = Path(directory)
    record = json.loads((directory / 'teacher.json').read_text())
    if (record['split'] != 'training' or record['cases'] != expected_cases or
            record['num_checkpoints'] != model.action_space.n or
            sha256(directory / 'teacher.jsonl.gz') != record['manifest_sha256'] or
            sha256(directory / 'teacher.npz') != record['sha256']):
        raise ValueError('Teacher dataset hash mismatch')
    with np.load(directory / 'teacher.npz') as data:
        obs = torch.as_tensor(data['observations'])
        masks = torch.as_tensor(data['masks'])
        labels = torch.as_tensor(data['labels'])
    if (obs.shape != (expected_cases, model.observation_space.shape[0]) or
            masks.shape != (expected_cases, model.action_space.n) or masks.dtype != torch.bool or
            labels.shape != (expected_cases,) or labels.dtype != torch.int64 or
            not torch.isfinite(obs).all() or (labels < 0).any() or (labels >= model.action_space.n).any() or
            not masks[torch.arange(expected_cases), labels].all()):
        raise ValueError('Invalid teacher dimensions or labels')
    with gzip.open(directory / 'teacher.jsonl.gz', 'rt') as stream:
        count = 0
        for line in stream:
            item = json.loads(line)
            scenario = item['snapshot']['scenario']
            if (scenario['split'] != 'training' or not 0 <= scenario['seed'] < 10000 or
                    len(scenario['checkpoint_ids']) != model.action_space.n):
                raise ValueError('Teacher manifest contains nontraining or incompatible inputs')
            count += 1
        if count != expected_cases:
            raise ValueError('Teacher manifest count mismatch')
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(len(labels)), max(1, int(np.ceil(len(labels) / 256)))):
            logits = model.policy.mlp_extractor.forward_actor(obs[batch])
            loss = torch.nn.functional.cross_entropy(logits.masked_fill(~masks[batch], -1e8), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return {**record, 'epochs': epochs, 'training_elapsed_s': time.monotonic() - start,
            'directory': str(directory), 'final_batch_loss': float(loss.detach())}
