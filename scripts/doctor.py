#!/usr/bin/env python
"""Inspect or freeze explicitly supplied flight artifact paths without modifying packages."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import yaml
from adaptive_drone_patrol.util import sha256, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--check', choices=['flight', 'routing'], default='flight')
    p.add_argument('--freeze-log-dir')
    p.add_argument('--source')
    a = p.parse_args()
    if a.freeze_log_dir:
        directory = Path(a.freeze_log_dir).resolve()
        models = [(int(f.stem.split('_')[1]), f) for f in directory.glob('model_*.pt')
                  if f.stem.split('_')[1].isdigit()]
        if not models or not (directory / 'cfgs.pkl').is_file() or not a.source:
            p.error('Provide existing operator-owned checkpoints, cfgs.pkl and --source')
        path = max(models)[1]
        source = Path(a.source).resolve()
        cfg = {'flight_checkpoint': str(path), 'flight_checkpoint_hash': sha256(path),
               'flight_cfg': str(directory / 'cfgs.pkl'), 'flight_cfg_hash': sha256(directory / 'cfgs.pkl'),
               'hover_source': str(source), 'hover_source_hash': sha256(source),
               'budget_ledger': 'outputs/gpu_ledger.json', 'gpu_budget_s': 28800}
        config_path = Path(a.config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            p.error('Freeze config output exists; choose a new name')
        config_path.write_text(yaml.safe_dump(cfg))
    cfg = load_config(a.config)
    result = {'python': sys.version, 'interpreter': sys.executable, 'platform': platform.platform()}
    names = ['numpy', 'genesis-world', 'rsl-rl-lib', 'torch', 'tensordict', 'tensorboard'] \
        if a.check == 'flight' else ['numpy', 'torch', 'gymnasium', 'stable-baselines3', 'sb3-contrib']
    result['packages'] = {name: metadata.version(name) for name in names}
    if a.check == 'flight':
        import torch
        result.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if result['packages']['genesis-world'] != '1.4.3':
            raise RuntimeError('Flight baseline requires Genesis 1.4.3')
        for key in ['flight_checkpoint', 'flight_cfg', 'hover_source']:
            if sha256(cfg[key]) != cfg[key + '_hash']:
                raise RuntimeError(f'Hash mismatch: {key}')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
