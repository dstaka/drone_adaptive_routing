#!/usr/bin/env python
"""Collision-safe invocation of the preserved notebook training subprocess."""
import argparse
from pathlib import Path
import subprocess
import sys
from adaptive_drone_patrol.util import load_config, unique_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--execution', required=True)
    p.add_argument('--num-envs', type=int, required=True)
    p.add_argument('--iterations', type=int, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    runtime = load_config(a.execution)
    if a.num_envs < 1 or a.num_envs > 256 or a.iterations < 1:
        p.error('Use 1–256 environments and positive iterations')
    out = unique_dir(a.out).resolve()
    source = Path(runtime['hover_source']).resolve().parent
    experiment = out.name
    if (source / 'logs' / experiment).exists():
        raise FileExistsError('Upstream log collision; select a unique experiment name')
    subprocess.run([sys.executable, '-u', str(source / 'hover_train.py'), '-e', experiment,
                    '-b', str(a.num_envs), '--max-iterations', str(a.iterations),
                    '--seed', str(cfg.get('seed', 1))], cwd=source, check=True)


if __name__ == '__main__':
    main()
