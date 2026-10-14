#!/usr/bin/env python
"""Run final GPU contracts and nine recorded attempts under one budget owner."""
import argparse
import os
import subprocess
import sys
from adaptive_drone_patrol.util import unique_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = unique_dir(a.out)
    env = {**os.environ, 'PATROL_EXECUTION_CONFIG': a.config}
    with (out / 'gpu_tests.log').open('x') as log:
        subprocess.run([sys.executable, '-m', 'pytest', '-m', 'gpu', '-q'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, '-u', 'scripts/run_demo_scenarios.py', '--config', a.config,
                    '--mode', 'genesis', '--record', '--trials', '3', '--out', str(out / 'genesis')], check=True)


if __name__ == '__main__':
    main()
