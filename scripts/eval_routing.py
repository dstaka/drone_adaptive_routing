#!/usr/bin/env python
"""Evaluate frozen models against paired fixed/greedy and static exact baselines."""
import argparse
from adaptive_drone_patrol.benchmark import benchmark


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', required=True)
    p.add_argument('--split', choices=['static', 'dynamic'], required=True)
    p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--num-checkpoints', type=int)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    benchmark(a.models, a.split, a.episodes, a.out, a.num_checkpoints)


if __name__ == '__main__':
    main()
