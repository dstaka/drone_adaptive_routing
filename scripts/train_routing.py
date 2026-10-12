#!/usr/bin/env python
"""Train and freeze three CPU routing models with validation-only selection."""
import argparse
from adaptive_drone_patrol.util import load_config
from adaptive_drone_patrol.routing_agent import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    train(load_config(a.config), a.seeds, a.out)


if __name__ == '__main__':
    main()
