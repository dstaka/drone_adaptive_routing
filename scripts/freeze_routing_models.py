#!/usr/bin/env python
"""Freeze validation-only model selection across disclosed training experiments."""
import argparse
import json
from adaptive_drone_patrol.routing_agent import freeze_validation_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = freeze_validation_models(args.models, args.out)
    print(json.dumps(result['selected']), flush=True)


if __name__ == '__main__':
    main()
