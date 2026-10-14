#!/usr/bin/env python
"""Persist the complete training candidate catalog and exact validation inputs."""
import argparse
import gzip
import json
from adaptive_drone_patrol.scenario import generate_scenario
from adaptive_drone_patrol.util import unique_dir, json_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-checkpoints', type=int, default=10)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = unique_dir(a.out)
    with gzip.open(out / 'training_candidates.jsonl.gz', 'wt') as f:
        for seed in range(10000):
            for kind in ['static', 'priority', 'cost']:
                f.write(json.dumps(generate_scenario(seed, 'training', kind, a.num_checkpoints).to_dict()) + '\n')
    validation = [generate_scenario(10000 + i, 'validation', ['static', 'priority', 'cost'][i % 3], a.num_checkpoints).to_dict()
                  for i in range(100)]
    json_write(out / 'validation.json', validation)
    json_write(out / 'splits.json', {'training_candidate_count': 30000, 'training_scenario_seeds': [0, 9999],
        'validation_seeds': [10000, 10099], 'heldout_static_seeds': [20000, 20099],
        'heldout_dynamic_seeds': [30000, 30049], 'demo_seeds': [40000, 40099],
        'note': 'Complete candidate catalog includes every online-generated training input. Boundary event values use observed state only.'})


if __name__ == '__main__':
    main()
