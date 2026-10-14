#!/usr/bin/env python
"""Independently recompute explanation costs using only Python's standard library."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def recompute(snapshot, order):
    """Use explicit absolute-time accounting, independent of the production objective."""
    s = snapshot['scenario']
    current, elapsed = snapshot['current'], snapshot['elapsed']
    eligible = {i for i, visited in enumerate(snapshot['visited']) if not visited}
    if len(order) != len(eligible) or set(order) != eligible:
        raise ValueError('Order does not cover the complete remaining set exactly once')
    travel = penalty = 0.
    for action in order:
        target = action + 1
        distance = math.dist(s['coords'][current], s['coords'][target])
        elapsed += distance / s['nominal_speed_mps'] + s['service_duration_s'][action]
        travel += distance * s['multipliers'][current][target]
        penalty += s['priorities'][action] * elapsed
        current = target
    if s['return_to_base'] and order:
        travel += math.dist(s['coords'][current], s['coords'][0]) * s['multipliers'][current][0]
    return {'cost': s['alpha'] * travel + s['beta'] * penalty,
            'travel_cost': travel, 'completion_penalty': penalty}


def reject_constant(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--explanations', required=True)
    parser.add_argument('--policy-hash', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    source = Path(args.explanations)
    values = json.loads(source.read_text(), parse_constant=reject_constant)['decisions']
    maximum, orders, identities = 0., 0, set()
    for value in values:
        identity = (value['run_id'], value['decision_id'])
        canonical = json.dumps(value['snapshot'], sort_keys=True, separators=(',', ':'), allow_nan=False)
        if (identity in identities or value['evidence_kind'] != 'analytical' or
                value['policy_hash'] != args.policy_hash or
                value['boundary_id'] != ':'.join(identity) or
                value['boundary_hash'] != hashlib.sha256(canonical.encode()).hexdigest()):
            raise ValueError('Explanation provenance mismatch')
        identities.add(identity)
        tol = max(1e-6, 1e-8 * abs(value['exact']['cost']))
        for record in [value['policy_rollout'], value['exact'], *value['forced_first']]:
            expected = recompute(value['snapshot'], record['order'])
            for key, result in expected.items():
                difference = abs(record[key] - result)
                maximum = max(maximum, difference)
                if not math.isfinite(record[key]) or difference > tol:
                    raise ValueError(f'Independent {key} mismatch at {identity}')
            orders += 1
    result = {'status': 'PASS', 'evidence_kind': 'analytical', 'policy_hash': args.policy_hash,
              'decisions_verified': len(values), 'orders_verified': orders,
              'maximum_absolute_component_error': maximum,
              'explanations_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'method': 'Independent standard-library Euclidean/directed/absolute-completion-time replay.'}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
