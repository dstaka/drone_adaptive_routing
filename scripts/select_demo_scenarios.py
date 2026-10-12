#!/usr/bin/env python
"""Disclosed demonstration-split search for both frozen-policy event responses."""
import argparse
import copy
import json
import numpy as np
from pathlib import Path
import yaml
from adaptive_drone_patrol.routing_agent import RoutingAgent
from adaptive_drone_patrol.objective import RoutingState
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.util import unique_dir, json_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    selected = json.loads(Path(a.models).read_text())['selected']
    agent = RoutingAgent(selected['path'], selected['sha256'])
    out = unique_dir(a.out)
    attempts = []
    for seed in range(40000, 40100):
        rng = np.random.default_rng(seed)
        theta = 0.  # Preserve the geometry verified by sequential physical trials.
        coords = [[0, 0, 1]] + [[float(.6 * np.cos(theta + i * 2 * np.pi / 3)),
                                  float(.6 * np.sin(theta + i * 2 * np.pi / 3)), 1] for i in range(3)]
        s = Scenario('normal', 'demo', seed, coords, rng.integers(1, 4, 3).tolist(),
                     np.ones((4, 4)).tolist(), [.5] * 3)
        s.validate_physical()
        st = RoutingState(s)
        first = agent.predict(st)
        st.complete(first, st.departure(first))
        next_action = agent.predict(st)
        alternate = next(i for i in range(3) if not st.visited[i] and i != next_action)
        emergency = copy.deepcopy(s)
        emergency.scenario_id = 'emergency'
        emergency.event = {'kind': 'priority', 'after': 1, 'checkpoint': alternate, 'value': 5}
        cost = copy.deepcopy(s)
        cost.scenario_id = 'cost_shift'
        cost.event = {'kind': 'cost', 'after': 1, 'edges': [[first + 1, next_action + 1]], 'value': 5}
        changed = []
        for scenario in [emergency, cost]:
            boundary = RoutingState(scenario)
            boundary.complete(first, boundary.departure(first))
            boundary.apply_event()
            changed.append(agent.predict(boundary) != next_action)
        attempts.append({'seed': seed, 'first': first, 'normal_next': next_action, 'changes': changed})
        if all(changed):
            for scenario in [s, emergency, cost]:
                (out / f'{scenario.scenario_id}.yaml').write_text(yaml.safe_dump(scenario.to_dict(), sort_keys=False))
            json_write(out / 'selection.json', {'status': 'PASS', 'policy_hash': agent.hash, 'selected_seed': seed,
                       'search_split': 'demo 40000-40099', 'attempts': attempts,
                       'note': 'Illustrative-case selection; independent of held-out measurements.'})
            return
    json_write(out / 'selection.json', {'status': 'NO_POLICY_CHANGE', 'attempts': attempts})
    raise SystemExit('No paired demo found; do not invent policy changes')


if __name__ == '__main__':
    main()
