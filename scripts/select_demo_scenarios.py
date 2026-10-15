#!/usr/bin/env python
"""Select shared ten-point analytical fixtures without changing the emergency target."""
import argparse
import copy
import json
from pathlib import Path
import yaml
from adaptive_drone_patrol.routing_agent import RoutingAgent
from adaptive_drone_patrol.objective import RoutingState
from adaptive_drone_patrol.analytical import demo_fixture, event_comparison, explain_decision, verify_explanation
from adaptive_drone_patrol.util import unique_dir, json_write, sha256


def select(models, out):
    selected = json.loads(Path(models).read_text())['selected']
    agent = RoutingAgent(selected['path'], selected['sha256'])
    if agent.num_checkpoints != 10:
        raise ValueError('The demonstration requires a retrained ten-action policy')
    out = unique_dir(out)
    attempts, retained = [], None
    for seed in range(40000, 40100):
        normal = demo_fixture(seed)
        normal.validate_physical()
        boundary = RoutingState(normal)
        first = agent.predict(boundary)
        boundary.complete(first, boundary.departure(first))
        no_action = agent.predict(boundary)
        emergency = copy.deepcopy(normal)
        emergency.scenario_id = 'emergency'
        emergency.event = {'kind': 'priority', 'after': 1, 'checkpoint': 'C', 'value': 5}
        cost = copy.deepcopy(normal)
        cost.scenario_id = 'cost_shift'
        cost.event = {'kind': 'cost', 'after': 1,
                      'edges': [[normal.checkpoint_ids[first], normal.checkpoint_ids[no_action]]], 'value': 5}
        comparisons = []
        for scenario in [emergency, cost]:
            st = RoutingState(scenario)
            st.complete(first, st.departure(first))
            comparisons.append(event_comparison(st, agent.predict))
        b, c = comparisons
        target = normal.checkpoint_ids.index('C')
        b_ok = (b['applicable'] and target in b['no_event_order'] and
                b['post_event_order'].index(target) < b['no_event_order'].index(target) and
                b['no_event_first'] != b['post_event_first'] and b['improvement'] > 1e-6)
        c_ok = (c['no_event_first'] != c['post_event_first'] and
                c['no_event_order'] != c['post_event_order'] and c['improvement'] > 1e-6)
        attempt = {'seed': seed, 'first': first, 'emergency_gate': b_ok,
                   'cost_shift_gate': c_ok, 'comparisons': comparisons}
        attempts.append(attempt)
        if not (b_ok and c_ok):
            continue
        replays, explanations = [], []
        for scenario in [normal, emergency, cost]:
            st = RoutingState(scenario)
            run_id = f'analytical-{seed}-{scenario.scenario_id}'
            decisions = []
            while not st.terminal:
                decision_id = f'decision-{int(st.visited.sum())}'
                explanation = explain_decision(st, agent.predict, agent.hash, run_id, decision_id)
                verify_explanation(explanation, agent.hash, run_id, decision_id)
                explanations.append(explanation)
                a = agent.predict(st)
                decisions.append({'decision_id': decision_id, 'boundary_id': explanation['boundary_id'],
                                  'boundary_hash': explanation['boundary_hash'], 'action': a,
                                  'checkpoint_id': scenario.checkpoint_ids[a]})
                st.complete(a, st.departure(a))
                st.apply_event()
            replays.append({'run_id': run_id, 'evidence_kind': 'analytical',
                            'scenario_id': scenario.scenario_id, 'policy_hash': agent.hash,
                            'decisions': decisions, 'cost': st.cost, 'legs': st.legs})
        explanation_ok = any(e['nearer_lower_priority_gate'] for e in explanations)
        attempt['explanation_gate'] = explanation_ok
        if retained is None or explanation_ok:
            retained = (normal, emergency, cost, replays, explanations, seed)
        if explanation_ok:
            break
    if retained:
        normal, emergency, cost, replays, explanations, seed = retained
        for scenario in [normal, emergency, cost]:
            validated = type(scenario).from_dict(scenario.to_dict())
            (out / f'{scenario.scenario_id}.yaml').write_text(yaml.safe_dump(validated.to_dict(), sort_keys=False))
        json_write(out / 'analytical_replay.json', {'schema_version': 2, 'evidence_kind': 'analytical', 'runs': replays})
        json_write(out / 'decision_explanations.json', {'schema_version': 2, 'evidence_kind': 'analytical',
                                                       'decisions': explanations})
        passed = any(e['nearer_lower_priority_gate'] for e in explanations)
        manifest = {s.scenario_id: sha256(out / f'{s.scenario_id}.yaml') for s in [normal, emergency, cost]}
    else:
        seed, passed, manifest = None, False, {}
    result = {'status': 'PASS' if retained and passed else 'INCOMPLETE', 'policy_hash': agent.hash,
              'selected_seed': seed, 'search_split': 'demo 40000-40099', 'attempts': attempts,
              'scenario_hashes': manifest, 'explanation_gate': passed,
              'event_gates': retained is not None, 'directed_updates': 'One direction only; reverse edge unchanged.',
              'criteria': 'Shared geometry/initial state; C-only emergency, earlier C, changed next action and positive full-suffix benefit; cost-only event with positive benefit; exact nearer/lower-priority example.',
              'geometry_note': 'Specified candidate grid, minimum separation 0.60 m. Physical envelope unverified.',
              'note': 'Illustrative demo selection only; held-out test cases are never read.'}
    json_write(out / 'selection.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'attempts'}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = select(a.models, a.out)
    raise SystemExit(0 if result['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
