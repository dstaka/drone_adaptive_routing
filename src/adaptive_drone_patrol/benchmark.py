"""Paired held-out evaluations over persisted common scenario manifests."""

import csv
import json
from pathlib import Path
import numpy as np
from .baselines import choose_baseline, exact_static
from .objective import RoutingState
from .routing_agent import RoutingAgent
from .scenario import generate_scenario
from .util import json_write, unique_dir


def evaluate_scenario(scenario, chooser):
    state = RoutingState(scenario)
    order, changes, opportunities = [], 0, 0
    while not state.terminal:
        a = int(chooser(state))
        if a not in range(3) or not state.mask()[a]:
            raise RuntimeError('Invalid action in benchmark')
        order.append(a)
        state.complete(a, state.departure(a))
        if state.visited.sum() == 1 and state.scenario.event:
            before = int(chooser(state.copy()))
            state.apply_event()
            after = int(chooser(state.copy()))
            opportunities += 1
            changes += before != after
    return {'analytical_cost': state.cost, 'priority_completion_penalty': state.penalty,
            'nominal_distance_m': state.distance, 'completion_ratio': float(state.visited.mean()),
            'invalid_actions': state.invalid_count, 'order': ''.join('ABC'[i] for i in order),
            'decision_changes': changes, 'event_opportunities': opportunities,
            'events': state.events}


def benchmark(models, split, episodes, out):
    limits = {'static': 100, 'dynamic': 50}
    if split not in limits or not 1 <= episodes <= limits[split]:
        raise ValueError('Invalid held-out split/episode count')
    out = unique_dir(out)
    model_data = json.loads(Path(models).read_text())
    seeds = range(20000, 20000 + episodes) if split == 'static' else range(30000, 30000 + episodes)
    scenarios = [generate_scenario(s, split) for s in seeds]
    json_write(out / 'scenarios.json', [s.to_dict() for s in scenarios])
    rows, details = [], []
    for item in model_data['models']:
        agent = RoutingAgent(item['path'], item['sha256'])
        for s in scenarios:
            planners = {'fixed': lambda st: choose_baseline(st, 'fixed'),
                        'greedy': lambda st: choose_baseline(st, 'greedy'), 'learned': agent.predict}
            oracle_cost = None
            if split == 'static':
                route, oracle_cost = exact_static(s)
                planners['exact'] = lambda st, r=route: next(a for a in r if not st.visited[a])
            results = {name: evaluate_scenario(s, fn) for name, fn in planners.items()}
            for name, result in results.items():
                detail = {'training_seed': item['seed'], 'scenario_id': s.scenario_id,
                          'scenario_seed': s.seed, 'planner': name, 'policy_hash': item['sha256'], **result}
                details.append(detail)
                row = {k: v for k, v in detail.items() if k != 'events'}
                row.update(delta_vs_fixed=result['analytical_cost'] - results['fixed']['analytical_cost'],
                           delta_vs_greedy=result['analytical_cost'] - results['greedy']['analytical_cost'],
                           static_exact_ratio=result['analytical_cost'] / oracle_cost if oracle_cost else '')
                rows.append(row)
    json_write(out / 'scenario_results.json', details)
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_seed = []
    for item in model_data['models']:
        subset = [r for r in rows if r['planner'] == 'learned' and r['training_seed'] == item['seed']]
        opportunities = sum(r['event_opportunities'] for r in subset)
        per_seed.append({'seed': item['seed'], 'episodes': episodes,
                         'cost_mean': float(np.mean([r['analytical_cost'] for r in subset])),
                         'delta_fixed_mean': float(np.mean([r['delta_vs_fixed'] for r in subset])),
                         'delta_greedy_mean': float(np.mean([r['delta_vs_greedy'] for r in subset])),
                         'exact_ratio_mean': float(np.mean([r['static_exact_ratio'] for r in subset]))
                            if split == 'static' else None,
                         'completion_ratio': float(np.mean([r['completion_ratio'] for r in subset])),
                         'invalid_actions': sum(r['invalid_actions'] for r in subset),
                         'decision_change_rate': sum(r['decision_changes'] for r in subset) / opportunities
                            if opportunities else None})
    summary = {'split': split, 'common_scenarios': episodes, 'per_seed': per_seed,
               'across_seed_cost_mean': float(np.mean([r['cost_mean'] for r in per_seed])),
               'across_seed_cost_std': float(np.std([r['cost_mean'] for r in per_seed], ddof=1)),
               'note': 'Shared scenarios are correlated; aggregate standard deviation is over seed means.'}
    json_write(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary
