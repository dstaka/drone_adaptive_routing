"""Persisted paired held-out comparisons and complete event counterfactuals."""

import csv
import json
from pathlib import Path
import numpy as np
from .baselines import choose_baseline, exact_remaining
from .analytical import event_comparison
from .objective import RoutingState
from .routing_agent import RoutingAgent
from .scenario import generate_scenario
from .util import json_write, unique_dir


def evaluate_scenario(scenario, chooser):
    state = RoutingState(scenario)
    order, comparisons = [], []
    while not state.terminal:
        a = int(chooser(state))
        if a not in range(scenario.num_checkpoints) or not state.mask()[a]:
            raise RuntimeError('Invalid action in benchmark')
        order.append(a)
        state.complete(a, state.departure(a))
        if not state.terminal and state.visited.sum() == 1 and state.scenario.event:
            comparisons.append(event_comparison(state, chooser))
            state.apply_event()
    opportunities = sum(c['applicable'] for c in comparisons)
    travel = sum(leg['travel_cost'] for leg in state.legs)
    if state.return_accounted:
        travel += float(np.linalg.norm(np.asarray(scenario.coords[state.current]) - scenario.coords[0])) * state.scenario.multipliers[state.current][0]
    return {'analytical_cost': state.cost, 'priority_completion_penalty': state.penalty,
            'weighted_travel_cost': scenario.alpha * travel,
            'weighted_completion_penalty': scenario.beta * state.penalty,
            'travel_cost': travel,
            'nominal_distance_m': state.distance, 'completion_ratio': float(state.visited.mean()),
            'invalid_actions': state.invalid_count, 'order': [scenario.checkpoint_ids[i] for i in order],
            'decision_changes': sum(c['applicable'] and c['no_event_first'] != c['post_event_first'] for c in comparisons),
            'event_opportunities': opportunities, 'scheduled_events': len(comparisons),
            'skipped_events': sum(not c['applicable'] for c in comparisons),
            'beneficial_responses': sum(c['applicable'] and c['no_event_first'] != c['post_event_first'] and
                                        c['improvement'] > 1e-6 for c in comparisons),
            'events': state.events, 'boundary_comparisons': comparisons}


def benchmark(models, split, episodes, out, num_checkpoints=None, geometry_profile=None):
    limits = {'static': 100, 'dynamic': 50}
    if split not in limits or not 1 <= episodes <= limits[split]:
        raise ValueError('Invalid held-out split/episode count')
    model_data = json.loads(Path(models).read_text())
    agents = [(item, RoutingAgent(item['path'], item['sha256'])) for item in model_data['models']]
    n = agents[0][1].num_checkpoints
    if (num_checkpoints is not None and num_checkpoints != n) or any(agent.num_checkpoints != n for _, agent in agents):
        raise ValueError('Evaluation/model destination counts disagree; retrain for this N')
    out = unique_dir(out)
    seeds = range(20000, 20000 + episodes) if split == 'static' else range(30000, 30000 + episodes)
    if geometry_profile not in (None, 'dispersed-constrained3d-v1'):
        raise ValueError('Unknown geometry profile')
    if geometry_profile:
        from .dispersed_layout import expanded_scenario
        if n != 10 or agents[0][1].priority_profile != 'integer-100-v1':
            raise ValueError('Expanded evaluation requires the adopted integer ten-point contract')
        scenarios = [expanded_scenario(s, split) for s in seeds]
    else:
        scenarios = [generate_scenario(s, split, num_checkpoints=n,
                                       priority_profile=agents[0][1].priority_profile) for s in seeds]
    for _, agent in agents:
        for s in scenarios:
            agent.validate_scenario(s)
    json_write(out / 'scenarios.json', [s.to_dict() for s in scenarios])
    references = {s.scenario_id: exact_remaining(RoutingState(s)) for s in scenarios} if split == 'static' else {}
    json_write(out / 'exact_references.json', references)
    rows, details = [], []
    for item, agent in agents:
        for s in scenarios:
            planners = {'fixed': lambda st: choose_baseline(st, 'fixed'),
                        'greedy': lambda st: choose_baseline(st, 'greedy'), 'learned': agent.predict}
            reference = references.get(s.scenario_id)
            if reference:
                planners['exact'] = lambda st, r=reference['order']: next(a for a in r if not st.visited[a])
            results = {name: evaluate_scenario(s, fn) for name, fn in planners.items()}
            for name, result in results.items():
                detail = {'schema_version': 2, 'evidence_kind': 'analytical',
                          'training_seed': item['seed'], 'scenario_id': s.scenario_id,
                          'scenario_seed': s.seed, 'planner': name, 'policy_hash': item['sha256'], **result}
                detail.update(delta_vs_fixed=result['analytical_cost'] - results['fixed']['analytical_cost'],
                              delta_vs_greedy=result['analytical_cost'] - results['greedy']['analytical_cost'],
                              static_exact_ratio=result['analytical_cost'] / reference['cost'] if reference else None,
                              exact_gap=result['analytical_cost'] - reference['cost'] if reference else None,
                              exact_status=reference['status'] if reference else 'NOT_STATIC')
                details.append(detail)
                rows.append({k: v for k, v in detail.items() if k not in ('events', 'boundary_comparisons')})
    json_write(out / 'scenario_results.json', details)
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_seed = []
    for item, _ in agents:
        subset = [r for r in rows if r['planner'] == 'learned' and r['training_seed'] == item['seed']]
        opportunities = sum(r['event_opportunities'] for r in subset)
        def mean(key):
            return float(np.mean([r[key] for r in subset]))
        summary = {'seed': item['seed'], 'episodes': episodes, 'cost_mean': mean('analytical_cost'),
                   'travel_cost_mean': mean('travel_cost'), 'completion_penalty_mean': mean('priority_completion_penalty'),
                   'weighted_travel_cost_mean': mean('weighted_travel_cost'),
                   'weighted_completion_penalty_mean': mean('weighted_completion_penalty'),
                   'delta_fixed_mean': mean('delta_vs_fixed'), 'delta_greedy_mean': mean('delta_vs_greedy'),
                   'exact_ratio_mean': mean('static_exact_ratio') if split == 'static' else None,
                   'exact_ratio_worst': max(r['static_exact_ratio'] for r in subset) if split == 'static' else None,
                   'exact_gap_worst': max(r['exact_gap'] for r in subset) if split == 'static' else None,
                   'completion_ratio': mean('completion_ratio'),
                   'invalid_actions': sum(r['invalid_actions'] for r in subset),
                   'scheduled_events': sum(r['scheduled_events'] for r in subset),
                   'skipped_events': sum(r['skipped_events'] for r in subset), 'applicable_events': opportunities,
                   'changed_events': sum(r['decision_changes'] for r in subset),
                   'beneficial_events': sum(r['beneficial_responses'] for r in subset)}
        summary['decision_change_rate'] = summary['changed_events'] / opportunities if opportunities else None
        summary['beneficial_response_rate'] = summary['beneficial_events'] / opportunities if opportunities else None
        summary['routing_quality_gate'] = (summary['exact_ratio_mean'] <= 1.10 and summary['delta_greedy_mean'] <= 0 and
                                           summary['delta_fixed_mean'] < 0) if split == 'static' else None
        per_seed.append(summary)
    summary = {'schema_version': 2, 'evidence_kind': 'analytical', 'split': split, 'num_checkpoints': n,
               'geometry_profile': geometry_profile,
               'common_scenarios': episodes, 'per_seed': per_seed, 'selected_seed': model_data['selected']['seed'],
               'across_seed_cost_mean': float(np.mean([r['cost_mean'] for r in per_seed])),
               'across_seed_cost_std': float(np.std([r['cost_mean'] for r in per_seed], ddof=1)) if len(per_seed) > 1 else 0.,
               'note': 'Shared cases are correlated; standard deviation is over training-seed means. All cases retained.'}
    json_write(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary
