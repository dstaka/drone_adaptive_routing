"""Independent cost and provenance checks for analytical boundary evidence."""

import copy
import numpy as np
import pytest
from adaptive_drone_patrol import analytical
from adaptive_drone_patrol.scenario import generate_scenario
from adaptive_drone_patrol.objective import RoutingState
from adaptive_drone_patrol.baselines import choose_baseline
from test_scalable import independent_cost


def test_full_counterfactuals_share_boundary_and_post_event_objective():
    s = generate_scenario(30001, 'dynamic', num_checkpoints=10)
    st = RoutingState(s)
    st.complete(0, st.departure(0))
    before = st.copy()
    result = analytical.event_comparison(st, lambda state: choose_baseline(state, 'greedy'))
    assert st.cost == before.cost and st.events == []
    assert len(result['no_event_order']) == len(result['post_event_order']) == 9
    post = st.copy()
    post.apply_event()
    assert result['retained_order_post_event']['cost'] == pytest.approx(
        independent_cost(post.scenario, result['no_event_order'], post.current, post.elapsed))


def test_explanation_all_q_values_and_provenance_reject_tampering():
    s = generate_scenario(20001, 'static', num_checkpoints=4)
    st = RoutingState(s)
    st.complete(1, st.departure(1))
    def chooser(state):
        return choose_baseline(state, 'fixed')
    result = analytical.explain_decision(st, chooser, 'model-hash', 'run-1', 'decision-1')
    assert result['evidence_kind'] == 'analytical'
    assert len(result['forced_first']) == 3
    for value in result['forced_first']:
        assert value['cost'] == pytest.approx(independent_cost(s, value['order'], st.current, st.elapsed))
    analytical.verify_explanation(result, 'model-hash', 'run-1', 'decision-1')
    for field, value in [('boundary_hash', 'bad'), ('policy_hash', 'other'),
                         ('decision_id', 'other'), ('run_id', 'other')]:
        bad = copy.deepcopy(result)
        bad[field] = value
        with pytest.raises(ValueError):
            analytical.verify_explanation(bad, 'model-hash', 'run-1', 'decision-1')
    bad = copy.deepcopy(result)
    bad['snapshot']['scenario']['alpha'] = 2.
    with pytest.raises(ValueError):
        analytical.verify_explanation(bad, 'model-hash', 'run-1', 'decision-1')


def test_incomplete_solver_cannot_claim_policy_optimality():
    assert analytical.optimality_label(10., {'cost': 10., 'status': 'TIMEOUT', 'proven': False}) == 'UNCONFIRMED'
    assert analytical.optimality_label(11., {'cost': 10., 'status': 'OPTIMAL', 'proven': True}) == 'SUBOPTIMAL'


def test_demo_geometry_and_balanced_shared_initial_conditions():
    s = analytical.demo_fixture(40000)
    assert s.num_checkpoints == 10
    assert min(s.priorities) >= 2 and max(s.priorities) <= 3
    assert max(s.priorities) / min(s.priorities) <= 1.5
    assert np.asarray(s.multipliers).min() == np.asarray(s.multipliers).max() == 1
    s.validate_physical()


@pytest.mark.parametrize('field,value', [('selected_L', 'fake'), ('nearer_lower_priority_gate', True),
                                        ('candidate_priorities_distances', [])])
def test_explanation_rejects_false_tradeoff_annotations(field, value):
    st = RoutingState(generate_scenario(20001, 'static', num_checkpoints=4))
    result = analytical.explain_decision(st, lambda state: choose_baseline(state, 'fixed'),
                                         'hash', 'run', 'decision')
    result[field] = value
    with pytest.raises(ValueError):
        analytical.verify_explanation(result, 'hash', 'run', 'decision')


def test_nonfinite_explanation_is_rejected():
    st = RoutingState(generate_scenario(20001, 'static', num_checkpoints=4))
    value = analytical.explain_decision(st, lambda state: choose_baseline(state, 'fixed'), 'hash', 'run', 'd')
    value['exact']['cost'] = float('nan')
    value['exact']['travel_cost'] = float('nan')
    value['policy_gap'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        analytical.verify_explanation(value, 'hash', 'run', 'd')


def test_benchmark_return_components_match_total():
    from adaptive_drone_patrol.benchmark import evaluate_scenario
    s = generate_scenario(20001, 'static')
    s.return_to_base = True
    result = evaluate_scenario(s, lambda state: choose_baseline(state, 'fixed'))
    assert result['analytical_cost'] == pytest.approx(s.alpha * result['travel_cost'] +
                                                    s.beta * result['priority_completion_penalty'])


def test_independent_cost_replay_cli_rejects_corrupt_components(tmp_path):
    import json
    import subprocess
    import sys
    st = RoutingState(generate_scenario(20001, 'static', num_checkpoints=10))
    record = analytical.explain_decision(st, lambda state: choose_baseline(state, 'fixed'), 'hash', 'run', 'd')
    source = tmp_path / 'explanations.json'
    source.write_text(json.dumps({'decisions': [record]}))
    command = [sys.executable, 'scripts/verify_analytical_costs.py', '--explanations', str(source),
               '--policy-hash', 'hash', '--out', str(tmp_path / 'ok.json')]
    good = subprocess.run(command, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert json.loads((tmp_path / 'ok.json').read_text())['orders_verified'] == 12
    record['forced_first'][0]['travel_cost'] += 1
    source.write_text(json.dumps({'decisions': [record]}))
    command[-1] = str(tmp_path / 'bad.json')
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_restored_prefix_preserves_reported_tolerance_despite_roundoff():
    for seed in range(10000, 10005):
        st = RoutingState(generate_scenario(seed, 'validation', num_checkpoints=10))
        st.complete(0, st.departure(0))
        value = analytical.explain_decision(st, lambda state: choose_baseline(state, 'fixed'),
                                           'hash', 'run', str(seed))
        assert analytical.verify_explanation(value, 'hash', 'run', str(seed))
