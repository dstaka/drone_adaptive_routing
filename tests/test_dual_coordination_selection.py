"""Synthetic numeric fixtures are NOT measured flight/training evidence."""
from copy import deepcopy
import importlib

import pytest


def module():
    try:
        return importlib.import_module('adaptive_drone_patrol.dual_coordination_selection')
    except ModuleNotFoundError:
        pytest.fail('Prospective selection/report gates are not implemented')


def cohort(time=90., held=2.):
    return [dict(case_id=f'{scenario}-{i}', scenario=scenario, success=True, team_time=time,
                 safety_violations=0, invalid_actions=0, duplicate_service=0,
                 decision_intervals=100, intervened_intervals=10, supervisor_held_drone_seconds=held,
                 useful_post_event_transfer=False, failure_reason=None)
            for scenario in ('normal', 'priority', 'availability-delay') for i in range(100)]


def failures(rows, indices):
    for i in indices:
        rows[i].update(success=False, team_time=None, failure_reason='timeout')
    return rows


def test_selection_tie_order_and_failure_penalty_include_all_cases():
    m = module()
    candidates = {42: cohort(), 43: cohort(), 44: cohort()}
    for row in candidates[43]:
        row['intervened_intervals'] = 9
    for row in candidates[44]:
        row['intervened_intervals'] = 9
    result = m.select_candidate(candidates, cohort(120.), cohort(110.))
    assert result['selected'] == 43 and result['eligible'] == [42, 43, 44]
    failures(candidates[42], [0, 100, 200])
    result = m.select_candidate(candidates, cohort(120.), cohort(110.))
    row = result['candidates'][42]
    assert row['mean_score'] == pytest.approx(95.1)
    assert row['successful_mean_time'] == 90.
    assert row['paired_success_count'] == 297
    assert row['denominator'] == 300


def test_gate_boundaries_and_inadequate_comparator():
    m = module()
    candidates = {seed: cohort(95.) for seed in (42, 43, 44)}
    result = m.select_candidate(candidates, cohort(100.), cohort(95/.97))
    assert result['eligible'] == [42, 43, 44]
    candidates[42][0]['safety_violations'] = 1
    failures(candidates[43], range(6))  # overall98%, normal94%
    candidates[44][0]['intervened_intervals'] = 11
    result = m.select_candidate(candidates, cohort(100.), cohort(95/.97))
    assert result['selected'] is None
    assert 'zero_safety' in result['candidates'][42]['reasons']
    assert 'class_completion:normal' in result['candidates'][43]['reasons']
    assert 'intervention_fraction' in result['candidates'][44]['reasons']
    dynamic = failures(cohort(200.), range(16))
    result = m.select_candidate({seed: cohort() for seed in (42, 43, 44)}, cohort(220.), dynamic)
    assert result['selected'] is None
    assert 'dynamic_comparator_completion' in result['candidates'][42]['reasons']


def test_paired_speed_gate_not_replaced_by_failure_penalized_improvement():
    m = module()
    dynamic = failures(cohort(100.), range(15))
    result = m.select_candidate({seed: cohort(99.) for seed in (42, 43, 44)}, cohort(130.), dynamic)
    assert result['selected'] is None
    assert 'paired_actual_improvement' in result['candidates'][42]['reasons']
    assert result['candidates'][42]['paired_success_count'] == 285


@pytest.mark.parametrize('mutation', ['short', 'duplicate', 'mismatch', 'nan', 'unknown', 'missing', 'bad_count'])
def test_malformed_or_unpaired_cohorts_rejected(mutation):
    m = module()
    rows = cohort()
    if mutation == 'short':
        rows.pop()
    elif mutation == 'duplicate':
        rows[1]['case_id'] = rows[0]['case_id']
    elif mutation == 'mismatch':
        rows[0]['case_id'] = 'wrong-case'
    elif mutation == 'nan':
        rows[0]['team_time'] = float('nan')
    elif mutation == 'unknown':
        rows[0]['scenario'] = 'unknown'
    elif mutation == 'missing':
        del rows[0]['safety_violations']
    else:
        rows[0]['intervened_intervals'] = 101
    with pytest.raises(ValueError):
        m.select_candidate({42: rows, 43: cohort(), 44: cohort()}, cohort(120.), cohort(110.))


def test_deadline_overrun_scores_failure_and_checkpoint_receipts_are_strict():
    m = module()
    rows = cohort()
    rows[0]['team_time'] = 300.00001
    result = m.select_candidate({42: rows, 43: cohort(), 44: cohort()}, cohort(120.), cohort(110.))
    assert result['candidates'][42]['success_count'] == 299
    manifests = {seed: dict(sha256=str(seed)*32, transitions=1048576, updates=256, stage3_transitions=65536)
                 for seed in (42, 43, 44)}
    assert m.validate_terminal_checkpoints(manifests) == manifests
    manifests[43]['transitions'] = 1044480
    with pytest.raises(ValueError, match='terminal'):
        m.validate_terminal_checkpoints(manifests)


def test_routing_bootstrap_transfer_and_yielding_gates():
    m = module()
    full = cohort(90., held=8.)
    for row in full[200:220]:
        row['useful_post_event_transfer'] = True
    crossing = dict(predeclared=True, success=True, safety_violations=0, intervened_intervals=0,
                    nn_priority_yield_visible=True)
    result = m.routing_report(full, cohort(120.), cohort(100.), cohort(100.), cohort(100., held=10.), crossing)
    assert result['passed'] is True
    assert result['bootstrap']['interval_95'] == [-10., -10.]
    assert result['bootstrap']['resamples'] == 10000
    assert result['bootstrap']['seed'] == 10272799
    assert result['useful_transfer_cases'] == 20 and result['transfer_denominator'] == 100
    result2 = m.routing_report(full, cohort(120.), cohort(100.), cohort(100.), cohort(100., held=0.), crossing)
    assert 'yielding_reduction' in result2['reasons']
    full[219]['useful_post_event_transfer'] = False
    result3 = m.routing_report(full, cohort(120.), cohort(100.), cohort(100.), cohort(100., held=10.), crossing)
    assert 'useful_transfer_count' in result3['reasons']


def physical():
    rows = cohort()[:127]
    for i, row in enumerate(rows):
        row['case_id'] = f'physical-{i}'
        row['group'] = 'basic' if i < 9 else 'endpoints' if i < 27 else 'mixed'
        group_index = i if i < 9 else i-9 if i < 27 else i-27
        bounds = (3, 6) if i < 9 else (6, 12) if i < 27 else (34, 67)
        row['scenario'] = 'normal' if group_index < bounds[0] else 'priority' if group_index < bounds[1] else 'availability-delay'
        row['independently_audited'] = True
    return rows


def test_physical_cohort_is_separate_and_safety_allowance_is_zero():
    m = module()
    rows = failures(physical(), range(27, 32))
    result = m.physical_report(rows)
    assert result['passed'] and result['denominator'] == 127
    assert result['groups']['mixed']['success_count'] == 95
    rows[27]['safety_violations'] = 1
    assert 'zero_safety' in m.physical_report(rows)['reasons']
    rows[27]['safety_violations'] = 0
    rows[0]['independently_audited'] = False
    assert 'independent_audit' in m.physical_report(rows)['reasons']
    with pytest.raises(ValueError):
        m.physical_report(rows[:126])
    wrong = deepcopy(rows)
    wrong[0]['group'] = 'mixed'
    with pytest.raises(ValueError):
        m.physical_report(wrong)


def test_bootstrap_keeps_each_scenario_stratum_fixed():
    m = module()
    full = cohort()
    for row in full:
        row['team_time'] = {'normal': 80., 'priority': 90., 'availability-delay': 100.}[row['scenario']]
    crossing = dict(predeclared=True, success=True, safety_violations=0, intervened_intervals=0,
                    nn_priority_yield_visible=True)
    result = m.routing_report(full, cohort(120.), cohort(100.), cohort(100.), cohort(100., held=10.), crossing)
    # Each stratum is constant, so stratified bootstrap has no variance despite
    # the between-scenario effect differences. Unstratified bootstrap would vary.
    assert result['bootstrap']['interval_95'] == [-10., -10.]
    equal = m.routing_report(cohort(100.), cohort(120.), cohort(100.), cohort(100.), cohort(100.), crossing)
    assert equal['bootstrap']['interval_95'] == [0., 0.]
    assert 'statistical_support' in equal['reasons']


def test_exact_95_percent_class_and_270_pair_boundaries():
    m = module()
    candidate = failures(cohort(90.), list(range(5))+list(range(100, 105))+list(range(200, 205)))
    dynamic = failures(cohort(100.), list(range(5, 10))+list(range(105, 110))+list(range(205, 210)))
    result = m.select_candidate({seed: candidate for seed in (42, 43, 44)}, cohort(130.), dynamic)
    assert result['eligible'] == [42, 43, 44]
    assert result['candidates'][42]['paired_success_count'] == 270
    assert result['candidates'][42]['success_count'] == 285
