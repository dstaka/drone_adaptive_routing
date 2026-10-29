"""Prospective numeric gates for spec §9, never a substitute for flight evidence.

Each outcome requires case_id, scenario, success, team_time, safety_violations,
invalid_actions, duplicate_service, decision_intervals, intervened_intervals,
supervisor_held_drone_seconds, useful_post_event_transfer and failure_reason.
The transfer flag means independently reconstructed post-event ownership change
AND successful service by its new owner, not just any assignment change.

select_candidate is numerical selection only. Before freeze its caller MUST
validate all three terminal manifests with validate_terminal_checkpoints, verify
the actual checkpoint hashes, authenticate common validation case/raw identities,
and bind this receipt to those artifacts. No numeric dictionary authenticates
raw evidence or proves qualification/freeze/media/delivery gates. Likewise the
crossing declaration and independent-audit flags are evidence assertions which
the orchestration/audit layer must substantiate with hashes and raw receipts.
"""
from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np

SCENARIOS = ('normal', 'priority', 'availability-delay')
_FIELDS = {'case_id', 'scenario', 'success', 'team_time', 'safety_violations', 'invalid_actions',
           'duplicate_service', 'decision_intervals', 'intervened_intervals',
           'supervisor_held_drone_seconds', 'useful_post_event_transfer', 'failure_reason'}
_COUNTS = ('safety_violations', 'invalid_actions', 'duplicate_service', 'decision_intervals', 'intervened_intervals')


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _outcomes(rows, *, physical=False):
    expected = 127 if physical else 300
    if not isinstance(rows, (list, tuple)) or len(rows) != expected:
        raise ValueError(f'Exactly {expected} declared outcomes are required; missing cases cannot be dropped')
    fields = _FIELDS | ({'group', 'independently_audited'} if physical else set())
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError('Exact outcome fields required')
        case_id = row['case_id']
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError('Unique nonempty case IDs required')
        seen.add(case_id)
        if row['scenario'] not in SCENARIOS:
            raise ValueError('Unknown scenario')
        if type(row['success']) is not bool or type(row['useful_post_event_transfer']) is not bool:
            raise ValueError('Explicit boolean outcome flags required')
        if row['team_time'] is not None and not _finite_nonnegative(row['team_time']):
            raise ValueError('Team time must be finite nonnegative or null on failure')
        if row['success'] and row['team_time'] is None:
            raise ValueError('Successful outcome requires actual team time')
        if row['success'] and row['failure_reason'] is not None:
            raise ValueError('Successful outcome cannot have a failure reason')
        if not row['success'] and (not isinstance(row['failure_reason'], str) or not row['failure_reason']):
            raise ValueError('Failed outcomes require explicit failure reasons')
        if any(type(row[k]) is not int or row[k] < 0 for k in _COUNTS):
            raise ValueError('Nonnegative integer counters required')
        if row['intervened_intervals'] > row['decision_intervals']:
            raise ValueError('Intervened intervals cannot exceed decision intervals')
        if row['success'] and not row['decision_intervals']:
            raise ValueError('A successful mission requires measured decision intervals')
        if not _finite_nonnegative(row['supervisor_held_drone_seconds']):
            raise ValueError('Finite nonnegative held drone-seconds required')
        if row['useful_post_event_transfer'] and row['scenario'] != 'availability-delay':
            raise ValueError('Useful transfer indicator is defined for the delay cohort')
        if physical and (row['group'] not in ('basic', 'endpoints', 'mixed') or
                         type(row['independently_audited']) is not bool):
            raise ValueError('Explicit physical cohort group and independent audit flag required')
    if not physical:
        if Counter(r['scenario'] for r in rows) != dict.fromkeys(SCENARIOS, 100):
            raise ValueError('Routing/validation denominator requires 100 cases per scenario')
    else:
        expected_groups = {'basic': (3, 3, 3), 'endpoints': (6, 6, 6), 'mixed': (34, 33, 33)}
        for group, counts in expected_groups.items():
            actual = Counter(r['scenario'] for r in rows if r['group'] == group)
            if actual != dict(zip(SCENARIOS, counts)):
                raise ValueError('Physical cohort must contain basic9, endpoints18 and mixed100 with declared scenario counts')
    return sorted(rows, key=lambda r: (SCENARIOS.index(r['scenario']), r['case_id']))


def _aligned(reference, other):
    if [(r['case_id'], r['scenario']) for r in reference] != [(r['case_id'], r['scenario']) for r in other]:
        raise ValueError('Methods must use identical case IDs and scenarios')


def _success(row):
    return row['success'] and row['team_time'] <= 300


def _scores(rows):
    return np.array([r['team_time'] if _success(r) else 600. for r in rows])


def _summary(rows):
    successful = [r['team_time'] for r in rows if _success(r)]
    intervals = sum(r['decision_intervals'] for r in rows)
    intervened = sum(r['intervened_intervals'] for r in rows)
    reasons = Counter('deadline_exceeded' if r['success'] else r['failure_reason'] for r in rows if not _success(r))
    return dict(denominator=len(rows), success_count=len(successful), success_rate=len(successful)/len(rows),
                mean_score=float(_scores(rows).mean()),
                successful_mean_time=float(np.mean(successful)) if successful else None,
                successful_median_time=float(np.median(successful)) if successful else None,
                successful_p95_time=float(np.percentile(successful, 95)) if successful else None,
                safety_violations=sum(r['safety_violations'] for r in rows),
                invalid_actions=sum(r['invalid_actions'] for r in rows),
                duplicate_service=sum(r['duplicate_service'] for r in rows),
                decision_intervals=intervals, intervened_intervals=intervened,
                intervention_fraction=intervened/intervals if intervals else None,
                supervisor_held_drone_seconds=sum(r['supervisor_held_drone_seconds'] for r in rows),
                failure_reasons=dict(reasons))


def _at_most(value, limit):
    # Floating reduction roundoff must not reject an exact specified boundary.
    return value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)


def _comparison(rows, static, dynamic):
    _aligned(rows, static)
    _aligned(rows, dynamic)
    result, base_static, base_dynamic = _summary(rows), _summary(static), _summary(dynamic)
    reasons = []
    checks = dict(zero_safety=result['safety_violations'] == 0,
                  completion=result['success_count'] >= 285,
                  valid_joint_actions=result['invalid_actions'] == 0,
                  unique_service=result['duplicate_service'] == 0,
                  intervention_fraction=result['intervention_fraction'] is not None and
                  result['intervention_fraction'] <= .10,
                  static_score_improvement=_at_most(result['mean_score'], .95*base_static['mean_score']),
                  dynamic_score_improvement=_at_most(result['mean_score'], .97*base_dynamic['mean_score']),
                  dynamic_comparator_completion=base_dynamic['success_count'] >= 285,
                  dynamic_comparator_safety=base_dynamic['safety_violations'] == 0)
    classes = {}
    for scenario in SCENARIOS:
        current = _summary([r for r in rows if r['scenario'] == scenario])
        baseline = _summary([r for r in dynamic if r['scenario'] == scenario])
        classes[scenario] = current
        checks[f'class_completion:{scenario}'] = current['success_count'] >= 95
        checks[f'class_score_guard:{scenario}'] = _at_most(current['mean_score'], 1.05*baseline['mean_score'])
    paired = [(a['team_time'], b['team_time']) for a, b in zip(rows, dynamic) if _success(a) and _success(b)]
    nmean = float(np.mean([a for a, _ in paired])) if paired else None
    bmean = float(np.mean([b for _, b in paired])) if paired else None
    checks['paired_success_count'] = len(paired) >= 270
    checks['paired_actual_improvement'] = bool(paired) and _at_most(nmean, .97*bmean)
    reasons.extend(name for name, passed in checks.items() if not passed)
    result.update(passed=not reasons, reasons=reasons, checks=checks, classes=classes,
                  static_mean_score=base_static['mean_score'], dynamic_mean_score=base_dynamic['mean_score'],
                  paired_success_count=len(paired), paired_nn_mean_time=nmean, paired_dynamic_mean_time=bmean,
                  baseline_static=base_static, baseline_dynamic=base_dynamic)
    return result


def validate_terminal_checkpoints(manifests):
    """Validate metadata only; caller must independently hash actual checkpoint bytes."""
    if not isinstance(manifests, dict) or set(manifests) != {42, 43, 44}:
        raise ValueError('All three production seed terminal manifests required')
    for seed, manifest in manifests.items():
        if type(seed) is not int or not isinstance(manifest, dict) or set(manifest) != {
                'sha256', 'transitions', 'updates', 'stage3_transitions'}:
            raise ValueError('Exact terminal manifest fields required')
        if not isinstance(manifest['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', manifest['sha256']):
            raise ValueError('Terminal checkpoint requires a lowercase SHA-256')
        if any(type(manifest[k]) is not int or manifest[k] != v for k, v in
               (('transitions', 1048576), ('updates', 256), ('stage3_transitions', 65536))):
            raise ValueError('Only fully trained terminal checkpoint counters are selectable')
    if len({m['sha256'] for m in manifests.values()}) != 3:
        raise ValueError('Three independently trained checkpoint hashes required')
    return {seed: dict(manifest) for seed, manifest in manifests.items()}


def select_candidate(candidate_results, static_results, dynamic_results):
    """Return prospective numeric eligibility; no checkpoint/raw authentication."""
    if not isinstance(candidate_results, dict) or set(candidate_results) != {42, 43, 44} or any(
            type(seed) is not int for seed in candidate_results):
        raise ValueError('All three seeds 42,43,44 must finish validation before selection')
    static, dynamic = _outcomes(static_results), _outcomes(dynamic_results)
    reports = {seed: _comparison(_outcomes(candidate_results[seed]), static, dynamic) for seed in (42, 43, 44)}
    eligible = [seed for seed, report in reports.items() if report['passed']]
    selected = min(eligible, key=lambda seed: (reports[seed]['mean_score'], reports[seed]['intervened_intervals'], seed)) if eligible else None
    return dict(schema='dual-selection-numeric-v1', evidence_scope='numeric_only', eligible=eligible,
                selected=selected, candidates=reports, reasons=[] if eligible else ['no_eligible_candidate'],
                terminal_checkpoint_hashes_verified=False)


def _bootstrap(rows, dynamic):
    difference = _scores(rows)-_scores(dynamic)
    rng = np.random.default_rng(10272799)
    draws = np.zeros(10000)
    # Class-stratified paired case-index resampling, 100 observations/class/draw.
    for scenario in SCENARIOS:
        values = difference[[r['scenario'] == scenario for r in rows]]
        for start in range(0, 10000, 1000):
            sampled = values[rng.integers(0, 100, size=(1000, 100))]
            draws[start:start+1000] += sampled.sum(axis=1)/300
    interval = np.percentile(draws, [2.5, 97.5]).tolist()
    return dict(mean_score_difference=float(difference.mean()), interval_95=interval,
                resamples=10000, seed=10272799, stratified_by='scenario', paired=True,
                passed=interval[1] < 0)


def routing_report(full_results, static_results, dynamic_results, no_transfer_results, fixed_priority_results, crossing):
    """Evaluate fixed selected policy; never reselect on held-out results."""
    full, static, dynamic, no_transfer, fixed = map(_outcomes,
        (full_results, static_results, dynamic_results, no_transfer_results, fixed_priority_results))
    for rows in (static, dynamic, no_transfer, fixed):
        _aligned(full, rows)
    if not isinstance(crossing, dict) or set(crossing) != {'predeclared', 'success', 'safety_violations',
                                                         'intervened_intervals', 'nn_priority_yield_visible'}:
        raise ValueError('Complete crossing challenge evidence required')
    if any(type(crossing[k]) is not bool for k in ('predeclared', 'success', 'nn_priority_yield_visible')) or any(
            type(crossing[k]) is not int or crossing[k] < 0 for k in ('safety_violations', 'intervened_intervals')):
        raise ValueError('Malformed crossing challenge evidence')
    comparison = _comparison(full, static, dynamic)
    bootstrap = _bootstrap(full, dynamic)
    delay_full = [r for r in full if r['scenario'] == 'availability-delay']
    delay_ablated = [r for r in no_transfer if r['scenario'] == 'availability-delay']
    useful = sum(r['useful_post_event_transfer'] for r in delay_full)
    held = sum(r['supervisor_held_drone_seconds'] for r in full)
    ablated_held = sum(r['supervisor_held_drone_seconds'] for r in fixed)
    checks = dict(statistical_support=bootstrap['passed'],
                  transfer_score_improvement=_at_most(_scores(delay_full).mean(), .97*_scores(delay_ablated).mean()),
                  useful_transfer_count=useful >= 20,
                  crossing_yield=crossing['predeclared'] and crossing['success'] and crossing['safety_violations'] == 0 and
                  crossing['intervened_intervals'] == 0 and crossing['nn_priority_yield_visible'],
                  yielding_reduction=ablated_held > 0 and _at_most(held, .8*ablated_held),
                  yielding_completion=sum(map(_success, full)) >= sum(map(_success, fixed)))
    checks = {k: bool(v) for k, v in checks.items()}
    reasons = comparison['reasons']+[k for k, passed in checks.items() if not passed]
    return dict(schema='dual-routing-numeric-v1', evidence_scope='numeric_only', passed=not reasons,
                reasons=reasons, checks=checks, comparison=comparison, bootstrap=bootstrap,
                useful_transfer_cases=useful, transfer_denominator=100,
                delay_full_mean_score=float(_scores(delay_full).mean()),
                delay_no_transfer_mean_score=float(_scores(delay_ablated).mean()),
                full_held_drone_seconds=held, fixed_priority_held_drone_seconds=ablated_held)


def physical_report(results):
    rows = _outcomes(results, physical=True)
    summary = _summary(rows)
    groups = {name: _summary([r for r in rows if r['group'] == name]) for name in ('basic', 'endpoints', 'mixed')}
    checks = dict(basic_completion=groups['basic']['success_count'] == 9,
                  endpoints_completion=groups['endpoints']['success_count'] == 18,
                  mixed_completion=groups['mixed']['success_count'] >= 95,
                  zero_safety=summary['safety_violations'] == 0,
                  independent_audit=all(r['independently_audited'] for r in rows),
                  valid_joint_actions=summary['invalid_actions'] == 0,
                  unique_service=summary['duplicate_service'] == 0)
    reasons = [k for k, passed in checks.items() if not passed]
    return dict(schema='dual-physical-numeric-v1', evidence_scope='numeric_only', **summary,
                groups=groups, checks=checks, reasons=reasons, passed=not reasons)
