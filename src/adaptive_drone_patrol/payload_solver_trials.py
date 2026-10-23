"""Exactly one fresh paired254-case attempt after immutable acceptance freeze."""

import json
from pathlib import Path

from .jobs import write_json
from .payload_solver_acceptance import verify_mission, paired_statistics
from .payload_solver_flight import SolverFlight
from .payload_solver_freeze import require_acceptance_freeze, verify_file_snapshot, ROUTING_SHA
from .payload_solver_validation import verify_initial_sample
from .payload_solver_heldout import claim_path, ORIGINAL_CASES_SHA, verify_opening
from .payload_recovery_mission import execute_payload_mission
from .scenario import Scenario
from .util import load_config, sha256


def claim_heldout_once(freeze_path, output):
    """Atomically claim the canonical case stream across all freeze identities."""
    freeze_path, output = Path(freeze_path), Path(output)
    if output.exists():
        raise FileExistsError('Preserve every original heldout attempt')
    marker = dict(schema='payload-solver-heldout-opened-v2',
        freeze_sha256=sha256(freeze_path), output=str(output), required_trials=254,
        canonical_cases_sha256=ORIGINAL_CASES_SHA,
        rule='No selective rerun, model change or method-specific rescue')
    claim = claim_path()
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open('x') as stream:
        json.dump(marker, stream, indent=2)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'opened.json', marker)
    return marker


def artifact_snapshot(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()}


def failed_trial(method, case, root, error):
    """Preserve corruption/execution failures in the unchanged paired denominator."""
    return dict(method=method, case_id=case['case_id'], success=False,
        physical_acceptance=False, error=f'{type(error).__name__}: {error}',
        artifact_sha256=artifact_snapshot(root))


def run_paired_trials(*, freeze_path, output, original, admission, planner):
    """Run both original127 sets, continuing other trials after a case failure."""
    freeze = json.loads(Path(freeze_path).read_text())
    cases = require_acceptance_freeze(freeze, admission)['cases']
    if sha256(freeze['routing_model']) != ROUTING_SHA or planner.policy_hash != ROUTING_SHA:
        raise ValueError('Actual adopted routing IPC model differs')
    root = Path(output)
    claim_heldout_once(freeze_path, root)
    write_json(root / 'binding.json', dict(freeze_sha256=sha256(freeze_path),
        original_asset_sha256=sha256(original), actual_binding=admission.binding,
        methods=['randomized', 'fixed'], case_order=[c['case_id'] for c in cases],
        physical_trials=254))
    results = []
    for method in ['randomized', 'fixed']:
        item = freeze['models'][method]
        for case in cases:
            verify_file_snapshot(freeze['artifact_sha256'])
            admission.require_current()
            trial = root / method / case['case_id']
            trial.mkdir(parents=True, exist_ok=False)
            write_json(trial / 'case.json', case)
            flight = None
            try:
                flight = SolverFlight.from_checkpoint(admission=admission,
                    checkpoint=item['checkpoint'], expected_hash=item['sha256'],
                    original=original, case=case, output=trial / 'flight', expected_method=method)
                verify_initial_sample(flight.sample(), case['initial'])
                scenario = Scenario.from_dict(load_config(
                    f"configs/demo_scenarios_dispersed/{case['scenario']}.yaml"))
                execute_payload_mission(flight, planner, scenario, trial / 'mission',
                    dict(evidence_kind='physical', method=method, case_id=case['case_id'],
                         checkpoint_sha256=item['sha256'], routing_model_hash=ROUTING_SHA,
                         acceptance_freeze_sha256=sha256(freeze_path)),
                    physical_evidence=dict(solver_admission=admission, actual_binding=admission.binding))
                flight.close()
                flight = None
                proof = verify_mission(trial / 'mission', planner, native=dict(
                    native_path=trial / 'flight/runtime/raw.jsonl.gz',
                    properties_path=trial / 'flight/runtime-properties.json',
                    checkpoint=item['checkpoint'], expected_hash=item['sha256'],
                    case=case, binding=admission.binding))
                result = dict(method=method, case_id=case['case_id'], **proof,
                              artifact_sha256=artifact_snapshot(trial))
            except Exception as exc:
                if flight is not None:
                    flight.close()
                result = failed_trial(method, case, trial, exc)
            results.append(result)
            write_json(trial / 'independent-check.json', result)
            write_json(root / 'results.partial.json', results)
    stats = paired_statistics(results, cases)
    write_json(root / 'results.json', results)
    write_json(root / 'paired-statistics.json', stats)
    return stats


def recheck_paired_trials(*, freeze_path, output, admission, planner):
    """Reconstruct every claimed success from native bytes and routing again."""
    freeze = json.loads(Path(freeze_path).read_text())
    cases = require_acceptance_freeze(freeze, admission)['cases']
    root = Path(output)
    if (json.loads((root / 'opened.json').read_text())['freeze_sha256'] != sha256(freeze_path)
            or planner.policy_hash != ROUTING_SHA):
        raise ValueError('Actual heldout run/frozen router differs')
    verify_opening(freeze_path, root)
    saved = json.loads((root / 'results.json').read_text())
    # Validate the complete denominator BEFORE replay, including all failures.
    paired_statistics(saved, cases)
    lookup = {(r['method'], r['case_id']): r for r in saved}
    results = []
    for method in ['randomized', 'fixed']:
        item = freeze['models'][method]
        for case in cases:
            trial = root / method / case['case_id']
            recorded = lookup[method, case['case_id']]
            for path, digest in recorded['artifact_sha256'].items():
                if Path(path).is_absolute() or '..' in Path(path).parts or sha256(trial / path) != digest:
                    raise ValueError('Retained original trial artifact changed')
            if json.loads((trial / 'case.json').read_text()) != case:
                raise ValueError('Actual paired case changed')
            try:
                proof = verify_mission(trial / 'mission', planner, native=dict(
                    native_path=trial / 'flight/runtime/raw.jsonl.gz',
                    properties_path=trial / 'flight/runtime-properties.json',
                    checkpoint=item['checkpoint'], expected_hash=item['sha256'],
                    case=case, binding=admission.binding))
                result = dict(method=method, case_id=case['case_id'], **proof)
            except Exception as exc:
                if recorded['success'] or recorded['physical_acceptance']:
                    raise ValueError('Claimed physical success failed independent replay') from exc
                result = failed_trial(method, case, trial, exc)
            if (result['success'] != recorded['success']
                    or result['physical_acceptance'] != recorded['physical_acceptance']):
                raise ValueError('Retained original trial verdict differs on fresh replay')
            for path, digest in recorded['artifact_sha256'].items():
                if sha256(trial / path) != digest:
                    raise ValueError('Retained original trial changed during fresh replay')
            result['artifact_sha256'] = dict(recorded['artifact_sha256'])
            results.append(result)
    return dict(trials=254, statistics=paired_statistics(results, cases), results=results)
