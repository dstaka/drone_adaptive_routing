#!/usr/bin/env python
"""Reproduce declared retry diagnostics with one cumulative-budget GPU job at a time.

Run from the repository root in the pinned flight environment. Process completion
is not physical qualification: independent raw checkers must inspect all outputs.
This command never grants admission or starts an optimizer.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

from adaptive_drone_patrol.jobs import BUDGET_S, run_job, write_json
from adaptive_drone_patrol.payload_qualification import validate_declaration
from adaptive_drone_patrol.util import sha256

MANIFEST = 'configs/payload_retry_probe_manifest.json'
MOTOR_PLAN = 'configs/payload_retry_motor_audit.json'
SOURCES = (
    'scripts/run_payload_retry_probes.py', 'scripts/probe_payload_ball.py',
    'scripts/probe_payload_ball_bank.py', 'src/adaptive_drone_patrol/jobs.py',
    'src/adaptive_drone_patrol/payload_ball_bank.py',
    'src/adaptive_drone_patrol/payload_ball_asset.py',
    'src/adaptive_drone_patrol/payload_contract.py',
    'src/adaptive_drone_patrol/payload_qualification.py',
    'src/adaptive_drone_patrol/payload_probe_controller.py',
    'src/adaptive_drone_patrol/payload_substep_probe.py',
    'src/adaptive_drone_patrol/payload_cases.py', 'src/adaptive_drone_patrol/util.py',
    MANIFEST, MOTOR_PLAN, 'requirements/flight-linux.lock.txt',
)
GRACE_S = 3.


def ledger_remaining(path):
    """Never initialize or repair a missing cumulative ledger implicitly."""
    path = Path(path)
    if not path.is_file():
        raise ValueError('Existing cumulative GPU budget ledger is required')
    data = json.loads(path.read_text())
    used = data.get('used_s')
    if (isinstance(used, bool) or not isinstance(used, (float, int)) or
            not math.isfinite(used) or used < 0 or not isinstance(data.get('jobs'), list)):
        raise ValueError('Invalid cumulative GPU budget ledger')
    if data.get('active_job'):
        raise ValueError('Active cumulative budget job requires reconciliation')
    return BUDGET_S - used


def build_plan(stage, out, python, phase_timeout, *, root=None,
               ledger='outputs/gpu_ledger.json', reference_audit=None):
    """Freeze every command, source and declaration before any subprocess starts."""
    if stage not in ('motor', 'domain', 'reset'):
        raise ValueError('Unknown diagnostic stage')
    minimum = 20. if stage == 'motor' else 30.
    if (isinstance(phase_timeout, bool) or not isinstance(phase_timeout, (float, int)) or
            not math.isfinite(phase_timeout) or phase_timeout <= minimum + GRACE_S):
        raise ValueError('Positive finite phase timeout with launch/cleanup allowance is required')
    out, root, ledger = Path(out), Path.cwd() if root is None else Path(root), Path(ledger)
    if out.exists():
        raise FileExistsError('Batch output already exists; preserve earlier attempts')
    if not isinstance(python, str) or not python:
        raise ValueError('A flight Python executable is required')
    remaining = ledger_remaining(ledger)
    if remaining < phase_timeout:
        raise ValueError('Phase timeout exceeds remaining cumulative GPU budget')
    manifest = json.loads((root/MANIFEST).read_text())
    validate_declaration(manifest)
    if (manifest['profile'] != 'cf2x-two-native-ball-root-aligned-v3' or
            manifest['settings'].get('root_alignment_requested') is not True):
        raise ValueError('Explicit root-aligned retry profile is required')
    motor = json.loads((root/MOTOR_PLAN).read_text())
    if (motor['hover_duration_s'] != 1. or motor['single_rotor_duration_s'] != .05 or
            motor['fixtures'] != ['original', 'unloaded', 'loaded'] or
            motor['single_rotors'] != [0, 1, 2, 3]):
        raise ValueError('Motor declaration differs from the fifteen-case audit')
    reference = None
    if stage == 'domain':
        if reference_audit is None or not Path(reference_audit).is_file():
            raise ValueError('Domain diagnostics require an actual free-runtime reference audit')
        reference = dict(path=str(reference_audit), sha256=sha256(reference_audit))
    jobs = []

    def append(name, command, timeout, case=None):
        jobs.append(dict(id=name, command=command, timeout_wall_s=float(timeout),
                         job_output=str(out/'jobs'/name), case=case))

    if stage == 'motor':
        for fixture in ('original', 'unloaded', 'loaded'):
            for rotor in (None, 0, 1, 2, 3):
                name = f'motor-{fixture}-' + ('hover' if rotor is None else f'rotor{rotor}') + '-v4'
                command = [python, '-u', 'scripts/probe_payload_ball.py', '--out', str(out/name),
                           '--fixture', fixture]
                if fixture != 'original':
                    command.append('--align-root')
                if rotor is not None:
                    command += ['--rotor', str(rotor)]
                append(name, command, 120.)
    elif stage == 'domain':
        order = {'anchor': 0, 'large_angle': 1, 'powered': 2, 'refinement': 3}
        for case in sorted(manifest['cases'], key=lambda c: order[c['kind']]):
            fixture = 'anchor' if case['kind'] in ('anchor', 'large_angle') else 'loaded'
            command = [python, '-u', 'scripts/probe_payload_ball.py', '--out', str(out/case['id']),
                       '--fixture', fixture, '--case', case['id'], '--align-root',
                       '--reference-audit', str(reference_audit)]
            append(case['id'], command, case['timeout_wall_s'], copy.deepcopy(case))
    else:
        append('bank-reset-v1', [python, '-u', 'scripts/probe_payload_ball_bank.py', '--out',
               str(out/'bank-reset-v1'), '--align-root', 'true'], 300.)
    return dict(schema='payload-retry-diagnostic-batch-plan-v1',
                status='EXPERIMENTAL_DIAGNOSTICS_ONLY', physical_training_admitted=False,
                stage=stage, output=str(out), repository_root=str(root.resolve()),
                ledger=str(ledger), phase_timeout_s=float(phase_timeout),
                minimum_launch_s=minimum, cleanup_grace_s=GRACE_S,
                cumulative_remaining_at_plan_s=remaining, ledger_sha256=sha256(ledger),
                manifest=manifest, motor_plan=motor, reference_audit=reference,
                source_binding={name: sha256(root/name) for name in SOURCES}, jobs=jobs,
                interpretation='Predeclared diagnostic processes only; no physical PASS or admission')


def execute_plan(plan, *, run_job_fn=run_job, clock=time.monotonic):
    """Fail fast, retain partial results and let the supervisor own child cleanup."""
    root, out = Path(plan['repository_root']), Path(plan['output'])
    if Path.cwd().resolve() != root:
        raise ValueError('Execute from the planned repository root')
    if out.exists():
        raise FileExistsError('Batch output already exists')
    def check_frozen_inputs():
        if any(sha256(root/name) != expected for name, expected in plan['source_binding'].items()):
            raise ValueError('Planned source changed before execution')
        if plan['reference_audit'] and sha256(plan['reference_audit']['path']) != plan['reference_audit']['sha256']:
            raise ValueError('Reference audit changed before execution')
    check_frozen_inputs()
    if sha256(plan['ledger']) != plan['ledger_sha256']:
        raise ValueError('Cumulative budget ledger changed; rebuild the batch plan')
    if ledger_remaining(plan['ledger']) < plan['phase_timeout_s']:
        raise ValueError('Insufficient cumulative GPU budget')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'batch-plan.json', plan)
    result = dict(schema='payload-retry-diagnostic-batch-result-v1', status='STARTED',
                  physical_training_admitted=False, batch_complete=False, completed_jobs=0,
                  planned_jobs=len(plan['jobs']), results=[], batch_plan_sha256=sha256(out/'batch-plan.json'))
    started = clock()
    try:
        for job in plan['jobs']:
            check_frozen_inputs()
            remaining = plan['phase_timeout_s'] - (clock()-started)
            if remaining <= plan['minimum_launch_s'] + plan['cleanup_grace_s']:
                result['status'] = 'STOPPED_PHASE_BUDGET'
                break
            timeout = min(job['timeout_wall_s'], remaining-plan['cleanup_grace_s'])
            value = run_job_fn(job['command'], job['job_output'], plan['ledger'], timeout,
                               grace_s=plan['cleanup_grace_s'])
            result['results'].append(dict(id=job['id'], supervisor=value, requested_timeout_s=timeout))
            if value.get('status') != 'SUCCEEDED':
                result['status'] = 'STOPPED_JOB_FAILURE'
                break
            result['completed_jobs'] += 1
            if clock()-started > plan['phase_timeout_s']:
                result['status'] = 'STOPPED_PHASE_OVERRUN'
                break
            write_json(out/'batch-results.json', result)
        else:
            result.update(status='DIAGNOSTIC_PROCESSES_COMPLETED_UNCHECKED', batch_complete=True)
    except BaseException as exc:
        result.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'STOPPED_EXCEPTION',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['elapsed_wall_s'] = clock()-started
        write_json(out/'batch-results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('motor', 'domain', 'reset'), required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--ledger', default='outputs/gpu_ledger.json', type=Path)
    parser.add_argument('--phase-timeout', required=True, type=float)
    parser.add_argument('--reference-audit', type=Path)
    args = parser.parse_args()
    plan = build_plan(args.stage, args.out, args.python, args.phase_timeout,
                      ledger=args.ledger, reference_audit=args.reference_audit)
    result = execute_plan(plan)
    print(result['status'], f"{result['completed_jobs']}/{result['planned_jobs']}")
    return 0 if result['batch_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
