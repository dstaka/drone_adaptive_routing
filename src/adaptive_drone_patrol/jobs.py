"""Durable single-GPU job execution with a cumulative wall-time budget."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import math
import threading

BUDGET_S = 8 * 60 * 60


def terminate_and_reap(proc, grace_s):
    """Stop the owned session, including descendants surviving the leader."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        code = proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        code = None
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return proc.wait(timeout=5) if code is None else code


def interrupted(signum, _frame):
    raise SystemExit(f'Supervisor interrupted by signal {signum}')


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def run_job(command, output, ledger, timeout_s, grace_s=.5, budget_s=BUDGET_S):
    output, ledger = Path(output), Path(ledger)
    if not math.isfinite(budget_s) or budget_s <= 0:
        raise ValueError('budget_s must be positive and finite')
    if not math.isfinite(timeout_s) or timeout_s <= 0 or not math.isfinite(grace_s) or grace_s <= 0:
        raise ValueError('timeout_s must be positive')
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another job owns the budget ledger') from exc
        data = json.loads(ledger.read_text()) if ledger.exists() else {'used_s': 0., 'jobs': []}
        if data.get('active_job'):
            raise RuntimeError('Unreconciled active job; inspect it before resuming')
        remaining = budget_s - data['used_s']
        if remaining <= grace_s:
            raise RuntimeError('Execution budget exhausted')
        output.mkdir(parents=True, exist_ok=False)
        deadline = min(float(timeout_s), remaining - grace_s)
        started = time.time()
        result = {'run_id': output.name, 'command': list(command), 'status': 'RUNNING',
                  'started_epoch': started, 'timeout_s': deadline, 'pid': None}
        data['active_job'] = result.copy()
        write_json(ledger, data)
        write_json(output / 'status.json', result)
        start = time.monotonic()
        timed_out = False
        proc, cleanup_confirmed = None, True
        old_handler = None
        if threading.current_thread() is threading.main_thread():
            old_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            with (output / 'job.log').open('w') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                cleanup_confirmed = False
                result['pid'] = proc.pid
                data['active_job'] = result.copy()
                write_json(ledger, data)
                write_json(output / 'status.json', result)
                try:
                    code = proc.wait(timeout=deadline)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    code = terminate_and_reap(proc, grace_s)
                cleanup_confirmed = True
            result.update(status='TIMED_OUT' if timed_out else ('SUCCEEDED' if code == 0 else 'FAILED'),
                          exit_code=code)
        except BaseException as exc:
            result.update(status='FAILED', exit_code=-1, error=str(exc))
            if proc is not None and not cleanup_confirmed:
                try:
                    terminate_and_reap(proc, grace_s)
                    cleanup_confirmed = True
                except BaseException as cleanup_error:
                    result['cleanup_error'] = repr(cleanup_error)
            raise
        finally:
            result.update(elapsed_s=time.monotonic() - start, finished_epoch=time.time())
            data['used_s'] += result['elapsed_s']
            data['jobs'].append(result)
            if cleanup_confirmed:
                data.pop('active_job', None)
            else:
                data['active_job'] = {**result, 'requires_reconciliation': True}
            write_json(output / 'status.json', result)
            write_json(ledger, data)
            if old_handler is not None:
                signal.signal(signal.SIGTERM, old_handler)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--budget-s', type=float, default=BUDGET_S)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error('Specify an executable after --')
    result = run_job(args.command[1:] if args.command[0] == '--' else args.command,
                     args.out, args.ledger, args.timeout, grace_s=20, budget_s=args.budget_s)
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'SUCCEEDED' else 1)


if __name__ == '__main__':
    main()
