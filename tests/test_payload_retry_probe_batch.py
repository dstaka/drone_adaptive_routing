"""Batch reproduction tests mock only the GPU supervisor, never physical PASS."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / 'scripts/run_payload_retry_probes.py'


def api():
    assert SCRIPT.is_file(), 'Public retry diagnostic batch runner is required'
    spec = importlib.util.spec_from_file_location('retry_probe_batch', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(tmp_path, stage='motor', **kwargs):
    ledger(tmp_path)
    return api().build_plan(stage, tmp_path/'batch', sys.executable, 600.,
                            root=ROOT, ledger=tmp_path/'ledger.json', **kwargs)


def ledger(tmp_path):
    path = tmp_path/'ledger.json'
    if not path.exists():
        path.write_text(json.dumps(dict(used_s=0., jobs=[])))
    return path


def test_motor_plan_contains_all_fifteen_exact_inputs_and_source_bindings(tmp_path):
    p = plan(tmp_path)
    assert len(p['jobs']) == 15
    assert p['status'] == 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
    assert p['physical_training_admitted'] is False
    assert p['manifest']['profile'] == 'cf2x-two-native-ball-root-aligned-v3'
    assert 'src/adaptive_drone_patrol/jobs.py' in p['source_binding']
    assert 'scripts/run_payload_retry_probes.py' in p['source_binding']
    assert all(len(value) == 64 for value in p['source_binding'].values())
    expected = [f'motor-{f}-{t}-v4' for f in ('original', 'unloaded', 'loaded')
                for t in ('hover', 'rotor0', 'rotor1', 'rotor2', 'rotor3')]
    assert [j['id'] for j in p['jobs']] == expected
    for j in p['jobs']:
        cmd = j['command']
        assert j['timeout_wall_s'] == 120.
        assert cmd[:3] == [sys.executable, '-u', 'scripts/probe_payload_ball.py']
        assert ('--align-root' in cmd) == ('-original-' not in j['id'])
        assert Path(cmd[cmd.index('--out')+1]).name == j['id']
        assert Path(j['job_output']).parent.name == 'jobs'
        assert '--rotor' in cmd or '-hover-' in j['id']


def test_domain_plan_preserves_fifty_five_declarations_and_stage_order(tmp_path):
    reference = tmp_path/'runtime-initial.json'
    reference.write_text('{"actual": "reference bytes"}\n')
    p = plan(tmp_path, 'domain', reference_audit=reference)
    declared = json.loads((ROOT/'configs/payload_retry_probe_manifest.json').read_text())
    assert len(p['jobs']) == len(declared['cases']) == 55
    by_id = {c['id']: c for c in declared['cases']}
    kinds = [j['case']['kind'] for j in p['jobs']]
    assert kinds == sorted(kinds, key={'anchor':0, 'large_angle':1, 'powered':2, 'refinement':3}.get)
    for j in p['jobs']:
        assert j['case'] == by_id[j['id']]
        assert j['timeout_wall_s'] == by_id[j['id']]['timeout_wall_s']
        cmd = j['command']
        assert cmd[cmd.index('--reference-audit')+1] == str(reference)
        assert cmd[cmd.index('--case')+1] == j['id']
        assert cmd[cmd.index('--fixture')+1] == ('anchor' if j['case']['kind'] in ('anchor', 'large_angle') else 'loaded')
    assert len(p['reference_audit']['sha256']) == 64
    assert {j['case']['substeps'] for j in p['jobs'] if j['case']['kind'] == 'refinement'} == {2, 4}
    assert {j['case']['initial_swing_deg'] for j in p['jobs'] if j['id'].startswith('sweep_')} == {5,45,80,89,91,120,170}


def test_reset_plan_is_one_thirty_reset_diagnostic(tmp_path):
    p = plan(tmp_path, 'reset')
    assert len(p['jobs']) == 1
    j = p['jobs'][0]
    assert j['id'] == 'bank-reset-v1' and j['timeout_wall_s'] == 300.
    assert j['command'][2] == 'scripts/probe_payload_ball_bank.py'
    assert j['command'][-2:] == ['--align-root', 'true']
    assert p['manifest']['resets']['count'] >= 30


@pytest.mark.parametrize('timeout', [0., -1., float('nan'), float('inf'), 1.])
def test_bad_or_insufficient_phase_timeout_creates_nothing(tmp_path, timeout):
    with pytest.raises(ValueError):
        api().build_plan('motor', tmp_path/'batch', sys.executable, timeout,
                         root=ROOT, ledger=tmp_path/'ledger.json')
    assert not (tmp_path/'batch').exists()


def test_output_and_missing_reference_are_rejected_before_creation(tmp_path):
    out = tmp_path/'batch'
    out.mkdir()
    with pytest.raises(FileExistsError):
        plan(tmp_path)
    assert list(out.iterdir()) == []
    with pytest.raises(ValueError):
        api().build_plan('domain', tmp_path/'domain', sys.executable, 600., root=ROOT)
    assert not (tmp_path/'domain').exists()


def test_failed_job_stops_and_preserves_plan_and_partial_results(tmp_path):
    m, calls = api(), []
    ledger(tmp_path)
    p = m.build_plan('motor', tmp_path/'batch', sys.executable, 600., root=ROOT,
                     ledger=tmp_path/'ledger.json')
    def supervisor(command, output, ledger, timeout_s, **kwargs):
        assert (tmp_path/'batch/batch-plan.json').is_file()
        calls.append((command, output, ledger, timeout_s, kwargs))
        return dict(status='SUCCEEDED' if len(calls) == 1 else 'FAILED', elapsed_s=2.)
    result = m.execute_plan(p, run_job_fn=supervisor, clock=lambda: 0.)
    assert len(calls) == 2
    assert result['status'] == 'STOPPED_JOB_FAILURE'
    assert result['completed_jobs'] == 1 and len(result['results']) == 2
    assert result['physical_training_admitted'] is False
    saved = json.loads((tmp_path/'batch/batch-results.json').read_text())
    assert saved == result and not saved['batch_complete']


def test_phase_budget_stops_without_launching_next_job(tmp_path):
    m, calls = api(), []
    ledger(tmp_path)
    p = m.build_plan('motor', tmp_path/'batch', sys.executable, 40., root=ROOT,
                     ledger=tmp_path/'ledger.json')
    ticks = iter([0., 0., 22., 22., 22.])
    def supervisor(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(status='SUCCEEDED', elapsed_s=22.)
    result = m.execute_plan(p, run_job_fn=supervisor, clock=lambda: next(ticks))
    assert len(calls) == 1
    assert calls[0][0][3] == 37.
    assert result['status'] == 'STOPPED_PHASE_BUDGET'
    assert not result['batch_complete'] and result['completed_jobs'] == 1


def test_supervisor_exception_and_keyboard_stop_preserve_partial_results(tmp_path):
    ledger(tmp_path)
    for error in (RuntimeError('supervisor refused'), KeyboardInterrupt()):
        root = tmp_path/type(error).__name__
        p = api().build_plan('reset', root, sys.executable, 600., root=ROOT,
                             ledger=tmp_path/'ledger.json')
        def supervisor(*args, **kwargs):
            raise error
        with pytest.raises(type(error)):
            api().execute_plan(p, run_job_fn=supervisor, clock=lambda: 0.)
        saved = json.loads((root/'batch-results.json').read_text())
        assert not saved['batch_complete'] and saved['completed_jobs'] == 0
        assert saved['status'] in ('STOPPED_EXCEPTION', 'INTERRUPTED')


def test_complete_process_batch_is_still_diagnostics_without_admission(tmp_path):
    m = api()
    p = plan(tmp_path, 'reset')
    result = m.execute_plan(p, run_job_fn=lambda *a, **k: dict(status='SUCCEEDED', elapsed_s=1.), clock=lambda: 0.)
    assert result['batch_complete'] and result['completed_jobs'] == 1
    assert result['status'] == 'DIAGNOSTIC_PROCESSES_COMPLETED_UNCHECKED'
    assert not result['physical_training_admitted']


def test_changed_source_and_exhausted_ledger_fail_before_gpu_launch(tmp_path):
    m = api()
    p = plan(tmp_path)
    p['source_binding']['src/adaptive_drone_patrol/jobs.py'] = '0'*64
    with pytest.raises(ValueError, match='source'):
        m.execute_plan(p, run_job_fn=lambda *a, **k: pytest.fail('Must not launch'))
    assert not (tmp_path/'batch').exists()
    ledger = tmp_path/'ledger.json'
    ledger.write_text(json.dumps(dict(used_s=28799., jobs=[])))
    with pytest.raises(ValueError, match='budget'):
        plan(tmp_path)


def test_phase_overrun_cannot_be_reported_as_complete(tmp_path):
    m = api()
    p = plan(tmp_path, 'reset')
    ticks = iter([0., 0., 601., 601.])
    result = m.execute_plan(p, run_job_fn=lambda *a, **k: dict(status='SUCCEEDED', elapsed_s=601.),
                            clock=lambda: next(ticks))
    assert not result['batch_complete']
    assert result['status'] == 'STOPPED_PHASE_OVERRUN'


def test_changed_source_between_jobs_stops_before_second_launch(tmp_path, monkeypatch):
    m, calls = api(), []
    p = plan(tmp_path)
    actual_digest = m.sha256
    changed = False
    def digest(path):
        if changed and str(path).endswith('scripts/probe_payload_ball.py'):
            return '0'*64
        return actual_digest(path)
    monkeypatch.setattr(m, 'sha256', digest)
    def supervisor(*args, **kwargs):
        nonlocal changed
        calls.append(args)
        changed = True
        return dict(status='SUCCEEDED', elapsed_s=1.)
    with pytest.raises(ValueError, match='source'):
        m.execute_plan(p, run_job_fn=supervisor, clock=lambda: 0.)
    assert len(calls) == 1
    saved = json.loads((tmp_path/'batch/batch-results.json').read_text())
    assert saved['status'] == 'STOPPED_EXCEPTION' and saved['completed_jobs'] == 1


def test_missing_or_active_cumulative_ledger_is_never_initialized(tmp_path):
    m = api()
    with pytest.raises(ValueError, match='ledger'):
        m.build_plan('motor', tmp_path/'batch', sys.executable, 600., root=ROOT,
                     ledger=tmp_path/'absent.json')
    assert not (tmp_path/'absent.json').exists()
    path = ledger(tmp_path)
    path.write_text(json.dumps(dict(used_s=0., jobs=[], active_job={'requires_reconciliation': True})))
    with pytest.raises(ValueError, match='reconciliation'):
        plan(tmp_path)
