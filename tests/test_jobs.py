import json
import sys

import pytest

from adaptive_drone_patrol.jobs import run_job


def test_budget_and_collision_prevent_execution(tmp_path):
    ledger = tmp_path / 'ledger.json'
    ledger.write_text(json.dumps({'used_s': 28800, 'jobs': []}))
    with pytest.raises(RuntimeError, match='budget'):
        run_job([sys.executable, '-c', 'raise RuntimeError()'], tmp_path / 'out', ledger, 1)
    ledger.write_text(json.dumps({'used_s': 0, 'jobs': []}))
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(FileExistsError):
        run_job([sys.executable, '-c', 'pass'], out, ledger, 1)


def test_exit_and_timeout_are_persisted(tmp_path):
    ledger = tmp_path / 'ledger.json'
    result = run_job([sys.executable, '-c', 'print("evidence")'], tmp_path / 'ok', ledger, 2)
    assert result['status'] == 'SUCCEEDED'
    assert result['exit_code'] == 0
    assert (tmp_path / 'ok' / 'job.log').read_text().strip() == 'evidence'
    result = run_job([sys.executable, '-c', 'import time; time.sleep(5)'], tmp_path / 'slow', ledger, .15)
    assert result['status'] == 'TIMED_OUT'
    assert json.loads(ledger.read_text())['used_s'] >= .15
    assert len(json.loads(ledger.read_text())['jobs']) == 2
def test_cpu_budget_is_separate_and_cumulative(tmp_path):
    import json
    import sys
    import pytest
    from adaptive_drone_patrol.jobs import run_job
    ledger = tmp_path / 'cpu.json'
    ledger.write_text(json.dumps({'used_s': 14400., 'jobs': []}))
    with pytest.raises(RuntimeError, match='budget exhausted'):
        run_job([sys.executable, '-c', 'pass'], tmp_path / 'never', ledger, 10, budget_s=14400)
    assert not (tmp_path / 'never').exists()
