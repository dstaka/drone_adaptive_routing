"""Read-only substep hooks must preserve exactly-once physical integration."""
import importlib
import io
import json
from types import SimpleNamespace
import pytest


def test_hook_records_each_actual_substep_and_restores_method():
    api = importlib.import_module('adaptive_drone_patrol.payload_substep_probe')
    solver = SimpleNamespace(substep_dt=.005, calls=[])
    def original(f):
        solver.calls.append(f)
    solver.substep = original
    entity = SimpleNamespace(n_qs=15, n_dofs=12)
    stream = io.StringIO()
    def capture(post=False):
        return {'calls': len(solver.calls), 'post': post}
    with api.SubstepRecorder(solver, entity, stream, capture=capture) as recorder:
        recorder.begin_control(3)
        solver.substep(8)
        solver.substep(9)
    assert solver.substep is original
    assert solver.calls == [8, 9]
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [(r['control_step'], r['substep'], r['time_s']) for r in rows] == [(3, 0, .03), (3, 1, .035)]
    assert rows[0]['before']['calls'] == 0 and rows[0]['after']['calls'] == 1
    assert rows[1]['before']['calls'] == 1 and rows[1]['after']['calls'] == 2


def test_hook_preserves_failure_and_raw_after_state():
    api = importlib.import_module('adaptive_drone_patrol.payload_substep_probe')
    def original(f):
        raise RuntimeError('integration failed')
    solver = SimpleNamespace(substep_dt=.005, substep=original)
    stream = io.StringIO()
    with pytest.raises(RuntimeError, match='integration failed'):
        with api.SubstepRecorder(solver, SimpleNamespace(n_qs=15, n_dofs=12), stream,
                                 capture=lambda post=False: {'post': post}) as recorder:
            recorder.begin_control(0)
            solver.substep(0)
    assert solver.substep is original
    row = json.loads(stream.getvalue())
    assert row['error'] == 'RuntimeError' and row['after']['post']


def test_hook_refuses_standalone_midpoint_body():
    api = importlib.import_module('adaptive_drone_patrol.payload_substep_probe')
    with pytest.raises(ValueError):
        api.SubstepRecorder(SimpleNamespace(substep_dt=.005), SimpleNamespace(n_qs=7, n_dofs=6), io.StringIO())
