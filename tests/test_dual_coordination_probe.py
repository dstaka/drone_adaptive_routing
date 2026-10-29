"""An unqualified probe cannot authorize production or silently replace a run."""

import pytest
import json


def test_probe_requires_new_namespace_and_declared_duration_before_physics(tmp_path):
    from adaptive_drone_patrol.dual_coordination_probe import capture_probe

    with pytest.raises(ValueError, match="duration"):
        capture_probe(original=tmp_path / "missing", design={}, output=tmp_path / "new", duration_s=-1)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        capture_probe(original=tmp_path / "missing", design={}, output=existing, duration_s=1)


@pytest.mark.parametrize('phase', ['configuration', 'compile', 'runtime', 'controller', 'initial_audit',
                                  'step', 'result', 'interrupt', 'success'])
def test_post_start_failures_are_closed_and_keep_original_exception(tmp_path, monkeypatch, phase):
    """Native doubles isolate lifecycle behavior; this is not physics evidence."""
    import numpy as np
    import adaptive_drone_patrol.dual_coordination_probe as probe

    failure = KeyboardInterrupt('probe interrupted') if phase == 'interrupt' else RuntimeError(f'{phase} failed')

    def fail():
        raise failure

    class Fixture:
        stage_rows = []
        substep_rows = []

        def __init__(self, *args):
            self.time = 0.
            if phase in ('compile', 'interrupt'):
                fail()

        def runtime(self):
            if phase == 'runtime':
                fail()
            return {'qualification': 'UNQUALIFIED'}

        def sample(self):
            return {'schema': 'dual-raw-v1', 'world_time': self.time, 'drones': [
                {'drone_xyz': p, 'drone_velocity': [0., 0., 0.]} for p in ([-3., -3., 1.], [3., 3., 1.])]}

        def geometry_state(self):
            return {'world_time': self.time}

        def step_rpm(self, rpm, *, observe_stages):
            if phase == 'step':
                fail()
            self.time += .01

    class Controller:
        last_audit = {}

        def __init__(self, *args):
            if phase == 'controller':
                fail()

        def action(self, *args, **kwargs):
            return np.zeros(4)

    def audit(fixture):
        if phase == 'initial_audit':
            fail()
        return dict(world_time=fixture.time, finite=True, min_clearance_m=5., floor_minima_m=[.7,.7],
                    contacts=[], wash_incursions=[], constraint_max_m=0., max_roll_pitch_deg=0.)

    save = probe.atomic_json

    def save_with_fault(path, data):
        if path.name == ('probe.json' if phase == 'configuration' else 'result.json' if phase == 'result' else ''):
            fail()
        save(path, data)

    monkeypatch.setattr(probe, 'DualMuJoCoFixture', Fixture)
    monkeypatch.setattr(probe, 'Controller', Controller)
    monkeypatch.setattr(probe, 'audit_geometry', audit)
    monkeypatch.setattr(probe, 'atomic_json', save_with_fault)
    output = tmp_path/'probe'
    if phase == 'success':
        result = probe.capture_probe(original='unused', design={'gain': [], 'hover_action': []},
                                     output=output, duration_s=.01)
        assert result['failure'] is None and result['completed_control_steps'] == 1
        assert json.loads((output/'progress.json').read_text())['jobs'][0]['status'] == 'done'
        assert (output/'native-control.jsonl.gz').exists()
        return
    with pytest.raises(type(failure)) as caught:
        probe.capture_probe(original='unused', design={'gain': [], 'hover_action': []},
                            output=output, duration_s=.01)
    assert caught.value is failure
    events = [json.loads(line) for line in (output/'timing/events.jsonl').read_text().splitlines()]
    assert sum(event['event'] == 'end' for event in events) == 1
    job = json.loads((output/'progress.json').read_text())['jobs'][0]
    assert job['status'] == 'failed'
    assert job['ended_at'] is not None
    assert str(failure) in job['reason']
    if phase in ('step', 'result'):
        assert (output/'native-control.jsonl.gz').exists()  # Failed capture evidence is retained.
