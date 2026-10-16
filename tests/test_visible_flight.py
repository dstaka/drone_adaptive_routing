"""CPU contracts only; these tests do not establish physical flight success."""
import json
import numpy as np
import pytest
from adaptive_drone_patrol.mission import execute_mission
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.evidence import validate_run
from test_patrol import FakeFlight


class OnlinePlanner:
    def __init__(self):
        self.queries = 0

    def predict(self, state):
        assert not state.terminal
        self.queries += 1
        return int(np.flatnonzero(state.mask())[0])


def layout(n):
    return Scenario('mock', 'validation', 10001,
                    [[0, 0, 1]] + [[i * .6, .6, 1] for i in range(n)],
                    [2.] * n, np.ones((n+1, n+1)).tolist(), [.5] * n,
                    schema_version=2, checkpoint_ids=[f'P{i}' for i in range(n)])


@pytest.mark.parametrize('n', [10, 12])
def test_configured_ids_unique_completion_and_physical_snapshots(tmp_path, n):
    planner = OnlinePlanner()
    result = execute_mission(FakeFlight(), planner, layout(n), tmp_path/'run', {}, record=False)
    assert result['status'] == 'SUCCEEDED', result['termination_cause']
    assert result['checkpoint_ids'] == layout(n).checkpoint_ids
    assert result['completed_count'] == n and planner.queries == n
    validate_run(tmp_path/'run')
    decisions = [json.loads(x) for x in (tmp_path/'run/decisions.jsonl').read_text().splitlines()]
    assert len(decisions) == n
    assert all('boundary_snapshot' in d and 'physical_pose' in d for d in decisions)
    assert result['legs'][0]['dwell_max_speed_mps'] == 0


@pytest.mark.parametrize('action', [True, 1.5, -1, 12, 'A'])
def test_malformed_actions_never_start_leg(tmp_path, action):
    class Invalid:
        def predict(self, state):
            return action
    result = execute_mission(FakeFlight(), Invalid(), layout(10), tmp_path/'run', {}, record=False)
    assert result['status'] == 'FAILED' and result['completed_count'] == 0
    assert result['intended_order'] == []


def test_frames_cover_initial_and_final_state_with_event_sequence(tmp_path, monkeypatch):
    captures = []
    class Recorder:
        def __init__(self, *args):
            pass
        def capture(self, step, force=False, event_seq=None):
            captures.append((step, event_seq))
            return len(captures)-1
        def close(self):
            pass
    monkeypatch.setattr('adaptive_drone_patrol.mission.FrameRecorder', Recorder)
    f = FakeFlight()
    f.cam = None
    result = execute_mission(f, OnlinePlanner(), layout(10), tmp_path/'run', {}, record=True)
    assert result['status'] == 'SUCCEEDED'
    events = [json.loads(x) for x in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    assert captures[0] == (0, 1)
    assert captures[-1][1] == events[-1]['event_seq']


def test_evidence_recomputes_dwell_from_pose_and_velocity(tmp_path):
    import csv
    run = tmp_path/'run'
    execute_mission(FakeFlight(), OnlinePlanner(), layout(10), run, {}, record=False)
    rows = list(csv.DictReader((run/'telemetry.csv').open()))
    rows[0]['vx'] = '1.0'
    with (run/'telemetry.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='telemetry|speed|pose'):
        validate_run(run)


def test_intro_has_real_samples_before_first_policy_departure(tmp_path):
    result = execute_mission(FakeFlight(), OnlinePlanner(), layout(10), tmp_path/'run', {},
                             record=False, limits={'intro_s': .1})
    assert result['status'] == 'SUCCEEDED'
    events = [json.loads(x) for x in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    first = next(e for e in events if e['type']=='TARGET_SELECTED')
    assert first['simulation_step'] == 10
    import csv
    rows = list(csv.DictReader((tmp_path/'run/telemetry.csv').open()))
    assert all(r['runner_state']=='INTRO' and r['target_id']=='S' for r in rows[:10])
    assert all(e['type']!='WAYPOINT_REACHED' for e in events if e['simulation_step']<=10)


def test_intro_failure_retains_failing_physical_sample(tmp_path):
    import csv
    result = execute_mission(FakeFlight('reset'), OnlinePlanner(), layout(10), tmp_path/'run', {},
                             record=False, limits={'intro_s': .1})
    assert result['status']=='FAILED' and result['completed_count']==0
    rows=list(csv.DictReader((tmp_path/'run/telemetry.csv').open()))
    assert len(rows)==1 and rows[0]['reset']=='True'


def test_capture_failure_still_closes_encoder(tmp_path, monkeypatch):
    from unittest.mock import Mock
    recorder=Mock()
    recorder.capture.side_effect=RuntimeError('camera failed')
    monkeypatch.setattr('adaptive_drone_patrol.mission.FrameRecorder',Mock(return_value=recorder))
    f=FakeFlight()
    f.cam=None
    result=execute_mission(f,OnlinePlanner(),layout(10),tmp_path/'run',{},record=True)
    assert result['status']=='FAILED'
    recorder.close.assert_called_once()


def test_initial_pose_tilt_does_not_depend_on_step_buffers():
    from adaptive_drone_patrol.patrol_runner import quaternion_tilt_degrees
    assert quaternion_tilt_degrees([1,0,0,0]) == 0
    angle=np.radians(65)/2
    assert quaternion_tilt_degrees([np.cos(angle),np.sin(angle),0,0]) == pytest.approx(65)
    assert quaternion_tilt_degrees([np.cos(angle),0,0,np.sin(angle)]) == 0


def test_duplicate_visited_action_fails_without_second_completion(tmp_path):
    class Duplicate:
        def predict(self, state):
            return 0
    result=execute_mission(FakeFlight(),Duplicate(),layout(10),tmp_path/'run',{},record=False)
    assert result['status']=='FAILED'
    assert result['completed_order']==['P0']
    assert result['intended_order']==['P0']
