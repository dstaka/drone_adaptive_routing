from adaptive_drone_patrol.mission import execute_mission
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.baselines import choose_baseline
import json
import numpy as np
import pytest
from adaptive_drone_patrol.patrol_runner import ArrivalGate, FlightSample
from adaptive_drone_patrol.target_adapter import validate_target


def test_continuous_dwell_resets_on_violation():
    gate = ArrivalGate(.01)
    reached = []
    for i in range(19):
        reached.extend(gate.update(FlightSample([0, 0, 1], [0, 0, 0]), [0, 0, 1]))
    assert reached == []
    assert gate.update(FlightSample([0, 0, 1], [0, 0, 0]), [0, 0, 1]) == ['ARRIVAL_CONFIRMED']
    gate.update(FlightSample([.3, 0, 1], [0, 0, 0]), [0, 0, 1])
    reached = []
    for i in range(50):
        reached.extend(gate.update(FlightSample([0, 0, 1], [0, 0, 0]), [0, 0, 1]))
    assert reached == ['ARRIVAL_CONFIRMED', 'WAYPOINT_REACHED']


@pytest.mark.parametrize('sample', [FlightSample([0, 0, 1], [0, 0, 0], done=True),
                                    FlightSample([0, 0, .05], [0, 0, 0]),
                                    FlightSample([np.nan, 0, 1], [0, 0, 0]),
                                    FlightSample([0, 0, 1], [0, 0, 0], reset=True)])
def test_reset_fall_nan_near_goal_never_becomes_visit(sample):
    gate = ArrivalGate(.01)
    with pytest.raises(RuntimeError):
        gate.update(sample, [0, 0, 1])
    assert gate.steps == 0


def test_target_shape_finiteness_and_envelope():
    for target in [[1, 2], [np.nan, 0, 1], [2, 0, 1]]:
        with pytest.raises(ValueError):
            validate_target(target, [[-1, 1], [-1, 1], [1, 1]])
    assert validate_target([.5, 0, 1], [[-1, 1], [-1, 1], [1, 1]]).tolist() == [.5, 0, 1]



class FakeFlight:
    dt = .01
    def __init__(self, fail=None):
        self.target = [0, 0, 1]
        self.fail = fail
    def set_target(self, target):
        self.target = target
    def advance(self):
        return {'xyz': self.target.copy(), 'velocity': [0, 0, 0], 'quaternion_wxyz': [1, 0, 0, 0],
                'effective_target': self.target if self.fail != 'mismatch' else [99, 0, 1],
                'actions': [0, 0, 0, 0], 'tilt_deg': 0, 'done': self.fail == 'reset',
                'reset': self.fail == 'reset'}


class FixedPlanner:
    def predict(self, state):
        return choose_baseline(state, 'fixed')


def mission_scenario():
    return Scenario('mock', 'validation', 10001, [[0, 0, 1], [.6, 0, 1], [0, .6, 1], [-.6, 0, 1]],
                    [1, 1, 1], np.ones((4, 4)).tolist(), [.5] * 3,
                    event={'kind': 'priority', 'after': 1, 'checkpoint': 2, 'value': 5})


def test_mocked_physical_event_order_and_failure_inclusive_result(tmp_path):
    out = tmp_path / 'success'
    result = execute_mission(FakeFlight(), FixedPlanner(), mission_scenario(), out, {}, record=False)
    assert result['status'] == 'SUCCEEDED' and result['completed_order'] == ['A', 'B', 'C']
    events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    kinds = [e['type'] for e in events]
    assert kinds.index('WAYPOINT_REACHED') < kinds.index('PRIORITY_UPDATED') < kinds.index('NO_POLICY_CHANGE')
    assert [e['event_seq'] for e in events] == list(range(1, len(events) + 1))
    for failure in ['reset', 'mismatch']:
        r = execute_mission(FakeFlight(failure), FixedPlanner(), mission_scenario(), tmp_path / failure,
                            {}, record=False)
        assert r['status'] == 'FAILED' and r['completed_order'] == []
        assert r['analytical_cost'] == 0
