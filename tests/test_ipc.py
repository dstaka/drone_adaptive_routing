import sys
import pytest
from adaptive_drone_patrol.planner_ipc import PlannerClient
from adaptive_drone_patrol.objective import SCHEMA


@pytest.mark.parametrize('program, match', [
    ('import sys; sys.exit(3)', 'died'),
    ('import time; time.sleep(3)', 'timeout'),
    ('import sys,json; q=json.loads(sys.stdin.readline()); print(json.dumps({"request_id":99,"status":"OK","action":0,"policy_hash":"abc"}),flush=True)', 'request'),
    ('import sys,json; q=json.loads(sys.stdin.readline()); print(json.dumps({"request_id":q["request_id"],"status":"OK","action":2,"policy_hash":"abc"}),flush=True)', 'invalid')])
def test_bad_planner_replies_fail_instead_of_inventing_action(program, match):
    with PlannerClient([sys.executable, '-u', '-c', program], 'abc', timeout=.2) as client:
        with pytest.raises(RuntimeError, match=match):
            client.request([0.] * SCHEMA['size'], [True, False, False])


def test_boolean_response_id_is_not_a_request_id():
    program = 'import sys,json; q=json.loads(sys.stdin.readline()); print(json.dumps({"request_id":True,"status":"OK","action":0,"policy_hash":"abc"}),flush=True)'
    with PlannerClient([sys.executable, '-u', '-c', program], 'abc', timeout=1) as client:
        with pytest.raises(RuntimeError, match='request'):
            client.request([0.] * SCHEMA['size'], [True, False, False])


def test_invalid_dimensions_rejected_before_planner_request():
    with PlannerClient([sys.executable, '-c', 'import time; time.sleep(10)'], 'abc') as client:
        with pytest.raises(ValueError, match='dimensions'):
            client.request([0.] * 324, [True] * 10)
