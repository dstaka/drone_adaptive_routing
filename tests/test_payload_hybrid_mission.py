import json

from test_payload_recovery_mission import CpuFlight, Planner, scenario
from adaptive_drone_patrol.payload_recovery_mission import execute_payload_mission


def test_revision2_mission_actually_continues_beyond_120_simulated_seconds(tmp_path):
    def wait(flight):
        if flight.target_steps<1250 and sum(x*x for x in flight.current['drone_velocity'])<.01:
            flight.current['package_velocity']=[.2,0,0]
            flight.current['attachment_b_velocity']=[.2,0,0]
    flight=CpuFlight(override=wait)
    out=tmp_path/'mission'
    result=execute_payload_mission(flight,Planner(),scenario(),out,
        dict(evidence_kind='cpu-fixture',routing_model_hash='a'*64),revision2=True,clock=lambda:0.)
    assert result['status']=='SUCCEEDED'
    assert result['sim_time_s']>120
    assert result['completed_count']==10
    metadata=json.loads((out/'metadata.json').read_text())
    assert metadata['settling_profile']=='payload-settling-video-v1'
    assert metadata['aggregate_simulated_deadline_s'] is None
    from adaptive_drone_patrol.payload_solver_acceptance import verify_mission
    assert verify_mission(out,Planner(),revision2=True)['success']
