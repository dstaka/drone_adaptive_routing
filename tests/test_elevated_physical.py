import numpy as np
import pytest
from adaptive_drone_patrol.elevated_physical import vertical_cases,clearance_envelope,vertical_report
from adaptive_drone_patrol.util import load_config


def test_vertical_suite_has_120_actual_distinct_transfers_from_s():
    cases=vertical_cases()
    assert len(cases)==12
    assert {c['execution_seed'] for c in cases}=={11,12,13,14}
    goals=load_config('configs/demo_scenarios_payload_elevated/normal.yaml')['coords']
    by_id=dict(zip('SABCDEFGHIJ',goals))
    for case in cases:
        assert case['targets']==[by_id[k] for k in 'AGIDBHFJEC']
        assert case['initial']['drone_xyz'][2]==1.
        seq=[[0.,0.,1.],*case['targets']]
        assert all(a!=b for a,b in zip(seq,seq[1:]))
    e=clearance_envelope()
    assert e['minimum_package_floor_bound_m']> .1
    assert e['minimum_drone_floor_bound_m']> .1
    assert not e['measured_flight_safety_claim']


def test_vertical_display_requires_between_distinct_goals():
    goals=load_config('configs/demo_scenarios_payload_elevated/normal.yaml')['coords']
    by_id=dict(zip('SABCDEFGHIJ',goals))
    stops=[dict(id=k,target_z_m=by_id[k][2],actual_z_m=by_id[k][2],step=i*100) for i,k in enumerate('AGIDBHFJEC',1)]
    result=vertical_report(stops)
    assert result['passed']
    assert result['ascents'] and result['descents']
    assert result['legs'][0]['from']=='S'
    # Initial climb can never supply the ascent gate.
    flat=[dict(id=k,target_z_m=z,actual_z_m=z,step=i*100) for i,(k,z) in enumerate(zip('ABCDEFGHIJ',np.arange(5.,0.,-.5)),1)]
    assert not vertical_report(flat)['passed']


def test_oriented_package_corners_breach_floor_while_center_is_above_limit():
    from adaptive_drone_patrol.payload_cases import initial_state
    from adaptive_drone_patrol.payload_hybrid_audit import independent_metrics
    s=initial_state(np.random.default_rng(1),.06)
    s['package_xyz']=[0.,0.,.12]
    s['package_quaternion_wxyz']=[float(np.cos(np.pi/8)),float(np.sin(np.pi/8)),0.,0.]
    m=independent_metrics(s,[0,0,1],.06)
    assert s['package_xyz'][2]>.10
    assert m['package_floor_m']==pytest.approx(.07757359312880714)
    assert m['package_floor_m']<.10
