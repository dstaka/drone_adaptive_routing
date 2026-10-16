"""Physical explanations must bind to logged decision snapshots and actual actions."""
import copy
import pytest
from adaptive_drone_patrol.analytical import snapshot, content_hash
from adaptive_drone_patrol.objective import RoutingState
from adaptive_drone_patrol.baselines import choose_baseline
from test_visible_flight import layout


def test_physical_provenance_and_measured_distances():
    from adaptive_drone_patrol.physical_evidence import physical_explanation, verify_physical_explanation
    state = RoutingState(layout(3))
    def chooser(st):
        return choose_baseline(st, 'fixed')
    decision = {'run_id':'run','decision_id':'decision-0','boundary_snapshot':snapshot(state),
                'boundary_hash':content_hash(snapshot(state)), 'physical_pose':[.1,0,1],
                'simulation_step':0,'event_seq':2,'selected_action':0,'policy_hash':'hash',
                'evidence_kind':'physical'}
    value = physical_explanation(decision, chooser)
    assert value['evidence_kind'] == 'physical'
    assert value['physical_reference']['simulation_step'] == 0
    assert value['candidate_priorities_distances'][0]['pose_distance_m'] != value['candidate_priorities_distances'][0]['distance_m']
    assert verify_physical_explanation(value, decision)
    bad = copy.deepcopy(decision)
    bad['selected_action'] = 1
    with pytest.raises(ValueError):
        verify_physical_explanation(value, bad)
    bad['evidence_kind'] = 'analytical'
    with pytest.raises(ValueError):
        physical_explanation(bad, chooser)


def test_event_action_must_match_actual_following_decision():
    from adaptive_drone_patrol.physical_evidence import verify_event_response
    event={'type':'PRIORITY_UPDATED','event_seq':4,'actual_action':2,'counterfactual_action':0,
           'update':{'checkpoint_id':'C','boundary':1,'skipped':False,'value':5},
           'comparison':{'improvement':2.,'no_event_order':[0,2,1],'post_event_order':[2,1,0],
                         'retained_order_post_event':{'cost':12.,'travel_cost':3.},
                         'replanned_order_post_event':{'cost':10.,'travel_cost':3.}}}
    decision={'event_seq':5,'selected_action':2}
    assert verify_event_response([event],[decision])
    decision['selected_action']=1
    with pytest.raises(ValueError):
        verify_event_response([event],[decision])


def test_physical_nearer_gate_uses_measured_pose_not_node_center():
    from adaptive_drone_patrol.physical_evidence import physical_explanation
    from adaptive_drone_patrol.scenario import Scenario
    from adaptive_drone_patrol.baselines import exact_remaining
    import numpy as np
    s=Scenario('probe','validation',10001,[[0,0,1],[.5,0,1],[2,0,1],[.6,.6,1]],
               [1,5,1],np.ones((4,4)).tolist(),[.5]*3)
    st=RoutingState(s)
    def oracle(state):
        return exact_remaining(state)['order'][0]
    decision={'run_id':'run','decision_id':'decision-0','boundary_snapshot':snapshot(st),
              'boundary_hash':content_hash(snapshot(st)), 'physical_pose':[2,0,1],
              'simulation_step':0,'event_seq':2,'selected_action':oracle(st),'policy_hash':'hash',
              'evidence_kind':'physical'}
    result=physical_explanation(decision,oracle)
    assert result['planning_nearer_lower_priority_gate']
    assert not result['nearer_lower_priority_gate']


def test_frozen_suite_rejects_geometry_camera_and_policy_changes():
    from adaptive_drone_patrol.physical_evidence import verify_frozen_inputs
    from adaptive_drone_patrol.analytical import demo_fixture
    from adaptive_drone_patrol.scene import scene_manifest
    s=demo_fixture(40004)
    summary={k:'hash' for k in ['flight_checkpoint_hash','flight_config_hash','routing_model_hash',
                               'flight_acceptance_config_hash','execution_source_hash','hover_source_hash']}
    item={'summary':summary,'scenario':s.to_dict(),'scene':scene_manifest(s),'initial_pose':[0,0,1]}
    assert verify_frozen_inputs([item,copy.deepcopy(item)],'hash')
    for section,key,value in [('scenario','coords',[[0,0,1]]*11),('scene','camera',{}),
                              ('summary','routing_model_hash','wrong')]:
        bad=copy.deepcopy(item)
        bad[section][key]=value
        with pytest.raises(ValueError):
            verify_frozen_inputs([item,bad],'hash')


def test_encoded_frames_must_match_recorded_count_and_pts():
    from adaptive_drone_patrol.physical_evidence import verify_media_index
    frames=[{'frame_index':str(i),'pts_s':str(i/50)} for i in range(3)]
    probe={'streams':[{'width':1280,'height':720}],
           'frames':[{'best_effort_timestamp_time':str(i/50)} for i in range(3)]}
    assert verify_media_index(frames,probe)
    bad=copy.deepcopy(probe)
    bad['frames'].pop()
    with pytest.raises(ValueError):
        verify_media_index(frames,bad)
    bad=copy.deepcopy(probe)
    bad['frames'][1]['best_effort_timestamp_time']='.03'
    with pytest.raises(ValueError):
        verify_media_index(frames,bad)


def test_physical_boundary_preserves_logged_integer_event_encoding():
    from adaptive_drone_patrol.physical_evidence import physical_explanation,verify_physical_explanation
    s=layout(3)
    s.event={'kind':'cost','after':1,'edges':[['P0','P1']],'value':5}
    st=RoutingState(s)
    st.complete(0,st.departure(0))
    st.apply_event()
    frozen=snapshot(st)
    assert type(frozen['scenario']['multipliers'][1][2]) is int
    def chooser(state):
        return choose_baseline(state,'fixed')
    decision={'run_id':'run','decision_id':'decision-1','boundary_snapshot':frozen,
              'boundary_hash':content_hash(frozen),'physical_pose':s.coords[1],
              'simulation_step':50,'event_seq':6,'selected_action':1,'policy_hash':'hash',
              'evidence_kind':'physical'}
    value=physical_explanation(decision,chooser)
    assert value['boundary_hash']==decision['boundary_hash']
    assert type(value['snapshot']['scenario']['multipliers'][1][2]) is int
    assert verify_physical_explanation(value,decision)


def test_critical_frame_selection_includes_first_post_event_frame():
    from adaptive_drone_patrol.physical_evidence import evidence_frame_indices
    events=[{'type':'TARGET_SELECTED','event_seq':2},{'type':'PRIORITY_UPDATED','event_seq':6},
            {'type':'TARGET_SELECTED','event_seq':7}]
    frames=[{'event_seq':str(s)} for s in [1,3,5,8,10]]
    assert evidence_frame_indices(events,frames)==[0,1,3,4]
    with pytest.raises(ValueError):
        evidence_frame_indices(events,frames[:3])


def test_recorded_pts_must_be_finite():
    from adaptive_drone_patrol.physical_evidence import verify_media_index
    frames=[{'frame_index':'0','pts_s':'nan'}]
    probe={'streams':[{'width':1280,'height':720}],'frames':[{'best_effort_timestamp_time':'0'}]}
    with pytest.raises(ValueError):
        verify_media_index(frames,probe)
