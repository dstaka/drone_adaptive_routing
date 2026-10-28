"""Causal adapters and actual native development integration (when available)."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from adaptive_drone_patrol.dual_coordination_env import DualCoordinationEnv, build_observation
from adaptive_drone_patrol.dual_coordination_mission import MissionState
from adaptive_drone_patrol.dual_coordination_policy import DRONE_FEATURES, GLOBAL_FEATURES
from adaptive_drone_patrol.dual_coordination_reference import ContinuousReference
from adaptive_drone_patrol.dual_coordination_schema import WAIT, encode_action


def measured_inputs():
    rows = []
    for d,p in enumerate(([-3.5,-3.5,1.],[3.5,3.5,1.])):
        rows.append(dict(drone_id=d+1,drone_xyz=p,drone_velocity=[0.,0.,0.],
                         drone_quaternion_wxyz=[1.,0.,0.,0.],drone_angular_velocity=[0.,0.,0.],
                         rod_quaternion_wxyz=[1.,0.,0.,0.],rod_angular_velocity=[0.,0.,0.],
                         package_xyz=[p[0],p[1],.6575],package_velocity=[0.,0.,0.],
                         package_quaternion_wxyz=[1.,0.,0.,0.],package_angular_velocity=[0.,0.,0.]))
    sample = dict(world_time=0.,drones=rows)
    goals = [[float(i%5)-2.,float(i//5)-.5,3.] for i in range(10)]
    mission = MissionState(priorities=[50]*10)
    refs = [ContinuousReference(row['drone_xyz']) for row in rows]
    prediction = dict(clearance_m=8.,conflict=False)
    return sample,mission,goals,refs,prediction


def test_observation_matches_named_shapes_and_frozen_physical_scales():
    sample,mission,goals,refs,prediction = measured_inputs()
    observation = build_observation(sample,mission,goals,refs,prediction)
    assert set(observation) == {'goals','drones','global_features'}
    assert observation['goals'].shape == (10,11)
    assert observation['drones'].shape == (2,66)
    assert observation['global_features'].shape == (11,)
    assert all(x.dtype == np.float32 and np.isfinite(x).all() for x in observation.values())
    np.testing.assert_allclose(observation['goals'][0],[-.4,-.1,.6,.5,1,0,0,0,1,0,0])
    np.testing.assert_allclose(observation['drones'][0,:3],[-.7,-.7,.2])
    assert observation['drones'][0,DRONE_FEATURES.index('target_10')] == 1
    assert observation['drones'][1,DRONE_FEATURES.index('last_target_10')] == 1
    np.testing.assert_allclose(observation['global_features'],[0,1,0,.7,.7,0,0,0,0,.8,0])


def test_observation_does_not_read_latent_load_size_event_or_case_metadata():
    sample,mission,goals,refs,prediction = measured_inputs()
    original = build_observation(sample,mission,goals,refs,prediction)
    sample.update(mass_ratio=9e9,event={'kind':'priority','tick':800,'goal':2},case_id='heldout-secret')
    for row in sample['drones']:
        row.update(side_m=10.,mass=123.,hidden_parameters={'payload_size':.06})
    changed = build_observation(sample,mission,goals,refs,prediction)
    for key in original:
        np.testing.assert_array_equal(original[key],changed[key])


def test_link_angle_and_rate_are_measured_relative_to_drone_not_hidden_geometry():
    sample,mission,goals,refs,prediction = measured_inputs()
    sample['drones'][0]['drone_quaternion_wxyz'] = [math.cos(.01),math.sin(.01),0.,0.]
    sample['drones'][0]['rod_quaternion_wxyz'] = [math.cos(.05),math.sin(.05),0.,0.]
    sample['drones'][0]['drone_angular_velocity'] = [1.,0.,0.]
    sample['drones'][0]['rod_angular_velocity'] = [2.,0.,0.]
    row = build_observation(sample,mission,goals,refs,prediction)['drones'][0]
    assert row[DRONE_FEATURES.index('link_angle_x')] == pytest.approx(.08/math.pi)
    assert row[DRONE_FEATURES.index('link_rate_x')] == pytest.approx(1/math.pi)


def test_target_owner_history_and_clock_features_follow_mission_state():
    sample,mission,goals,refs,prediction = measured_inputs()
    mission.apply(encode_action(2,3,2),0)
    for tick in range(1,21):
        mission.advance(tick)
    sample['world_time'] = .2
    observation = build_observation(sample,mission,goals,refs,prediction)
    assert observation['goals'][2,5] == 1
    assert observation['goals'][2,9] == 1
    assert observation['goals'][3,10] == 1
    assert observation['drones'][0,DRONE_FEATURES.index('target_2')] == 1
    assert observation['drones'][1,DRONE_FEATURES.index('last_target_3')] == 1
    assert observation['drones'][0,DRONE_FEATURES.index('commitment_age')] == pytest.approx(.2)
    assert observation['drones'][0,DRONE_FEATURES.index('leg_age')] == pytest.approx(.2/30)
    assert observation['global_features'][GLOBAL_FEATURES.index('team_time')] == pytest.approx(.2/300)
    assert observation['global_features'][2] == 1


def test_adapter_rejects_nonfinite_sensor_instead_of_leaking_nan_to_policy():
    sample,mission,goals,refs,prediction = measured_inputs()
    sample['drones'][0]['drone_velocity'][0] = float('nan')
    with pytest.raises(ValueError):
        build_observation(sample,mission,goals,refs,prediction)


def test_policy_observation_cannot_request_extra_mid_interval_decision():
    sample,mission,goals,refs,prediction = measured_inputs()
    mission.advance(1)
    sample['world_time'] = .01
    with pytest.raises(ValueError):
        build_observation(sample,mission,goals,refs,prediction)


def test_production_or_non_development_case_refused_before_creating_files(tmp_path):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    case = generate_case(15,0,'normal')
    with pytest.raises(ValueError):
        DualCoordinationEnv('missing',tmp_path/'production',case,{},mode='production')
    case['split'] = 10
    with pytest.raises(ValueError):
        DualCoordinationEnv('missing',tmp_path/'training',case,{})
    assert not (tmp_path/'production').exists()
    assert not (tmp_path/'training').exists()


def native_env(tmp_path, name='environment', *, record_raw=False, case=None, replay_stride_ticks=1, collect_timings=False):
    mj = pytest.importorskip('mujoco',reason='Actual native MuJoCo3.15 required; no fake physics')
    if mj.__version__ != '3.15.0':
        pytest.skip('Requires pinned native MuJoCo3.15.0')
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    original = Path(os.environ.get('DUAL_CF2X','outputs/dual_payload_coordination_20261027T144754Z/inputs/cf2x/cf2x.urdf'))
    design_path = Path(os.environ.get('DUAL_CONTROLLER','outputs/dual_payload_coordination_20261027T144754Z/inputs/controller.json'))
    if not original.exists() or not design_path.exists():
        pytest.skip('Native source asset and frozen controller are required')
    if case is None:
        case = generate_case(15,0,'normal')
        case['starts'] = [[-3.5,-3.5,1.],[3.5,3.5,1.]]
        case['initials'] = [dict(position=p,quaternion_wxyz=[1.,0.,0.,0.],swing_deg=0.,direction_rad=0.)
                            for p in case['starts']]
    return DualCoordinationEnv(original,tmp_path/name,case,json.loads(design_path.read_text()),record_raw=record_raw,replay_stride_ticks=replay_stride_ticks,collect_timings=collect_timings)


def test_native_step_advances_one_world_twenty_ticks_and_distinguishes_wait(tmp_path):
    env = native_env(tmp_path)
    before = env.fixture.sample()
    obs,reward,terminal,info = env.step(encode_action(WAIT,WAIT,1))
    assert not terminal,info
    assert env.mission.tick == 20
    assert env.fixture.data.time == pytest.approx(.2)
    assert info['elapsed_simulated_s'] == pytest.approx(.2)
    assert info['supervisor_held_drone_seconds'] == 0
    assert info['reward_components']['time'] == pytest.approx(-.2)
    assert reward == pytest.approx(sum(info['reward_components'].values()))
    assert env.mission.leg_start_ticks == [None,None]
    assert np.linalg.norm(np.subtract(env.fixture.sample()['drones'][0]['drone_xyz'],before['drones'][0]['drone_xyz'])) < .1
    assert obs['drones'].shape == (2,66)
    env.close()


def test_native_assignment_moves_continuously_without_teleport(tmp_path):
    env = native_env(tmp_path)
    before = np.array([r['drone_xyz'] for r in env.fixture.sample()['drones']])
    _,_,terminal,info = env.step(encode_action(0,1,1))
    assert not terminal,info
    after = np.array([r['drone_xyz'] for r in env.fixture.sample()['drones']])
    assert np.max(np.linalg.norm(after-before,axis=1)) < .2
    assert env.mission.targets == (0,1)
    assert env.mission.leg_start_ticks == [0,0]
    env.close()


def test_native_checkpoint_resume_reproduces_next_world_and_controller_update(tmp_path):
    env = native_env(tmp_path,'original')
    env.step(encode_action(WAIT,WAIT,1))
    checkpoint = json.loads(json.dumps(env.to_dict(),allow_nan=False))
    resumed = native_env(tmp_path,'resumed',case=deepcopy(env.case))
    resumed.restore(checkpoint)
    a = env.step(encode_action(WAIT,WAIT,1))
    b = resumed.step(encode_action(WAIT,WAIT,1))
    np.testing.assert_allclose(env.fixture.data.qpos,resumed.fixture.data.qpos,rtol=0,atol=1e-12)
    np.testing.assert_allclose(env.fixture.data.qvel,resumed.fixture.data.qvel,rtol=0,atol=1e-12)
    assert a[1:3] == b[1:3]
    assert env.mission.to_dict() == resumed.mission.to_dict()
    assert [c.integral for c in env.controllers] == [c.integral for c in resumed.controllers]
    assert [c.tick for c in env.controllers] == [c.tick for c in resumed.controllers]
    assert env.online_auditor.to_dict() == resumed.online_auditor.to_dict()
    assert env.online_auditor.report() == resumed.online_auditor.report()
    for key in a[0]:
        np.testing.assert_allclose(a[0][key],b[0][key],rtol=0,atol=1e-7)
    env.close()
    resumed.close()


def test_native_raw_contains_reset_each_tick_and_all_internal_solver_rows(tmp_path):
    from adaptive_drone_patrol.dual_coordination_raw import read_raw
    env = native_env(tmp_path,record_raw=True)
    env.step(encode_action(WAIT,WAIT,1))
    env.close()
    records = list(read_raw(env.raw_segments[0]['path']))
    assert records[0]['kind'] == 'RESET'
    assert len(records) == 21
    for tick,row in enumerate(records[1:],1):
        assert row['kind'] == 'STEP' and row['tick'] == tick
        assert len(row['stages']) == 64
        assert len(row['substeps']) == 17
        assert row['action'] == encode_action(WAIT,WAIT,1)
        assert len(row['structural_mask']) == 242
        assert row['mission']['targets'] == [WAIT,WAIT]
        assert len(row['requested_references']) == len(row['executed_references']) == 2


def test_native_invalid_action_is_rejected_without_advancing_world_or_mission(tmp_path):
    env = native_env(tmp_path)
    before = env.fixture.data.qpos.copy()
    with pytest.raises(ValueError):
        env.step(encode_action(0,0,1))
    assert env.mission.tick == 0 and env.fixture.data.time == 0
    np.testing.assert_array_equal(before,env.fixture.data.qpos)
    env.close()


def test_observation_rejects_reordered_or_duplicated_public_drone_ids():
    sample,mission,goals,refs,prediction = measured_inputs()
    sample['drones'][1]['drone_id'] = 1
    with pytest.raises(ValueError):
        build_observation(sample,mission,goals,refs,prediction)


def test_native_resume_cannot_reset_wall_deadline(tmp_path):
    env = native_env(tmp_path,'wall-source')
    snapshot = env.to_dict()
    snapshot['elapsed_wall_s'] = 601.
    resumed = native_env(tmp_path,'wall-resumed',case=deepcopy(env.case))
    resumed.restore(snapshot)
    assert resumed.terminated
    assert resumed.mission.terminal_reason == 'infrastructure:wall_timeout'
    assert resumed.mission.reward_components['failure'] == -1000
    assert resumed.mission.tick == 0
    with pytest.raises(ValueError):
        resumed.step(encode_action(WAIT,WAIT,1))
    env.close()
    resumed.close()


def test_native_corrupt_checkpoint_does_not_mutate_fresh_world(tmp_path):
    env = native_env(tmp_path,'checkpoint-source')
    env.step(encode_action(WAIT,WAIT,1))
    snapshot = env.to_dict()
    snapshot['backend']['qpos'][0] += 1.
    resumed = native_env(tmp_path,'checkpoint-receiver',case=deepcopy(env.case))
    before = resumed.fixture.data.qpos.copy()
    with pytest.raises(ValueError):
        resumed.restore(snapshot)
    np.testing.assert_array_equal(resumed.fixture.data.qpos,before)
    assert resumed.fixture.data.time == 0
    env.close()
    resumed.close()


def test_compact_replay_retains_exact_inputs_outputs_and_event_without_internal_arrays():
    from adaptive_drone_patrol.dual_coordination_env import compact_replay_record
    sample,mission,goals,refs,prediction = measured_inputs()
    sample.update(schema='dual-raw-v1',qpos=[.125,1.],qvel=[-.25,0.],rpm=[[123.]*4]*2,
                  ctrl=[.5]*8,qacc=[999.],qfrc_passive=[888.],warning_counts=[0],contacts=[])
    full = dict(kind='STEP',tick=1,actual=sample,action=240,action_tick=0,
                requested_references=[r.to_dict() for r in refs],
                executed_references=[r.to_dict() for r in refs],
                event={'kind':'priority','tick':1,'goal':0,'priority':100},
                stages=[{'qpos':[999.]}]*64,substeps=[{'qpos':[888.]}]*17,
                audit={'min_clearance_m':2.,'floor_minima_m':[.4,.5],
                       'constraint_max_m':1e-9,'finite':True,'metrics':[{'detailed':123.}]})
    compact = compact_replay_record(full)
    assert compact['actual']['qpos'] == [.125,1.]
    assert compact['actual']['qvel'] == [-.25,0.]
    assert compact['actual']['ctrl'] == [.5]*8
    assert compact['event'] == full['event']
    assert compact['requested_references'] == full['requested_references']
    assert compact['executed_references'] == full['executed_references']
    assert compact['audit']['min_clearance_m'] == 2.
    assert 'stages' not in compact and 'substeps' not in compact
    assert 'qacc' not in compact['actual'] and 'qfrc_passive' not in compact['actual']
    assert 'metrics' not in compact['audit']
    compact['actual']['qpos'][0] = 999.
    assert full['actual']['qpos'][0] == .125


def test_native_compact_capture_is_default_and_detailed_raw_is_optional(tmp_path):
    from adaptive_drone_patrol.dual_coordination_raw import read_raw
    env = native_env(tmp_path,record_raw=False)
    env.step(encode_action(WAIT,WAIT,1))
    env.close()
    assert env.raw_segments == []
    records = list(read_raw(env.replay_segments[0]['path']))
    assert len(records) == 21
    assert all('stages' not in row and 'substeps' not in row for row in records)
    assert records[-1]['actual']['world_time'] == pytest.approx(.2)
    assert records[-1]['actual']['qpos'] == env.fixture.data.qpos.tolist()
    assert records[-1]['actual']['ctrl'] == env.fixture.data.ctrl.tolist()


def test_native_default_online_audit_is_checkpointed_and_counts_all_internal_states(tmp_path):
    env = native_env(tmp_path)
    env.step(encode_action(WAIT,WAIT,1))
    assert env.online_auditor is not None
    report = env.online_auditor.report()
    assert report['safe'] is True
    assert report['rk_states'] == 1280
    assert report['native_poses'] == 321
    assert report['control_steps'] == 20
    assert report['full_physical_acceptance_qualified'] is False
    snapshot = env.to_dict()
    assert snapshot['online_auditor_state']['rk_states'] == 1280
    assert env.fixture.stage_rows == [] and env.fixture.substep_rows == []
    env.close()


def test_native_compact_live_audit_and_detailed_replay_have_same_flight_and_metrics(tmp_path):
    compact = native_env(tmp_path,'compact',record_raw=False)
    detailed = native_env(tmp_path,'detailed',record_raw=True,case=deepcopy(compact.case))
    a = compact.step(encode_action(WAIT,WAIT,1))
    b = detailed.step(encode_action(WAIT,WAIT,1))
    assert not a[2] and not b[2],(a[3],b[3])
    np.testing.assert_array_equal(compact.fixture.data.qpos,detailed.fixture.data.qpos)
    np.testing.assert_array_equal(compact.fixture.data.qvel,detailed.fixture.data.qvel)
    assert compact.mission.to_dict() == detailed.mission.to_dict()
    live_report,replay_report = compact.online_auditor.report(),detailed.online_auditor.report()
    assert live_report['rk_states'] == replay_report['rk_states']
    assert live_report['native_poses'] == replay_report['native_poses']
    assert live_report['safe'] == replay_report['safe']
    assert live_report['minimum_clearance_m'] == pytest.approx(replay_report['minimum_clearance_m'],abs=1e-10)
    compact.close()
    detailed.close()


def test_native_moving_reference_records_beyond_one_second(tmp_path):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    from adaptive_drone_patrol.dual_coordination_raw import read_raw
    case = generate_case(15,0,'normal')
    case['starts'] = [[-3.5,-3.5,1.],[3.5,3.5,1.]]
    case['initials'] = [dict(position=p,quaternion_wxyz=[1.,0.,0.,0.],swing_deg=0.,direction_rad=0.)
                        for p in case['starts']]
    case['goals'][0] = [.5,-3.5,1.]
    env = native_env(tmp_path,case=case)
    for _ in range(6):
        _,_,terminated,info = env.step(encode_action(0,WAIT,1))
        assert not terminated,info
    env.close()
    records = list(read_raw(env.replay_segments[0]['path']))
    assert records[-1]['tick'] == 120
    assert records[-1]['actual']['world_time'] == pytest.approx(1.2)
    assert max(row['supervisor']['horizon_s'] for row in records[1:]) > 2.


def test_training_and_validation_require_file_verified_admission_before_side_effects(tmp_path):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    for mode,split in [('training',10),('validation',20)]:
        case=generate_case(split,0,'normal')
        for capability in (None,True,{'passed':True}):
            output=tmp_path/(mode+str(type(capability)))
            with pytest.raises(ValueError,match='verified'):
                DualCoordinationEnv('missing',output,case,{},mode=mode,admission=capability)
            assert not output.exists()


def test_development_replay_cannot_be_silently_disabled(tmp_path):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    with pytest.raises(ValueError,match='replay'):
        DualCoordinationEnv('missing',tmp_path/'disabled',generate_case(15,0,'normal'),{},retain_replay=False)
    assert not (tmp_path/'disabled').exists()


def test_replay_stride_is_explicit_and_detailed_raw_requires_full_rate(tmp_path):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    case=generate_case(15,0,'normal')
    for stride in (0,2,True):
        with pytest.raises(ValueError,match='stride'):
            DualCoordinationEnv('missing',tmp_path/str(stride),case,{},replay_stride_ticks=stride)
    with pytest.raises(ValueError,match='detailed'):
        DualCoordinationEnv('missing',tmp_path/'detail',case,{},record_raw=True,replay_stride_ticks=20)


def test_native_sparse_replay_retains_each_action_and_terminal_boundary_without_skipping_audit(tmp_path):
    from adaptive_drone_patrol.dual_coordination_raw import read_raw
    env=native_env(tmp_path,replay_stride_ticks=20)
    for _ in range(2):
        _,_,terminal,info=env.step(encode_action(WAIT,WAIT,1))
        assert not terminal,info
    snapshot=env.to_dict()
    assert snapshot['replay_stride_ticks']==20
    rows=list(read_raw(env.replay_segments[0]['path']))
    assert [r['tick'] for r in rows]==[0,1,20,21,40]
    assert [r['action_tick'] for r in rows if r.get('action_receipt') is not None]==[0,20]
    assert env.online_auditor.report()['rk_states']==2560
    assert env.online_auditor.report()['reconstructed_intervals']==640
    env.close()


def test_heldout_execution_requires_its_own_file_verified_capability(tmp_path):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    case=generate_case(15,0,'normal')
    for mode in ('routing-heldout','physical-heldout'):
        for capability in (None,True,{}):
            with pytest.raises(ValueError,match='verified'):
                DualCoordinationEnv('missing',tmp_path/mode,case,{},mode=mode,admission=capability)
            assert not (tmp_path/mode).exists()


def timing_only_timeout_env(monkeypatch, *, enabled=True, failing_io=False):
    """Software clock/IO fixture; intentionally does not simulate native physics."""
    import adaptive_drone_patrol.dual_coordination_env as module
    clock = [0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    def spend(seconds, result=None):
        clock[0] += seconds
        return result
    env = object.__new__(DualCoordinationEnv)
    env.collect_timings = enabled
    env.closed = False
    env.mission = MissionState([50]*10)
    legal_mask = env.mission.legal_mask
    env.mission.legal_mask = lambda:spend(2.,legal_mask())
    env._elapsed = lambda:601.
    def ensure():
        spend(3.)
        if failing_io:
            raise OSError('flush failed')
    env._ensure_raw = ensure
    env._finish_raw_segment = lambda:spend(5.)
    env.observation = lambda:spend(7.,{'synthetic':np.array([1.])})
    env._write_result = lambda:spend(11.)
    env.scope = 'SOFTWARE_TEST_ONLY'
    env.production_admitted = False
    env.mode = 'development'
    env.last_supervisor_reason = 'clear'
    env.raw_segments = []
    env.replay_segments = []
    env.online_audit_summary = {}
    return env


def test_step_phase_timings_include_initial_mask_timeout_final_observation_and_flush(monkeypatch):
    env = timing_only_timeout_env(monkeypatch)
    _,reward,terminated,info = env.step(encode_action(WAIT,WAIT,1))
    assert terminated and reward == -1000
    assert info['timings'] == dict(controller_reference_supervisor=0.,physics_audit=0.,
                                   mask_observation=9.,record_io=24.,total_step=33.)
    assert env.last_step_timings == info['timings']


def test_disabled_timings_do_not_read_clock_or_add_info(monkeypatch):
    import adaptive_drone_patrol.dual_coordination_env as module
    env = timing_only_timeout_env(monkeypatch,enabled=False)
    def forbidden_clock():
        raise AssertionError('Opt-out must not read diagnostic clock')
    monkeypatch.setattr(module.time,'perf_counter',forbidden_clock)
    assert 'timings' not in env.step(encode_action(WAIT,WAIT,1))[3]


def test_step_phase_timings_close_on_propagated_io_exception(monkeypatch):
    env = timing_only_timeout_env(monkeypatch,failing_io=True)
    with pytest.raises(OSError,match='flush failed'):
        env.step(encode_action(WAIT,WAIT,1))
    assert env.last_step_timings == dict(controller_reference_supervisor=0.,physics_audit=0.,
                                        mask_observation=2.,record_io=3.,total_step=5.)
    assert env.mission.tick == 0 and env.mission.terminal_reason is None


def test_timing_flag_is_strict_boolean_before_output_creation(tmp_path):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    with pytest.raises(ValueError,match='collect_timings'):
        DualCoordinationEnv('missing',tmp_path/'invalid',generate_case(15,0,'normal'),{},collect_timings=1)
    assert not (tmp_path/'invalid').exists()


def test_native_timings_preserve_physics_and_bind_checkpoint_recording_mode(tmp_path):
    timed = native_env(tmp_path,'timed',collect_timings=True)
    plain = native_env(tmp_path,'plain')
    try:
        action = encode_action(WAIT,WAIT,1)
        first = timed.step(action)
        second = plain.step(action)
        assert first[1:3] == second[1:3]
        np.testing.assert_array_equal(timed.fixture.data.qpos,plain.fixture.data.qpos)
        np.testing.assert_array_equal(timed.fixture.data.qvel,plain.fixture.data.qvel)
        assert timed.mission.to_dict() == plain.mission.to_dict()
        times = first[3]['timings']
        assert all(math.isfinite(v) and v >= 0 for v in times.values())
        assert sum(v for k,v in times.items() if k != 'total_step') <= times['total_step']
        assert all(times[k] > 0 for k in ('controller_reference_supervisor','physics_audit','mask_observation','record_io'))
        assert 'timings' not in second[3]
        snapshot = timed.to_dict()
        assert snapshot['collect_timings'] is True and snapshot['binding']['collect_timings'] is True
        assert json.loads((timed.output/'environment.json').read_text())['collect_timings'] is True
        restored = native_env(tmp_path,'restored',collect_timings=True)
        mismatch = native_env(tmp_path,'mismatch')
        try:
            with pytest.raises(ValueError,match='matching'):
                mismatch.restore(snapshot)
            restored.restore(snapshot)
            assert restored.collect_timings is True
            altered = deepcopy(snapshot)
            altered['collect_timings'] = False
            with pytest.raises(ValueError,match='matching'):
                mismatch.restore(altered)
        finally:
            restored.close()
            mismatch.close()
    finally:
        timed.close()
        plain.close()
