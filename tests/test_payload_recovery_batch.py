"""CPU contracts for real batched size buckets; these do not emulate physics."""
import importlib
from pathlib import Path

import numpy as np
import pytest

from adaptive_drone_patrol.payload_recovery_sampling import EpisodeLoads

PATH = Path(__file__).resolve().parents[1]/'src/adaptive_drone_patrol/payload_recovery_batch.py'
REF = .02700047753751278


def api():
    assert PATH.is_file(), 'Batched physical adapter has not been implemented'
    return importlib.import_module('adaptive_drone_patrol.payload_recovery_batch')


class Bucket:
    """Observable CPU array double; deliberately no simulated dynamics claim."""
    def __init__(self, side, n):
        self.side, self.n = side, n
        self.reset_calls, self.rpms = [], []
        self.bad_env = None
        self.data = dict(qpos=np.zeros((n,15)), velocity=np.zeros((n,12)), xyz=np.zeros((n,7,3)),
                         quaternion=np.zeros((n,7,4)), linear_velocity=np.zeros((n,7,3)),
                         angular_velocity=np.zeros((n,7,3)), mass=np.zeros((n,7)),
                         inertia=np.zeros((n,7,3,3)), package_aabb=np.zeros((n,2,3)),
                         contact=np.zeros(n,dtype=bool), force=np.zeros((n,8,3)), torque=np.zeros((n,8,3)))
        self.data['quaternion'][:,:,0] = 1
        self.data['qpos'][:,[3,7,11]] = 1
        self.data['xyz'][:,:,2] = 1
        self.data['xyz'][:,5,2] = .98
        self.data['xyz'][:,6,2] = .68-side/2
        self.data['package_aabb'][:,0] = [-side/2,-side/2,.68-side]
        self.data['package_aabb'][:,1] = [side/2,side/2,.68]
        self.data['mass'][:] = [.027,1.1920928955078125e-7,1.1920928955078125e-7,
                               1.1920928955078125e-7,1.1920928955078125e-7,.0001,.2*REF]

    def snapshot(self):
        return {key:np.array(value,copy=True) for key,value in self.data.items()}

    def reset_rows(self, rows, draws, initials):
        self.reset_calls.append(list(rows))
        for row, draw, initial in zip(rows,draws,initials):
            self.data['mass'][row,6] = draw.payload.mass
            self.data['inertia'][row,6] = np.eye(3)*draw.payload.inertia[0]
            self.data['qpos'][row,:3] = initial['drone_xyz']
            self.data['velocity'][row] = 0
            self.data['xyz'][row,0] = initial['drone_xyz']
            self.data['xyz'][row,5] = np.array(initial['drone_xyz'])+[0,0,-.02]
            self.data['xyz'][row,6] = initial['package_xyz']
            self.data['package_aabb'][row] = np.array(initial['package_xyz'])+np.array([[-1,-1,-1],[1,1,1]])*self.side/2

    def set_rpm(self, values):
        assert values.shape == (self.n,4)
        self.rpms.append(values.copy())

    def step(self):
        self.data['qpos'][:,0] += .0001
        if self.bad_env is not None:
            self.data['linear_velocity'][self.bad_env,6,0] = 21.


class Builder:
    def __init__(self):
        self.calls, self.buckets = [], []

    def __call__(self, output, side, n_envs, substeps):
        self.calls.append((side,n_envs,substeps))
        bucket = Bucket(side,n_envs)
        self.buckets.append(bucket)
        return bucket


def make(tmp_path, n=4):
    builder, records = Builder(), []
    bank = api().RecoveryBatch(tmp_path/'batch', n, substeps=128, scene_builder=builder, sink=records.append)
    return bank,builder,records


def inputs(assignments, randomized=True):
    states = {}
    for env, identity in assignments.items():
        side = EpisodeLoads(REF).draw(identity,randomized=randomized).payload.side_m
        states[env] = dict(drone_xyz=[0.,0.,1.], drone_quaternion_wxyz=[1.,0.,0.,0.],
                           package_xyz=[0.,0.,.68-side/2], package_quaternion_wxyz=[1.,0.,0.,0.],
                           drone_velocity=[0.,0.,0.], drone_angular_velocity=[0.,0.,0.],
                           package_velocity=[0.,0.,0.], package_angular_velocity=[0.,0.,0.])
    return states


def reset(bank, assignments, randomized=True):
    bank.reset(assignments, initials=inputs(assignments,randomized),
               targets={env:[.1,.2,1.5] for env in assignments},randomized=randomized)


def test_builds_three_batched_size_scenes_and_never_counts_unused_slots(tmp_path):
    bank,builder,_ = make(tmp_path)
    assert builder.calls == [(.03,4,128),(.045,4,128),(.06,4,128)]
    reset(bank,{0:0,1:1,2:2,3:3})
    result = bank.step(np.zeros((4,4)))
    assert result['active_mask'].shape == (3,4)
    assert np.all(result['active_mask'].sum(axis=0) == 1)
    assert result['physical_mask'].sum() == 12
    assert result['generated_transitions'] == 4
    assert not bank.training_admitted and not bank.effective_geometry_verified
    for bucket,mask in zip(builder.buckets,result['active_mask']):
        assert bucket.rpms[-1].shape == (4,4)
        assert np.all(bucket.rpms[-1][~mask] == 0)
        assert np.all(bucket.rpms[-1][mask] == 14468.429183500699)


def test_selective_reset_keeps_other_rows_in_every_bucket_and_all_history(tmp_path):
    bank,builder,records = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    bank.step(np.full((4,4),.2))
    before = [b.snapshot() for b in builder.buckets]
    history = bank.actions.copy()
    bank.mark_done([1])
    reset(bank,{1:99})
    for b,old in zip(builder.buckets,before):
        for key in old:
            np.testing.assert_array_equal(b.snapshot()[key][[0,2,3]],old[key][[0,2,3]])
    np.testing.assert_array_equal(bank.actions[[0,2,3]],history[[0,2,3]])
    assert np.all(bank.actions[1] == 0)
    assert bank.episode_steps.tolist() == [1,0,1,1]
    assert records[-1]['kind'] == 'reset'
    assert sum(len(b.reset_calls) for b in builder.buckets) <= 4


def test_retarget_preserves_bodies_actions_and_episode_identity(tmp_path):
    bank,builder,_ = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    bank.step(np.full((4,4),.1))
    before = [b.snapshot() for b in builder.buckets]
    actions = bank.actions.copy()
    episodes = bank.schedule.snapshot()
    bank.retarget({2:[1.,2.,3.]})
    np.testing.assert_array_equal(bank.actions,actions)
    assert bank.schedule.snapshot() == episodes
    for b,old in zip(builder.buckets,before):
        for key in old:
            np.testing.assert_array_equal(b.snapshot()[key],old[key])


def test_episode_ids_not_completion_order_choose_load_and_size_bucket(tmp_path):
    a,_,_ = make(tmp_path/'a',2)
    b,_,_ = make(tmp_path/'b',2)
    reset(a,{0:7,1:8})
    reset(b,{1:7,0:8})
    assert a.schedule.snapshot()[0].draw == b.schedule.snapshot()[1].draw
    assert a.bucket_for_env[0] == b.bucket_for_env[1]
    assert a.schedule.snapshot()[1].draw == b.schedule.snapshot()[0].draw
    with pytest.raises(ValueError):
        reset(a,{0:9})
    a.mark_done([0])
    with pytest.raises(ValueError):
        reset(a,{0:7})


def test_preclip_raw_abnormal_state_quarantines_whole_rollout_and_preserves_mask(tmp_path):
    bank,builder,records = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    bank.step(np.zeros((4,4)))
    builder.buckets[int(bank.bucket_for_env[2])].bad_env = 2
    with pytest.raises(api().BatchQuarantined):
        bank.step(np.zeros((4,4)))
    assert bank.rollout_quarantined and bank.optimizer_transitions == 0
    assert records[-1]['raw_after'][int(bank.bucket_for_env[2])]['linear_velocity'][2,6,0] == 21
    assert records[-1]['invalid_logical_rows'] == [2]
    with pytest.raises(api().BatchQuarantined):
        bank.step(np.zeros((4,4)))
    with pytest.raises(api().BatchQuarantined):
        reset(bank,{0:90})


def test_nonfinite_action_is_retained_before_clipping_without_advancing_any_scene(tmp_path):
    bank,builder,records = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    actions = np.zeros((4,4))
    actions[0,0] = np.nan
    with pytest.raises(api().BatchQuarantined):
        bank.step(actions)
    assert np.isnan(records[-1]['raw_actions'][0,0])
    assert not any(b.rpms for b in builder.buckets)


def test_changed_other_row_on_reset_or_unapplied_property_stops_backend(tmp_path):
    bank,builder,_ = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    bank.mark_done([1])
    target = EpisodeLoads(REF).draw(90,randomized=True).payload.side_m
    bucket = builder.buckets[[.03,.045,.06].index(target)]
    original = bucket.reset_rows
    def bad(rows,draws,initials):
        original(rows,draws,initials)
        bucket.data['mass'][0,6] = .5
    bucket.reset_rows = bad
    with pytest.raises(api().BatchQuarantined):
        reset(bank,{1:90})


def test_at_least30_asynchronous_geometry_resets_have_unique_ids_and_selected_rows_only(tmp_path):
    bank,_,_ = make(tmp_path,3)
    reset(bank,{0:0,1:1,2:2})
    sizes = set()
    for identity in range(3,33):
        env = identity%3
        bank.step(np.zeros((3,4)))
        bank.mark_done([env])
        reset(bank,{env:identity})
        sizes.add(bank.schedule.snapshot()[env].draw.payload.side_m)
    assert sizes == {.03,.045,.06}
    assert len({p.draw.logical_episode_id for p in bank.schedule.snapshot().values()}) == 3


def test_complete_sink_and_strict_full_action_shape_are_required(tmp_path):
    with pytest.raises(ValueError):
        api().RecoveryBatch(tmp_path/'none',2,scene_builder=Builder())
    bank,_,_ = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    with pytest.raises(ValueError):
        bank.step(np.zeros(4))


def test_incomplete_global_wrench_field_cannot_claim_observable_inactive_rows(tmp_path):
    bank,builder,_ = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    builder.buckets[0].data['force'] = np.zeros((4,1,3))
    with pytest.raises(api().BatchQuarantined):
        bank.step(np.zeros((4,4)))


def test_real_adapter_uses_per_env_entity_setters_and_full_motor_kernel_tensor():
    from types import SimpleNamespace
    import torch
    names = ['base_link','prop0_link','prop1_link','prop2_link','prop3_link','suspension_rod','payload']
    class Entity:
        n_qs, n_dofs, link_start, q_start, dof_start = 15,12,1,0,0
        def __init__(self):
            self.links = [SimpleNamespace(name=n,idx=i+1) for i,n in enumerate(names)]
            self.joints = [SimpleNamespace(n_qs=7,n_dofs=6,dof_start=0)]
            self.calls = []
            self.solver = SimpleNamespace(set_drone_rpm=lambda *args:self.calls.append(('rpm',args)))
        def get_link(self,name):
            return self.links[names.index(name)]
        def get_joint(self,name):
            return SimpleNamespace(q_start=7 if name == 'suspension_ball' else 11)
        def get_qpos(self,envs_idx):
            return torch.zeros((len(envs_idx),15))
        def set_links_mass(self,value,**kwargs):
            self.calls.append(('mass',value.clone(),kwargs))
        def set_links_inertia(self,value,**kwargs):
            self.calls.append(('inertia',value.clone(),kwargs))
        def set_pos(self,value,**kwargs):
            self.calls.append(('pos',value.clone(),kwargs))
        def set_quat(self,value,**kwargs):
            self.calls.append(('quat',value.clone(),kwargs))
        def set_qpos(self,value,**kwargs):
            self.calls.append(('qpos',value.clone(),kwargs))
        def set_dofs_damping(self,value,**kwargs):
            self.calls.append(('damping',value.clone(),kwargs))
    entity = Entity()
    gs = SimpleNamespace(device='cpu',tc_float=torch.float32,tc_int=torch.int32)
    bucket = api().GenesisSizeBucket(None,entity,None,gs,.027,{})
    draws = [EpisodeLoads(REF).draw(i,randomized=False) for i in [4,5]]
    states = inputs({0:4,2:5},False)
    bucket.reset_rows([0,2],draws,[states[0],states[2]])
    for call in entity.calls:
        assert call[-1]['envs_idx'] == [0,2]
    mass, inertia = entity.calls[:2]
    assert mass[1].shape == (2,1) and inertia[1].shape == (2,1,3,3)
    assert mass[2]['links_idx_local'] == [6] == inertia[2]['links_idx_local']
    bucket.set_rpm(np.zeros((3,4)))
    assert entity.calls[-1][0] == 'rpm'
    assert entity.calls[-1][1][3].shape == (3,4)


def test_solver_advance_then_exception_preserves_best_effort_current_state_and_unknown_completion(tmp_path):
    bank,builder,records = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    first, second = builder.buckets[:2]
    original_step, original_snapshot = first.step, second.snapshot
    advanced = [False]
    def advance_then_fail():
        original_step()
        advanced[0] = True
        raise RuntimeError('diagnostic after solver advance')
    def unavailable_after_advance():
        if advanced[0]:
            raise RuntimeError('one getter unavailable')
        return original_snapshot()
    first.step = advance_then_fail
    second.snapshot = unavailable_after_advance
    with pytest.raises(api().BatchQuarantined):
        bank.step(np.zeros((4,4)))
    receipt = records[-1]
    assert receipt['bucket_step_status'] == ['UNKNOWN','NOT_ATTEMPTED','NOT_ATTEMPTED']
    assert receipt['raw_after'][0]['qpos'][0,0] == pytest.approx(.0001)
    assert receipt['raw_after'][1] is None
    assert receipt['raw_after'][2] is not None
    assert receipt['capture_errors'][0]['bucket'] == 1
    assert 'generated_transitions' not in receipt
    assert bank.generated_valid == bank.generated_invalid == bank.vector_steps == 0
    assert receipt['quarantined'] and bank.rollout_quarantined


def test_unreadable_pre_step_bucket_stops_without_losing_other_current_states(tmp_path):
    bank,builder,records = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    def fail_snapshot():
        raise RuntimeError('unreadable current bucket')
    builder.buckets[1].snapshot = fail_snapshot
    with pytest.raises(api().BatchQuarantined):
        bank.step(np.zeros((4,4)))
    assert records[-1]['raw_before'][0] is not None
    assert records[-1]['raw_before'][1] is None
    assert records[-1]['raw_before'][2] is not None
    assert not any(b.rpms for b in builder.buckets)
    assert bank.rollout_quarantined


def test_partial_reset_exception_preserves_effective_abnormal_properties(tmp_path):
    bank,builder,records = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    bank.mark_done([1])
    side = EpisodeLoads(REF).draw(91,randomized=True).payload.side_m
    bucket = builder.buckets[[.03,.045,.06].index(side)]
    def mutate_then_fail(rows,draws,initials):
        bucket.data['mass'][rows[0],6] = .5
        raise RuntimeError('setter after partial write')
    bucket.reset_rows = mutate_then_fail
    with pytest.raises(api().BatchQuarantined):
        reset(bank,{1:91})
    assert records[-1]['raw_after'][[.03,.045,.06].index(side)]['mass'][1,6] == .5


@pytest.mark.parametrize('field,component,value', [
    ('linear_velocity',(0,6,0),1e12), ('angular_velocity',(0,5,1),201.),
    ('quaternion',(0,6,slice(None)),[0.,0.,0.,0.]),
    ('qpos',(0,slice(7,11)),[0.,0.,0.,0.]), ('xyz',(0,6,0),20.1),
])
def test_inactive_finite_numerical_excursions_quarantine_whole_rollout(tmp_path,field,component,value):
    bank,builder,records = make(tmp_path)
    reset(bank,{0:0,1:1,2:2,3:3})
    inactive_bucket = (int(bank.bucket_for_env[0])+1)%3
    builder.buckets[inactive_bucket].data[field][component] = value
    with pytest.raises(api().BatchQuarantined):
        bank.step(np.zeros((4,4)))
    assert not records[-1]['active_mask'][inactive_bucket,0]
    np.testing.assert_array_equal(records[-1]['raw_before'][inactive_bucket][field][component],value)
    assert bank.rollout_quarantined and bank.optimizer_transitions == 0
    assert not any(bucket.rpms for bucket in builder.buckets)
