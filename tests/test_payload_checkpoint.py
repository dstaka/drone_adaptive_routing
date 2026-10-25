"""Explicit initialization mapping is not a trained/adopted loaded controller."""

import copy
import torch
import pytest
from adaptive_drone_patrol.payload_checkpoint import transfer_state, validate_parent_config


def parent_config():
    return [dict(num_actions=4, base_init_pos=[0.,0.,1.], base_init_quat=[1.,0.,0.,0.],
                 clip_actions=1., simulate_action_latency=True, termination_if_close_to_ground=.1,
                 termination_if_roll_greater_than=60., termination_if_pitch_greater_than=60.,
                 termination_if_x_greater_than=6., termination_if_y_greater_than=4.,
                 termination_if_z_greater_than=3.),
            {'obs_scales': {'rel_pos': 1/3, 'lin_vel': 1/3, 'ang_vel': 1/3.14159}}, {},
            dict(num_commands=3, pos_x_range=[-2.7,2.7], pos_y_range=[-1.8,1.8], pos_z_range=[1.,3.]),
            {'actor': {'hidden_dims': [128,128], 'activation':'tanh'},
             'critic': {'hidden_dims': [128,128], 'activation':'tanh'}}]


def test_semantic_config_mismatch_rejected_despite_same_shape():
    cfg = parent_config()
    validate_parent_config(cfg)
    for bad in [copy.deepcopy(cfg) for _ in range(3)]:
        bad[1]['obs_scales']['rel_pos'] = .5
        with pytest.raises(ValueError):
            validate_parent_config(bad)
    cfg[0]['base_init_pos'][2] = 2
    with pytest.raises(ValueError):
        validate_parent_config(cfg)


def test_transfer_copies_legacy_columns_zeroes_new_inputs_and_discards_optimizer():
    parent = dict(actor_state_dict={'mlp.0.weight': torch.randn(128,17), 'mlp.0.bias': torch.randn(128)},
                  critic_state_dict={'mlp.0.weight': torch.randn(128,17)}, optimizer_state_dict={'stale':1}, iter=4296)
    out = transfer_state(parent)
    assert 'optimizer_state_dict' not in out
    assert out['iter'] == 0
    for k in ['actor_state_dict','critic_state_dict']:
        assert torch.equal(out[k]['mlp.0.weight'][:,:17],parent[k]['mlp.0.weight'])
        assert torch.count_nonzero(out[k]['mlp.0.weight'][:,17:]) == 0
        assert out[k]['mlp.0.weight'].shape == (128,29)
    assert parent['actor_state_dict']['mlp.0.weight'].shape == (128,17)
    parent['actor_state_dict']['mlp.0.weight'] = torch.zeros(128,29)
    with pytest.raises(ValueError):
        transfer_state(parent)


def test_matched_training_rejects_seed_reward_and_optimizer_mismatch():
    from adaptive_drone_patrol.payload_checkpoint import validate_matched_manifests
    common=dict(seed=7,num_envs=1,environment_steps=300000,parent_sha256='parent',
                parent_config_sha256='config',training_source_sha256='reward',bank_source_sha256='bank',
                training_script_sha256='script',optimizer_config={'learning_rate':.00003},
                architecture={'hidden_dims':[128,128]},curriculum=['hover','long'],
                observation_contract={'size':29},full_distribution_every_stage=True)
    a=dict(common,method='randomized')
    b=dict(common,method='fixed')
    validate_matched_manifests(a,b)
    for key,value in [('seed',8),('training_source_sha256','other'),('optimizer_config',{'learning_rate':.01})]:
        bad=copy.deepcopy(b)
        bad[key]=value
        with pytest.raises(ValueError):
            validate_matched_manifests(a,bad)
