"""Explicit CPU initialization transfer; no loaded-flight compatibility claim."""

import copy
from .payload_contract import OBSERVATION_CONTRACT


def validate_parent_config(configs):
    if len(configs) != 5:
        raise ValueError('Expected actual five-part flight configuration')
    env, obs, _, commands, training = configs
    expected = dict(num_actions=4, base_init_pos=[0.,0.,1.], base_init_quat=[1.,0.,0.,0.],
                    clip_actions=1., simulate_action_latency=True, termination_if_close_to_ground=.1,
                    termination_if_roll_greater_than=60., termination_if_pitch_greater_than=60.,
                    termination_if_x_greater_than=6., termination_if_y_greater_than=4.,
                    termination_if_z_greater_than=3.)
    scales = dict(rel_pos=1/3, lin_vel=1/3, ang_vel=1/3.14159)
    bounds = dict(num_commands=3, pos_x_range=[-2.7,2.7], pos_y_range=[-1.8,1.8], pos_z_range=[1.,3.])
    if any(env.get(k) != v for k,v in expected.items()) or obs.get('obs_scales') != scales:
        raise ValueError('Parent safety/normalization/initial-state configuration mismatch')
    if any(commands.get(k) != v for k,v in bounds.items()):
        raise ValueError('Parent dispersed command envelope mismatch')
    for key in ('actor', 'critic'):
        if training[key].get('hidden_dims') != [128,128] or training[key].get('activation') != 'tanh':
            raise ValueError('Parent architecture mismatch')


def transfer_state(parent):
    import torch
    result = {}
    for kind in ('actor_state_dict', 'critic_state_dict'):
        state = copy.deepcopy(parent[kind])
        old = state['mlp.0.weight']
        if old.shape != (128,17) or not torch.isfinite(old).all():
            raise ValueError('Expected a finite adopted 128x17 first layer')
        new = old.new_zeros((128,29))
        new[:,:17] = old
        state['mlp.0.weight'] = new
        result[kind] = state
    result.update(iter=0, infos={'observation_contract': copy.deepcopy(OBSERVATION_CONTRACT),
                                'status': 'INITIALIZATION_ONLY_NOT_TRAINED',
                                'transfer': 'Copy all original tensors; first-layer columns 0:17 copied, 17:29 zero; new optimizer'})
    return result


def actor_signature(state):
    """Hash actual actor tensors independently of checkpoint packaging metadata."""
    import hashlib
    import json
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        h.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)]).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def validate_matched_manifests(randomized,fixed):
    """Require identical declared learning procedures except the payload method."""
    required={'seed','num_envs','environment_steps','parent_sha256','parent_config_sha256',
              'training_source_sha256','bank_source_sha256','training_script_sha256',
              'optimizer_config','architecture','curriculum','observation_contract','full_distribution_every_stage'}
    if not required.issubset(randomized) or not required.issubset(fixed):
        raise ValueError('Training manifests omit matched-procedure evidence')
    if randomized.get('method')!='randomized' or fixed.get('method')!='fixed':
        raise ValueError('Training method labels do not match')
    excluded={'method','initialization_sha256'}
    a={k:v for k,v in randomized.items() if k not in excluded}
    b={k:v for k,v in fixed.items() if k not in excluded}
    if a!=b:
        raise ValueError('Seed, reward, optimizer, curriculum or other shared training contract differs')
