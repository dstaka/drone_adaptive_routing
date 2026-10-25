"""Explicit changed-history initialization; no loaded training claim."""

import copy

from .payload_checkpoint import actor_signature
from .payload_retry_contract import OBSERVATION_CONTRACT, PROFILE, require_checkpoint_contract


def _validate_tensors(state, role, inputs):
    import torch
    outputs = 4 if role == 'actor' else 1
    shapes = {'mlp.0.weight': (128, inputs), 'mlp.0.bias': (128,),
              'mlp.2.weight': (128, 128), 'mlp.2.bias': (128,),
              'mlp.4.weight': (outputs, 128), 'mlp.4.bias': (outputs,)}
    if role == 'actor':
        shapes['distribution.std_param'] = (4,)
    if any(key not in state or tuple(state[key].shape) != shape for key, shape in shapes.items()):
        raise ValueError('Actual checkpoint architecture differs from the adopted 128/128 controller')
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint contains nonfinite or unsupported tensors')


def transfer_state(parent):
    """Copy unchanged kinematics; zero changed history and new payload columns."""
    result = {}
    for role in ('actor', 'critic'):
        key = role+'_state_dict'
        _validate_tensors(parent[key], role, 17)
        state = copy.deepcopy(parent[key])
        original = state['mlp.0.weight']
        first = original.new_zeros((128, 29))
        first[:, :13] = original[:, :13]
        state['mlp.0.weight'] = first
        result[key] = state
    result.update(iter=0, infos={
        'observation_contract': copy.deepcopy(OBSERVATION_CONTRACT),
        'physics_signature': PROFILE,
        'status': 'INITIALIZATION_ONLY_NOT_TRAINED',
        'transfer': {'copied_legacy_columns': [0, 13], 'zeroed_history_columns': [13, 17],
                     'zeroed_payload_columns': [17, 29], 'other_compatible_tensors': 'copied',
                     'fresh_optimizer': True, 'iteration': 0},
    })
    return result


def validate_retry_checkpoint(checkpoint):
    require_checkpoint_contract(checkpoint.get('infos', {}).get('observation_contract'), OBSERVATION_CONTRACT)
    if checkpoint['infos'].get('physics_signature') != PROFILE:
        raise ValueError('Retry checkpoint physical profile mismatch')
    for role in ('actor', 'critic'):
        _validate_tensors(checkpoint[role+'_state_dict'], role, 29)


def initialization_comparison(parent, initial):
    import torch
    validate_retry_checkpoint(initial)
    preserved, history_zero, payload_zero, others = True, True, True, True
    for role in ('actor', 'critic'):
        key = role+'_state_dict'
        _validate_tensors(parent[key], role, 17)
        before, after = parent[key], initial[key]
        preserved &= torch.equal(before['mlp.0.weight'][:, :13], after['mlp.0.weight'][:, :13])
        history_zero &= not bool(torch.count_nonzero(after['mlp.0.weight'][:, 13:17]))
        payload_zero &= not bool(torch.count_nonzero(after['mlp.0.weight'][:, 17:]))
        others &= before.keys() == after.keys() and all(torch.equal(value, after[name])
                  for name, value in before.items() if name != 'mlp.0.weight')
    comparison = dict(unchanged_kinematic_columns_equal=preserved,
                      changed_history_columns_zero=history_zero, payload_columns_zero=payload_zero,
                      other_tensors_equal=others,
                      parent_actor_tensor_sha256=actor_signature(parent['actor_state_dict']),
                      parent_critic_tensor_sha256=actor_signature(parent['critic_state_dict']),
                      initial_actor_tensor_sha256=actor_signature(initial['actor_state_dict']),
                      initial_critic_tensor_sha256=actor_signature(initial['critic_state_dict']))
    if not all((preserved, history_zero, payload_zero, others)) or initial.get('iter') != 0 or 'optimizer_state_dict' in initial:
        raise ValueError('Initialization mapping or fresh-optimizer contract differs')
    return comparison


def validate_matched_manifests(randomized, fixed):
    required = {'profile', 'seed', 'num_envs', 'environment_steps', 'source_binding',
                'observation_contract', 'optimizer_config', 'architecture', 'curriculum',
                'full_distribution_every_stage', 'initialization_comparison',
                'parent_sha256', 'parent_config_sha256'}
    if not required <= randomized.keys() or not required <= fixed.keys():
        raise ValueError('Retry matched-learning evidence is incomplete')
    if randomized.get('method') != 'randomized' or fixed.get('method') != 'fixed':
        raise ValueError('Expected randomized and fixed methods in that order')
    for manifest in (randomized, fixed):
        require_checkpoint_contract(manifest['observation_contract'], OBSERVATION_CONTRACT)
        if (manifest['profile'] != PROFILE or manifest['seed'] != 7 or manifest['num_envs'] != 1 or
                not manifest['full_distribution_every_stage']):
            raise ValueError('Retry learning profile, seed, parallelism or curriculum differs')
    excluded = {'method', 'initialization_sha256'}
    if ({key: value for key, value in randomized.items() if key not in excluded} !=
            {key: value for key, value in fixed.items() if key not in excluded}):
        raise ValueError('Initialization, environment-step budget, reward, optimizer or shared procedure differs')
