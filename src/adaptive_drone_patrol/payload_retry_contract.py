"""Retry-only ideal observation contract with consistent current-action history."""

import copy
import numpy as np

from .payload_contract import OBSERVATION_CONTRACT as LEGACY_PAYLOAD_CONTRACT
from .payload_contract import actor_observation as append_payload_observation, rotation

PROFILE = 'cf2x-two-native-ball-root-aligned-v3'
OBSERVATION_CONTRACT = copy.deepcopy(LEGACY_PAYLOAD_CONTRACT)
OBSERVATION_CONTRACT.update(
    id='payload29-ideal-world-current-action-v2',
    legacy_unchanged_columns=[0, 13],
    changed_history_columns=[13, 17],
    history='Most recently applied clipped four motor actions; zero after episode reset; preserved on retarget',
    action_timing={
        'motor_application': 'Current clipped action is applied immediately for this control interval',
        'next_observation': 'Measured post-step state and most recently applied action',
        'retarget': 'Refresh target-relative fields without changing physical state or action history',
        'reset': 'Zero action history before constructing the new episode observation',
    },
)


def _vector(value, size):
    v = np.asarray(value, dtype=float)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError('Expected a finite observation vector')
    return v


def actor_observation(sample, target, last_applied_actions):
    """Read only measured state and the applied action, never episode parameters."""
    target, actions = _vector(target, 3), _vector(last_applied_actions, 4)
    if np.any(np.abs(actions) > 1):
        raise ValueError('Observation history must contain actual clipped motor commands')
    r = rotation(sample['drone_quaternion_wxyz'])
    legacy = np.concatenate([
        np.clip((target-_vector(sample['drone_xyz'], 3))/3, -1, 1),
        _vector(sample['drone_quaternion_wxyz'], 4),
        np.clip(r.T @ _vector(sample['drone_velocity'], 3)/3, -1, 1),
        np.clip(r.T @ _vector(sample['drone_angular_velocity'], 3)/3.14159, -1, 1),
        actions,
    ])
    return append_payload_observation(legacy, sample)


def require_checkpoint_contract(embedded, actual):
    if embedded != OBSERVATION_CONTRACT or actual != OBSERVATION_CONTRACT:
        raise ValueError('Retry observation/action-history semantics mismatch; 29 fields alone are insufficient')
