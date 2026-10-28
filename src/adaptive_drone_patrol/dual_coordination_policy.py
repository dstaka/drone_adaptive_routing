"""Standalone central joint PPO primitives; no environment or qualified physics.

Inputs are batched float tensors: goals [B,10,G], drones [B,2,D], and
 global_features [B,F]. Defaults below name every observable feature. Positions
are world coordinates; quaternion order is wxyz. The caller must apply frozen
physical scaling before input (defaults are unit scale); no mass, sampled box
size, future event or teacher/action labels belong in these tensors. Custom
feature dimensions require a separately frozen, audited input adapter.

Permutation augmentation supports the named default schema. A dense actor head
is deliberately not advertised as permutation equivariant. Checkpoints recover
policy/optimizer/RNG state; physical recovery additionally requires complete
caller-owned environment AND backend payloads, and their successful restoration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .dual_coordination_schema import decode_action, encode_action

GOAL_FEATURES = ('x', 'y', 'z', 'priority', 'unassigned', 'assigned', 'servicing', 'done',
                 'owner_none', 'owner_1', 'owner_2')
DRONE_FEATURES = tuple(
    f'{group}_{axis}' for group, axes in (
        ('position', 'xyz'), ('velocity', 'xyz'), ('quaternion', 'wxyz'), ('angular_velocity', 'xyz'),
        ('payload_position', 'xyz'), ('payload_quaternion', 'wxyz'), ('payload_velocity', 'xyz'),
        ('payload_angular_velocity', 'xyz'), ('link_angle', 'xy'), ('link_rate', 'xy'),
        ('reference_position', 'xyz'), ('reference_velocity', 'xyz'), ('reference_acceleration', 'xyz'),
    ) for axis in axes
) + tuple(f'target_{i}' for i in range(11)) + tuple(f'last_target_{i}' for i in range(11)) + (
    'commitment_age', 'leg_age', 'service_dwell', 'available', 'previous_supervisor_hold')
GLOBAL_FEATURES = ('team_time', 'last_priority_1', 'last_priority_2',
                   'relative_x', 'relative_y', 'relative_z', 'relative_vx', 'relative_vy', 'relative_vz',
                   'predicted_clearance', 'predicted_conflict')


@dataclass(frozen=True)
class PolicyConfig:
    goal_dim: int = len(GOAL_FEATURES)
    drone_dim: int = len(DRONE_FEATURES)
    global_dim: int = len(GLOBAL_FEATURES)
    hidden: int = 128
    heads: int = 4
    blocks: int = 2

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in asdict(self).values()) or self.hidden % self.heads:
            raise ValueError('Positive dimensions and hidden width divisible by heads required')


class JointActorCritic(nn.Module):
    def __init__(self, config=PolicyConfig()):
        super().__init__()
        self.config = config
        self.goal_encoder = nn.Linear(config.goal_dim, config.hidden)
        self.drone_encoder = nn.Linear(config.drone_dim, config.hidden)
        self.global_encoder = nn.Linear(config.global_dim, config.hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(config.hidden, config.heads, dim_feedforward=config.hidden*4,
                                       dropout=0., batch_first=True, activation='gelu')
            for _ in range(config.blocks)
        ])
        self.actor = nn.Linear(13*config.hidden, 242)
        self.critic = nn.Linear(13*config.hidden, 1)
        # Frozen normalization travels in the state_dict. Input adapter applies
        # physical scales; these buffers allow explicitly supplied training stats.
        for name, size in (('goals', config.goal_dim), ('drones', config.drone_dim),
                           ('global_features', config.global_dim)):
            self.register_buffer(f'{name}_mean', torch.zeros(size))
            self.register_buffer(f'{name}_scale', torch.ones(size))

    def forward(self, observations):
        expected = {'goals': (10, self.config.goal_dim), 'drones': (2, self.config.drone_dim),
                    'global_features': (self.config.global_dim,)}
        if set(observations) != set(expected):
            raise ValueError('Exactly goals, drones and global_features are required')
        normalized = {}
        batch = None
        for name, shape in expected.items():
            x = observations[name]
            if not isinstance(x, torch.Tensor) or tuple(x.shape[1:]) != shape or x.shape[0] == 0:
                raise ValueError(f'Invalid {name} shape')
            if not x.is_floating_point() or not torch.isfinite(x).all():
                raise ValueError('Observation must contain finite floating features')
            if batch is not None and x.shape[0] != batch:
                raise ValueError('Observation batch sizes differ')
            batch = x.shape[0]
            scale = getattr(self, f'{name}_scale')
            mean = getattr(self, f'{name}_mean')
            if not torch.isfinite(scale).all() or not torch.isfinite(mean).all() or (scale <= 0).any():
                raise ValueError('Normalization must be finite with positive scales')
            normalized[name] = (x-mean)/scale
        tokens = torch.cat((self.goal_encoder(normalized['goals']), self.drone_encoder(normalized['drones']),
                            self.global_encoder(normalized['global_features']).unsqueeze(1)), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        features = tokens.flatten(1)
        logits, value = self.actor(features), self.critic(features).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise ValueError('Policy output must be finite')
        return logits, value

    def distribution(self, observations, mask):
        logits, value = self(observations)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != logits.shape:
            raise ValueError('Action mask must be bool with shape [batch,242]')
        if not mask.any(dim=-1).all():
            raise ValueError('Each world must have at least one valid action')
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf)), value

    def act(self, observations, mask, *, deterministic=False):
        distribution, value = self.distribution(observations, mask)
        actions = distribution.logits.argmax(-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), value


def action_permutation(goal_order, *, swap_drones=False):
    """Return old-action -> new-action indices; goal_order is new -> old."""
    if len(goal_order) != 10 or any(type(i) is not int for i in goal_order) or sorted(goal_order) != list(range(10)):
        raise ValueError('Goal order must be a permutation of ten integer indices')
    inverse = np.argsort(goal_order).tolist()+[10]
    result = []
    for i in range(242):
        t1, t2, priority = decode_action(i)
        t1, t2 = inverse[t1], inverse[t2]
        if swap_drones:
            t1, t2, priority = t2, t1, 3-priority
        result.append(encode_action(t1, t2, priority))
    return torch.tensor(result, dtype=torch.long)


def permute_observation(observations, mask, goal_order, *, swap_drones=False):
    """Augment the named default feature schema, including embedded ID fields.

    A transformed mask preserves the original valid set exactly. Canonical
    redundant-priority masking must not subsequently replace this set before
    computing the associated PPO likelihood.
    """
    permutation = action_permutation(goal_order, swap_drones=swap_drones).to(mask.device)
    dimensions = (len(GOAL_FEATURES), len(DRONE_FEATURES), len(GLOBAL_FEATURES))
    if tuple(observations[k].shape[-1] for k in ('goals', 'drones', 'global_features')) != dimensions:
        raise ValueError('Permutation requires the named default feature schema')
    changed = {k: v.clone() for k, v in observations.items()}
    changed['goals'] = changed['goals'][:, goal_order]
    if swap_drones:
        changed['drones'] = changed['drones'][:, [1, 0]]
        changed['goals'][..., [9, 10]] = changed['goals'][..., [10, 9]]
        changed['global_features'][..., [1, 2]] = changed['global_features'][..., [2, 1]]
        changed['global_features'][..., 3:9] *= -1
    for name in ('target_0', 'last_target_0'):
        start = DRONE_FEATURES.index(name)
        changed['drones'][..., start:start+10] = changed['drones'][..., [start+i for i in goal_order]]
    transformed_mask = torch.empty_like(mask)
    transformed_mask[..., permutation] = mask
    return changed, transformed_mask


def generalized_advantage(rewards, values, terminals, last_values, *, gamma=.999, gae_lambda=.95):
    """[time,world] GAE. terminals includes failure and deadline truncation.

    An ordinary rollout boundary is not terminal and uses last_values. Completed
    or failed episode boundaries cut both bootstrapping and the GAE recurrence.
    """
    if rewards.ndim != 2 or rewards.shape != values.shape or terminals.shape != values.shape:
        raise ValueError('GAE requires equally shaped [time,world] tensors')
    if terminals.dtype != torch.bool or last_values.shape != values.shape[1:]:
        raise ValueError('Boolean terminals and one bootstrap value per world required')
    if not all(torch.isfinite(x).all() for x in (rewards, values, last_values)):
        raise ValueError('GAE inputs must be finite')
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_values)
    next_value = last_values
    for t in reversed(range(len(rewards))):
        live = (~terminals[t]).to(values.dtype)
        delta = rewards[t]+gamma*live*next_value-values[t]
        carry = delta+gamma*gae_lambda*live*carry
        advantages[t] = carry
        next_value = values[t]
    return advantages, advantages+values


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip: float = .2
    gamma: float = .999
    gae_lambda: float = .95
    entropy_coefficient: float = .01
    value_coefficient: float = .5
    gradient_norm: float = .5
    epochs: int = 5
    minibatch: int = 512
    transitions_per_update: int = 4096


def ppo_loss(log_prob, old_log_prob, advantages, values, returns, entropy, config=PPOConfig()):
    ratio = (log_prob-old_log_prob).exp()
    policy = -torch.minimum(ratio*advantages, ratio.clamp(1-config.clip, 1+config.clip)*advantages).mean()
    value = (values-returns).square().mean()
    entropy_mean = entropy.mean()
    return dict(total=policy+config.value_coefficient*value-config.entropy_coefficient*entropy_mean,
                policy=policy, value=value, entropy=entropy_mean)


@dataclass
class TrainingBudget:
    seed: int
    transitions: int = 0
    updates: int = 0

    def __post_init__(self):
        if self.seed not in (42, 43, 44):
            raise ValueError('Production seed must be 42, 43 or 44')
        if (type(self.transitions) is not int or type(self.updates) is not int or
                not 0 <= self.transitions <= 1048576 or self.transitions != self.updates*4096):
            raise ValueError('Inconsistent production budget counters')

    @property
    def stage(self):
        return 1 if self.transitions < 262144 else 2 if self.transitions < 983040 else 3

    @property
    def selectable(self):
        return self.transitions == 1048576

    def check_update(self, transitions, *, backend):
        if type(transitions) is not int or transitions != 4096:
            raise ValueError('One update requires exactly 4096 aggregate joint transitions')
        if self.transitions+transitions > 1048576:
            raise ValueError('Production transition budget exhausted')
        if not isinstance(backend, str) or not backend:
            raise ValueError('Explicit qualified backend identifier required')
        if self.stage == 3 and backend != 'native-mujoco':
            raise ValueError('Stage 3 requires native-mujoco')

    def record_update(self, transitions, *, backend):
        self.check_update(transitions, backend=backend)
        self.transitions += transitions
        self.updates += 1


@dataclass
class PPOBatch:
    observations: dict
    masks: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class JointPPO:
    def __init__(self, policy, *, seed, config=PPOConfig()):
        if config != PPOConfig():
            raise ValueError('Production PPO settings are frozen by the approved campaign')
        self.policy, self.config = policy, config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.budget = TrainingBudget(seed)

    def update(self, batch, *, backend):
        n = len(batch.actions)
        self.budget.check_update(n, backend=backend)
        if batch.actions.dtype != torch.long or batch.actions.shape != (n,):
            raise ValueError('Actions must be a flat int64 tensor')
        if batch.masks.dtype != torch.bool or batch.masks.shape != (n, 242):
            raise ValueError('Masks must be bool [4096,242]')
        if (batch.actions < 0).any() or (batch.actions >= 242).any():
            raise ValueError('Action outside categorical domain')
        if not batch.masks.gather(1, batch.actions[:, None]).all():
            raise ValueError('Rollout includes a masked action')
        for x in (batch.old_log_probs, batch.advantages, batch.returns):
            if x.shape != (n,) or not torch.isfinite(x).all() or x.requires_grad:
                raise ValueError('Rollout scalar fields must be finite detached vectors')
        # Validate all observations before any optimizer mutation.
        with torch.no_grad():
            for start in range(0, n, self.config.minibatch):
                self.policy.distribution({k: v[start:start+512] for k, v in batch.observations.items()},
                                         batch.masks[start:start+512])
        advantages = (batch.advantages-batch.advantages.mean())/(batch.advantages.std(unbiased=False)+1e-8)
        metrics = dict(total=0., policy=0., value=0., entropy=0., minibatches=0)
        self.policy.train()
        for _ in range(self.config.epochs):
            order = torch.randperm(n, device=batch.actions.device)
            for ids in order.split(self.config.minibatch):
                distribution, values = self.policy.distribution({k: v[ids] for k, v in batch.observations.items()},
                                                               batch.masks[ids])
                losses = ppo_loss(distribution.log_prob(batch.actions[ids]), batch.old_log_probs[ids],
                                  advantages[ids], values, batch.returns[ids], distribution.entropy(), self.config)
                if not torch.isfinite(losses['total']):
                    raise ValueError('Nonfinite PPO loss; recover the last complete checkpoint')
                self.optimizer.zero_grad(set_to_none=True)
                losses['total'].backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.gradient_norm, error_if_nonfinite=True)
                self.optimizer.step()
                for name, loss in losses.items():
                    metrics[name] += loss.detach().item()
                metrics['minibatches'] += 1
        self.budget.record_update(n, backend=backend)
        for key in ('total', 'policy', 'value', 'entropy'):
            metrics[key] /= metrics['minibatches']
        return metrics

    def save(self, path, *, environment_state=None, backend_state=None):
        """Trusted local pickle checkpoint; payload completeness is caller-owned.

        Saves only complete PPO-update counters. Mid-rollout continuation requires
        the caller to include the rollout buffer/cursor in environment_state.
        """
        state = dict(schema='dual-policy-v1', policy_config=asdict(self.policy.config),
                     ppo_config=asdict(self.config), policy=self.policy.state_dict(),
                     optimizer=self.optimizer.state_dict(), budget=asdict(self.budget),
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     environment_state=environment_state, backend_state=backend_state,
                     physical_resume_available=environment_state is not None and backend_state is not None,
                     selectable=self.budget.selectable,
                     versions=dict(torch=torch.__version__, numpy=np.__version__),
                     parameter_count=sum(p.numel() for p in self.policy.parameters()))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, *, device='cpu'):
        """Load only a trusted checkpoint; restore supplied physics externally."""
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('schema') != 'dual-policy-v1':
            raise ValueError('Incompatible joint policy schema')
        policy = JointActorCritic(PolicyConfig(**state['policy_config'])).to(device)
        policy.load_state_dict(state['policy'])
        trainer = cls(policy, seed=state['budget']['seed'], config=PPOConfig(**state['ppo_config']))
        trainer.optimizer.load_state_dict(state['optimizer'])
        trainer.budget = TrainingBudget(**state['budget'])
        if state['selectable'] != trainer.budget.selectable:
            raise ValueError('Checkpoint selection flag disagrees with finite budget')
        if state['cuda_rng'] is not None:
            if not torch.cuda.is_available() or len(state['cuda_rng']) != torch.cuda.device_count():
                raise ValueError('Exact RNG resume requires the saved CUDA device topology')
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        return trainer, {k: state[k] for k in ('environment_state', 'backend_state', 'physical_resume_available')}
