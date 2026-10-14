"""Shared neural candidate scoring for MaskablePPO; no inference-time planner."""

import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from .objective import schema_for
from .scenario import generate_scenario


class CandidateNetwork(nn.Module):
    """Encode observed directed neighbor relations with permutation-equivariant weights."""

    def __init__(self, n):
        super().__init__()
        n = int(n)
        self.n = n
        self.fields = schema_for(generate_scenario(0, 'training', num_checkpoints=n))['fields']
        self.relations = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.nodes = nn.Sequential(nn.Linear(44, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.latent_dim_pi, self.latent_dim_vf = n, 64

    def field(self, features, name):
        start, stop = self.fields[name]
        return features[:, start:stop]

    def encode(self, features):
        n = self.n
        visited = self.field(features, 'visited')
        mask = 1 - visited
        current = self.field(features, 'current_onehot')
        coords = self.field(features, 'coords').reshape(-1, n + 1, 3)
        distances = self.field(features, 'distance_matrix').reshape(-1, n + 1, n + 1)
        edges = self.field(features, 'edge_costs').reshape(-1, n + 1, n + 1)
        priorities = self.field(features, 'priorities')
        service = self.field(features, 'service')
        count = mask.sum(1, keepdim=True).clamp_min(1)
        current_xyz = (coords * current[:, :, None]).sum(1)
        d_current = (distances * current[:, :, None]).sum(1)[:, 1:]
        e_current = (edges * current[:, :, None]).sum(1)[:, 1:]
        pair_mask = mask[:, None, :] * (1 - torch.eye(n, device=features.device)[None])
        pair_features = torch.stack([distances[:, 1:, 1:], edges[:, 1:, 1:],
                                     priorities[:, None, :].expand(-1, n, -1),
                                     service[:, None, :].expand(-1, n, -1)], dim=-1)
        relations = (self.relations(pair_features) * pair_mask[:, :, :, None]).sum(2)
        relations = relations / pair_mask.sum(2, keepdim=True).clamp_min(1)
        globals_ = torch.cat([count / n, (priorities * mask).sum(1, keepdim=True) / n,
                              self.field(features, 'elapsed'), self.field(features, 'constants')[:, 3:4]], dim=1)
        local = torch.cat([torch.stack([visited, priorities, service, d_current, e_current], dim=-1),
                           coords[:, 1:] - current_xyz[:, None],
                           globals_[:, None].expand(-1, n, -1), relations], dim=-1)
        nodes = self.nodes(local)
        pooled = (nodes * mask[:, :, None]).sum(1) / count
        return nodes, pooled

    def forward(self, features):
        nodes, pooled = self.encode(features)
        logits = self.score(torch.cat([nodes, pooled[:, None].expand(-1, self.n, -1)], dim=-1)).squeeze(-1)
        return logits, self.critic(pooled)

    def forward_actor(self, features):
        return self.forward(features)[0]

    def forward_critic(self, features):
        return self.forward(features)[1]


class CandidatePolicy(MaskableActorCriticPolicy):
    def _build(self, lr_schedule):
        self.mlp_extractor = CandidateNetwork(self.action_space.n)
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(64, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
