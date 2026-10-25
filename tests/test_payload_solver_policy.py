"""Deterministic actor transfer preserves the measured 29-field contract."""

import numpy as np
import pytest
import torch
from test_payload_retry_learning import parent_state
from adaptive_drone_patrol.payload_retry_checkpoint import transfer_state


def test_initial_actor_matches_parent_prefix_and_ignores_new_zero_columns():
    from adaptive_drone_patrol.payload_solver_policy import mean_policy

    parent = parent_state()
    initial = transfer_state(parent)
    policy = mean_policy(initial["actor_state_dict"])
    obs = np.random.default_rng(8191).normal(size=29).astype(np.float32)
    before = torch.as_tensor(np.r_[obs[:13], [0.0] * 4], dtype=torch.float32)
    state = parent["actor_state_dict"]
    x = torch.tanh(torch.nn.functional.linear(before, state["mlp.0.weight"], state["mlp.0.bias"]))
    x = torch.tanh(torch.nn.functional.linear(x, state["mlp.2.weight"], state["mlp.2.bias"]))
    expected = torch.nn.functional.linear(x, state["mlp.4.weight"], state["mlp.4.bias"]).detach().numpy()
    assert policy(obs) == pytest.approx(expected, abs=2e-5)
    obs[13:] = 10.0
    assert policy(obs) == pytest.approx(expected, abs=2e-5)


def test_policy_rejects_wrong_size_or_nonfinite_input():
    from adaptive_drone_patrol.payload_solver_policy import mean_policy

    policy = mean_policy(transfer_state(parent_state())["actor_state_dict"])
    for obs in [np.zeros(17), np.full(29, np.nan)]:
        with pytest.raises(ValueError):
            policy(obs)
