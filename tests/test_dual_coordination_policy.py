"""Exercise joint policy boundaries, update mathematics and reproducible recovery."""
import importlib
import random

import numpy as np
import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(1)


def module():
    try:
        return importlib.import_module('adaptive_drone_patrol.dual_coordination_policy')
    except ModuleNotFoundError:
        pytest.fail('Standalone joint coordinator policy is not implemented')


def observation(p, n=2):
    c = p.config
    return dict(goals=torch.randn(n, 10, c.goal_dim), drones=torch.randn(n, 2, c.drone_dim),
                global_features=torch.randn(n, c.global_dim))


def test_masked_policy_determinism_gradients_and_invalid_inputs():
    m = module()
    p = m.JointActorCritic(m.PolicyConfig(goal_dim=5, drone_dim=7, global_dim=3))
    obs = observation(p)
    mask = torch.zeros(2, 242, dtype=torch.bool)
    mask[:, 241] = True
    action, logp, value = p.act(obs, mask, deterministic=True)
    assert action.tolist() == [241, 241]
    assert torch.equal(logp, torch.zeros(2))
    assert value.shape == (2,)
    assert torch.equal(p.act(obs, mask, deterministic=True)[2], value)
    mask[:, 24] = True
    loss = -p.distribution(obs, mask)[0].log_prob(torch.tensor([24, 241])).mean()
    loss.backward()
    assert p.goal_encoder.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='valid action'):
        p.act(obs, torch.zeros_like(mask))
    bad = {**obs, 'goals': obs['goals'].clone()}
    bad['goals'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        p.act(bad, mask)
    with pytest.raises(ValueError, match='bool'):
        p.act(obs, mask.float())


def test_gae_cuts_terminal_bootstrap_and_episode_carry():
    m = module()
    rewards = torch.tensor([[1., 2.], [3., 4.]])
    values = torch.tensor([[10., 20.], [30., 40.]])
    terminated = torch.tensor([[True, False], [False, True]])
    adv, ret = m.generalized_advantage(rewards, values, terminated, torch.tensor([50., 60.]), gamma=1., gae_lambda=1.)
    assert adv.tolist() == [[-9., -14.], [23., -36.]]
    assert ret.tolist() == [[1., 6.], [53., 4.]]


def test_clipped_ppo_loss_hand_calculation():
    m = module()
    result = m.ppo_loss(torch.log(torch.tensor([2., .5])), torch.zeros(2),
                        torch.tensor([1., -1.]), torch.tensor([2., 0.]),
                        torch.tensor([0., 2.]), torch.tensor([.5, .5]))
    # clipped surrogate mean=(1.2 - .8)/2=.2; value MSE=4; entropy=.5
    assert result['policy'].item() == pytest.approx(-.2)
    assert result['value'].item() == pytest.approx(4.)
    assert result['total'].item() == pytest.approx(1.795)


def test_budget_stages_native_tail_and_terminal_selection():
    m = module()
    b = m.TrainingBudget(seed=42)
    assert b.stage == 1 and not b.selectable
    for _ in range(64):
        b.record_update(4096, backend='qualified-gpu')
    assert b.stage == 2 and not b.selectable
    for _ in range(176):
        b.record_update(4096, backend='qualified-gpu')
    assert b.stage == 3
    with pytest.raises(ValueError, match='native'):
        b.record_update(4096, backend='qualified-gpu')
    for _ in range(16):
        b.record_update(4096, backend='native-mujoco')
    assert b.transitions == 1048576 and b.updates == 256 and b.selectable
    with pytest.raises(ValueError, match='budget'):
        b.record_update(4096, backend='native-mujoco')
    with pytest.raises(ValueError, match='seed'):
        m.TrainingBudget(seed=45)
    with pytest.raises(ValueError, match='4096'):
        m.TrainingBudget(seed=43).record_update(8192, backend='native-mujoco')


def test_permutation_moves_actions_masks_and_embedded_identity_features():
    m = module()
    p = m.JointActorCritic()
    obs = observation(p, 1)
    order = [2, 0, 1, 3, 4, 5, 6, 7, 8, 9]  # new index -> old index
    mask = torch.zeros(1, 242, dtype=torch.bool)
    mask[0, (0*11+2)*2] = True  # old (0,2,priority1)
    changed, moved = m.permute_observation(obs, mask, order, swap_drones=True)
    assert moved.nonzero().tolist() == [[0, (0*11+1)*2+1]]  # new (0,1,priority2)
    assert torch.equal(changed['goals'][0, 1, :8], obs['goals'][0, 0, :8])
    assert changed['goals'][0, 1, 9] == obs['goals'][0, 0, 10]
    assert changed['drones'][0, 0, 0] == obs['drones'][0, 1, 0]
    target = m.DRONE_FEATURES.index('target_0')
    assert changed['drones'][0, 0, target+1] == obs['drones'][0, 1, target]
    restored, restored_mask = m.permute_observation(changed, moved, np.argsort(order).tolist(), swap_drones=True)
    for key in obs:
        assert torch.equal(restored[key], obs[key])
    assert torch.equal(restored_mask, mask)
    permutation = m.action_permutation(order, swap_drones=True)
    assert sorted(permutation.tolist()) == list(range(242))
    assert permutation[240] == 241  # WAIT WAIT priority swap


def test_real_ppo_update_and_complete_rng_optimizer_recovery(tmp_path):
    m = module()
    torch.manual_seed(42)
    p = m.JointActorCritic(m.PolicyConfig(goal_dim=2, drone_dim=2, global_dim=2))
    trainer = m.JointPPO(p, seed=42)
    obs = observation(p, 4096)
    mask = torch.zeros(4096, 242, dtype=torch.bool)
    mask[:, :4] = True
    with torch.no_grad():
        actions, old_logp, _ = p.act(obs, mask)
    initial = p.actor.weight.detach().clone()
    batch = m.PPOBatch(obs, mask, actions, old_logp, torch.randn(4096), torch.randn(4096))
    metrics = trainer.update(batch, backend='native-mujoco')
    assert metrics['minibatches'] == 40 and trainer.budget.transitions == 4096
    assert not torch.equal(initial, p.actor.weight)
    path = tmp_path/'checkpoint.pt'
    trainer.save(path, environment_state={'tick': 200}, backend_state={'engine': 'native-mujoco'})
    expected = (torch.rand(3), np.random.random(3), random.random())
    restored, payload = m.JointPPO.load(path)
    assert payload['physical_resume_available'] is True
    assert payload['environment_state'] == {'tick': 200}
    actual = (torch.rand(3), np.random.random(3), random.random())
    assert torch.equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1]) and expected[2] == actual[2]
    assert restored.budget.transitions == 4096
    assert restored.optimizer.state_dict()['state'].keys() == trainer.optimizer.state_dict()['state'].keys()
    assert torch.equal(restored.policy.act(obs, mask, deterministic=True)[0], p.act(obs, mask, deterministic=True)[0])
    # Recovery includes optimizer moments and RNG needed for the next identical update.
    trainer.update(batch, backend='native-mujoco')
    expected_weights = p.actor.weight.detach().clone()
    restored, _ = m.JointPPO.load(path)
    torch.rand(3)  # same draws used above, before uninterrupted update
    restored.update(batch, backend='native-mujoco')
    assert torch.equal(expected_weights, restored.policy.actor.weight)
    restored.save(path)
    _, payload = m.JointPPO.load(path)
    assert payload['physical_resume_available'] is False
