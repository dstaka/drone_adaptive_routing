import numpy as np
import pytest

from adaptive_drone_patrol.routing_agent import RoutingAgent, train_one
from adaptive_drone_patrol.scenario import generate_scenario
from adaptive_drone_patrol.objective import RoutingState


def test_real_masked_ppo_save_reload_and_terminal_guard(tmp_path):
    path = train_one(42, tmp_path / 'model', static_steps=256, event_steps=256, n_epochs=1,
                     num_checkpoints=3)
    agent = RoutingAgent(path)
    state = RoutingState(generate_scenario(20001, 'static'))
    while not state.terminal:
        action = agent.predict(state)
        assert state.mask()[action]
        assert np.isfinite(state.observation()).all()
        state.complete(action, state.departure(action))
    with pytest.raises(RuntimeError, match='terminal'):
        agent.predict(state)


def test_ten_action_model_contract_and_real_ipc(tmp_path):
    import sys
    from adaptive_drone_patrol.planner_ipc import PlannerClient
    path = train_one(43, tmp_path / 'ten', static_steps=32, event_steps=32,
                     n_steps=32, batch_size=16, n_epochs=1, num_checkpoints=10)
    agent = RoutingAgent(path)
    state = RoutingState(generate_scenario(20001, 'static', num_checkpoints=10))
    with PlannerClient([sys.executable, '-m', 'adaptive_drone_patrol.planner_ipc',
                        '--model', str(path), '--hash', agent.hash], agent.hash,
                       schema=agent.schema, timeout=30) as client:
        assert client.predict(state) == agent.predict(state)
        for bad in [generate_scenario(20001, 'static'),
                    generate_scenario(20001, 'static', num_checkpoints=12)]:
            with pytest.raises(ValueError, match='retrain|Retrain'):
                agent.predict(RoutingState(bad))
        state.scenario.checkpoint_ids.reverse()
        with pytest.raises(ValueError, match='retrain|Retrain'):
            client.predict(state)


def test_cli_count_conflict_rejected_before_training(tmp_path):
    import subprocess
    import sys
    config = tmp_path / 'config.yaml'
    config.write_text('num_checkpoints: 12\n')
    result = subprocess.run([sys.executable, 'scripts/train_routing.py', '--config', str(config),
                             '--num-checkpoints', '10', '--out', str(tmp_path / 'out')],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'disagree' in result.stderr


def test_model_preflight_fails_before_loading_physics(tmp_path):
    import subprocess
    import sys
    import yaml
    path = train_one(44, tmp_path / 'preflight', static_steps=16, event_steps=16,
                     n_steps=16, batch_size=8, n_epochs=1, num_checkpoints=3)
    from adaptive_drone_patrol.util import sha256
    scenario = tmp_path / 'scenario.yaml'
    scenario.write_text(yaml.safe_dump(generate_scenario(20001, 'static', num_checkpoints=10).to_dict()))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'planner_python': sys.executable,
                                      'routing_model': str(path), 'routing_model_hash': sha256(path)}))
    result = subprocess.run([sys.executable, 'scripts/run_demo_scenarios.py', '--worker',
                             '--config', str(config), '--mode', 'genesis', '--scenario', str(scenario),
                             '--out', str(tmp_path / 'flight')], capture_output=True, text=True)
    assert result.returncode != 0 and 'retrain' in result.stderr.lower()
    assert 'genesis' not in result.stderr.lower()
    assert not (tmp_path / 'flight').exists()


def test_shared_candidate_policy_is_equivariant_and_saved(tmp_path):
    import torch
    path = train_one(42, tmp_path / 'shared', static_steps=16, event_steps=16,
                     n_steps=16, batch_size=8, n_epochs=1, num_checkpoints=10,
                     policy_arch='candidate')
    agent = RoutingAgent(path)
    assert type(agent.model.policy).__name__ == 'CandidatePolicy'
    s = generate_scenario(20001, 'static', num_checkpoints=10)
    state = RoutingState(s)
    permutation = [3, 0, 5, 2, 8, 4, 1, 7, 6, 9]
    permuted = generate_scenario(20001, 'static', num_checkpoints=10)
    nodes = [0] + [i + 1 for i in permutation]
    permuted.coords = np.asarray(s.coords)[nodes].tolist()
    permuted.priorities = np.asarray(s.priorities)[permutation].tolist()
    permuted.multipliers = np.asarray(s.multipliers)[np.ix_(nodes, nodes)].tolist()
    with torch.no_grad():
        a = agent.model.policy.get_distribution(torch.tensor(state.observation()[None]), action_masks=[state.mask()]).distribution.probs
        st = RoutingState(permuted)
        b = agent.model.policy.get_distribution(torch.tensor(st.observation()[None]), action_masks=[st.mask()]).distribution.probs
    assert torch.allclose(a[:, permutation], b, atol=1e-6)


def test_training_only_teacher_records_inputs_and_actual_steps(tmp_path):
    import gzip
    import json
    path = train_one(43, tmp_path / 'teacher', static_steps=16, event_steps=16,
                     n_steps=16, batch_size=8, n_epochs=1, num_checkpoints=10,
                     policy_arch='candidate', warmstart_cases=8, warmstart_epochs=1)
    record = json.loads((tmp_path / 'teacher_run/training.json').read_text())
    assert record['total_steps_actual'] == 32
    assert record['warmstart']['cases'] == 8
    with gzip.open(tmp_path / 'teacher_run/teacher.jsonl.gz', 'rt') as stream:
        for line in stream:
            item = json.loads(line)
            assert 0 <= item['snapshot']['scenario']['seed'] < 10000
            assert item['exact']['proven'] and item['exact']['status'] == 'OPTIMAL'
    assert RoutingAgent(path).num_checkpoints == 10


def test_legacy_agent_rejects_objective_changes_and_nonboolean_masks():
    from adaptive_drone_patrol.objective import SCHEMA
    agent = object.__new__(RoutingAgent)
    agent.schema, agent.legacy, agent.num_checkpoints = SCHEMA, True, 3
    s = generate_scenario(20001, 'static')
    s.alpha = 999.
    with pytest.raises(ValueError, match='retrain'):
        agent.validate_scenario(s)
    with pytest.raises(ValueError, match='mask'):
        agent.predict_features([0.] * 65, ['false', 'false', 'true'])


def test_reused_teacher_rejects_manifest_or_split_mismatch(tmp_path):
    import json
    from adaptive_drone_patrol.routing_teacher import prepare_teacher
    teacher = prepare_teacher(tmp_path / 'dataset', 10, 4)
    record = json.loads((teacher / 'teacher.json').read_text())
    record['split'] = 'static'
    (teacher / 'teacher.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='teacher|Teacher'):
        train_one(42, tmp_path / 'bad', static_steps=16, event_steps=16,
                  n_steps=16, batch_size=8, n_epochs=1, num_checkpoints=10,
                  policy_arch='candidate', warmstart_cases=4, warmstart_epochs=1,
                  teacher_directory=teacher)


def test_raw_features_reject_mask_and_objective_disagreement():
    from adaptive_drone_patrol.objective import schema_for
    agent = object.__new__(RoutingAgent)
    state = RoutingState(generate_scenario(20001, 'static', num_checkpoints=10))
    agent.schema, agent.legacy, agent.num_checkpoints = schema_for(state.scenario), False, 10
    mask = state.mask()
    mask[0] = False
    with pytest.raises(ValueError, match='mask'):
        agent.predict_features(state.observation(), mask)
    features = state.observation()
    features[agent.schema['fields']['constants'][0]] = 999.
    with pytest.raises(ValueError, match='objective'):
        agent.predict_features(features, state.mask())


def test_optimizer_reward_scale_preserves_raw_objective_and_masks():
    from adaptive_drone_patrol import routing_agent
    env = routing_agent.training_env(10, reward_scale=.01)
    env.reset(seed=42)
    _, reward, _, _, _ = env.step(0)
    state = env.unwrapped.state
    assert reward == pytest.approx(-.01 * state.cost)
    assert not env.get_wrapper_attr('action_masks')()[0]
    assert state.legs[0]['cost'] == state.cost


def test_cross_experiment_freeze_uses_validation_only_and_retains_three_seeds(tmp_path):
    import json
    from adaptive_drone_patrol import routing_agent
    from adaptive_drone_patrol.objective import schema_for
    from adaptive_drone_patrol.util import sha256
    schema = schema_for(generate_scenario(0, 'training', num_checkpoints=10))
    manifests = []
    for method in ['first', 'second']:
        folder = tmp_path / method
        folder.mkdir()
        models = []
        for seed in [42, 43, 44]:
            model = folder / f'seed_{seed}.zip'
            model.write_bytes(f'{method}-{seed}'.encode())
            model.with_suffix('.schema.json').write_text(json.dumps(schema))
            models.append({'seed': seed, 'path': str(model), 'sha256': sha256(model),
                           'validation': {'invalid_actions': 0, 'decision_changes': 2,
                                          'mean_cost': 10. if method == 'second' else 20.}})
        path = folder / 'models.json'
        path.write_text(json.dumps({'schema': schema, 'models': models}))
        manifests.append(path)
    result = routing_agent.freeze_validation_models(manifests, tmp_path / 'frozen')
    assert len(result['models']) == 3 and len(result['all_trials']) == 6
    assert result['selected']['seed'] == 42
    assert all('/second/' in item['path'] for item in result['models'])
    assert result['heldout_opened'] is False
