"""Recovery learning arithmetic and guards are independent of old GPU budgets."""

import pytest


def test_common_development_ladder_counts_real_environment_transitions():
    from adaptive_drone_patrol.payload_solver_training import sample_ladder

    assert sample_ladder(1, 100) == [300000, 1000000, 3000000]
    assert sample_ladder(8, 100) == [300000, 1000000, 3000000]
    with pytest.raises(ValueError):
        sample_ladder(128, 100)


def test_reward_has_explicit_load_settling_terms_without_changing_actor_fields():
    from adaptive_drone_patrol.payload_solver_training import reward_components

    metrics = dict(
        drone_error_m=0.0,
        drone_speed_m_s=0.0,
        package_speed_m_s=0.0,
        swing_angle_deg=0.0,
        swing_rate_deg_s=0.0,
        package_spin_deg_s=0.0,
    )
    good = reward_components([0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], metrics, False, True)
    metrics["package_speed_m_s"] = 1.0
    metrics["swing_angle_deg"] = 10.0
    bad = reward_components([0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], metrics, False, False)
    assert sum(good.values()) > sum(bad.values())
    assert {"swing", "swing_rate", "package_velocity", "shared_settling"} <= good.keys()


def test_rung_mismatch_does_not_silently_round_methods_differently():
    from adaptive_drone_patrol.payload_solver_training import validate_learning_steps

    assert validate_learning_steps(300000, 1, 100) == 3000
    with pytest.raises(ValueError):
        validate_learning_steps(300001, 1, 100)


def test_training_cli_rejects_missing_physics_before_optimizer_or_outputs(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "never-training"
    p = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/train_payload_recovery.py"),
            "--qualification",
            str(tmp_path / "missing.json"),
            "--locations",
            str(tmp_path / "missing-locations.json"),
            "--parent",
            str(tmp_path / "model.pt"),
            "--config",
            str(tmp_path / "cfg.pkl"),
            "--original",
            str(tmp_path / "cf2x.urdf"),
            "--method",
            "randomized",
            "--steps",
            "20000",
            "--development",
            "--pilot",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0 and not out.exists()
    assert "qualification" in p.stderr.lower()


def test_timed_out_commands_never_count_as_shared_completions():
    import io
    import types
    import numpy as np

    torch = pytest.importorskip("torch")
    from adaptive_drone_patrol.payload_solver_training import SolverTrainingEnv
    from adaptive_drone_patrol.payload_contract import Payload

    metrics = dict(
        drone_error_m=0.0,
        drone_speed_m_s=0.0,
        package_error_m=0.0,
        package_speed_m_s=0.0,
        swing_angle_deg=0.0,
        swing_rate_deg_s=0.0,
        package_spin_deg_s=0.0,
        package_floor_m=0.6,
        drone_floor_m=0.98,
        drone_roll_deg=0.0,
        drone_pitch_deg=0.0,
    )
    row = dict(
        drone_xyz=[0.0, 0.0, 1.0],
        drone_angular_velocity=[0.0, 0.0, 0.0],
        metrics=metrics,
        unsafe=False,
        complete=False,
    )
    core = types.SimpleNamespace(
        current_sample=row, observation=np.zeros(29), step=lambda action: row, set_target=lambda *a, **k: None
    )
    env = object.__new__(SolverTrainingEnv)
    env.core = core
    env.payload = Payload(0.02700047753751278, 0.2, 0.045)
    env.trace, env.terminal = io.StringIO(), io.StringIO()
    for name in [
        "generated_steps",
        "invalid_steps",
        "total_steps",
        "episode_steps",
        "command_steps",
        "command_count",
        "episode_completed_stops",
        "completed_stops",
        "failures",
        "target_stage",
        "episode_stage",
    ]:
        setattr(env, name, 0)
    env.episode_count = 1
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    env.max_episode_length = 12000
    env.stage_counts, env.stage_experience, env.reward_totals = [0] * 6, [], {}
    env.curriculum_steps, env.pilot = 300000, True
    env.target = np.array([0.0, 0.0, 1.0])
    env._observation = lambda: None
    env._reset = lambda: pytest.fail("Ten attempted targets are not ten completed targets")
    for _ in range(9):
        env.command_steps = 799
        _, _, done, _ = env.step(torch.zeros((1, 4)))
        assert not done.item()
    row["complete"] = True
    _, _, done, _ = env.step(torch.zeros((1, 4)))
    assert not done.item()
    assert env.command_count == 10
    assert env.episode_completed_stops == 1
    assert env.completed_stops == 1
    row["complete"], row["unsafe"] = False, True
    env.episode_steps = 11999
    env._reset = lambda: None
    _, _, done, extras = env.step(torch.zeros((1, 4)))
    assert done.item()
    assert not extras["time_outs"].item()
    assert env.stage_experience[-1]["completed_target_count"] == 1


def test_changed_learning_dependency_is_rejected_before_optimizer(tmp_path):
    from adaptive_drone_patrol.payload_solver_training import require_learning_dependencies

    lock = tmp_path / "wrong.lock"
    lock.write_text("numpy==0.0.0\n")
    with pytest.raises(ValueError, match="dependency differs"):
        require_learning_dependencies(lock)


@pytest.mark.parametrize('shared,unsafe,prior_steps,want_bootstrap', [
    (True, False, 11998, False),
    (False, False, 11999, True),
    (True, False, 11999, False),
    (True, True, 11999, False),
])
def test_true_terminal_takes_precedence_over_coincident_time_limit(
        shared, unsafe, prior_steps, want_bootstrap):
    """Only truncation may add a value bootstrap at the PPO boundary."""
    import io
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np

    torch = pytest.importorskip('torch')
    from adaptive_drone_patrol.payload_contract import Payload, payload_metrics
    from adaptive_drone_patrol.payload_solver_trace import observation_sample
    from adaptive_drone_patrol.payload_solver_training import SolverTrainingEnv

    native = json.loads((Path(__file__).parent / 'data/payload_solver/native-reset.json').read_text())
    payload = Payload(0.02700047753751278, 0.2, 0.045)
    sample = observation_sample(native['actual'], payload.side_m)
    row = dict(sample, metrics=payload_metrics(sample, [0, 0, 1], payload),
               unsafe=unsafe, complete=shared)
    env = object.__new__(SolverTrainingEnv)
    # Replace only the physical step; exercise real termination/counter/log logic.
    env.core = SimpleNamespace(current_sample=sample, observation=np.zeros(29), step=lambda _: row)
    env.payload, env.target = payload, np.array([0.0, 0.0, 1.0])
    env.trace, env.terminal = io.StringIO(), io.StringIO()
    env.generated_steps = env.invalid_steps = env.total_steps = 0
    env.command_steps = env.command_count = env.failures = 0
    env.target_stage = env.episode_stage = 0
    env.episode_count, env.episode_steps = 1, prior_steps
    env.episode_completed_stops = env.completed_stops = 9
    env.max_episode_length = 12000
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    env.stage_counts, env.stage_experience, env.reward_totals = [0] * 6, [], {}
    env._observation = lambda: None
    env._reset = lambda: None

    _, _, done, extras = env.step(torch.zeros((1, 4)))

    assert done.item()
    assert extras['time_outs'].item() is want_bootstrap
    terminal = json.loads(env.terminal.getvalue())
    assert terminal['timeout'] is (prior_steps == 11999)
    assert terminal['failed'] is unsafe
    assert env.episode_completed_stops == (10 if shared and not unsafe else 9)


def test_reward_penalizes_observed_cube_spin_and_preserves_terminal_penalty():
    from adaptive_drone_patrol.payload_solver_training import reward_components, transition_reward

    metrics = dict(
        drone_error_m=0.0,
        drone_speed_m_s=0.0,
        package_speed_m_s=0.0,
        swing_angle_deg=0.0,
        swing_rate_deg_s=0.0,
        package_spin_deg_s=0.0,
    )
    low = reward_components([0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], metrics, False, False)
    metrics["package_spin_deg_s"] = 200.0
    high = reward_components([0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], metrics, False, False)
    assert high["package_spin"] < low["package_spin"]
    crashed = dict(low, crash=-200.0)
    assert transition_reward(crashed) < -199.0


def test_development_gae_adjustment_preserves_all_other_parent_ppo_fields():
    from adaptive_drone_patrol.payload_solver_training import learning_ppo_config

    parent = dict(gamma=0.99, lam=0.95, clip_param=0.2, num_mini_batches=4,
                  num_learning_epochs=5, learning_rate=0.0003, schedule='adaptive',
                  entropy_coef=0.001, max_grad_norm=1.0)
    result = learning_ppo_config(parent, gae_lambda=0.99)
    assert result['lam'] == 0.99 and result['gamma'] == 0.99
    assert result['clip_param'] == 0.2 and result['num_mini_batches'] == 4
    assert result['num_learning_epochs'] == 5 and result['learning_rate'] == 0.00003
    assert parent['lam'] == 0.95 and parent['learning_rate'] == 0.0003
    for invalid in [float('nan'), -0.1, 1.1, True]:
        with pytest.raises(ValueError):
            learning_ppo_config(parent, gae_lambda=invalid)


def test_spin_training_cost_is_bounded_without_relaxing_shared_settling():
    from adaptive_drone_patrol.payload_solver_training import reward_components
    from adaptive_drone_patrol.payload_contract import settled

    metrics = dict(drone_error_m=0.0, drone_speed_m_s=0.0, package_error_m=0.0,
        package_speed_m_s=0.0, swing_angle_deg=0.0, swing_rate_deg_s=0.0,
        package_spin_deg_s=200.0, package_floor_m=0.65, drone_floor_m=0.98,
        drone_roll_deg=0.0, drone_pitch_deg=0.0)
    a = reward_components([0,0,1],[0,0,1],[0,0,1],[0,0,0],metrics,False,False)
    metrics['package_spin_deg_s'] = 10000.0
    b = reward_components([0,0,1],[0,0,1],[0,0,1],[0,0,0],metrics,False,False)
    assert a['package_spin'] == -0.1 and b['package_spin'] == -0.1
    assert not settled(metrics)


def test_observed_vehicle_rotation_gets_direct_quadratic_training_cost():
    from adaptive_drone_patrol.payload_solver_training import reward_components

    metrics=dict(drone_error_m=0.0,drone_speed_m_s=0.0,package_speed_m_s=0.0,
        swing_angle_deg=0.0,swing_rate_deg_s=0.0,package_spin_deg_s=0.0)
    terms=reward_components([0,0,1],[0,0,1],[0,0,1],[0,3,0],metrics,False,False)
    assert terms['angular'] == pytest.approx(-0.9)


@pytest.mark.parametrize('current', [[.3,.4,1.0],[0,0,1.5]])
def test_dense_position_cost_penalizes_horizontal_and_vertical_offset_beyond_hover_peak(current):
    from adaptive_drone_patrol.payload_solver_training import reward_components

    metrics=dict(drone_error_m=.5,drone_speed_m_s=0,package_speed_m_s=0,
                 swing_angle_deg=0,swing_rate_deg_s=0,package_spin_deg_s=0)
    terms=reward_components(current,current,[0,0,1],[0,0,0],metrics,False,False)
    assert terms['position'] == pytest.approx(-.5)
    assert terms['crash'] == 0


def test_optimizer_python_source_is_part_of_actual_learning_binding(tmp_path, monkeypatch):
    import importlib.metadata as metadata
    from types import SimpleNamespace
    from adaptive_drone_patrol.payload_solver_training import require_learning_dependencies

    files = ['rsl_rl/algorithms/ppo.py', 'torch/optim/adam.py', 'torch/_C.fake.so', 'torch/version.py']
    for name in files:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    def dist(name):
        prefix = 'rsl_rl/' if name == 'rsl-rl-lib' else 'torch/'
        return SimpleNamespace(files=[p for p in files if p.startswith(prefix)],
                               locate_file=lambda p: tmp_path / p)
    monkeypatch.setattr(metadata, 'distribution', dist)
    monkeypatch.setattr(metadata, 'distributions', lambda: [
        SimpleNamespace(metadata={'Name': 'rsl-rl-lib'}, version='5.5.1'),
        SimpleNamespace(metadata={'Name': 'torch'}, version='2.14.0+cpu')])
    lock = tmp_path / 'lock.txt'
    lock.write_text('rsl-rl-lib==5.5.1\ntorch==2.14.0+cpu\n')
    before = require_learning_dependencies(lock)
    (tmp_path / 'torch/optim/adam.py').write_text('changed optimizer')
    after = require_learning_dependencies(lock)
    assert before != after
    assert before['torch_source_sha256']['torch/optim/adam.py'] != after['torch_source_sha256']['torch/optim/adam.py']


def test_longer_credit_recipe_changes_only_declared_discount_and_gae():
    from adaptive_drone_patrol.payload_solver_training import learning_ppo_config

    parent = dict(gamma=.99,lam=.95,clip_param=.2,num_learning_epochs=5)
    before = learning_ppo_config(parent)
    after = learning_ppo_config(parent,discount=.999,gae_lambda=.99)
    assert after == dict(before,gamma=.999,lam=.99)
    assert parent['gamma'] == .99 and parent['lam'] == .95
    for invalid in [float('nan'),-0.1,1.0,True]:
        with pytest.raises(ValueError):
            learning_ppo_config(parent,discount=invalid)
