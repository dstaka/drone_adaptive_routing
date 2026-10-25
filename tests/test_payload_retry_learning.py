"""Pure CPU initialization/timing tests; fake banks are never flight evidence."""

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT/'scripts/train_payload_retry.py'
PROFILE = 'cf2x-two-native-ball-root-aligned-v3'
PARENT_SHA = '26658a31560f71c36fd9e9605b15d8e4f7b9cae8174c1465fd12dcd0d10b3431'
CFG_SHA = '35c6b0be8db90e4e799d453d447a484275317f85098a22cbdf04c19c84056e55'


def module(name):
    path = ROOT/f'src/adaptive_drone_patrol/{name}.py'
    assert path.is_file(), f'{name} implementation is required'
    return importlib.import_module(f'adaptive_drone_patrol.{name}')


def parent_state():
    result = {'iter': 4296, 'optimizer_state_dict': {'stale': True}}
    for role, outputs in [('actor', 4), ('critic', 1)]:
        result[role+'_state_dict'] = {
            'mlp.0.weight': torch.arange(128*17, dtype=torch.float32).reshape(128, 17)/1000,
            'mlp.0.bias': torch.arange(128, dtype=torch.float32),
            'mlp.2.weight': torch.eye(128), 'mlp.2.bias': torch.zeros(128),
            'mlp.4.weight': torch.ones(outputs, 128), 'mlp.4.bias': torch.ones(outputs),
        }
    result['actor_state_dict']['distribution.std_param'] = torch.ones(4)
    return result


def sample(initial=None, side=.045):
    from adaptive_drone_patrol.payload_contract import rotation
    s = copy.deepcopy(initial) if initial else dict(
        drone_xyz=[0., 0., 1.], drone_quaternion_wxyz=[1., 0., 0., 0.],
        package_xyz=[0., 0., .6575], package_quaternion_wxyz=[1., 0., 0., 0.],
        drone_velocity=[0., 0., 0.], drone_angular_velocity=[0., 0., 0.],
        package_velocity=[0., 0., 0.], package_angular_velocity=[0., 0., 0.])
    for label, body, offset in [('a', 'drone', [0, 0, -.02]), ('b', 'package', [0, 0, side/2])]:
        rotated = rotation(s[body+'_quaternion_wxyz']) @ offset
        s['attachment_'+label+'_world'] = (np.array(s[body+'_xyz'])+rotated).tolist()
        s['attachment_'+label+'_velocity'] = (np.array(s[body+'_velocity'])+
                np.cross(s[body+'_angular_velocity'], rotated)).tolist()
    return s


def test_current_action_observation_excludes_hidden_parameters_and_keeps_legacy_units():
    contract = module('payload_retry_contract')
    s = sample()
    s['drone_velocity'] = [3., -3., 6.]
    s['drone_angular_velocity'] = [3.14159, 0., -6.28318]
    action = [.1, -.2, .3, -.4]
    obs = contract.actor_observation(s, [3., 0., 1.], action)
    assert obs.shape == (29,)
    assert obs[:13] == pytest.approx([1, 0, 0, 1, 0, 0, 0, 1, -1, 1, 1, 0, -1])
    assert obs[13:17] == pytest.approx(action)
    assert obs[17:20] == pytest.approx([0, 0, -1])
    s.update(mass=99, side_m=.060, inertia=[100]*3, seed=123, geometry_id=9)
    assert np.array_equal(obs, contract.actor_observation(s, [3., 0., 1.], action))
    assert contract.OBSERVATION_CONTRACT['id'] == 'payload29-ideal-world-current-action-v2'
    assert contract.OBSERVATION_CONTRACT['hidden_parameters'] == []


def test_stale_twenty_nine_field_contract_is_rejected_without_changing_legacy():
    contract = module('payload_retry_contract')
    from adaptive_drone_patrol.payload_contract import OBSERVATION_CONTRACT as old
    prior = copy.deepcopy(old)
    with pytest.raises(ValueError):
        contract.require_checkpoint_contract(old, contract.OBSERVATION_CONTRACT)
    with pytest.raises(ValueError):
        contract.require_checkpoint_contract(contract.OBSERVATION_CONTRACT, old)
    contract.require_checkpoint_contract(contract.OBSERVATION_CONTRACT, contract.OBSERVATION_CONTRACT)
    assert old == prior and old['id'] == 'payload29-ideal-world-v1'


def test_transfer_zeroes_changed_history_and_payload_but_preserves_other_weights():
    checkpoint = module('payload_retry_checkpoint')
    parent = parent_state()
    before = copy.deepcopy(parent)
    out = checkpoint.transfer_state(parent)
    assert out['iter'] == 0 and 'optimizer_state_dict' not in out
    assert out['infos']['physics_signature'] == PROFILE
    for kind in ('actor_state_dict', 'critic_state_dict'):
        assert out[kind]['mlp.0.weight'].shape == (128, 29)
        assert torch.equal(out[kind]['mlp.0.weight'][:, :13], parent[kind]['mlp.0.weight'][:, :13])
        assert torch.count_nonzero(out[kind]['mlp.0.weight'][:, 13:]) == 0
        for key in parent[kind]:
            if key != 'mlp.0.weight':
                assert torch.equal(out[kind][key], parent[kind][key])
            assert torch.equal(parent[kind][key], before[kind][key])
    info = checkpoint.initialization_comparison(parent, out)
    assert info['unchanged_kinematic_columns_equal']
    assert info['changed_history_columns_zero'] and info['payload_columns_zero']
    assert info['other_tensors_equal']
    assert info['parent_actor_tensor_sha256'] != info['initial_actor_tensor_sha256']
    checkpoint.validate_retry_checkpoint(out)


@pytest.mark.parametrize('bad', ['wrong_shape', 'stale_v1', 'nonfinite', 'wrong_profile'])
def test_incompatible_or_nonfinite_checkpoint_is_rejected(bad):
    checkpoint = module('payload_retry_checkpoint')
    parent = parent_state()
    if bad == 'wrong_shape':
        parent['actor_state_dict']['mlp.0.weight'] = torch.zeros(128, 29)
        with pytest.raises(ValueError):
            checkpoint.transfer_state(parent)
        return
    out = checkpoint.transfer_state(parent)
    if bad == 'stale_v1':
        from adaptive_drone_patrol.payload_contract import OBSERVATION_CONTRACT
        out['infos']['observation_contract'] = OBSERVATION_CONTRACT
    elif bad == 'nonfinite':
        out['critic_state_dict']['mlp.2.weight'][0, 0] = float('nan')
    else:
        out['infos']['physics_signature'] = 'payload-articulation-v1-isolated-scenes'
    with pytest.raises(ValueError):
        checkpoint.validate_retry_checkpoint(out)


@pytest.fixture
def training_env(tmp_path, monkeypatch):
    def build():
        training = module('payload_retry_training')
        from adaptive_drone_patrol.payload_training import PayloadTrainingEnv
        def unsafe_old_init(*args, **kwargs):
            raise AssertionError('Failed Euler initializer must never execute')
        monkeypatch.setattr(PayloadTrainingEnv, '__init__', unsafe_old_init)
        monkeypatch.setitem(sys.modules, 'genesis', types.SimpleNamespace(device='cpu'))
        # The isolated CPU-routing lock deliberately omits flight-only TensorDict.
        # Its container boundary is replaced; all observations and step logic remain real.
        class CpuTensorDict(dict):
            def __init__(self, data, batch_size):
                super().__init__(data)
                self.batch_size = batch_size
        monkeypatch.setitem(sys.modules, 'tensordict', types.SimpleNamespace(TensorDict=CpuTensorDict))
        class CpuBody:
            def __init__(self, bank, name):
                self.bank, self.name = bank, name
            def _read(self, field):
                key = self.name+'_'+field
                fallback = {'xyz': (np.array(self.bank.current['drone_xyz'])+[0, 0, -.02]).tolist(),
                            'quaternion_wxyz': [1., 0., 0., 0.],
                            'velocity': [0., 0., 0.], 'angular_velocity': [0., 0., 0.]}
                return torch.tensor([self.bank.current.get(key, fallback[field])], dtype=torch.float64)
            def get_pos(self):
                return self._read('xyz')
            def get_quat(self):
                return self._read('quaternion_wxyz')
            def get_vel(self):
                return self._read('velocity')
            def get_ang(self):
                return self._read('angular_velocity')
            def get_qpos(self):
                s = self.bank.current
                return torch.tensor([[*s['drone_xyz'], *s['drone_quaternion_wxyz'], *([1, 0, 0, 0]*2)]],
                                    dtype=torch.float64)
            def get_dofs_velocity(self):
                s = self.bank.current
                return torch.tensor([[*s['drone_velocity'], *s['drone_angular_velocity'], *([0.]*6)]],
                                    dtype=torch.float64)
            def get_contacts(self):
                return {'valid_mask': torch.tensor([self.bank.contact])}
            def get_link(self, name):
                assert name == 'suspension_rod'
                return self.bank.rod
        class CpuBank:
            """Only the simulator boundary is substituted; the real parent step runs."""
            def __init__(self, output, dt=.01, substeps=2, align_root=False):
                self.reference_mass = .02700047753751278
                self.align_root = align_root
                self.drone, self.package, self.rod = [CpuBody(self, name) for name in ('drone', 'package', 'rod')]
                self.contact, self.steps, self.overrides = False, 0, {}
                self.current = None
                self.force_terminal = False
            def reset(self, payload, initial):
                self.side = payload.side_m
                self.payload = payload
                self.current = sample(initial, payload.side_m)
                return copy.deepcopy(self.current)
            def step(self, action):
                self.steps += 1
                s = copy.deepcopy(self.current)
                if self.force_terminal:
                    for body in ('drone', 'package'):
                        s[body+'_xyz'][2] -= 1.
                    for label in ('a', 'b'):
                        s['attachment_'+label+'_world'][2] -= 1.
                s.update(copy.deepcopy(self.overrides))
                self.current = copy.deepcopy(s)  # raw state remains accessible if conversion raises
                s = sample(s, self.side)
                self.current = copy.deepcopy(s)
                return s
        def synthetic_properties(bank):
            p=bank.payload
            s=bank.current
            extent=np.full(3,p.side_m/2)
            center=np.asarray(s['package_xyz'])
            return dict(original_mass_kg=p.reference_mass_kg,package_mass_kg=p.mass,
                package_inertia_kg_m2=(np.eye(3)*p.inertia[0]).tolist(),
                package_aabb=[(center-extent).tolist(),(center+extent).tolist()],
                qpos=bank.drone.get_qpos()[0].tolist(),velocity=bank.drone.get_dofs_velocity()[0].tolist(),
                asset_sha256='a'*64,asset_bytes_verified=True,
                **{body+'_'+field:s[body+'_'+field] for body in ('drone','package')
                   for field in ('xyz','quaternion_wxyz')})
        monkeypatch.setattr(training, 'read_reset_properties', synthetic_properties)
        monkeypatch.setattr(training, 'BallSceneBank', CpuBank)
        return training.PayloadRetryTrainingEnv(tmp_path/'scenes', True, 300000, seed=7)
    return build


def test_next_actor_observation_has_just_applied_action_and_retarget_keeps_it(training_env):
    env = training_env()
    assert env.bank.align_root is True and env.cfg['profile'] == PROFILE
    assert env.get_observations()['policy'][0, 13:17].tolist() == [0.]*4
    action = torch.tensor([[.1, -.2, 2., -2.]])
    obs, _, done, _ = env.step(action)
    assert not done[0]
    assert obs['policy'][0, 13:17].tolist() == pytest.approx([.1, -.2, 1., -1.])
    before = copy.deepcopy(env.sample)
    history = env.last_actions.copy()
    env.set_target([1., -.5, 2.])
    assert env.sample == before
    assert np.array_equal(env.last_actions, history)
    assert env.get_observations()['policy'][0, 13:17].tolist() == pytest.approx(history)
    assert env.total_steps == 1 and env.num_envs == 1 and env.budget_steps == 300000


def test_terminal_samples_are_saved_before_reset_and_reset_history_is_zero(training_env):
    env = training_env()
    env.bank.force_terminal = True
    obs, reward, done, _ = env.step(torch.tensor([[.2, .3, .4, .5]]))
    assert done[0] and reward[0] < 0
    terminal = json.loads((env.output/'terminal-states.jsonl').read_text().splitlines()[-1])
    assert terminal['sample']['drone_xyz'][2] == 0.
    assert terminal['failed'] and terminal['metrics']['drone_floor_m'] <= .1
    assert obs['policy'][0, 13:17].tolist() == [0.]*4
    assert env.last_actions.tolist() == [0.]*4
    assert env.episode_count == 2 and env.failures == 1


def test_training_manifest_matching_rejects_initialization_and_optimizer_changes():
    checkpoint = module('payload_retry_checkpoint')
    contract = module('payload_retry_contract')
    common = dict(profile=PROFILE, seed=7, num_envs=1, environment_steps=300000,
                  observation_contract=contract.OBSERVATION_CONTRACT,
                  source_binding={'reward': 'same'}, optimizer_config={'learning_rate':.00003},
                  architecture={'actor': [128, 128]}, curriculum=['hover', 'long'],
                  full_distribution_every_stage=True,
                  initialization_comparison={'initial_actor_tensor_sha256': 'same',
                                             'initial_critic_tensor_sha256': 'same'},
                  parent_sha256=PARENT_SHA, parent_config_sha256=CFG_SHA)
    a, b = dict(common, method='randomized'), dict(common, method='fixed')
    checkpoint.validate_matched_manifests(a, b)
    for key, value in [('seed', 8), ('environment_steps', 299999), ('profile', 'old'),
                       ('optimizer_config', {'learning_rate': .01}),
                       ('initialization_comparison', {'initial_actor_tensor_sha256':'changed'})]:
        bad = copy.deepcopy(b)
        bad[key] = value
        with pytest.raises(ValueError):
            checkpoint.validate_matched_manifests(a, bad)


@pytest.mark.parametrize('status', ['PASS', 'QUALIFIED_FOR_TRAINING'])
def test_train_cli_admission_runs_before_any_genesis_import_or_output_creation(tmp_path, status):
    assert SCRIPT.is_file(), 'retry training CLI is required'
    # A trap module would make an early simulator import unambiguous.
    (tmp_path/'genesis.py').write_text("raise RuntimeError('GENESIS_IMPORT_SHOULD_NOT_HAPPEN')\n")
    parent, config = tmp_path/'parent.pt', tmp_path/'cfg.pkl'
    parent.write_bytes(b'not a trained checkpoint')
    config.write_bytes(b'not a configuration')
    admission, reconstruction = tmp_path/'admission.json', tmp_path/'reconstruction.json'
    admission.write_text(json.dumps({'status':status,'physical_training_admitted':True}))
    reconstruction.write_text(json.dumps({'status':'PASS'}))
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    out = tmp_path/'must-not-start'
    command = [sys.executable, str(SCRIPT), '--parent', str(parent), '--parent-sha256', PARENT_SHA,
               '--config', str(config), '--config-sha256', CFG_SHA, '--admission', str(admission),
               '--admission-sha256', sha(admission), '--reconstruction', str(reconstruction),
               '--reconstruction-sha256', sha(reconstruction), '--method', 'randomized', '--out', str(out)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(ROOT/'src')]))
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert 'require_retry_admission' in result.stderr
    assert 'GENESIS_IMPORT_SHOULD_NOT_HAPPEN' not in result.stderr
    assert not out.exists()


def test_effective_training_keeps_optimizer_architecture_and_exact_rollout_budget():
    assert SCRIPT.is_file(), 'retry training CLI is required'
    spec = importlib.util.spec_from_file_location('retry_train', SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    parent = [{'num_actions':4}, {}, {}, {}, dict(num_steps_per_env=100,
              algorithm={'learning_rate':.00003,'schedule':'fixed','entropy_coef':0.,'max_grad_norm':.3},
              actor={'hidden_dims':[128,128]},critic={'hidden_dims':[128,128]},
              obs_groups={'actor':['policy'],'critic':['policy']})]
    a, b = script.training_configuration(parent, 'randomized'), script.training_configuration(parent, 'fixed')
    assert a['algorithm'] == b['algorithm'] == parent[4]['algorithm']
    assert a['actor'] == b['actor'] == parent[4]['actor']
    assert a['num_steps_per_env']*3000 == 300000
    assert parent[4].get('run_name') is None


def trace_records(env):
    path = env.output/'training-trace.jsonl'
    assert path.exists(), 'bounded raw training trace is required'
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_each_step_records_preclip_state_observations_commands_and_saturation(training_env):
    env = training_env()
    env.bank.overrides = {'drone_velocity': [4., 0., 0.]}
    obs, reward, done, _ = env.step(torch.tensor([[.2, -.3, 2., -2.]]))
    records = trace_records(env)
    assert [row['kind'] for row in records] == ['RAW_STEP', 'PRE_PARENT_OBSERVATION', 'STEP_OUTCOME']
    raw, observed, outcome = records
    assert raw['bodies']['drone']['velocity'] == [4., 0., 0.]
    assert raw['raw_actions'] == pytest.approx([.2, -.3, 2., -2.])
    assert raw['clipped_actions'] == pytest.approx([.2, -.3, 1., -1.])
    rpm = np.asarray((1+.8*np.array([.2, -.3, 1., -1.]))*14468.429183500699, dtype=np.float32)
    assert raw['rpm_command_float32'] == pytest.approx(rpm.tolist())
    assert raw['episode_geometry_declared']['side_m'] == env.payload.side_m
    assert raw['episode_geometry_declared']['mass_kg'] == env.payload.mass
    assert observed['observation_preclip'][20] == pytest.approx(-4/3)
    assert observed['observation_clipped'][20] == -1.
    assert observed['saturation_mask'][20]
    assert observed['saturation_duration_s'][20] == pytest.approx(.01)
    assert observed['entity_contact'] is False
    assert observed['constraint_residual_m'] < 2e-5
    assert outcome['reward'] == pytest.approx(float(reward[0]))
    assert outcome['done'] == bool(done[0])
    assert outcome['next_actor_observation'] == pytest.approx(obs['policy'][0].tolist())
    assert env.bank.steps == 1 and env.total_steps == 1


def test_trace_keeps_terminal_reward_and_old_state_before_reset(training_env):
    env = training_env()
    env.bank.force_terminal = True
    obs, reward, done, _ = env.step(torch.tensor([[.2, .3, .4, .5]]))
    records = trace_records(env)
    assert records[0]['bodies']['drone']['xyz'][2] == 0.
    outcome = records[-1]
    assert done[0] and outcome['done'] and outcome['reward'] == pytest.approx(float(reward[0]))
    assert outcome['episode'] == 1 and outcome['next_episode'] == 2
    assert outcome['next_actor_observation'][13:17] == [0.]*4
    assert obs['policy'][0, 13:17].tolist() == [0.]*4
    assert not (env.output/'training-abort.json').exists()


@pytest.mark.parametrize('field,value,code', [
    ('drone_velocity', [100., 0, 0], 'LINEAR_SPEED_EXCURSION'),
    ('package_angular_velocity', [1000., 0, 0], 'ANGULAR_SPEED_EXCURSION'),
    ('rod_angular_velocity', [1000., 0, 0], 'ANGULAR_SPEED_EXCURSION'),
    ('drone_quaternion_wxyz', [1.00002, 0, 0, 0], 'QUATERNION_NORM_EXCURSION'),
    ('package_velocity', [float('nan'), 0, 0], 'NONFINITE_RAW_STATE'),
    ('package_xyz', [.2, 0, .6575], 'CONSTRAINT_RESIDUAL_EXCURSION'),
])
def test_numerical_guard_saves_raw_before_metrics_or_reward_and_aborts_job(training_env, monkeypatch,
                                                                         field, value, code):
    env = training_env()
    env.bank.overrides = {field: value}
    import adaptive_drone_patrol.payload_training as legacy
    def metrics_must_not_run(*args, **kwargs):
        raise AssertionError('METRICS_MUST_NOT_RUN')
    monkeypatch.setattr(legacy, 'payload_metrics', metrics_must_not_run)
    with pytest.raises(RuntimeError):
        env.step(torch.tensor([[.1, .2, .3, .4]]))
    abort = json.loads((env.output/'training-abort.json').read_text())
    assert abort['code'] == code
    assert abort['optimizer_sample_admitted'] is False
    body, key = field.split('_', 1)
    measured = abort['offending_raw']['bodies'][body][key]
    if field == 'package_velocity':
        assert np.isnan(measured[0])
    else:
        assert measured == value
    assert env.reward_totals == {} and env.total_steps == 0
    assert env.bank.steps == 1
    assert env._training_aborted
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(1, 4))
    assert env.bank.steps == 1


@pytest.mark.parametrize('actions', [torch.tensor([[float('nan'), 0, 0, 0]]), torch.zeros(1, 100000)])
def test_abnormal_input_is_saved_before_abort_without_physical_step(training_env, actions):
    env = training_env()
    with pytest.raises(RuntimeError):
        env.step(actions)
    abort = json.loads((env.output/'training-abort.json').read_text())
    assert abort['code'] in ('ACTION_NONFINITE', 'ACTION_SHAPE')
    assert abort['physics_step_completed'] is False
    assert len(abort['action_input']['values_prefix']) <= 16
    assert env.bank.steps == 0 and env.last_actions.tolist() == [0.]*4
    assert (env.output/'training-abort.json').stat().st_size < 65536


def test_metrics_exception_saves_recent_window_without_duplicate_step(training_env, monkeypatch):
    env = training_env()
    env.step(torch.zeros(1, 4))
    import adaptive_drone_patrol.payload_training as legacy
    def fail_metrics(*args, **kwargs):
        raise ValueError('injected metrics/rotation failure')
    monkeypatch.setattr(legacy, 'payload_metrics', fail_metrics)
    with pytest.raises(RuntimeError):
        env.step(torch.tensor([[.1, .2, .3, .4]]))
    abort = json.loads((env.output/'training-abort.json').read_text())
    assert abort['code'] == 'PARENT_STEP_EXCEPTION'
    assert [row['step'] for row in abort['recent_raw_window']] == [1, 2]
    assert abort['offending_raw']['step'] == 2
    assert abort['optimizer_sample_admitted'] is False
    assert env.bank.steps == 2


def test_recent_window_and_each_record_are_bounded_without_dropping_normal_steps(training_env):
    env = training_env()
    for _ in range(130):
        env.step(torch.zeros(1, 4))
    env.bank.overrides = {'drone_velocity': [100., 0, 0]}
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(1, 4))
    abort = json.loads((env.output/'training-abort.json').read_text())
    assert len(abort['recent_raw_window']) == 128
    assert abort['recent_raw_window'][0]['step'] == 4 and abort['recent_raw_window'][-1]['step'] == 131
    records = trace_records(env)
    assert len([row for row in records if row['kind'] == 'RAW_STEP']) == 131
    assert len([row for row in records if row['kind'] == 'STEP_OUTCOME']) == 130
    assert max(len(line.encode()) for line in (env.output/'training-trace.jsonl').read_text().splitlines()) <= 32768


def test_observation_exception_saves_partial_reward_and_never_returns_to_optimizer(training_env, monkeypatch):
    env = training_env()
    def fail_observation():
        raise ValueError('injected post-reward observation failure')
    monkeypatch.setattr(env, '_observation', fail_observation)
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(1, 4))
    abort = json.loads((env.output/'training-abort.json').read_text())
    assert abort['offending_raw']['step'] == 1
    assert abort['partial_reward_components']
    assert abort['partial_reward'] is not None
    assert abort['optimizer_sample_admitted'] is False
    assert env.bank.steps == 1


def test_reset_exception_retains_old_terminal_and_actual_bad_reset_state(training_env):
    env = training_env()
    env.bank.force_terminal = True
    original_reset = env.bank._bank.reset
    def bad_reset(payload, initial):
        s = original_reset(payload, initial)
        s['drone_quaternion_wxyz'] = [2., 0, 0, 0]
        env.bank.current = copy.deepcopy(s)
        return s
    env.bank.reset = bad_reset
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(1, 4))
    abort = json.loads((env.output/'training-abort.json').read_text())
    assert abort['code'] == 'QUATERNION_NORM_EXCURSION'
    assert abort['offending_raw']['bodies']['drone']['quaternion_wxyz'] == [2., 0, 0, 0]
    assert abort['preceding_post_step_raw']['bodies']['drone']['xyz'][2] == 0.
    assert abort['current_runtime_raw_at_abort']['captured_episode'] == env.episode_count
    assert env.episode_count == 1  # Failed reset never becomes an initialized episode.
    terminal = json.loads((env.output/'terminal-states.jsonl').read_text().splitlines()[-1])
    assert terminal['sample']['drone_xyz'][2] == 0. and terminal['reward'] < 0


def test_invalid_action_does_not_invent_advanced_simulation_time(training_env):
    env = training_env()
    with pytest.raises(RuntimeError):
        env.step(torch.tensor([[float('nan'), 0, 0, 0]]))
    abort = json.loads((env.output/'training-abort.json').read_text())
    assert abort['offending_raw']['simulation_time_s'] == 0.
    assert abort['offending_raw']['episode_time_s'] == 0.
    assert env.bank.steps == 0


def test_training_binding_includes_both_actual_physical_probe_entrypoints(tmp_path):
    spec = importlib.util.spec_from_file_location('retry_train_binding', SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    parent, config = tmp_path/'parent', tmp_path/'config'
    parent.write_bytes(b'CPU binding fixture')
    config.write_bytes(b'CPU binding fixture')
    binding=cli.actual_training_binding(ROOT,parent,config)
    for name in ('scripts/probe_payload_ball.py','scripts/probe_payload_ball_bank.py'):
        assert binding['source_binding'][name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
