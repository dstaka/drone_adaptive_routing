"""CPU boundary tests; no fixture below establishes physical qualification."""
import copy
import json

import numpy as np
import pytest
import torch

from adaptive_drone_patrol.payload_contract import Payload


class Body:
    def __init__(self, bank, name):
        self.bank, self.name = bank, name

    def _value(self, field):
        return torch.tensor([self.bank.current[self.name+'_'+field]], dtype=torch.float64)

    def get_pos(self):
        return self._value('xyz')

    def get_quat(self):
        return self._value('quaternion_wxyz')

    def get_vel(self):
        return self._value('velocity')

    def get_ang(self):
        return self._value('angular_velocity')

    def get_AABB(self):
        p = np.asarray(self.bank.current[self.name+'_xyz'])
        extent = .0225 if self.name == 'package' else .04
        return torch.tensor(np.array([[p-extent, p+extent]]), dtype=torch.float64)


class Bank(Body):
    """Substitute only simulator I/O, preserving raw state and step side effects."""
    reference_mass = .02700047753751278
    dt, substeps, align_root = .01, 2, True

    def __init__(self):
        super().__init__(self, 'drone')
        self.drone = self
        self.package, self.rod = Body(self, 'package'), Body(self, 'rod')
        self.steps, self.resets = 0, 0
        self.overrides = {}
        self.current = None

    def reset(self, payload, initial):
        self.resets += 1
        self.payload = payload
        self.current = copy.deepcopy(initial)
        for label, offset, body in [('a', -.02, 'drone'), ('b', payload.side_m/2, 'package')]:
            self.current['attachment_'+label+'_world'] = (np.asarray(initial[body+'_xyz'])+[0, 0, offset]).tolist()
            self.current['attachment_'+label+'_velocity'] = [0., 0., 0.]
        self.current.update(rod_xyz=[0., 0., .98], rod_quaternion_wxyz=[1., 0., 0., 0.],
                            rod_velocity=[0., 0., 0.], rod_angular_velocity=[0., 0., 0.])
        return copy.deepcopy(self.current)

    def step(self, actions):
        self.steps += 1
        self.current.update(copy.deepcopy(self.overrides))
        return copy.deepcopy(self.current)

    def sample(self):
        return copy.deepcopy(self.current)

    def get_qpos(self):
        return torch.tensor([[*self.current['drone_xyz'], *self.current['drone_quaternion_wxyz'],
                              *([1., 0., 0., 0.]*2)]], dtype=torch.float64)

    def get_dofs_velocity(self):
        return torch.tensor([[*self.current['drone_velocity'], *self.current['drone_angular_velocity'],
                              *([0.]*6)]], dtype=torch.float64)

    def get_link(self, name):
        return self.rod if name == 'suspension_rod' else self

    def get_contacts(self):
        return {'valid_mask': torch.tensor([False])}


def initial():
    return dict(drone_xyz=[0., 0., 1.], package_xyz=[0., 0., .6575],
                drone_quaternion_wxyz=[1., 0., 0., 0.], package_quaternion_wxyz=[1., 0., 0., 0.],
                drone_velocity=[0., 0., 0.], package_velocity=[0., 0., 0.],
                drone_angular_velocity=[0., 0., 0.], package_angular_velocity=[0., 0., 0.])


def properties(bank):
    p, s = bank.payload, bank.current
    center = np.asarray(s['package_xyz'])
    return dict(original_mass_kg=p.reference_mass_kg, package_mass_kg=p.mass,
                package_inertia_kg_m2=(np.eye(3)*p.inertia[0]).tolist(),
                package_aabb=[(center-p.side_m/2).tolist(), (center+p.side_m/2).tolist()],
                qpos=bank.get_qpos()[0].tolist(), velocity=bank.get_dofs_velocity()[0].tolist(),
                asset_sha256='a'*64, asset_bytes_verified=True,
                **{body+'_'+field:s[body+'_'+field] for body in ('drone', 'package')
                   for field in ('xyz', 'quaternion_wxyz')})


@pytest.fixture
def runtime(tmp_path):
    from adaptive_drone_patrol.payload_recovery_runtime import RecoveryPayloadRuntime
    core = RecoveryPayloadRuntime(Bank(), tmp_path/'trace', property_reader=properties)
    core.reset_case(Payload(core.bank.reference_mass, .2, .045), initial())
    return core


def test_current_action_history_and_retarget_preserve_native_state(runtime):
    assert runtime.observation[13:17].tolist() == [0.]*4
    runtime.step([.1, -.2, 2., -2.])
    assert runtime.observation[13:17].tolist() == pytest.approx([.1, -.2, 1., -1.])
    before = copy.deepcopy(runtime.bank.current)
    runtime.set_target([1., .5, 2.], target_id='A')
    assert runtime.bank.current == before
    assert runtime.observation[:3].tolist() == pytest.approx([1/3, 1/6, 1/3])
    assert runtime.observation[13:17].tolist() == pytest.approx([.1, -.2, 1., -1.])
    assert runtime.bank.resets == 1 and runtime.bank.steps == 1


def test_failed_reset_conversion_preserves_actual_new_state_before_quarantine(runtime):
    original_reset = runtime.bank.reset

    def fail_after_reset(payload, state):
        original_reset(payload, state)
        runtime.bank.current['drone_velocity'] = [float('nan'), 0., 0.]
        raise ValueError('Conversion failed after native reset')

    runtime.bank.reset = fail_after_reset
    with pytest.raises(RuntimeError, match='quarantined'):
        runtime.reset_case(runtime.payload, initial())
    abort = json.loads((runtime.output/'abort.json').read_text())
    raw = abort['offending_raw']
    assert raw['kind'] == 'RAW_RESET_EXCEPTION'
    assert np.isnan(raw['bodies']['drone']['velocity'][0])
    assert raw['time_s'] is None
    assert raw['reset_completion'] == 'UNKNOWN'
    assert raw['reset_attempt_episode_id'] == 2
    assert raw['last_confirmed_episode_id'] == 1
    assert abort['optimizer_sample_admitted'] is False


def test_shared_settling_requires_full_intervals_and_retarget_restarts_dwell(runtime):
    runtime.set_target([0., 0., 1.], target_id='A')
    for _ in range(100):
        assert not runtime.step([0.]*4)['complete']
    assert runtime.step([0.]*4)['complete']
    runtime.set_target([0., 0., 1.], target_id='B')
    assert not runtime.step([0.]*4)['complete']
    assert runtime.bank.resets == 1


@pytest.mark.parametrize('override', [
    {'drone_velocity':[float('nan'), 0., 0.]},
    {'package_angular_velocity':[201., 0., 0.]},
    {'drone_quaternion_wxyz':[2., 0., 0., 0.]},
    {'package_xyz':[0., 0., .5]},
])
def test_invalid_raw_is_saved_and_permanently_quarantined_before_metrics(runtime, override):
    runtime.bank.overrides = override
    with pytest.raises(RuntimeError, match='quarantined'):
        runtime.step([0.]*4)
    rows = [json.loads(s) for s in runtime.trace_path.read_text().splitlines()]
    assert rows[-1]['kind'] == 'ABORT'
    assert rows[-2]['kind'] == 'RAW_STEP'
    assert rows[-1]['optimizer_sample_admitted'] is False
    abort = json.loads((runtime.output/'abort.json').read_text())
    assert abort['offending_raw']['bodies']['package']['xyz'] == runtime.bank.current['package_xyz']
    steps = runtime.bank.steps
    with pytest.raises(RuntimeError, match='quarantined'):
        runtime.step([0.]*4)
    with pytest.raises(RuntimeError, match='quarantined'):
        runtime.reset_case(runtime.payload, initial())
    assert runtime.bank.steps == steps and runtime.bank.resets == 1


def test_unsafe_oriented_vehicle_floor_stops_without_reset(runtime):
    runtime.bank.get_AABB = lambda: torch.tensor([[[-.1, -.1, .09], [.1, .1, 1.1]]])
    row = runtime.step([0.]*4)
    assert row['unsafe'] and row['metrics']['drone_floor_m'] == pytest.approx(.09)
    assert runtime.bank.resets == 1


def test_reset_zeroes_history_and_measures_effective_properties(runtime):
    runtime.step([.2]*4)
    runtime.reset_case(runtime.payload, initial())
    assert runtime.observation[13:17].tolist() == [0.]*4
    assert runtime.step_count == 0
    runtime.property_reader = lambda bank: {**properties(bank), 'package_mass_kg':.5}
    with pytest.raises(RuntimeError, match='quarantined'):
        runtime.reset_case(runtime.payload, initial())
    assert (runtime.output/'abort.json').is_file()


def test_admission_rejects_before_loading_checkpoint_or_importing_genesis(tmp_path, monkeypatch):
    import builtins
    from adaptive_drone_patrol.payload_recovery_runtime import RecoveryPayloadFlight
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'genesis':
            pytest.fail('Genesis imported before admission')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError):
        RecoveryPayloadFlight.from_artifacts(
            checkpoint=tmp_path/'missing.pt', checkpoint_sha256='a'*64,
            parent_cfg=tmp_path/'missing.pkl', parent_cfg_sha256='b'*64,
            receipt={}, actual_binding={}, locations={}, output=tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_legacy_mission_gate_cannot_be_mistaken_for_payload_acceptance(runtime):
    from adaptive_drone_patrol.payload_recovery_runtime import RecoveryPayloadFlight
    flight = RecoveryPayloadFlight(runtime, lambda obs: np.zeros(4))
    row = flight.advance()
    assert row['xyz'] == [0., 0., 1.] and row['done'] is False and row['reset'] is False
    assert row['shared_settled'] is False
    assert row['payload_acceptance_contract'] == 'payload-shared-settling-100-intervals-v1'
    assert flight.sample()['package_xyz'] == [0., 0., .6575]
    assert flight.dt == .01


def test_visual_only_rotors_do_not_make_oriented_vehicle_clearance_unavailable(runtime):
    original = runtime.bank.get_link
    class VisualRotor:
        n_geoms = 0
        def get_AABB(self):
            raise ValueError('Link has no collision geometries')
    runtime.bank.get_link = lambda name: VisualRotor() if name.startswith('prop') else original(name)
    assert runtime.step([0.]*4)['metrics']['drone_floor_m'] == pytest.approx(.96)


def test_invalid_action_records_offending_input_without_advancing_physics(runtime):
    with pytest.raises(RuntimeError, match='quarantined'):
        runtime.step([float('nan'), 0., 0., 0.])
    failure = json.loads((runtime.output/'abort.json').read_text())
    assert np.isnan(failure['action_input'][0])
    assert failure['step'] == 0 and runtime.bank.steps == 0


@pytest.mark.parametrize('bad', ['runtime_contract', 'source', 'checkpoint_physics', 'checkpoint_observation'])
def test_stale_runtime_or_checkpoint_cannot_reach_genesis(tmp_path, monkeypatch, bad):
    import builtins
    from adaptive_drone_patrol import payload_recovery_runtime as mod
    from adaptive_drone_patrol.payload_retry_checkpoint import transfer_state
    monkeypatch.setattr(mod, 'require_recovery_runtime_admission',
                        lambda *args: {'physical_signature_sha256':'c'*64})
    state = {}
    for role, outputs in [('actor', 4), ('critic', 1)]:
        state[role+'_state_dict'] = {
            'mlp.0.weight':torch.zeros(128, 17), 'mlp.0.bias':torch.zeros(128),
            'mlp.2.weight':torch.zeros(128, 128), 'mlp.2.bias':torch.zeros(128),
            'mlp.4.weight':torch.zeros(outputs, 128), 'mlp.4.bias':torch.zeros(outputs)}
    state['actor_state_dict']['distribution.std_param'] = torch.zeros(4)
    checkpoint = transfer_state(state)
    checkpoint['infos'].update(status='TRAINED_NOT_SELECTED', physical_signature_sha256='c'*64)
    binding = dict(runtime_contract=mod.RUNTIME_CONTRACT, observation_contract=copy.deepcopy(mod.OBSERVATION_CONTRACT),
                   control_dt_s=.01, rigid_dt_s=.005, substeps=2, requested_root_alignment=True,
                   source_binding={name:mod.sha256(name) for name in mod.RUNTIME_SOURCES})
    if bad == 'runtime_contract':
        binding['runtime_contract'] = 'legacy'
    elif bad == 'source':
        binding['source_binding'][mod.RUNTIME_SOURCES[0]] = 'd'*64
    elif bad == 'checkpoint_physics':
        checkpoint['infos']['physical_signature_sha256'] = 'd'*64
    else:
        checkpoint['infos']['observation_contract']['history'] = 'stale history'
    model, config = tmp_path/'checkpoint.pt', tmp_path/'config.pkl'
    torch.save(checkpoint, model)
    config.write_bytes(b'unopened configuration')
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'genesis':
            pytest.fail('Unqualified runtime reached Genesis')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError):
        mod.RecoveryPayloadFlight.from_artifacts(checkpoint=model, checkpoint_sha256=mod.sha256(model),
            parent_cfg=config, parent_cfg_sha256=mod.sha256(config), receipt={}, locations={},
            actual_binding=binding, output=tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_core_preserves_control_interval_with_source_declared_refinement(tmp_path):
    from adaptive_drone_patrol.payload_recovery_runtime import RecoveryPayloadRuntime
    bank = Bank()
    bank.substeps = 4
    core = RecoveryPayloadRuntime(bank, tmp_path/'refined', property_reader=properties)
    core.reset_case(Payload(bank.reference_mass, .2, .045), initial())
    row = core.step([0.]*4)
    assert row['time_s'] == .01
    assert core.numerical_settings['rigid_dt_s'] == .0025
    assert core.numerical_settings['substeps'] == 4


@pytest.mark.parametrize('schema', ['payload-operating-domain-independent-v3',
                                  'payload-recovery-operating-domain-independent-v1'])
def test_claimed_pass_cannot_unlock_unimplemented_recovery_raw_admission(schema):
    from adaptive_drone_patrol.payload_recovery_runtime import require_recovery_runtime_admission
    with pytest.raises(ValueError, match='62'):
        require_recovery_runtime_admission(
            {'schema':schema, 'status':'QUALIFIED_FOR_TRAINING', 'physical_training_admitted':True}, {}, {})


def test_failed_bank_conversion_preserves_current_raw_without_inventing_sample_time(runtime):
    def failed_conversion(actions):
        runtime.bank.steps += 1
        runtime.bank.current['drone_velocity'] = [float('nan'), 0., 0.]
        raise ValueError('Derived sample conversion failed after solver step')
    runtime.bank.step = failed_conversion
    with pytest.raises(RuntimeError, match='quarantined'):
        runtime.step([.1]*4)
    failure = json.loads((runtime.output/'abort.json').read_text())
    raw = failure['offending_raw']
    assert raw['kind'] == 'RAW_BANK_EXCEPTION'
    assert raw['time_s'] is None and raw['physics_step_completed'] == 'UNKNOWN'
    assert raw['attempted_step'] == 1 and raw['last_confirmed_step'] == 0
    assert np.isnan(raw['bodies']['drone']['velocity'][0])


def test_outcome_retains_measured_vehicle_geometry_for_independent_mission_predicates(runtime):
    row = runtime.step([0.]*4)
    assert row['vehicle_collision_aabbs']['base_link'] == [[-.04, -.04, .96], [.04, .04, 1.04]]
