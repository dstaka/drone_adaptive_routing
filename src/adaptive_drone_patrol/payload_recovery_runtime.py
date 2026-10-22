"""Native payload runtime boundary; CPU tests never grant physical admission.

The core accepts an existing bank so training and inference can share action,
observation, reset, guard and retarget semantics. The artifact factory alone
performs physical/source/checkpoint admission before constructing Genesis.
Legacy mission ArrivalGate and media formats do not implement payload settling;
callers must consume ``complete`` and the full payload telemetry explicitly.
"""

from collections import deque
import copy
import json
from pathlib import Path
import pickle

import numpy as np

from .payload_checkpoint import actor_signature, validate_parent_config
from .payload_contract import DRONE_ATTACHMENT, SharedSettling, payload_metrics, rotation, settled
from .payload_retry_checkpoint import validate_retry_checkpoint
from .payload_retry_contract import OBSERVATION_CONTRACT, PROFILE, actor_observation
from .payload_retry_properties import read_reset_properties, validate_reset_properties
from .util import sha256

RUNTIME_CONTRACT = 'payload-native-recovery-runtime-v1'
SETTLING_CONTRACT = 'payload-shared-settling-100-intervals-v1'
BODY_FIELDS = (('xyz', 'get_pos', 3), ('quaternion_wxyz', 'get_quat', 4),
               ('velocity', 'get_vel', 3), ('angular_velocity', 'get_ang', 3))
RUNTIME_SOURCES = ('src/adaptive_drone_patrol/payload_recovery_runtime.py',
                   'src/adaptive_drone_patrol/payload_retry_contract.py',
                   'src/adaptive_drone_patrol/payload_contract.py',
                   'src/adaptive_drone_patrol/payload_ball_bank.py',
                   'src/adaptive_drone_patrol/payload_retry_properties.py')


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _vector(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError('Expected a finite runtime vector')
    return value


def require_recovery_runtime_admission(receipt, actual_binding, locations):
    """Keep production closed until complete recovery raw reconstruction exists.

    The retry55 replay cannot admit the new recovery62 domain. This deliberate
    barrier must be replaced by the complete source/settings/batch-bound replay,
    never by checking a receipt's status or declared case count alone.
    """
    raise ValueError('Complete recovery62 raw admission is not integrated; retry55 receipts cannot admit runtime')


class RecoveryPayloadRuntime:
    """One active native episode; no implicit reset, resample, reward or optimizer.

    Direct construction is an integration boundary, not evidence of qualification.
    The bank must expose the native BallSceneBank methods and effective getters.
    """
    dt = .01

    def __init__(self, bank, output, *, property_reader=read_reset_properties, delta_z=0.):
        if (not np.isfinite(delta_z) or not 0 <= delta_z <= 1 or
                bank.dt != self.dt or type(bank.substeps) is not int or bank.substeps <= 0 or
                type(bank.align_root) is not bool):
            raise ValueError('Unqualified runtime numerical settings or layout translation')
        self.bank, self.property_reader, self.delta_z = bank, property_reader, float(delta_z)
        self.numerical_settings = dict(control_dt_s=self.dt, substeps=bank.substeps,
            rigid_dt_s=self.dt/bank.substeps, requested_root_alignment=bank.align_root)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.trace_path = self.output/'runtime-trace.jsonl'
        self.trace_path.touch(exist_ok=False)
        self.recent_raw = deque(maxlen=128)
        self.quarantined, self.step_count, self.episode_id = False, 0, 0
        self.last_actions = np.zeros(4)
        self.action_input = None
        self.current_sample = None
        self.target = np.array([0., 0., 1.+delta_z])
        self.target_id = None
        self.observation = None
        self.last_row = None
        self.dwell = SharedSettling(self.dt)

    def _require_active(self):
        if self.quarantined:
            raise RuntimeError('Native runtime is quarantined; no reset or further step is permitted')

    def _write(self, row):
        with self.trace_path.open('a') as stream:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=True)+'\n')
            stream.flush()

    def _raw(self, kind):
        row = dict(kind=kind, schema=RUNTIME_CONTRACT, episode_id=self.episode_id,
                   step=self.step_count, time_s=self.step_count*self.dt,
                   target=self.target.tolist(), actions=self.last_actions.tolist(), bodies={})
        if kind == 'RAW_BANK_EXCEPTION':
            row.update(time_s=None, physics_step_completed='UNKNOWN',
                       attempted_step=self.step_count+1, last_confirmed_step=self.step_count)
        if kind == 'RAW_RESET_EXCEPTION':
            row.update(time_s=None, reset_completion='UNKNOWN',
                       reset_attempt_episode_id=self.episode_id+1,
                       last_confirmed_episode_id=self.episode_id)
        for name in ('drone', 'package', 'rod'):
            row['bodies'][name] = {}
            try:
                body = self.bank.drone.get_link('suspension_rod') if name == 'rod' else getattr(self.bank, name)
            except Exception as exc:
                row['bodies'][name]['capture_error'] = type(exc).__name__
                continue
            for field, method, _ in BODY_FIELDS:
                try:
                    row['bodies'][name][field] = _array(getattr(body, method)())[0].tolist()
                except Exception as exc:
                    row['bodies'][name][field] = {'capture_error':type(exc).__name__}
        for field, method in [('qpos', 'get_qpos'), ('dofs_velocity', 'get_dofs_velocity')]:
            try:
                row[field] = _array(getattr(self.bank.drone, method)())[0].tolist()
            except Exception as exc:
                row[field] = {'capture_error':type(exc).__name__}
        return row

    @staticmethod
    def _guard(raw):
        for body in raw['bodies'].values():
            for field, _, size in BODY_FIELDS:
                value = _vector(body[field], size)
                if field == 'xyz' and np.max(np.abs(value)) > 20:
                    raise ValueError('POSITION_EXCURSION')
                if field == 'velocity' and np.linalg.norm(value) > 20:
                    raise ValueError('LINEAR_SPEED_EXCURSION')
                if field == 'angular_velocity' and np.linalg.norm(value) > 200:
                    raise ValueError('ANGULAR_SPEED_EXCURSION')
                if field == 'quaternion_wxyz':
                    rotation(value)
        qpos = _vector(raw['qpos'], 15)
        velocity = _vector(raw['dofs_velocity'], 12)
        for start in (3, 7, 11):
            rotation(qpos[start:start+4])
        if np.linalg.norm(velocity[:3]) > 20 or any(np.linalg.norm(velocity[i:i+3]) > 200 for i in (3, 6, 9)):
            raise ValueError('GENERALIZED_SPEED_EXCURSION')

    def _capture(self, kind):
        raw = self._raw(kind)
        self.recent_raw.append(raw)
        self._write(raw)
        self._guard(raw)
        return raw

    def _abort(self, exc):
        self.quarantined = True
        record = dict(schema=RUNTIME_CONTRACT, kind='ABORT', optimizer_sample_admitted=False,
                      exception_type=type(exc).__name__, error=str(exc)[:256],
                      episode_id=self.episode_id, step=self.step_count,
                      action_input=self.action_input,
                      offending_raw=self.recent_raw[-1] if self.recent_raw else None,
                      recent_raw_window=list(self.recent_raw))
        with (self.output/'abort.json').open('x') as stream:
            json.dump(record, stream, allow_nan=True)
            stream.write('\n')
            stream.flush()
        self._write({k:v for k, v in record.items() if k not in ('offending_raw', 'recent_raw_window')})
        raise RuntimeError('Native runtime quarantined before observation or optimizer consumption') from exc

    def _check_attachments(self, sample):
        rd, rp = [rotation(sample[body+'_quaternion_wxyz']) for body in ('drone', 'package')]
        expected = {}
        for label, offset, body in [('a', rd @ DRONE_ATTACHMENT, 'drone'),
                                   ('b', rp @ np.array([0., 0., self.payload.side_m/2]), 'package')]:
            expected['attachment_'+label+'_world'] = np.asarray(sample[body+'_xyz'])+offset
            expected['attachment_'+label+'_velocity'] = (np.asarray(sample[body+'_velocity'])+
                np.cross(sample[body+'_angular_velocity'], offset))
        for key, value in expected.items():
            if not np.allclose(_vector(sample[key], 3), value, atol=1e-7, rtol=0):
                raise ValueError('ATTACHMENT_SAMPLE_MISMATCH')
        if abs(np.linalg.norm(expected['attachment_b_world']-expected['attachment_a_world'])-.30) > 2e-5:
            raise ValueError('CONSTRAINT_RESIDUAL_EXCURSION')

    def reset_case(self, payload, initial):
        self._require_active()
        try:
            self.payload = payload
            try:
                self.current_sample = self.bank.reset(payload, initial)
            except Exception:
                # The physical reset may have completed before conversion failed.
                # Preserve current getters without asserting an episode/timestamp.
                raw = self._raw('RAW_RESET_EXCEPTION')
                self.recent_raw.append(raw)
                self._write(raw)
                raise
            self.episode_id += 1
            self.step_count = 0
            self.last_actions = np.zeros(4)
            self.target = np.array([0., 0., 1.+self.delta_z])
            self.target_id = None
            self.dwell = SharedSettling(self.dt)
            self.last_row = None
            self._capture('RAW_RESET')
            record = self.property_reader(self.bank)
            errors = validate_reset_properties(record, payload, initial)
            self._write(dict(kind='RESET_PROPERTIES', episode_id=self.episode_id, actual=record, failures=errors))
            if errors:
                raise ValueError(','.join(errors))
            self._check_attachments(self.current_sample)
            self.observation = actor_observation(self.current_sample, self.target, self.last_actions)
            return copy.deepcopy(self.current_sample)
        except Exception as exc:
            self._abort(exc)

    def set_target(self, target, target_id=None):
        self._require_active()
        value = _vector(target, 3)
        if np.any(value < [-2.7, -1.8, 1.+self.delta_z]) or np.any(value > [2.7, 1.8, 3.+self.delta_z]):
            raise ValueError('Target outside the frozen inherited envelope')
        before = (_array(self.bank.drone.get_qpos()).copy(),
                  _array(self.bank.drone.get_dofs_velocity()).copy(), self.last_actions.copy())
        self.target, self.target_id = value.copy(), target_id
        self.observation = actor_observation(self.current_sample, self.target, self.last_actions)
        after = (_array(self.bank.drone.get_qpos()), _array(self.bank.drone.get_dofs_velocity()), self.last_actions)
        if not all(np.array_equal(a, b) for a, b in zip(before, after)):
            self._abort(ValueError('RETARGET_CHANGED_PHYSICAL_STATE_OR_HISTORY'))
        self.dwell = SharedSettling(self.dt)
        self.dwell.update(self.step_count*self.dt, False, target_id)
        self._write(dict(kind='RETARGET', episode_id=self.episode_id, step=self.step_count,
                         target=self.target.tolist(), target_id=target_id, actions=self.last_actions.tolist()))

    def step(self, actions):
        self._require_active()
        try:
            try:
                self.action_input = _array(actions).tolist()
            except Exception:
                self.action_input = {'capture_error':'Unsupported action input'}
            command = np.clip(_vector(actions, 4), -1, 1)
            self.last_actions = command.copy()
            before_observation = self.observation.copy()
            try:
                self.current_sample = self.bank.step(command)
            except Exception:
                # A conversion failure can follow an advanced native solver. Preserve
                # the actual getters, without inventing a completed step timestamp.
                self._capture('RAW_BANK_EXCEPTION')
                raise
            self.step_count += 1
            raw = self._capture('RAW_STEP')
            for body in ('drone', 'package'):
                for field, _, size in BODY_FIELDS:
                    measured = _vector(self.current_sample[body+'_'+field], size)
                    if not np.array_equal(measured, raw['bodies'][body][field]):
                        raise ValueError('SAMPLE_DIFFERS_FROM_RAW_GETTERS')
            self._check_attachments(self.current_sample)
            metrics = payload_metrics(self.current_sample, self.target, self.payload)
            floors = []
            collision_aabbs = {}
            for name in ('base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link'):
                link = self.bank.drone.get_link(name)
                if getattr(link, 'n_geoms', None) == 0:
                    continue
                bounds = _array(link.get_AABB())[0]
                if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                    raise ValueError('VEHICLE_GEOMETRY_UNAVAILABLE')
                floors.append(float(bounds[0, 2]))
                collision_aabbs[name] = bounds.tolist()
            if not floors:
                raise ValueError('VEHICLE_GEOMETRY_UNAVAILABLE')
            metrics['drone_floor_m'] = min(floors)
            contacts = bool(_array(self.bank.drone.get_contacts()['valid_mask']).any())
            unsafe = bool(contacts or metrics['drone_floor_m'] <= .1 or metrics['package_floor_m'] <= .1 or
                          abs(metrics['drone_roll_deg']) > 60 or abs(metrics['drone_pitch_deg']) > 60)
            complete = self.dwell.update(self.step_count*self.dt, bool(settled(metrics) and not unsafe), self.target_id)
            self.observation = actor_observation(self.current_sample, self.target, self.last_actions)
            row = dict(kind='STEP_OUTCOME', schema=RUNTIME_CONTRACT, episode_id=self.episode_id,
                       step=self.step_count, time_s=self.step_count*self.dt, target_id=self.target_id,
                       effective_target=self.target.tolist(), actions=command.tolist(),
                       rpm_command_float32=np.asarray((1+.8*command)*14468.429183500699, dtype=np.float32).tolist(),
                       contacts=contacts, metrics=metrics, unsafe=unsafe, complete=complete,
                       vehicle_collision_aabbs=collision_aabbs,
                       dwell_s=self.dwell.dwell_s, actor_observation_before_step=before_observation.tolist(),
                       actor_observation=self.observation.tolist(), **copy.deepcopy(self.current_sample))
            self.last_row = row
            self._write(row)
            return copy.deepcopy(row)
        except Exception as exc:
            self._abort(exc)


class RecoveryPayloadFlight:
    """Inference adapter with explicit payload completion; no automatic episodes.

    ``sample/advance/set_target/dt`` match the mission surface, but the legacy
    vehicle-only mission gate and CSV writer must not be used for acceptance.
    """
    dt = .01

    def __init__(self, runtime, policy):
        self.runtime, self.policy, self.bank = runtime, policy, runtime.bank
        self.payload_acceptance_contract = SETTLING_CONTRACT

    @classmethod
    def from_artifacts(cls, *, checkpoint, checkpoint_sha256, parent_cfg, parent_cfg_sha256,
                       receipt, actual_binding, locations, output):
        # All source/physics and checkpoint checks precede Genesis import, output
        # creation and runtime construction. Neither PROFILE nor a PASS label alone
        # proves the current physical implementation qualified.
        admitted = require_recovery_runtime_admission(receipt, actual_binding, locations)
        substeps = actual_binding.get('substeps')
        if (actual_binding.get('runtime_contract') != RUNTIME_CONTRACT or
                actual_binding.get('observation_contract') != OBSERVATION_CONTRACT or
                actual_binding.get('control_dt_s') != .01 or type(substeps) is not int or substeps <= 0 or
                actual_binding.get('rigid_dt_s') != .01/substeps or
                type(actual_binding.get('requested_root_alignment')) is not bool):
            raise ValueError('Runtime settings/observation contract differs from the qualified binding')
        for source in RUNTIME_SOURCES:
            if actual_binding.get('source_binding', {}).get(source) != sha256(source):
                raise ValueError('Runtime source differs from the frozen binding')
        if sha256(checkpoint) != checkpoint_sha256 or sha256(parent_cfg) != parent_cfg_sha256:
            raise ValueError('Checkpoint or parent configuration hash mismatch')
        import torch
        data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        validate_retry_checkpoint(data)
        if (data['infos'].get('status') not in ('TRAINED_NOT_SELECTED', 'SELECTED') or
                data['infos'].get('physical_signature_sha256') != admitted['physical_signature_sha256']):
            raise ValueError('Checkpoint is not trained under the actual qualified physical signature')
        with Path(parent_cfg).open('rb') as stream:
            config = pickle.load(stream)
        validate_parent_config(config)
        delta_z = actual_binding.get('delta_z_m', 0.)
        if not isinstance(delta_z, (int, float)) or isinstance(delta_z, bool) or not np.isfinite(delta_z) or not 0 <= delta_z <= 1:
            raise ValueError('Invalid frozen common translation')
        if delta_z != 0:
            raise ValueError('Translated layout admission is not implemented by this runtime factory')
        import genesis as gs
        from tensordict import TensorDict
        from rsl_rl.runners import OnPolicyRunner
        from .payload_ball_bank import BallSceneBank
        gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=7)
        root = Path(output)
        root.mkdir(parents=True, exist_ok=False)
        bank = BallSceneBank(root/'scenes', dt=.01, substeps=substeps,
                             align_root=actual_binding['requested_root_alignment'])
        core = RecoveryPayloadRuntime(bank, root/'trace', delta_z=delta_z)
        # RSL inference construction uses only this observation boundary, never an
        # old Euler environment or a fresh optimizer-driven physical rollout.
        class PolicyEnvironment:
            num_envs, num_actions, device = 1, 4, gs.device
            cfg = dict(profile=PROFILE, observation_contract=OBSERVATION_CONTRACT)

            def get_observations(self):
                return TensorDict({'policy':torch.zeros((1, 29), device=gs.device)}, batch_size=[1])
        runner = OnPolicyRunner(PolicyEnvironment(), config[4], None, device=gs.device)
        runner.load(str(checkpoint), load_cfg={'actor':True, 'critic':False, 'optimizer':False,
                                              'iteration':False, 'rnd':False})
        policy = runner.get_inference_policy(device=gs.device)
        if actor_signature(policy.state_dict()) != actor_signature(data['actor_state_dict']):
            raise ValueError('Actual inference actor differs from frozen checkpoint tensors')

        def infer(observation):
            value = TensorDict({'policy':torch.as_tensor(observation, dtype=torch.float32,
                device=gs.device)[None]}, batch_size=[1])
            with torch.no_grad():
                return policy(value)[0].detach().cpu().numpy()
        flight = cls(core, infer)
        flight.checkpoint_sha256, flight.config_sha256 = checkpoint_sha256, parent_cfg_sha256
        flight.actor_sha256 = actor_signature(data['actor_state_dict'])
        flight.actual_execution_binding = copy.deepcopy(actual_binding)
        return flight

    def reset_case(self, payload, initial):
        return self.runtime.reset_case(payload, initial)

    def set_target(self, target, target_id=None):
        self.runtime.set_target(target, target_id)

    def sample(self):
        state = copy.deepcopy(self.runtime.current_sample)
        if state is None:
            raise ValueError('Reset an explicit physical case before inference')
        state.update(xyz=state['drone_xyz'], velocity=state['drone_velocity'],
                     quaternion_wxyz=state['drone_quaternion_wxyz'],
                     effective_target=self.runtime.target.tolist(), actions=self.runtime.last_actions.tolist(),
                     tilt_deg=payload_metrics(state, self.runtime.target, self.runtime.payload)['drone_tilt_deg'],
                     done=False, reset=False, shared_settled=False,
                     payload_acceptance_contract=SETTLING_CONTRACT)
        return state

    def advance(self):
        self.runtime._require_active()
        if self.runtime.observation is None:
            raise ValueError('Reset an explicit physical case before inference')
        row = self.runtime.step(self.policy(self.runtime.observation.copy()))
        if row['unsafe']:
            raise RuntimeError('Payload physical safety violation; terminal state retained without reset')
        result = self.sample()
        result.update(row)
        result.update(shared_settled=row['complete'], relative_observation_absmax=float(
            np.max(np.abs(self.runtime.observation[:3]))))
        return result
