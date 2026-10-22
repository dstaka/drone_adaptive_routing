"""Native-ball retry PPO environment; physical admission belongs to its caller.

Reuse the original curriculum/reward/terminal guards through inheritance while
never constructing the failed Euler environment. No optimizer is created here.
"""

from pathlib import Path
from collections import deque
import json
import math
import numpy as np

from .payload_training import PayloadTrainingEnv
from .payload_ball_bank import BallSceneBank
from .payload_contract import PayloadSampler, DRONE_ATTACHMENT, rotation
from .payload_retry_contract import PROFILE, OBSERVATION_CONTRACT, actor_observation
from .payload_retry_properties import read_reset_properties, validate_reset_properties


# Declared before any retry learning. These are conservative numerical screening
# guards, not relaxed mission-acceptance thresholds or proof of qualification.
RETRY_TRAINING_GUARDS = dict(speed_max_m_s=20., angular_speed_max_rad_s=200.,
    quaternion_norm_error=1e-5, constraint_residual_m=2e-5, position_abs_max_m=20.)
TRACE_RECORD_MAX_BYTES = 32768
RECENT_RAW_STEPS = 128
BODY_METHODS = (('xyz', 'get_pos'), ('quaternion_wxyz', 'get_quat'),
                ('velocity', 'get_vel'), ('angular_velocity', 'get_ang'))


class _InvalidTrainingSample(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class _TraceBank:
    """Observe one existing bank step without repeating or changing physics."""
    def __init__(self, bank, owner):
        object.__setattr__(self, '_bank', bank)
        object.__setattr__(self, '_owner', owner)

    def __getattr__(self, name):
        return getattr(self._bank, name)

    def __setattr__(self, name, value):
        setattr(self._bank, name, value)

    def step(self, actions):
        measured = self._bank.step(actions)
        self._owner._capture_after_physics(measured)
        return measured

    def reset(self, payload, initial):
        measured = self._bank.reset(payload, initial)
        record = read_reset_properties(self._bank)
        errors = validate_reset_properties(record, payload, initial)
        row = dict(schema='payload-retry-reset-property-trace-v1',
            next_episode=self._owner.episode_count+1, actual=record, failures=errors,
            declared=dict(side_m=payload.side_m, mass_kg=payload.mass, initial=initial))
        with self._owner.reset_trace_path.open('a') as stream:
            stream.write(json.dumps(row,allow_nan=True)+'\n')
            stream.flush()
        if errors:
            raise _InvalidTrainingSample('EFFECTIVE_RESET_PROPERTY_FAILURE')
        return measured


class PayloadRetryTrainingEnv(PayloadTrainingEnv):
    def __init__(self, output, randomized, budget_steps=300000, seed=7):
        if (type(randomized) is not bool or type(budget_steps) is not int or budget_steps <= 0 or
                type(seed) is not int or seed != 7):
            raise ValueError('Retry comparison requires seed7, a positive actual step budget and a declared method')
        import torch
        import genesis as gs
        # Deliberately do not call PayloadTrainingEnv.__init__: it builds failed Euler scenes.
        self.output = Path(output).parent
        self.num_envs, self.num_actions, self.device = 1, 4, gs.device
        self.cfg = dict(profile=PROFILE, num_actions=4, observation_size=29, dt=.01,
                        episode_length_s=30., randomized_payload=randomized,
                        observation_contract=OBSERVATION_CONTRACT,
                        source='ideal simulator motion; immediate four-rotor control with current-action history')
        self.max_episode_length = 3000
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self.bank = _TraceBank(BallSceneBank(output, dt=.01, substeps=2, align_root=True), self)
        self.output.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output/'training-trace.jsonl'
        with self.trace_path.open('x'):
            pass
        self.reset_trace_path = self.output/'reset-properties.jsonl'
        with self.reset_trace_path.open('x'):
            pass
        self._recent_raw = deque(maxlen=RECENT_RAW_STEPS)
        self._saturation_counts = np.zeros(29, dtype=np.int64)
        self._training_aborted = False
        self.cfg['training_trace'] = dict(record_max_bytes=TRACE_RECORD_MAX_BYTES,
            recent_raw_steps=RECENT_RAW_STEPS, guards=dict(RETRY_TRAINING_GUARDS),
            geometry='Episode assignment logged; runtime properties require independent reset audit')
        self.sampler = PayloadSampler(self.bank.reference_mass, seed, 'training', randomized=randomized)
        self.pose_rng = np.random.default_rng(np.random.SeedSequence([seed, 710, 101]))
        self.command_rng = np.random.default_rng(np.random.SeedSequence([seed, 710, 102]))
        self.budget_steps, self.total_steps = budget_steps, 0
        self.episode_count, self.failures, self.completed_stops = 0, 0, 0
        self.reward_totals = {}
        self.last_actions = np.zeros(4)
        self._reset()

    def _observation(self):
        import torch
        from tensordict import TensorDict
        observed = actor_observation(self.sample, self.target, self.last_actions)
        return TensorDict({'policy': torch.as_tensor(observed, dtype=torch.float32,
                                                     device=self.device)[None]}, batch_size=[1])

    def _append_trace(self, record):
        encoded = json.dumps(record, separators=(',', ':'), allow_nan=True)+'\n'
        if len(encoded.encode()) > TRACE_RECORD_MAX_BYTES:
            raise RuntimeError('Training trace record exceeds its predeclared byte bound')
        with self.trace_path.open('a') as stream:
            stream.write(encoded)
            stream.flush()

    @staticmethod
    def _bounded_value(value, limit=64):
        try:
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            a = np.asarray(value)
            if a.size <= limit:
                return a.tolist()
            return {'shape': list(a.shape)[:8], 'values_prefix': a.reshape(-1)[:limit].tolist()}
        except Exception as exc:
            return {'capture_error': type(exc).__name__}

    def _direct_body(self, body):
        result = {}
        for field, method in BODY_METHODS:
            try:
                result[field] = self._bounded_value(getattr(body, method)()[0])
            except Exception as exc:
                result[field] = {'capture_error': type(exc).__name__}
        return result

    def _raw_record(self, measured=None):
        bodies = {}
        for name, body in [('drone', self.bank.drone), ('package', self.bank.package)]:
            if measured is None:
                bodies[name] = self._direct_body(body)
            else:
                bodies[name] = {field: self._bounded_value(measured.get(name+'_'+field))
                                for field, _ in BODY_METHODS}
        bodies['rod'] = self._direct_body(self.bank.drone.get_link('suspension_rod'))
        generalized = {}
        for name, method in [('qpos', 'get_qpos'), ('dofs_velocity', 'get_dofs_velocity')]:
            try:
                generalized[name] = self._bounded_value(getattr(self.bank.drone, method)()[0])
            except Exception as exc:
                generalized[name] = {'capture_error': type(exc).__name__}
        context = self._step_context
        measured_step = (context['step'] if measured is not None or self._physics_step_completed is True else
                         self.total_steps if self._physics_step_completed is False or
                         self.total_steps >= context['step'] else None)
        measured_episode_step = (context['episode_step'] if measured is not None else
            self.episode_steps if self._physics_step_completed is False or
            self.episode_count != context['episode'] or self.total_steps >= context['step'] else
            context['episode_step'] if self._physics_step_completed is True else None)
        return dict(schema='payload-retry-training-trace-v1', kind='RAW_STEP',
                    step=context['step'], episode=context['episode'], episode_step=context['episode_step'],
                    captured_episode=self.episode_count,
                    simulation_time_s=None if measured_step is None else measured_step*.01,
                    captured_episode_step=measured_episode_step,
                    episode_time_s=None if measured_episode_step is None else measured_episode_step*.01,
                    target=context['target'], raw_actions=context['raw_actions'],
                    clipped_actions=context['clipped_actions'], rpm_command_float32=context['rpm_command_float32'],
                    actor_observation_before_step=context['actor_observation_before_step'],
                    bodies=bodies, generalized_state=generalized,
                    episode_geometry_declared=dict(reference_mass_kg=self.payload.reference_mass_kg,
                        mass_ratio=self.payload.mass_ratio, mass_kg=self.payload.mass, side_m=self.payload.side_m,
                        inertia_kg_m2=self.payload.inertia, sampler_draw_index=self.sampler.draw_indices[0],
                        runtime_property_remeasurement='Not performed each step; requires independent reset/property audit'),
                    guard_limits=RETRY_TRAINING_GUARDS,
                    frames='Body centers/velocities/angular velocities in world SI; body-to-world wxyz; qpos/dofs native solver coordinates')

    def _guard_body_state(self, raw):
        for body in raw['bodies'].values():
            for field, size in [('xyz', 3), ('velocity', 3), ('angular_velocity', 3), ('quaternion_wxyz', 4)]:
                try:
                    v = np.asarray(body[field], dtype=float)
                except (TypeError, ValueError):
                    raise _InvalidTrainingSample('INCOMPLETE_RAW_STATE') from None
                if v.shape != (size,):
                    raise _InvalidTrainingSample('INCOMPLETE_RAW_STATE')
                if not np.isfinite(v).all():
                    raise _InvalidTrainingSample('NONFINITE_RAW_STATE')
                if field == 'xyz' and np.max(np.abs(v)) > RETRY_TRAINING_GUARDS['position_abs_max_m']:
                    raise _InvalidTrainingSample('POSITION_EXCURSION')
                if field == 'velocity' and np.linalg.norm(v) > RETRY_TRAINING_GUARDS['speed_max_m_s']:
                    raise _InvalidTrainingSample('LINEAR_SPEED_EXCURSION')
                if field == 'angular_velocity' and np.linalg.norm(v) > RETRY_TRAINING_GUARDS['angular_speed_max_rad_s']:
                    raise _InvalidTrainingSample('ANGULAR_SPEED_EXCURSION')
                if field == 'quaternion_wxyz' and abs(np.linalg.norm(v)-1) > RETRY_TRAINING_GUARDS['quaternion_norm_error']:
                    raise _InvalidTrainingSample('QUATERNION_NORM_EXCURSION')
        state = raw['generalized_state']
        try:
            qpos, velocity = np.asarray(state['qpos'], dtype=float), np.asarray(state['dofs_velocity'], dtype=float)
        except (TypeError, ValueError):
            raise _InvalidTrainingSample('INCOMPLETE_GENERALIZED_STATE') from None
        if qpos.shape != (15,) or velocity.shape != (12,):
            raise _InvalidTrainingSample('INCOMPLETE_GENERALIZED_STATE')
        if not np.isfinite(qpos).all() or not np.isfinite(velocity).all():
            raise _InvalidTrainingSample('NONFINITE_RAW_STATE')
        if any(abs(np.linalg.norm(qpos[start:start+4])-1) > RETRY_TRAINING_GUARDS['quaternion_norm_error']
               for start in (3, 7, 11)):
            raise _InvalidTrainingSample('QUATERNION_NORM_EXCURSION')
        if np.linalg.norm(velocity[:3]) > RETRY_TRAINING_GUARDS['speed_max_m_s']:
            raise _InvalidTrainingSample('LINEAR_SPEED_EXCURSION')
        if any(np.linalg.norm(velocity[start:start+3]) > RETRY_TRAINING_GUARDS['angular_speed_max_rad_s']
               for start in (3, 6, 9)):
            raise _InvalidTrainingSample('ANGULAR_SPEED_EXCURSION')

    @staticmethod
    def _observation_pair(measured, target, actions):
        r = rotation(measured['drone_quaternion_wxyz'])
        direction = np.asarray(measured['attachment_b_world'])-measured['attachment_a_world']
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length < 1e-8:
            raise _InvalidTrainingSample('DEGENERATE_ATTACHMENT_STATE')
        attachment_velocity = (np.asarray(measured['attachment_b_velocity'])-
                               measured['attachment_a_velocity'])
        raw = np.concatenate([
            (np.asarray(target)-measured['drone_xyz'])/3,
            measured['drone_quaternion_wxyz'], r.T @ np.asarray(measured['drone_velocity'])/3,
            r.T @ np.asarray(measured['drone_angular_velocity'])/3.14159, actions,
            (np.asarray(measured['package_xyz'])-measured['drone_xyz'])/.30,
            (np.asarray(measured['package_velocity'])-measured['drone_velocity'])/3,
            direction/length, np.cross(direction, attachment_velocity)/length**2/math.pi,
        ])
        if raw.shape != (29,) or not np.isfinite(raw).all():
            raise _InvalidTrainingSample('NONFINITE_OBSERVATION_SOURCE')
        clipped = raw.copy()
        mask = np.zeros(29, dtype=bool)
        mask[:3], mask[7:13], mask[17:] = True, True, True
        clipped[mask] = np.clip(raw[mask], -1, 1)
        saturated = mask & (np.abs(raw) > 1)
        return raw, clipped, saturated

    def _capture_after_physics(self, measured):
        self._physics_step_completed = True
        raw = self._raw_record(measured)
        self._offending_raw = raw
        self._recent_raw.append(raw)
        # Persist raw simulator measurements before every numerical predicate,
        # rotation, derived metric, normalization, inherited reward or reset.
        self._append_trace(raw)
        self._guard_body_state(raw)
        self._trace_stage = 'ATTACHMENT_GUARD'
        rd, rp = rotation(measured['drone_quaternion_wxyz']), rotation(measured['package_quaternion_wxyz'])
        a_offset, b_offset = rd @ DRONE_ATTACHMENT, rp @ np.array([0, 0, self.payload.side_m/2])
        a, b = np.asarray(measured['drone_xyz'])+a_offset, np.asarray(measured['package_xyz'])+b_offset
        residual = float(abs(np.linalg.norm(b-a)-.30))
        if residual > RETRY_TRAINING_GUARDS['constraint_residual_m']:
            raise _InvalidTrainingSample('CONSTRAINT_RESIDUAL_EXCURSION')
        expected = {'attachment_a_world': a, 'attachment_b_world': b,
                    'attachment_a_velocity': np.asarray(measured['drone_velocity'])+
                        np.cross(measured['drone_angular_velocity'], a_offset),
                    'attachment_b_velocity': np.asarray(measured['package_velocity'])+
                        np.cross(measured['package_angular_velocity'], b_offset)}
        for key, value in expected.items():
            actual = np.asarray(measured[key], dtype=float)
            if actual.shape != (3,) or not np.isfinite(actual).all() or not np.allclose(actual, value, atol=1e-7, rtol=0):
                raise _InvalidTrainingSample('ATTACHMENT_SAMPLE_MISMATCH')
        self._trace_stage = 'PRE_PARENT_OBSERVATION'
        preclip, clipped, saturated = self._observation_pair(measured, self._step_context['target'], self.last_actions)
        contacts = self.bank.drone.get_contacts()
        valid = contacts['valid_mask'].detach().cpu().numpy()
        self._saturation_counts += saturated.astype(np.int64)
        self._append_trace(dict(schema='payload-retry-training-trace-v1', kind='PRE_PARENT_OBSERVATION',
            step=self._step_context['step'], observation_preclip=preclip.tolist(),
            observation_clipped=clipped.tolist(), saturation_mask=saturated.tolist(),
            saturation_duration_s=(self._saturation_counts*.01).tolist(),
            saturation_duration_definition='Cumulative duration per field across recorded training steps',
            entity_contact=bool(valid.any()), contact_valid_count=int(valid.sum()),
            constraint_residual_m=residual, attachment_a_world=a.tolist(), attachment_b_world=b.tolist(),
            package_lowest_point_m=float(measured['package_xyz'][2]-self.payload.side_m/2*np.abs(rp[2]).sum()),
            drone_center_altitude_m=float(measured['drone_xyz'][2])))
        self._trace_stage = 'PARENT_GUARDS_REWARD_OR_OBSERVATION'

    def _abort_training(self, exc):
        self._training_aborted = True
        code = exc.code if isinstance(exc, _InvalidTrainingSample) else 'PARENT_STEP_EXCEPTION'
        preceding = self._offending_raw
        # A terminal reset can fail after the preceding valid terminal sample.
        # Always preserve the actual current getters as well as that sample.
        current = self._raw_record()
        try:
            self._guard_body_state(current)
        except _InvalidTrainingSample as guard:
            code = guard.code
            self._offending_raw = current
        if self._offending_raw is None:
            self._offending_raw = current
        if preceding != self._offending_raw:
            self._recent_raw.append(self._offending_raw)
            self._append_trace(self._offending_raw)
            if self._physics_step_completed is not False:
                self._physics_step_completed = 'UNKNOWN_AFTER_BANK_EXCEPTION'
        try:
            self._guard_body_state(self._offending_raw)
        except _InvalidTrainingSample as guard:
            code = guard.code
        before = self._step_context['reward_totals_before']
        partial = {key: value-before.get(key, 0.) for key, value in self.reward_totals.items()
                   if value != before.get(key, 0.)}
        record = dict(schema='payload-retry-training-abort-v1', status='TRAINING_ABORTED_INVALID_SAMPLE',
                      code=code, stage=self._trace_stage, optimizer_sample_admitted=False,
                      action_input=self._step_context['action_input'],
                      physics_step_completed=self._physics_step_completed,
                      offending_raw=self._offending_raw, recent_raw_window=list(self._recent_raw),
                      preceding_post_step_raw=preceding, current_runtime_raw_at_abort=current,
                      exception_type=type(exc).__name__, exception_message=str(exc)[:256],
                      partial_reward_components=partial,
                      partial_reward=sum(partial.values()) if partial else None)
        with (self.output/'training-abort.json').open('x') as stream:
            json.dump(record, stream, separators=(',', ':'), allow_nan=True)
            stream.write('\n')
            stream.flush()
        self._append_trace(dict(schema='payload-retry-training-trace-v1', kind='ABORT',
                                step=self._step_context['step'], code=code, optimizer_sample_admitted=False))
        return code

    def step(self, actions):
        if self._training_aborted:
            raise RuntimeError('Retry training already aborted; no more physical or PPO steps are permitted')
        shape = list(getattr(actions, 'shape', ()))
        try:
            prefix = actions.detach().reshape(-1)[:16].cpu().tolist()
        except Exception:
            prefix = []
        self._step_context = dict(step=self.total_steps+1, episode=self.episode_count,
            episode_step=self.episode_steps+1, target=np.asarray(self.target).tolist(),
            action_input={'shape': shape[:8], 'values_prefix': prefix}, raw_actions=prefix,
            clipped_actions=None, rpm_command_float32=None,
            actor_observation_before_step=self.obs['policy'][0].detach().cpu().tolist(),
            reward_totals_before=dict(self.reward_totals))
        self._offending_raw, self._physics_step_completed = None, False
        self._trace_stage = 'ACTION_INPUT'
        try:
            if shape != [1, 4]:
                raise _InvalidTrainingSample('ACTION_SHAPE')
            if not bool(actions.isfinite().all()):
                raise _InvalidTrainingSample('ACTION_NONFINITE')
            if self.total_steps >= self.budget_steps:
                raise _InvalidTrainingSample('DECLARED_STEP_BUDGET_EXHAUSTED')
            command = actions[0].detach().cpu().numpy().clip(-1, 1).copy()
            self._step_context['clipped_actions'] = command.tolist()
            self._step_context['rpm_command_float32'] = np.asarray(
                (1+.8*np.asarray(command, dtype=float))*14468.429183500699, dtype=np.float32).tolist()
            self.last_actions = command
            self._trace_stage = 'BANK_STEP'
            self._physics_step_completed = 'UNKNOWN_DURING_BANK_STEP'
            result = super().step(actions)
            observation, reward, done, extra = result
            # This may be the next episode after a normal terminal reset. The
            # saved RAW_STEP remains the actual preceding terminal state.
            raw_next, _, _ = self._observation_pair(self.sample, self.target, self.last_actions)
            before = self._step_context['reward_totals_before']
            components = {key: value-before.get(key, 0.) for key, value in self.reward_totals.items()}
            self._append_trace(dict(schema='payload-retry-training-trace-v1', kind='STEP_OUTCOME',
                step=self._step_context['step'], episode=self._step_context['episode'],
                next_episode=self.episode_count, reward=float(reward[0].item()),
                reward_components=components, done=bool(done[0].item()),
                next_target=np.asarray(self.target).tolist(), next_observation_preclip=raw_next.tolist(),
                next_actor_observation=observation['policy'][0].detach().cpu().tolist()))
            return result
        except Exception as exc:
            code = self._abort_training(exc)
            raise RuntimeError(f'Retry training aborted before PPO consumption: {code}') from exc

    def set_target(self, target):
        target = np.asarray(target, dtype=float)
        if (target.shape != (3,) or not np.isfinite(target).all() or
                np.any(target < [-2.7, -1.8, 1.]) or np.any(target > [2.7, 1.8, 3.])):
            raise ValueError('Retry target is outside the inherited dispersed envelope')
        self.target = target.copy()
        self.obs = self._observation()
