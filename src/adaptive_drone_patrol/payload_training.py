"""Single-environment payload PPO curriculum with complete episode resampling."""

import math
import json
from pathlib import Path
import numpy as np
from .payload_bank import PayloadSceneBank
from .payload_contract import (PayloadSampler, SharedSettling, payload_metrics, settled,
                               actor_observation, rotation, training_outside_envelope, numerical_excursion)
from .payload_cases import initial_state


class PayloadTrainingEnv:
    def __init__(self, output, randomized, budget_steps, seed=7):
        import torch
        import genesis as gs
        self.output=Path(output).parent
        self.num_envs, self.num_actions = 1, 4
        self.device = gs.device
        self.cfg = dict(profile='suspended-payload-v1',num_actions=4,observation_size=29,
                        dt=.01,episode_length_s=30.,randomized_payload=randomized,
                        source='ideal simulator motion; four learned motor commands')
        self.max_episode_length = 3000
        self.episode_length_buf = torch.zeros(1,dtype=torch.long,device=gs.device)
        self.bank = PayloadSceneBank(output)
        self.sampler = PayloadSampler(self.bank.reference_mass,seed,'training',randomized=randomized)
        self.pose_rng = np.random.default_rng(np.random.SeedSequence([seed,710,101]))
        self.command_rng = np.random.default_rng(np.random.SeedSequence([seed,710,102]))
        self.budget_steps, self.total_steps = budget_steps, 0
        self.episode_count, self.failures, self.completed_stops = 0, 0, 0
        self.reward_totals = {}
        self.last_actions = np.zeros(4)
        self._reset()

    def _target(self):
        stage = min(5,int(self.total_steps/self.budget_steps*6))
        if stage == 0:
            return np.array([0.,0.,1.])
        bounds = [(0,0,1),(.5,.5,1.5),(1.5,1.2,2.),(2.7,1.8,3.),(2.7,1.8,3.),(2.7,1.8,3.)][stage]
        if stage == 4:
            # Explicit full-span reversals, ascents and descents.
            corners = [(-2.7,-1.8,1.),(2.7,1.8,3.),(2.7,-1.8,1.),(-2.7,1.8,3.)]
            return np.array(corners[self.command_count%4])
        low, high = np.array([-bounds[0],-bounds[1],1.]),np.array(bounds)
        return self.command_rng.uniform(low,high)

    def _reset(self):
        self.payload = self.sampler.reset([0])[0]
        self.initial = initial_state(self.pose_rng,self.payload.side_m)
        self.sample = self.bank.reset(self.payload,self.initial)
        self.last_actions = np.zeros(4)
        self.episode_steps, self.command_steps, self.command_count = 0,0,0
        self.target = self._target()
        self.dwell = SharedSettling(.01)
        self.episode_count += 1
        self.obs = self._observation()

    def _observation(self):
        import torch
        from tensordict import TensorDict
        s = self.sample
        r = rotation(s['drone_quaternion_wxyz'])
        old = np.concatenate([np.clip((self.target-np.array(s['drone_xyz']))/3,-1,1),
                              s['drone_quaternion_wxyz'],
                              np.clip(r.T@np.array(s['drone_velocity'])/3,-1,1),
                              np.clip(r.T@np.array(s['drone_angular_velocity'])/3.14159,-1,1),
                              self.last_actions])
        obs = actor_observation(old,s)
        return TensorDict({'policy':torch.as_tensor(obs,dtype=torch.float32,device=self.device)[None]},batch_size=[1])

    def get_observations(self):
        return self.obs

    def step(self, actions):
        import torch
        command = actions[0].detach().cpu().numpy().clip(-1,1)
        before = np.array(self.sample['drone_xyz'])
        try:
            self.sample = self.bank.step(command)
        except Exception as exc:
            raw={'error':repr(exc),'step':self.total_steps,'target':self.target.tolist()}
            for name,body in [('drone',self.bank.drone),('package',self.bank.package)]:
                for key,method in [('xyz','get_pos'),('quaternion_wxyz','get_quat'),('velocity','get_vel'),('angular_velocity','get_ang')]:
                    raw[f'{name}_{key}']=getattr(body,method)()[0].detach().cpu().tolist()
            (self.output/'invalid-raw-state.json').write_text(json.dumps(raw,allow_nan=True)+'\n')
            raise
        self.total_steps += 1
        self.episode_steps += 1
        self.command_steps += 1
        self.episode_length_buf[:] = self.episode_steps
        m = payload_metrics(self.sample,self.target,self.payload)
        contacts = self.bank.drone.get_contacts()
        contact = bool(contacts['valid_mask'].any().item())
        failed = (m['drone_floor_m'] <= .1 or m['package_floor_m'] <= .1 or
                  abs(m['drone_roll_deg'])>60 or abs(m['drone_pitch_deg'])>60 or
                  m['constraint_residual_m']>2e-5 or contact or training_outside_envelope(self.sample,self.target))
        current = np.array(self.sample['drone_xyz'])
        terms = dict(progress=2*(np.linalg.norm(before-self.target)**2-np.linalg.norm(current-self.target)**2),
                     hover=.5*math.exp(-m['drone_error_m']**2/.04)*math.exp(-m['drone_speed_m_s']**2/.09),
                     velocity=-.03*m['drone_speed_m_s']**2,
                     altitude=-.5*(current[2]-self.target[2])**2,
                     angular=-.01*np.linalg.norm(self.sample['drone_angular_velocity']),
                     swing=-.05*math.radians(m['swing_angle_deg'])**2,
                     swing_rate=-.01*math.radians(m['swing_rate_deg_s'])**2,
                     crash=-200.*failed)
        reward = .01*sum(terms.values())
        for key,value in terms.items():
            self.reward_totals[key] = self.reward_totals.get(key,0)+.01*float(value)
        complete = self.dwell.update(self.command_steps*.01,settled(m) and not failed)
        if complete:
            self.completed_stops += 1
        timeout = self.episode_steps >= self.max_episode_length
        done = bool(failed or timeout or (complete and self.command_count>=9))
        self.obs = self._observation()
        self.last_actions = command.copy()
        extras = {'time_outs':torch.tensor([timeout],device=self.device)}
        # Terminal reward and failure metrics exist before resetting any body.
        if done or numerical_excursion(self.sample,reward):
            record=dict(step=self.total_steps,episode=self.episode_count,episode_steps=self.episode_steps,
                        payload=dict(reference_mass_kg=self.payload.reference_mass_kg,mass_ratio=self.payload.mass_ratio,
                                     side_m=self.payload.side_m,draw_index=self.sampler.draw_indices[0]),
                        target=self.target.tolist(),sample=self.sample,metrics=m,reward=reward,reward_components=terms,
                        failed=bool(failed),timeout=bool(timeout))
            with (self.output/'terminal-states.jsonl').open('a') as stream:
                stream.write(json.dumps(record,allow_nan=True)+'\n')
        if numerical_excursion(self.sample,reward):
            raise RuntimeError('Unusable physical numerics: terminal state saved before reset; training not admitted')
        if done:
            self.failures += int(failed)
            extras['episode'] = {'failure':float(failed),'completed_stops':float(self.completed_stops)}
            self._reset()
            self.episode_length_buf[:] = 0
        elif complete or self.command_steps >= 800:
            self.command_count += 1
            self.command_steps = 0
            self.target = self._target()
            self.dwell = SharedSettling(.01)
            self.obs = self._observation()
        return (self.obs,torch.tensor([reward],device=self.device,dtype=torch.float32),
                torch.tensor([done],device=self.device),extras)
