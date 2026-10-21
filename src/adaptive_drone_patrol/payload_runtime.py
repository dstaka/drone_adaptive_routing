"""Payload-only inference; evaluation never automatically resets or resamples."""

import pickle
import numpy as np
from .payload_bank import PayloadSceneBank
from .payload_contract import actor_observation, rotation, OBSERVATION_CONTRACT, require_checkpoint_contract
from .payload_checkpoint import validate_parent_config, actor_signature
from .util import sha256


class PayloadFlight:
    def __init__(self, checkpoint, checkpoint_hash, parent_cfg, output):
        import genesis as gs
        import torch
        from tensordict import TensorDict
        from rsl_rl.runners import OnPolicyRunner
        if sha256(checkpoint)!=checkpoint_hash:
            raise ValueError('Payload checkpoint hash mismatch')
        self.checkpoint_sha256=checkpoint_hash
        self.config_sha256=sha256(parent_cfg)
        data = torch.load(checkpoint,map_location='cpu',weights_only=False)
        require_checkpoint_contract(data['infos']['observation_contract'],OBSERVATION_CONTRACT)
        if data['infos'].get('physics_signature')!='payload-articulation-v1-isolated-scenes':
            raise ValueError('Payload physics signature mismatch')
        cfg = pickle.load(open(parent_cfg,'rb'))
        validate_parent_config(cfg)
        self.num_envs,self.num_actions,self.device = 1,4,gs.device
        self.cfg = dict(profile='suspended-payload-v1',observation_contract=OBSERVATION_CONTRACT)
        self.obs = TensorDict({'policy':torch.zeros((1,29),device=gs.device)},batch_size=[1])
        self.bank = PayloadSceneBank(output)
        runner = OnPolicyRunner(self,cfg[4],None,device=gs.device)
        runner.load(checkpoint,load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False,'rnd':False})
        self.policy = runner.get_inference_policy(device=gs.device)
        self.actor_sha256=actor_signature(self.policy.state_dict())
        if self.actor_sha256!=actor_signature(data['actor_state_dict']):
            raise ValueError('Actual inference actor differs from checkpoint tensors')
        self.last_actions = np.zeros(4)
        self.target = None

    def get_observations(self):
        return self.obs

    def reset_case(self,payload,initial):
        self.sample = self.bank.reset(payload,initial)
        self.last_actions = np.zeros(4)
        self.target = np.array([0.,0.,1.])
        self._observation()
        return self.sample

    def _observation(self):
        import torch
        from tensordict import TensorDict
        s = self.sample
        r = rotation(s['drone_quaternion_wxyz'])
        old = np.concatenate([np.clip((self.target-np.array(s['drone_xyz']))/3,-1,1),
                              s['drone_quaternion_wxyz'],np.clip(r.T@np.array(s['drone_velocity'])/3,-1,1),
                              np.clip(r.T@np.array(s['drone_angular_velocity'])/3.14159,-1,1),self.last_actions])
        self.obs = TensorDict({'policy':torch.as_tensor(actor_observation(old,s),dtype=torch.float32,
                                                        device=self.device)[None]},batch_size=[1])

    def set_target(self,target):
        before = (self.bank.drone.get_qpos().clone(),self.bank.drone.get_dofs_velocity().clone(),self.last_actions.copy())
        self.target = np.array(target,dtype=float)
        if self.target.shape!=(3,) or not np.isfinite(self.target).all():
            raise ValueError('Invalid target')
        self._observation()
        if (not np.array_equal(before[0].cpu().numpy(),self.bank.drone.get_qpos().cpu().numpy()) or
            not np.array_equal(before[1].cpu().numpy(),self.bank.drone.get_dofs_velocity().cpu().numpy()) or
            not np.array_equal(before[2],self.last_actions)):
            raise RuntimeError('Target switch changed physical state or policy history')

    def advance(self):
        import torch
        with torch.no_grad():
            actions = self.policy(self.obs)[0].cpu().numpy().clip(-1,1)
        self.sample = self.bank.step(actions)
        self._observation()
        self.last_actions = actions.copy()
        contacts = self.bank.drone.get_contacts()
        self.sample.update(actions=actions.tolist(),contacts=bool(contacts['valid_mask'].any().item()),
                           effective_target=self.target.tolist(),
                           relative_position_preclip=((self.target-np.array(self.sample['drone_xyz']))/3).tolist(),
                           relative_position_clipped=np.clip((self.target-np.array(self.sample['drone_xyz']))/3,-1,1).tolist())
        return self.sample
