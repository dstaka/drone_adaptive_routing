#!/usr/bin/env python
"""Fine-tune the existing motor policy for bounded retargeting and continuous hover."""
import argparse
import pickle
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from adaptive_drone_patrol.util import load_config, unique_dir, sha256, json_write
from adaptive_drone_patrol.target_adapter import load_hover_class
from adaptive_drone_patrol.flight_training import finetune_configs, dispersed_configs, dispersed_curriculum_configs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--iterations', type=int, default=1001)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--domain', choices=['visible', 'dispersed'], default='visible')
    p.add_argument('--curriculum', action='store_true')
    a = p.parse_args()
    if not 1 <= a.num_envs <= 256 or a.iterations < 1:
        p.error('Use 1–256 environments and positive iterations')
    if a.curriculum and (a.domain != 'dispersed' or a.iterations % 4):
        p.error('Expanded curriculum requires four equal positive stages')
    cfg = load_config(a.config)
    for path, key in [('flight_checkpoint','flight_checkpoint_hash'),('flight_cfg','flight_cfg_hash')]:
        if sha256(cfg[path]) != cfg[key]:
            raise ValueError(f'Artifact mismatch: {path}')
    base = load_hover_class(cfg['hover_source'],cfg['hover_source_hash'], external=False)

    class TrainingEnv(base):
        def _at_target(self):
            stable = (torch.norm(self.rel_pos,dim=1) <= .15) & (torch.norm(self.base_lin_vel,dim=1) <= .30)
            self.hover_steps = torch.where(stable, self.hover_steps+1, 0)
            reached = (self.hover_steps >= 50) | ((self.episode_length_buf % (800 if a.curriculum else 300)) == 0)
            return reached.nonzero(as_tuple=False).reshape(-1)

        def _resample_commands(self, indices):
            super()._resample_commands(indices)
            if a.curriculum and len(indices):
                bounds = [[1.,1.,1.5],[1.5,1.2,2.],[2.1,1.5,2.5],[2.7,1.8,3.]][getattr(self,'domain_stage',0)]
                low=torch.tensor([-bounds[0],-bounds[1],1.],device=gs.device)
                high=torch.tensor(bounds,device=gs.device)
                self.commands[indices]=low+torch.rand((len(indices),3),device=gs.device)*(high-low)
                corners=torch.rand(len(indices),device=gs.device)<.20
                ix=indices[corners]
                self.commands[ix]=torch.where(torch.rand((len(ix),3),device=gs.device)<.5,low,high)
            if hasattr(self, 'hover_steps'):
                self.hover_steps[indices] = 0

        def _reward_hover(self):
            return torch.exp(-torch.sum(self.rel_pos**2,dim=1)/.04) * torch.exp(-torch.sum(self.base_lin_vel**2,dim=1)/.09)

        def _reward_velocity(self):
            return torch.sum(self.base_lin_vel**2,dim=1)

        def _reward_altitude(self):
            return self.rel_pos[:,2]**2

        def _reward_upright(self):
            return torch.sum((self.base_euler[:,:2]*torch.pi/180)**2,dim=1)

    out = unique_dir(a.out)
    original = pickle.load(open(cfg['flight_cfg'],'rb'))
    configs = dispersed_configs(original) if a.domain == 'dispersed' else finetune_configs(original)
    if a.curriculum:
        configs=dispersed_curriculum_configs(original)
    env_cfg,obs_cfg,reward_cfg,command_cfg,train_cfg = configs
    train_cfg['run_name'] = out.name
    gs.init(backend=gs.gpu, precision='32', logging_level='warning',seed=7,performance_mode=True)
    # Initialize counter before the first training step; reset construction does not query arrival.
    env = TrainingEnv(a.num_envs,env_cfg,obs_cfg,reward_cfg,command_cfg,show_viewer=False)
    env.hover_steps = torch.zeros(a.num_envs,dtype=torch.int64,device=gs.device)
    with (out/'cfgs.pkl').open('xb') as stream:
        pickle.dump(configs,stream)
    runner = OnPolicyRunner(env,train_cfg,str(out),device=gs.device)
    runner.load(cfg['flight_checkpoint'])
    initial_iteration = runner.current_learning_iteration
    json_write(out/'training_manifest.json',{'parent_checkpoint_hash':cfg['flight_checkpoint_hash'],
               'parent_config_hash':cfg['flight_cfg_hash'],'source_hash':cfg['hover_source_hash'],
               'seed':7,'num_envs':a.num_envs,'iterations_requested':a.iterations,
               'initial_iteration':initial_iteration,'observation_size':17,'action_size':4,
               'command_bounds':command_cfg,'reward_config':reward_cfg,
               'retarget_period_steps':800 if a.curriculum else 300,'training_dwell_steps':50,
               'curriculum_stages':[[1.,1.,1.5],[1.5,1.2,2.],[2.1,1.5,2.5],[2.7,1.8,3.]] if a.curriculum else None,
               'corner_fraction':.20 if a.curriculum else 0})
    json_write(out/'compatibility.json', {'domain':a.domain,
               'observation_contract':'hover17-relative-world-v1',
               'position_scale':obs_cfg['obs_scales']['rel_pos'], 'position_clip':[-1,1],
               'coordinate_meaning':'world metres; target minus vehicle position',
               'changed_normalization':False, 'parent_is_initialization_only':True,
               'transition_validation':'PENDING; training completion is not acceptance'})
    if a.curriculum:
        # Loading restores optimizer state; explicitly enforce the declared fixed rate.
        runner.alg.learning_rate=.00003
        for group in runner.alg.optimizer.param_groups:
            group['lr']=.00003
        for stage in range(4):
            env.domain_stage=stage
            runner.learn(num_learning_iterations=a.iterations//4,init_at_random_ep_len=True)
    else:
        runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
    checkpoints = sorted(out.glob('model_*.pt'), key=lambda path:int(path.stem.split('_')[-1]))
    json_write(out/'completed.json',{'checkpoint':str(checkpoints[-1].relative_to(out)),
               'sha256':sha256(checkpoints[-1]),'config_sha256':sha256(out/'cfgs.pkl'),
               'final_iteration':runner.current_learning_iteration})


if __name__ == '__main__':
    main()
