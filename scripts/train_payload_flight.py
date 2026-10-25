#!/usr/bin/env python
"""Matched loaded PPO training; selection and acceptance are separate stages."""

import argparse
import copy
import json
import pickle
from pathlib import Path
from adaptive_drone_patrol.payload_checkpoint import validate_parent_config, transfer_state
from adaptive_drone_patrol.payload_contract import OBSERVATION_CONTRACT, require_physical_admission
from adaptive_drone_patrol.util import sha256, json_write, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['randomized','fixed'],required=True)
    p.add_argument('--preflight',required=True)
    p.add_argument('--preflight-sha256',required=True)
    p.add_argument('--iterations',type=int,default=3000)
    a = p.parse_args()
    if sha256(a.preflight)!=a.preflight_sha256:
        raise ValueError('Physical preflight receipt hash mismatch')
    admission=json.loads(Path(a.preflight).read_text())
    require_physical_admission(admission,load_config('configs/payload_preflight.yaml'),
                               sha256('configs/payload_preflight.yaml'))
    for path,digest in admission['source_binding'].items():
        if sha256(path)!=digest:
            raise ValueError('Physical implementation differs from measured preflight')
    if sha256(a.parent)!='26658a31560f71c36fd9e9605b15d8e4f7b9cae8174c1465fd12dcd0d10b3431':
        raise ValueError('Adopted dispersed parent mismatch')
    if sha256(a.config)!='35c6b0be8db90e4e799d453d447a484275317f85098a22cbdf04c19c84056e55':
        raise ValueError('Adopted dispersed configuration mismatch')
    cfg = pickle.load(open(a.config,'rb'))
    validate_parent_config(cfg)
    out = Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    import genesis as gs
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from adaptive_drone_patrol.payload_training import PayloadTrainingEnv
    gs.init(backend=gs.gpu,precision='32',logging_level='warning',seed=7)
    env = PayloadTrainingEnv(out/'scenes',a.method=='randomized',a.iterations*100,seed=7)
    train = copy.deepcopy(cfg[4])
    train['run_name'] = a.method
    train['save_interval'] = 500
    train['algorithm'].update(learning_rate=.00003,schedule='fixed',entropy_coef=0.,max_grad_norm=.3)
    effective_training=copy.deepcopy(train)
    runner = OnPolicyRunner(env,train,str(out),device=gs.device)
    initial = transfer_state(torch.load(a.parent,map_location='cpu',weights_only=False))
    torch.save(initial,out/'initialization.pt')
    runner.load(str(out/'initialization.pt'),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False,'rnd':False})
    json_write(out/'training_manifest.json',dict(method=a.method,seed=7,num_envs=1,
               environment_steps=a.iterations*100,observation_contract=OBSERVATION_CONTRACT,
               parent_sha256=sha256(a.parent),parent_config_sha256=sha256(a.config),
               training_source_sha256=sha256('src/adaptive_drone_patrol/payload_training.py'),
               training_script_sha256=sha256(__file__),optimizer_config=effective_training['algorithm'],
               architecture={k:effective_training[k] for k in ('actor','critic','obs_groups')},
               preflight_sha256=sha256(a.preflight),
               bank_source_sha256=sha256('src/adaptive_drone_patrol/payload_bank.py'),
               initialization_sha256=sha256(out/'initialization.pt'),
               curriculum=['hover','short','medium','long','reversal/ascent/descent','dispersed_xyz'],
               full_distribution_every_stage=True,heldout_opened=False))
    pickle.dump(cfg,open(out/'parent_cfgs.pkl','xb'))
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    runner.save(str(out/'final.pt'),infos={'observation_contract':OBSERVATION_CONTRACT,
                'physics_signature':'payload-articulation-v1-isolated-scenes',
                'status':'TRAINED_NOT_SELECTED'})
    json_write(out/'completed.json',dict(checkpoint_sha256=sha256(out/'final.pt'),
               actual_environment_steps=env.total_steps,episodes=env.episode_count,
               failures=env.failures,completed_training_stops=env.completed_stops,
               reward_components=env.reward_totals,heldout_opened=False))


if __name__=='__main__':
    main()
