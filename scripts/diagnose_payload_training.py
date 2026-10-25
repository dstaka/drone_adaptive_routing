#!/usr/bin/env python
"""Preserve raw state around expanded-curriculum failure without changing physics."""

import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from adaptive_drone_patrol.payload_training import PayloadTrainingEnv
from adaptive_drone_patrol.util import sha256,json_write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--substeps',type=int,choices=[2,4,8,16],default=2)
    a=p.parse_args()
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    import genesis as gs
    import torch
    from rsl_rl.runners import OnPolicyRunner
    gs.init(backend=gs.gpu,precision='32',logging_level='warning',seed=7)
    import adaptive_drone_patrol.payload_training as training_module
    from adaptive_drone_patrol.payload_bank import PayloadSceneBank
    training_module.PayloadSceneBank=lambda path: PayloadSceneBank(path,substeps=a.substeps)
    env=PayloadTrainingEnv(out/'scenes',True,300000,seed=7)
    env.total_steps=200000
    env.target=env._target()
    env.obs=env._observation()
    cfg=pickle.load(open('inputs/cfgs.pkl','rb'))[4]
    runner=OnPolicyRunner(env,cfg,None,device=gs.device)
    runner.load(a.checkpoint,load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False,'rnd':False})
    policy=runner.get_inference_policy(device=gs.device)
    old_step=env.bank.step
    captured=[]
    def observed(actions):
        try:
            return old_step(actions)
        finally:
            row=dict(index=len(captured),target=env.target.tolist(),actions=np.asarray(actions).tolist(),
                     mass_ratio=env.payload.mass_ratio,side_m=env.payload.side_m)
            for label,body in [('drone',env.bank.drone),('package',env.bank.package)]:
                for key,method in [('xyz','get_pos'),('quaternion_wxyz','get_quat'),('velocity','get_vel'),('angular_velocity','get_ang')]:
                    row[f'{label}_{key}']=getattr(body,method)()[0].detach().cpu().tolist()
            captured.append(row)
            with (out/'raw-states.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=True)+'\n')
    env.bank.step=observed
    error=None
    try:
        for _ in range(a.steps):
            with torch.no_grad():
                action=policy(env.get_observations())
                _,reward,_,_=env.step(action)
            last=captured[-1]
            velocity=max(np.linalg.norm(last['drone_velocity']),np.linalg.norm(last['package_velocity']))
            if not np.isfinite(reward.cpu().numpy()).all() or abs(reward.item())>1e6 or velocity>1e4:
                raise RuntimeError('Diagnostic numerical/energy excursion guard')
    except Exception as exc:
        error=repr(exc)
    json_write(out/'summary.json',dict(status='FAILED' if error else 'PROBE_RECORDED',error=error,
               checkpoint_sha256=sha256(a.checkpoint),substeps=a.substeps,control_dt_s=.01,physics_dt_s=.01/a.substeps,source_sha256=sha256(__file__),samples=len(captured),
               raw_state_sha256=sha256(out/'raw-states.jsonl'),mission_success=False))
    print(error,len(captured))


if __name__=='__main__':
    main()
