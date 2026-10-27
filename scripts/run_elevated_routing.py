#!/usr/bin/env python3
"""Finite declared routing comparison; intentionally never opens held-out."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback
from adaptive_drone_patrol.util import json_write,sha256
from adaptive_drone_patrol.elevated_routing_data import Dataset,KEY_CONTRACT
from adaptive_drone_patrol.elevated_routing_experiment import fresh_candidate,train_stages,evaluate_models,PPO_SETTINGS,BASELINE_SHA256
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

class ProgressReceipt(BaseCallback):
    def __init__(self,path,env):
        super().__init__(); self.path=path; self.env=env
    def _on_step(self): return True
    def _on_rollout_end(self):
        json_write(self.path/'progress-receipt.json',dict(stage=getattr(self.model,'elevated_stage',None),actual_transitions=self.model.num_timesteps,sampling_episodes=len(self.env.samples),resumable=False))
        json_write(self.path/'sampling-progress.json',self.env.samples)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(); run=args.run
    declaration=run/'routing-declaration.json'
    if declaration.exists(): raise FileExistsError('No implicit restart/resume of declared campaign')
    if sha256(run/'baseline/B.zip')!=BASELINE_SHA256: raise ValueError('Immutable baseline SHA mismatch')
    sources=[*Path('src/adaptive_drone_patrol').glob('*.py'),Path(__file__).resolve().relative_to(Path.cwd())]
    settings=dict(schema='elevated-routing-declaration-v1',key_contract=KEY_CONTRACT,
        seeds=[42,43,44],requested_per_stage=50000,actual_per_stage=50176,total_budget=301056,
        ppo=PPO_SETTINGS,reward_scale=.01,optimizer=dict(name='Adam',lr=.0003,betas=[.9,.999],eps=1e-5,weight_decay=0,amsgrad=False),
        architecture='unchanged CandidatePolicy; baseline policy/value weights only',teacher=False,
        dependencies={n:importlib.metadata.version(n) for n in ['numpy','torch','gymnasium','stable-baselines3','sb3-contrib']},
        source_sha256={str(p):sha256(p) for p in sources},selection_spec_sha256=sha256('specs/2026-10-25-elevated-payload-destinations-detail-zoom/spec.md'),
        baseline_sha256=sha256(run/'baseline/B.zip'),status='DECLARED_BEFORE_GENERATION',heldout_opened=False)
    json_write(declaration,settings)
    snap=run/'routing-executed-source'; snap.mkdir()
    for p in sources:
        dest=snap/p; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    data=Dataset(run/'routing-inputs'); validation=data.validation()
    json_write(run/'validation-manifest.json',validation)
    models={'B':dict(path=str((run/'baseline/B.zip').resolve()),sha256=sha256(run/'baseline/B.zip'),seed=None)}
    model=env=None; seed=None
    try:
        for seed in [42,43,44]:
            out=run/f'F{seed}'; out.mkdir(exist_ok=False)
            model,env=fresh_candidate(run/'baseline/B.zip',seed,data)
            model.set_logger(configure(str(out),['csv']))
            json_write(out/'initialization.json',dict(seed=seed,weights_sha256=settings['baseline_sha256'],optimizer_entries=len(model.policy.optimizer.state),num_timesteps=model.num_timesteps))
            print(f'F{seed} started',flush=True)
            receipt=train_stages(model,env,callback=ProgressReceipt(out,env))
            model.save(out/'terminal.zip'); json_write(out/'terminal.schema.json',model.routing_schema)
            json_write(out/'sampling.json',env.samples); receipt['checkpoint_sha256']=sha256(out/'terminal.zip')
            json_write(out/'training.json',receipt)
            models[f'F{seed}']=dict(path=str((out/'terminal.zip').resolve()),sha256=receipt['checkpoint_sha256'],seed=seed)
            json_write(run/'terminal-models.partial.json',models)
            print(f'F{seed} terminal {receipt}',flush=True)
        json_write(run/'terminal-models.json',models)
        result=evaluate_models(models,validation,run/'routing-validation')
        json_write(run/'routing-result.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        if env is not None:
            json_write(run/f'F{seed}'/'sampling-interrupted.json',env.samples)
        json_write(run/'routing-interruption.json',dict(error=repr(exc),traceback=traceback.format_exc(),models=models,active_seed=seed,
            active_stage=getattr(model,'elevated_stage',None),actual_transitions=getattr(model,'num_timesteps',0),
            sampling_episodes=len(env.samples) if env else 0,
            hypothesis='Inspect retained traceback; no seed restart or expanded training authorized',next_action='Preserve partial state and report incomplete campaign',heldout_opened=False))
        raise

if __name__=='__main__': main()
