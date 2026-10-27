#!/usr/bin/env python3
"""Open routing held-out once under joint freeze; report all4 without reselection."""
import argparse,json
from pathlib import Path
from adaptive_drone_patrol.elevated_campaign import JointFreeze
from adaptive_drone_patrol.elevated_routing_data import generate_case
from adaptive_drone_patrol.elevated_routing_experiment import evaluate_models
from adaptive_drone_patrol.util import sha256,json_write
p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);a=p.parse_args()
f=JointFreeze.load(a.freeze);r=Path(f.record['run']);manifest=r/'routing-heldout-manifest.json'
if manifest.exists() or (r/'routing-heldout').exists():raise FileExistsError('One declared held-out pass; preserve exposed cohort')
with (r/'routing-heldout-claim.json').open('x') as claim:
 json.dump(dict(freeze_sha256=sha256(a.freeze),status='CLAIMED_BEFORE_INPUT_GENERATION'),claim)
inputs=json.loads((r/'frozen-routing-input-hashes.json').read_text());seen=set(inputs['hashes']);seen.add(inputs['demo_coordinate_sha256'])
cases=[]
for i in range(300):
 c=generate_case('held-out',i,freeze=f)
 if c['coordinate_sha256'] in seen or c['content_sha256'] in seen:
  json_write(r/'routing-heldout-collision.json',dict(index=i,case=c,status='CAMPAIGN_INVALID_NO_REPLACEMENT'))
  raise ValueError('Held-out collision invalidates campaign; no replacement/reseed')
 seen.update([c['coordinate_sha256'],c['content_sha256']]);cases.append(c)
json_write(manifest,cases);json_write(r/'routing-heldout-exposure.json',dict(freeze_sha256=sha256(a.freeze),input_sha256=sha256(manifest),cases=300,reselection=False))
models={}
for name,item in f.record['routing_models'].items():
 path=r/'baseline/B.zip' if name=='B' else r/name/'terminal.zip'
 models[name]=dict(item,path=str(path))
result=evaluate_models(models,cases,r/'routing-heldout',input_manifest=manifest,selected_model=f.record['selected'])
json_write(r/'routing-heldout-result.json',result);print('ROUTING_HELD_OUT',result['status'],flush=True)
