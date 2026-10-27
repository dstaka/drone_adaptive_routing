#!/usr/bin/env python3
"""Complete input sets and independently re-audit new-height joint prerequisites."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from adaptive_drone_patrol.elevated_campaign import JointFreeze,audit_prerequisite,development_cases
from adaptive_drone_patrol.elevated_physical import vertical_cases
from adaptive_drone_patrol.elevated_routing_data import generate_case
from adaptive_drone_patrol.util import json_write
from probe_elevated_framing import probe


def training_input(params):
 root,index=params;path=Path(root)/'train'/f'{index:04d}.json';expected=generate_case('train',index,kind='static')
 if path.exists():
  if json.loads(path.read_text())!=expected:raise ValueError('Training input cache/key contract changed')
 else:json_write(path,expected)
 return expected['coordinate_sha256'],expected['content_sha256']


def audit_job(params):
 trial,case,paths,vertical,out,framing=params
 result=audit_prerequisite(trial,case,paths,vertical=vertical)
 json_write(Path(out)/f"{case['case_id']}.json",result)
 if not vertical:probe(trial,paths['camera'],Path(framing)/f"{case['case_id']}.json")
 return case['case_id']


def main():
 p=argparse.ArgumentParser()
 for name in ('run','development','vertical','reference','original','design','routing','camera','selection','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 audits=a.run/'joint-prerequisite-audits';framing=a.run/'selected-framing-v1'
 audits.mkdir(exist_ok=False);framing.mkdir(exist_ok=False)
 # Materialize all10000 declared training indices before sealing either held-out.
 # No policy query, teacher data or optimizer is involved.
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  hashes=list(pool.map(training_input,[(a.run/'routing-inputs',i) for i in range(10000)],chunksize=10))
 seen=set()
 for pair in hashes:
  if any(v in seen for v in pair):raise ValueError('Train collision invalidates campaign')
  seen.update(pair)
 for path in (a.run/'routing-inputs/validation').glob('*.json'):
  c=json.loads(path.read_text())
  if c['coordinate_sha256'] in seen or c['content_sha256'] in seen:raise ValueError('Validation collision invalidates campaign')
  seen.update([c['coordinate_sha256'],c['content_sha256']])
 from adaptive_drone_patrol.elevated_layout import transform_coordinates
 from adaptive_drone_patrol.dispersed_layout import coordinate_hash
 from adaptive_drone_patrol.util import load_config
 demo=coordinate_hash(transform_coordinates(load_config('configs/demo_scenarios_dispersed/normal.yaml')['coords']))
 if demo in seen:raise ValueError('Demo contamination invalidates campaign')
 json_write(a.run/'frozen-routing-input-hashes.json',dict(train=10000,validation=300,hashes=sorted(seen),demo_coordinate_sha256=demo))
 paths={k:str(getattr(a,k)) for k in ('reference','original','design','routing','camera','selection')}
 jobs=[]
 for directory,cases,vertical in [(a.development,development_cases(),False),(a.vertical,vertical_cases(),True)]:
  jobs.extend((directory/c['case_id'],c,paths,vertical,audits,framing) for c in cases)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for name in pool.map(audit_job,jobs):print('INDEPENDENT_PREREQUISITE',name,flush=True)
 frozen=JointFreeze.create(a.run,development=a.development,vertical=a.vertical,paths=paths,framing=framing,output=a.output)
 print('JOINT_FROZEN',frozen.record['selected'],frozen.record['selected_sha256'],flush=True)

if __name__=='__main__':main()
