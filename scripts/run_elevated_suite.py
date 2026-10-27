#!/usr/bin/env python3
"""Selected-model development suite; manifest first, one attempt per case."""
import argparse,json,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from adaptive_drone_patrol.payload_hybrid_final import robustness_cases
from adaptive_drone_patrol.elevated_physical import vertical_cases
from adaptive_drone_patrol.elevated_flight import run_development,run_vertical
from adaptive_drone_patrol.elevated_binding import development_binding
from adaptive_drone_patrol.util import json_write,sha256


def worker(params):
 case,kwargs,vertical=params
 try:
  sources={str(p):sha256(p) for p in Path('src/adaptive_drone_patrol').glob('*.py')}
  result=(run_vertical if vertical else run_development)(case,**kwargs)
  if sources!={str(p):sha256(p) for p in Path('src/adaptive_drone_patrol').glob('*.py')}:raise ValueError('Source changed during trial')
  root=Path(kwargs['output'])
  if not (root/'executed-source.json').exists():
   json_write(root/'executed-source.json',sources)
   for name in sources:
    saved=root/'executed-source'/name;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(Path(name).read_bytes())
  return dict(case_id=case['case_id'],success=result['success'])
 except BaseException as exc:
  root=Path(kwargs['output']);root.mkdir(parents=True,exist_ok=True)
  json_write(root/'interruption.json',dict(error=repr(exc),traceback=traceback.format_exc(),hypothesis='Inspect retained failure; no alternate case or automatic tuning',next_action='Report failed gate'))
  return dict(case_id=case['case_id'],success=False,error=repr(exc))


def main():
 p=argparse.ArgumentParser()
 for name in ('output','reference','original','design','routing','camera','selection'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--vertical',action='store_true');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
 if not 1<=a.workers<=6:raise ValueError('One to six independent N1 CPU workers')
 binding=development_binding(a.routing,a.camera,selection=a.selection)
 cases=vertical_cases() if a.vertical else robustness_cases()
 if not a.vertical:
  for c in cases:c['case_id']=c['case_id'].replace('final-development','elevated-development');c['suite']='elevated-development'
 a.output.mkdir(parents=True,exist_ok=False)
 json_write(a.output/'declaration.json',dict(schema='elevated-selected-development-suite-v1',binding=binding,cases=cases,attempts_per_case=1))
 args=[(c,dict(output=a.output/c['case_id'],reference=a.reference,original=a.original,design=a.design,routing=a.routing,camera=a.camera,selection=a.selection),a.vertical) for c in cases]
 results=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for result in pool.map(worker,args):
   results.append(result);json_write(a.output/'progress.json',results);print(json.dumps(result),flush=True)
 json_write(a.output/'summary.json',dict(successes=sum(r['success'] for r in results),total=len(cases),cases=results,passed=all(r['success'] for r in results)))

if __name__=='__main__':main()
