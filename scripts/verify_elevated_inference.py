#!/usr/bin/env python3
"""Verify CPU inference portability against retained validation orders, no selection."""
import argparse,json
from pathlib import Path
from adaptive_drone_patrol.routing_agent import RoutingAgent
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.benchmark import evaluate_scenario
from adaptive_drone_patrol.util import sha256,json_write
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
models=json.loads((r/'terminal-models.json').read_text());cases=json.loads((r/'validation-manifest.json').read_text());rows=[]
for name,item in models.items():
 path=r/'baseline/B.zip' if name=='B' else r/name/'terminal.zip';agent=RoutingAgent(path,item['sha256'])
 for i,case in enumerate(cases):
  original=json.loads((r/f'routing-validation/case-{i:03d}.json').read_text())['results'][name]
  current=evaluate_scenario(Scenario.from_dict(case['scenario']),agent.predict)
  rows.append(dict(model=name,index=i,order_equal=current['order']==original['order'],cost_error=abs(current['analytical_cost']-original['analytical_cost'])))
 print(name,flush=True)
passed=all(x['order_equal'] and x['cost_error']<1e-10 for x in rows)
json_write(r/'inference-portability.json',dict(passed=passed,cases_per_model=300,rows=rows,source_sha256=sha256(__file__),scope='Same saved validation inputs/checkpoints; inference compatibility only, no reselection'))
print('PORTABILITY',passed,flush=True)
if not passed:raise ValueError('CPU inference semantics differ across dependency environments')
