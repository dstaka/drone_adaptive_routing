#!/usr/bin/env python3
"""Supplement original validation records; never rewrite completed experiment bytes."""
import argparse,json
from pathlib import Path
from adaptive_drone_patrol.elevated_routing_experiment import counterfactual_response
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.util import json_write,sha256
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
cases=json.loads((r/'validation-manifest.json').read_text());rows=[]
for i,case in enumerate(cases):
 p=r/f'routing-validation/case-{i:03d}.json';data=json.loads(p.read_text());s=Scenario.from_dict(case['scenario'])
 for model,result in data['results'].items():
  if s.event:rows.append(dict(case=i,model=model,original_sha256=sha256(p),**counterfactual_response(s,result)))
json_write(r/'routing-validation/event-applicability-audit.json',dict(schema='elevated-event-applicability-v1',rows=rows,
  source_sha256=sha256(__file__),note='Original update logs and selected checkpoint unchanged; completed-target and unreachable-source directed edges reported inapplicable to the remaining route.'))
