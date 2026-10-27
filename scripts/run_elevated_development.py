#!/usr/bin/env python3
"""One new-height mission on declared development inputs; never formal draws."""
import argparse,json
from pathlib import Path
from adaptive_drone_patrol.payload_hybrid_final import robustness_cases
from adaptive_drone_patrol.elevated_flight import run_development
from adaptive_drone_patrol.elevated_physical import clearance_envelope,vertical_cases
from adaptive_drone_patrol.util import json_write


def main():
    p=argparse.ArgumentParser()
    for name in ('output','reference','original','design','routing','camera'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--selection',type=Path);p.add_argument('--provisional',action='store_true');p.add_argument('--case-index',type=int,default=0)
    args=p.parse_args()
    cases=robustness_cases()
    for c in cases:c['case_id']=c['case_id'].replace('final-development','elevated-development');c['suite']='elevated-development'
    if args.case_index not in range(15):raise ValueError('Declared15-case index required')
    declaration=args.output.parent/'development-declaration.json'
    if not declaration.exists():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        json_write(declaration,dict(schema='elevated-development-declaration-v1',cases=cases,vertical_cases=vertical_cases(),
            envelope=clearance_envelope(),routing_heldout_opened=False,physical_heldout_opened=False))
    elif json.loads(declaration.read_text())['cases']!=cases:raise ValueError('Development declaration differs')
    result=run_development(cases[args.case_index],output=args.output,reference=args.reference,original=args.original,
        design=args.design,routing=args.routing,camera=args.camera,selection=args.selection,provisional=args.provisional)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
