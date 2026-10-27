#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from adaptive_drone_patrol.elevated_artifacts import inventory,verify_inventory
from adaptive_drone_patrol.util import json_write
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',type=Path);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
r=verify_inventory(a.root,json.loads(a.verify.read_text())) if a.verify else inventory(a.root)
json_write(a.output,r);print({k:v for k,v in r.items() if k!='files' or not isinstance(v,dict)},flush=True)
