#!/usr/bin/env python3
"""Matched old/new native camera previews from old raw: display evidence only."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from PIL import Image
from adaptive_drone_patrol.elevated_media import load_elevated_camera,ElevatedRenderer,projected_bounds
from adaptive_drone_patrol.payload_solver_media import NativeRenderer,load_camera,native_frame_rows,mission_display_state
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.util import json_write,sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=False)
 scenario=Scenario.from_dict(json.loads((args.trial/'mission/scenario.json').read_text()))
 events=[json.loads(l) for l in (args.trial/'mission/events.jsonl').read_text().splitlines()]
 # Two declared travel times and two raw verified dwell-end points.
 stops=[e['simulation_step'] for e in events if e['type']=='SHARED_SETTLED']
 picks=sorted(set([100,250,stops[0],stops[min(3,len(stops)-1)]]))
 old=NativeRenderer(args.trial/'flight/fixture/model.xml',load_camera('configs/payload_mujoco_overview_camera.json'),cpu_only=True)
 new=ElevatedRenderer(args.trial/'flight/fixture/model.xml',load_elevated_camera('configs/payload_mujoco_elevated_camera.json'))
 rows=[]
 try:
  with gzip.open(args.trial/'flight/runtime/raw.jsonl.gz','rt') as stream:
   for row in native_frame_rows(json.loads(l) for l in stream):
    if row['step'] not in picks:continue
    state=mission_display_state(scenario,events,row['step']); native=row['actual']
    frames=[r.render(native,scenario.coords[1:],state['priorities'],state['visited'],state['selected']) for r in (old,new)]
    for label,frame in zip(('old','new'),frames):Image.fromarray(frame).save(args.output/f"{row['step']:05d}-{label}.png")
    ids=[i for i in range(new.model.ngeom) if new.model.geom_bodyid[i] not in (0,) and new.mj.mj_id2name(new.model,new.mj.mjtObj.mjOBJ_BODY,new.model.geom_bodyid[i]) in ['base_link',*[f'prop{j}_link' for j in range(4)]]]
    sizes=[]
    for renderer in (old,new):
     b=projected_bounds(renderer.renderers['detail'].scene.camera,renderer.data.geom_xpos[ids],renderer.model.geom_rbound[ids],width=512,height=720,fovy=renderer.model.vis.global_.fovy);sizes.append(b[2]-b[0])
    rows.append(dict(step=row['step'],phase='settling' if row['step'] in stops else 'travel',old_drone_width_px=sizes[0],new_drone_width_px=sizes[1],ratio=sizes[1]/sizes[0],bounds=new.bounds[-1]))
 finally:old.close();new.close()
 json_write(args.output/'preview-proof.json',dict(schema='elevated-matched-old-raw-preview-v1',new_physical_success_count=0,scope='DISPLAY_ONLY_OLD_GEOMETRY',rows=rows,raw_sha256=sha256(args.trial/'flight/runtime/raw.jsonl.gz'),images={p.name:sha256(p) for p in args.output.glob('*.png')}))
 print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
