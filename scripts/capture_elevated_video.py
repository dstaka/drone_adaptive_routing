#!/usr/bin/env python3
"""Full100Hz new-profile native capture, recomposition and viewing-copy audit."""
import argparse,gzip,json
from itertools import zip_longest
from pathlib import Path
import numpy as np
from adaptive_drone_patrol.elevated_media import ElevatedRenderer,load_elevated_camera,projected_bounds,elevated_hud,framing_report
from adaptive_drone_patrol.elevated_binding import verify_binding_scenario
from adaptive_drone_patrol.payload_solver_media import NativeRenderer,load_camera,native_frame_rows,mission_display_state
from adaptive_drone_patrol.payload_solver_video import PairedVideoWriter,verify_video_pair,decoded_rgb,create_viewing_copies,verify_viewing_copies
from adaptive_drone_patrol.payload_hybrid_media import compare_recomposition
from adaptive_drone_patrol.payload_hybrid_audit import independent_metrics,_valid
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.util import sha256,json_write


def main():
 p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--font',type=Path,required=True);p.add_argument('--final',action='store_true');p.add_argument('--freeze',type=Path);p.add_argument('--formal',type=Path);a=p.parse_args()
 trial=a.trial; result=json.loads((trial/'result.json').read_text());binding=json.loads((trial/'elevated-binding.json').read_text());case=json.loads((trial/'case.json').read_text())
 if not result['success'] or not result['numeric']['passed'] or not result['vertical']['passed']:raise ValueError('Independent mission/numerical/altitude admission required')
 frozen=None
 if a.final:
  from adaptive_drone_patrol.elevated_campaign import JointFreeze,formal_cases,audit_prerequisite
  from adaptive_drone_patrol.elevated_media import require_recording_case
  require_recording_case(case)
  if a.freeze is None or a.formal is None:raise ValueError('Joint freeze and independent formal campaign required')
  from adaptive_drone_patrol.elevated_formal import verify_formal_acceptance
  from adaptive_drone_patrol.analytical import content_hash
  frozen=JointFreeze.load(a.freeze)
  stats=verify_formal_acceptance(a.formal,formal_cases(frozen),sha256(a.freeze),content_hash(frozen.record))
  if (case not in formal_cases(frozen)
      or a.trial.resolve()!=(a.formal/case['case_id']).resolve() or binding!=frozen.record['binding']):raise ValueError('Final capture case/source/campaign differs')
  audit_prerequisite(a.trial,case,frozen.record['paths'])
 scenario=Scenario.from_dict(json.loads((trial/'mission/scenario.json').read_text()));verify_binding_scenario(scenario,case,binding)
 config=load_elevated_camera(binding['camera_path']);events=[json.loads(l) for l in (trial/'mission/events.jsonl').read_text().splitlines()]
 before={str(p.relative_to(trial)):sha256(p) for p in trial.rglob('*') if p.is_file()}
 assets={str(p):sha256(p) for p in [a.font,Path(binding['camera_path']),Path(binding['routing_path'])]}
 predicates={r['simulation_step']:r for r in [json.loads(l) for l in (trial/'mission/predicates.jsonl').read_text().splitlines()]}
 def frames():
  with gzip.open(trial/'flight/runtime/raw.jsonl.gz','rt') as stream:yield from native_frame_rows(json.loads(l) for l in stream)
 def state_at(row):
  state=mission_display_state(scenario,events,row['step']);metric=independent_metrics(row['actual'],state['target'],case['side_m'])
  valid=_valid(metric,row['actual']['contact']);pred=predicates.get(row['step'],{});dwell=pred.get('dwell_s',0.) if pred.get('target_id')==state['selected'] else 0.
  return state,valid,dwell
 sources={str(p.resolve().relative_to(Path.cwd().resolve())):sha256(p) for p in [Path(__file__),*Path('src/adaptive_drone_patrol').glob('*.py')]}
 snapshot=a.output.parent/(a.output.name+'-source');snapshot.mkdir(parents=True,exist_ok=False)
 for name in sources:
  dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
 json_write(snapshot/'manifest.json',sources)
 renderer=ElevatedRenderer(trial/'flight/fixture/model.xml',config)
 old=NativeRenderer(trial/'flight/fixture/model.xml',load_camera('configs/payload_mujoco_overview_camera.json'),cpu_only=True)
 bounds=[]
 ids=[i for i in range(renderer.model.ngeom) if renderer.mj.mj_id2name(renderer.model,renderer.mj.mjtObj.mjOBJ_BODY,renderer.model.geom_bodyid[i]) in ['base_link',*[f'prop{j}_link' for j in range(4)]]]
 def render(row,compare_old):
  state,valid,dwell=state_at(row);native=row['actual']
  raw=renderer.render(native,scenario.coords[1:],state['priorities'],state['visited'],state['selected'])
  hud=elevated_hud(raw,state,native,case,config,a.font,binding,goals=scenario.coords[1:],dwell_s=dwell,valid=valid,final=a.final)
  if compare_old:
   old.render(native,scenario.coords[1:],state['priorities'],state['visited'],state['selected'])
   widths=[]
   for r in (old,renderer):
    b=projected_bounds(r.renderers['detail'].scene.camera,r.data.geom_xpos[ids],r.model.geom_rbound[ids],width=512,height=720,fovy=r.model.vis.global_.fovy);widths.append(b[2]-b[0])
   bounds.append(dict(**renderer.bounds[-1],step=row['step'],phase='settling' if valid else 'travel',old_drone_width=widths[0],drone_width=widths[1]))
  return raw,hud
 try:
  with PairedVideoWriter(a.output,width=1792,raw_height=720,hud_height=480) as writer:
   for row in frames():
    raw,hud=render(row,True);writer.append(row['step'],raw,hud)
    if row['step']%500==0:print('CAPTURE',row['step'],flush=True)
  framing=framing_report(bounds,frames=result['mission']['steps']+1);json_write(a.output/'framing.json',bounds)
  lossless=verify_video_pair(a.output)
  streams=[decoded_rgb(a.output/'raw.mkv',1792,720),decoded_rgb(a.output/'hud.mkv',1792,1200)]
  try:
   for row,raw,hud in zip_longest(frames(),*streams):
    if row is None or raw is None or hud is None:raise ValueError('Native/decoded complete-frame count differs')
    fresh,fresh_hud=render(row,False);compare_recomposition(raw,hud,fresh,fresh_hud,raw_height=720)
    if row['step']%500==0:print('RECOMPOSE',row['step'],flush=True)
  finally:
   for stream in streams:stream.close()
  create_viewing_copies(a.output);viewing=verify_viewing_copies(a.output)
  if before!={str(p.relative_to(trial)):sha256(p) for p in trial.rglob('*') if p.is_file()}:raise ValueError('Physical raw changed during capture')
  if any(sha256(p)!=h for p,h in assets.items()):raise ValueError('Camera/routing/font asset changed')
  if any(sha256(name)!=digest for name,digest in sources.items()):raise ValueError('Executed media source changed during capture')
  if frozen is not None:
   frozen.require_current()
   verify_formal_acceptance(a.formal,formal_cases(frozen),sha256(a.freeze),content_hash(frozen.record))
  proof=dict(schema='elevated-final-media-v1' if a.final else 'elevated-development-media-v1',new_physical_success_count=0,
    binding=binding,trial_files=before,assets=assets,framing=framing,lossless=lossless,viewing=viewing,
    renderer=renderer.renderer_string,whole_recording_recomposed=True,human_review='PENDING_FRESH_CONTINUOUS1X',
    sources=sources,source_snapshot=str(snapshot),freeze_sha256=sha256(a.freeze) if a.final else None,
    formal_independent_statistics_sha256=sha256(a.formal/'independent-statistics.json') if a.final else None,
    videos={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in a.output.glob('*') if p.suffix in ('.mp4','.mkv')})
  json_write(a.output/'video-proof.json',proof);print('VIDEO_VERIFIED',a.output,framing,flush=True)
 finally:renderer.close();old.close()

if __name__=='__main__':main()
