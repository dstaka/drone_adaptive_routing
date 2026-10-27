#!/usr/bin/env python3
"""Complete selected-development pose projections with both physical panels."""
import argparse,gzip,json
from pathlib import Path
from adaptive_drone_patrol.elevated_media import CameraProbe,load_elevated_camera,framing_report
from adaptive_drone_patrol.payload_solver_media import native_frame_rows,mission_display_state
from adaptive_drone_patrol.payload_hybrid_audit import independent_metrics,_valid
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.util import json_write,sha256


def probe(trial,camera,output):
 trial=Path(trial);case=json.loads((trial/'case.json').read_text());scenario=Scenario.from_dict(json.loads((trial/'mission/scenario.json').read_text()))
 events=[json.loads(l) for l in (trial/'mission/events.jsonl').read_text().splitlines()]
 c=load_elevated_camera(camera);p=CameraProbe(trial/'flight/fixture/model.xml',c);rows=[]
 with gzip.open(trial/'flight/runtime/raw.jsonl.gz','rt') as stream:
  for row in native_frame_rows(json.loads(l) for l in stream):
   state=mission_display_state(scenario,events,row['step']);native=row['actual'];bounds=p.bounds(native,scenario.coords[1:])
   metric=independent_metrics(native,state['target'],case['side_m'])
   bounds.update(step=row['step'],phase='settling' if _valid(metric,native['contact']) else 'travel');rows.append(bounds)
 report=framing_report(rows,frames=len(rows));output=Path(output)
 if output.exists():raise FileExistsError(output)
 with gzip.open(output.with_suffix('.rows.jsonl.gz'),'xt') as stream:
  for row in rows:stream.write(json.dumps(row)+'\n')
 json_write(output,dict(schema='elevated-selected-development-framing-v1',**report,
                       raw_sha256=sha256(trial/'flight/runtime/raw.jsonl.gz'),camera_sha256=sha256(camera),
                       source_sha256=sha256(__file__),rows_sha256=sha256(output.with_suffix('.rows.jsonl.gz'))))
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--camera',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(probe(a.trial,a.camera,a.output),flush=True)
