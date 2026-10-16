#!/usr/bin/env python
"""Audit genuine camera visibility and extract critical raw/annotated contact sheets."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from adaptive_drone_patrol.util import unique_dir,json_write
from adaptive_drone_patrol.scene import project
from adaptive_drone_patrol.physical_evidence import evidence_frame_indices


def audit_raw(run, manifest, frames):
    width,height=manifest['camera']['res']
    positions=[project(m['world_xyz'],manifest['camera']) for m in manifest['markers']]
    decoder=subprocess.Popen(['ffmpeg','-v','error','-i',str(run/'raw.mp4'),'-f','rawvideo',
                              '-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    minimum=[10**9]*len(positions)
    failures=[]
    count=0
    try:
        for frame in frames:
            data=decoder.stdout.read(width*height*3)
            if len(data)!=width*height*3:
                raise ValueError('Visibility audit has fewer decoded frames than capture records')
            image=np.frombuffer(data,dtype=np.uint8).reshape(height,width,3)
            pixels=[]
            for i,(x,y) in enumerate(positions):
                x,y=int(round(x)),int(round(y))
                patch=image[y-14:y+15,x-14:x+15].astype(float)
                neutral=(patch.max(axis=2)-patch.min(axis=2)<30)&(patch.mean(axis=2)>85)
                visible=int(neutral.sum())
                minimum[i]=min(minimum[i],visible)
                pixels.append(visible)
            if any(v<20 for v in pixels):
                failures.append({'frame_index':int(frame['frame_index']),'neutral_pixels':pixels})
            count+=1
        if decoder.stdout.read(1):
            raise ValueError('Visibility audit found unmapped raw frames')
    finally:
        decoder.stdout.close()
        code=decoder.wait(timeout=30)
    if code:
        raise RuntimeError('Visibility decoder failed')
    return {'frames_audited':count,'minimum_neutral_landmark_pixels':dict(zip(manifest['checkpoint_ids'],minimum)),
            'pixel_gate_failures':failures,
            'method':'Neutral landmark patch contrast is a supplementary screen-space check; inspect retained real frames.'}


def extract_sheet(video, indices, frames, output, font):
    output.mkdir(exist_ok=False)
    expression='+'.join(f'eq(n\\,{i})' for i in indices)
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f'select={expression}',
                    '-fps_mode','vfr',str(output/'frame_%03d.png')],check=True)
    images=sorted(output.glob('frame_*.png'))
    if len(images)!=len(indices):
        raise ValueError('Critical frame extraction count mismatch')
    sheet=Image.new('RGB',(1280,264*((len(images)+2)//3)),'#102333')
    draw=ImageDraw.Draw(sheet)
    for j,(image,index) in enumerate(zip(images,indices)):
        x,y=(j%3)*426,(j//3)*264
        sheet.paste(Image.open(image).resize((426,240)),(x,y))
        f=frames[index]
        draw.text((x+5,y+241),f'frame {index} / step {f["simulation_step"]} / event {f["event_seq"]}',
                  font=font,fill='white')
    sheet.save(output/'contact.jpg',quality=90)
    return [str(p.name) for p in images]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--annotations',required=True)
    p.add_argument('--font',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    out=unique_dir(a.out)
    font=ImageFont.truetype(a.font,13)
    results=[]
    for path in sorted(Path(a.root).glob('*/mission_summary.json')):
        run=path.parent
        summary=json.loads(path.read_text())
        manifest=json.loads((run/'scene_manifest.json').read_text())
        frames=list(csv.DictReader((run/'frames.csv').open()))
        events=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
        audit=audit_raw(run,manifest,frames)
        telemetry=list(csv.DictReader((run/'telemetry.csv').open()))
        positions=[project([float(row[k]) for k in ['x','y','z']],manifest['camera']) for row in telemetry]
        audit['drone_outside_frame_samples']=sum(not (20<x<1260 and 110<y<650) for x,y in positions)
        audit['run_id']=summary['run_id']
        if summary['execution_seed']==1:
            indices=evidence_frame_indices(events,frames)
            audit['critical_frame_indices']=indices
            extract_sheet(run/'raw.mp4',indices,frames,out/f'{run.name}_raw',font)
            extract_sheet(Path(a.annotations)/f'{run.name}.annotated.mp4',indices,frames,out/f'{run.name}_annotated',font)
        results.append(audit)
        json_write(out/'visibility.json',{'schema_version':1,'evidence_kind':'physical','runs':results})
    print(json.dumps({'runs_audited':len(results),'frames_audited':sum(r['frames_audited'] for r in results),
                      'pixel_gate_failures':sum(len(r['pixel_gate_failures']) for r in results),
                      'drone_outside_frame_samples':sum(r['drone_outside_frame_samples'] for r in results)}))


if __name__=='__main__':
    main()
