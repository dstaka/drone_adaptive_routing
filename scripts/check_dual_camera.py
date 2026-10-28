"""Project every full-rate development frame before freezing a new camera."""
import argparse
import json
from pathlib import Path
import numpy as np
from adaptive_drone_patrol.dual_coordination_media import DualNativeRenderer,camera_profile,verify_frame_coverage
from adaptive_drone_patrol.dual_coordination_media import compose_japanese_hud
from adaptive_drone_patrol.dual_coordination_progress import atomic_json,RunJournal
from adaptive_drone_patrol.dual_coordination_raw import read_raw
from adaptive_drone_patrol.dual_payload_mujoco import DualMuJoCoFixture
from adaptive_drone_patrol.payload_contract import Payload
from adaptive_drone_patrol.payload_solver_mujoco import FREE_MASS_KG
from adaptive_drone_patrol.util import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','original','output','journal-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--font')
    args=p.parse_args()
    source,output=Path(args.source),Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    j=RunJournal(args.journal_root)
    attempt=j.start('camera-full-development-traversal',str(source),category='media verification')
    renderer=None
    try:
        metadata=json.loads((source/'environment.json').read_text())
        fixture=DualMuJoCoFixture(args.original,output/'fixture',
            [Payload(FREE_MASS_KG,load['mass_ratio'],load['side_m']) for load in metadata['case']['loads']])
        fixture.reset(metadata['case']['initials'])
        renderer=DualNativeRenderer(fixture,metadata['case']['goals'])
        receipts=[]
        for path in sorted(source.glob('replay-*.draw')):
            for row in read_raw(path):
                if row.get('replay_stride_ticks',1)!=1 or row['tick']!=len(receipts):
                    raise ValueError('Full-rate consecutive development frames required')
                _,receipt=renderer.render(row,render_pixels=False)
                receipts.append(receipt)
                if row['tick']==0:
                    from PIL import Image
                    frame,_=renderer.render(row)
                    Image.fromarray(frame).save(output/'first-frame.png')
                    if args.font:
                        Image.fromarray(compose_japanese_hud(frame,row,font=args.font,method='B-static（開発確認）')).save(output/'first-hud.png')
                last=row
                if len(receipts)%250==0:
                    j.heartbeat(attempt,completed=len(receipts),reason='Checking every original native frame and all3panels')
        if not receipts:
            raise ValueError('No development frames')
        from PIL import Image
        frame,_=renderer.render(last)
        Image.fromarray(frame).save(output/'terminal-frame.png')
        if args.font:
            Image.fromarray(compose_japanese_hud(frame,last,font=args.font,method='B-static（開発確認）')).save(output/'terminal-hud.png')
        report=verify_frame_coverage(receipts,camera_profile(),terminal_tick=last['tick'])
        atomic_json(output/'framing.json',receipts)
        atomic_json(output/'camera.json',camera_profile())
        report.update(scope='DEVELOPMENT_CAMERA_ONLY',camera_sha256=sha256(output/'camera.json'),
                      framing_sha256=sha256(output/'framing.json'),minimum_pixel_intensity=int(np.min(frame)))
        if args.font:
            report['japanese_font_sha256']=sha256(args.font)
        atomic_json(output/'camera-verification.json',report)
        j.finish(attempt,status='passed',reason=f'{len(receipts)}frames all3panels/5%border passed')
        print(json.dumps(report,indent=2))
    except BaseException as error:
        j.finish(attempt,status='failed',reason=repr(error))
        raise
    finally:
        if renderer is not None:
            renderer.close()


if __name__=='__main__':
    main()
