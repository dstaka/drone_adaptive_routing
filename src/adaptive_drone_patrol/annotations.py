"""Same-frame priority badges over retained Genesis footage; no timing edits."""
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .scene import priority_color, project, marker_style
from .scenario import Scenario
from .util import json_write


def display_state(scenario, events, event_seq):
    state = {'selected':None, 'visited':[], 'priorities':scenario.priorities.copy(),
             'alert':None, 'cost_update':None, 'event_text':None}
    for event in events:
        if event['event_seq'] > event_seq:
            break
        kind = event['type']
        if kind == 'TARGET_SELECTED':
            state['selected'] = event['target_id']
        if kind == 'WAYPOINT_REACHED':
            state['visited'].append(event['target_id'])
            state['selected'] = None
        if 'marker_state' in event:
            state.update(event['marker_state'])
        if kind == 'PRIORITY_UPDATED':
            update = event['update']
            name = update.get('checkpoint_id', 'C')
            state['alert'] = name
            old = update.get('old', scenario.priorities[scenario.checkpoint_ids.index(name)])
            new = update.get('value', update.get('new', 5))
            state['event_text'] = f'{name}：優先度 {old:.2f} → {new:g}'
        if kind == 'COST_UPDATED':
            state['cost_update'] = event['update']
            update = event['update']
            edges = ', '.join(f'{src}→{dst}' for src,dst in update['edges'])
            old = update.get('old', [1])[0]
            state['event_text'] = f'有向移動コスト {edges}：{old:g} → {update["value"]:g}'
    return state


def draw_frame(image, scenario, manifest, state, pose, sim_time, font_path, explanation=None):
    dx,dy = project(pose,manifest['camera'])
    detail = same_frame_detail(image,[dx,dy])
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path),19)
    small = ImageFont.truetype(str(font_path),16)
    title = ImageFont.truetype(str(font_path),25)
    for face in [font, small, title]:
        try:
            face.set_variation_by_axes([550])
        except (OSError, AttributeError):
            pass

    def text(xy, value, size=font, fill='white'):
        draw.text(xy,value,font=size,fill=fill,stroke_width=1,stroke_fill='#101820')

    text((24,14),f'{scenario.scenario_id}｜全{scenario.num_checkpoints}地点の巡回｜{sim_time:.2f} simulation s',title)
    text((24,52),'全地点は開始時から固定配置。輪郭＝選択中 ／ ✓＝訪問済み ／ !＝緊急',small)
    text((24,77),'世界座標に固定した表示注釈（元のGenesis映像を保持）',small)
    for i,marker in enumerate(manifest['markers']):
        name = marker['id']
        x,y = project(marker['world_xyz'],manifest['camera'])
        color = priority_color(state['priorities'][i])
        draw.ellipse((x-18,y-18,x+18,y+18),outline='#111111',width=7)
        draw.ellipse((x-16,y-16,x+16,y+16),outline=color,width=5)
        if name == state['selected']:
            draw.ellipse((x-24,y-24,x+24,y+24),outline='#00e5ff',width=3)
        if name in state['visited']:
            draw.rectangle((x-22,y-22,x+22,y+22),outline='white',width=2)
            draw.line([(x+17,y-22),(x+22,y-17),(x+32,y-30)],fill='white',width=3)
        if name == state['alert']:
            text((x-32,y-37),'!',title,fill='#ffe600')
        label = f'{name}｜優先度 {state["priorities"][i]:.2f}'
        box = draw.textbbox((0,0),label,font=font)
        width = box[2]+14
        draw.rounded_rectangle((x-width/2,y+22,x+width/2,y+50),radius=5,fill=color,outline='white',width=2)
        draw.text((x-width/2+7,y+22),label,font=font,fill=marker_style(state['priorities'][i])['ink'])
        symbols = []
        if name == state['selected']:
            symbols.append('選択中')
        if name in state['visited']:
            symbols.append('訪問済み')
        if symbols:
            text((x-36,y+51),' / '.join(symbols),small)
    # Tactical map stays outside all world-anchored goal labels.
    draw.rounded_rectangle((1030,150,1265,438),radius=8,fill='#102333',outline='#d0e0e5')
    text((1040,160),'全地点 / 有向コスト',small)
    coords = np.asarray(scenario.coords)
    def map_xy(point):
        return (1147+float(point[0])*95, 300-float(point[1])*115)
    for i,name in enumerate(scenario.checkpoint_ids):
        x,y=map_xy(coords[i+1])
        draw.ellipse((x-7,y-7,x+7,y+7),fill=priority_color(state['priorities'][i]),outline='white')
        text((x-5,y+7),name,small)
    if state['cost_update']:
        # The retained scenario explicitly changes B->A only; use event edge IDs for general inputs.
        update = state['cost_update']
        for edge in update.get('edges', [[2,1]]):
            src,dst = edge
            if isinstance(src,str):
                src = (['S'] + scenario.checkpoint_ids).index(src)
                dst = (['S'] + scenario.checkpoint_ids).index(dst)
            start,end = map_xy(coords[src]),map_xy(coords[dst])
            dashed_arrow(draw,start,end,'#00e5ff')
        text((1041,404),'破線矢印：1 → 5',small,fill='#00e5ff')
    # Drone outline follows this frame's measured physical position.
    draw.ellipse((dx-20,dy-20,dx+20,dy+20),outline='#66ffb2',width=2)
    text((dx+24,dy-12),'Drone',small,fill='#66ffb2')
    if explanation is not None:
        low,high = explanation['selected_L'],explanation['selected_H']
        text((22,154),'距離と優先度の比較',font)
        for j,name in enumerate([low,high]):
            idx = scenario.checkpoint_ids.index(name)
            measured = float(np.linalg.norm(coords[idx+1]-pose))
            text((22,190+j*29),f'{name}: 実距離 {measured:.2f} m / p={state["priorities"][idx]:.2f}',small)
            x,y = project(coords[idx+1],manifest['camera'])
            text((x-32,y-63),f'{"L" if j==0 else "H"}：{measured:.2f} m',small,fill='#ffe600')
        text((22,260),'現在の条件を固定した',small)
        text((22,285),'残り巡回の最適解との比較',small)
        for j,key in enumerate(['exact','policy_rollout']):
            value=explanation[key]['cost']
            text((22,320+j*27),f'{"J*" if j==0 else "J_policy"} = {value:.5f}',small)
        hidx=scenario.checkpoint_ids.index(high)
        q=next(q for q in explanation['forced_first'] if q['action']==hidx)
        text((22,374),f'Q*({high}) = {q["cost"]:.5f}',small)
        text((22,405),'物理時間・エネルギーの最適性ではない',small)
    image.paste(detail,(24,455))
    draw.rectangle((23,454,265,636),outline='#66ffb2',width=2)
    text((24,432),'同フレームの機体拡大',small,fill='#66ffb2')
    if state['event_text']:
        text((24,100),state['event_text'],font,fill='#ffe600')
    text((390,665),'薄い色＝低優先度 / 濃い赤＝高優先度',small)
    for i in range(1,6):
        x=855+(i-1)*52
        draw.rectangle((x,667,x+25,691),fill=priority_color(i),outline='white')
        text((x+29,666),str(i),small)
    return image


def dashed_arrow(draw, start, end, color):
    delta=np.array(end)-start
    length=np.linalg.norm(delta)
    if length == 0:
        return
    direction=delta/length
    for offset in np.arange(0,length,12):
        a=np.array(start)+direction*offset
        b=np.array(start)+direction*min(offset+7,length)
        draw.line([tuple(a),tuple(b)],fill=color,width=3)
    tip=np.array(end)
    side=np.array([-direction[1],direction[0]])
    draw.polygon([tuple(tip),tuple(tip-direction*13+side*6),tuple(tip-direction*13-side*6)],fill=color)


def annotate_run(run, output, font_path, explanations=None):
    run,output=Path(run),Path(output)
    if output.exists():
        raise FileExistsError(output)
    scenario=Scenario.from_dict(json.loads((run/'scenario.json').read_text()))
    manifest=json.loads((run/'scene_manifest.json').read_text())
    events=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
    frames=list(csv.DictReader((run/'frames.csv').open()))
    telemetry={int(r['simulation_step']):r for r in csv.DictReader((run/'telemetry.csv').open())}
    initial=json.loads((run/'initial_state.json').read_text())['xyz']
    width,height=manifest['camera']['res']
    fps=1/(float(frames[1]['pts_s'])-float(frames[0]['pts_s']))
    decoder=subprocess.Popen(['ffmpeg','-v','error','-i',str(run/'raw.mp4'),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    encoder=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
                              '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20',
                              '-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE)
    checks=[]
    featured=[x for x in (explanations or []) if x['nearer_lower_priority_gate']]
    state_cache={}
    try:
        for frame in frames:
            data=decoder.stdout.read(width*height*3)
            if len(data)!=width*height*3:
                raise ValueError('Raw video/frame-map count mismatch')
            step,seq=int(frame['simulation_step']),int(frame['event_seq'])
            pose=initial if step==0 else [float(telemetry[step][k]) for k in ['x','y','z']]
            state=state_cache.setdefault(seq,display_state(scenario,events,seq))
            feature=next((x for x in featured if x['physical_reference']['event_seq'] <= seq
                          and state['selected']==scenario.checkpoint_ids[x['policy_action']]),None)
            image=Image.frombytes('RGB',(width,height),data)
            draw_frame(image,scenario,manifest,state,pose,float(frame['sim_time_s']),font_path,feature)
            encoder.stdin.write(image.tobytes())
            checks.append({'frame_index':int(frame['frame_index']),'simulation_step':step,'event_seq':seq,
                           'pts_s':float(frame['pts_s']),'priorities':state['priorities'],
                           'selected':state['selected'],'visited':state['visited'],
                           'projected_ids':{m['id']:project(m['world_xyz'],manifest['camera']) for m in manifest['markers']},
                           'drone_projected_xy':project(pose,manifest['camera']),
                           'explanation_decision_id':feature['decision_id'] if feature else None})
        if decoder.stdout.read(1):
            raise ValueError('Raw video contains unmapped frames')
    finally:
        decoder.stdout.close()
        encoder.stdin.close()
        codes=[decoder.wait(timeout=30),encoder.wait(timeout=30)]
    if any(codes):
        raise RuntimeError(f'Annotation codec failure: {codes}')
    json_write(output.with_suffix('.frames.json'),{'schema_version':1,'kind':'world_anchored_annotation',
               'raw_preserved':True,'timing_edits':False,
               'detail_view':{'source':'unannotated same raw frame','crop_size_px':[96,72],
                              'display_rect_px':[24,455,240,180]},'frames':checks})
    return checks


def same_frame_detail(image, position):
    """Crop genuine current camera pixels before any badge can cover the vehicle."""
    x,y = position
    left = max(0,min(image.width-96,int(round(x))-48))
    top = max(0,min(image.height-72,int(round(y))-36))
    return image.crop((left,top,left+96,top+72)).resize((240,180),Image.Resampling.LANCZOS)
