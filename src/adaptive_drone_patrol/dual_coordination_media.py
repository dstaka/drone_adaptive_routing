"""New dual-scene camera/HUD; rendering never confers physical acceptance."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from .dual_coordination_schema import WAIT, decode_action

_PROFILE=dict(schema='dual-camera-v1',fps=100,border_fraction=.05,terminal_display_ticks=1,
              collision_proxies='hidden-in-display-still-in-projection-checks',
              hud_height=300,views={
                  'overview':dict(width=1280,height=720,distance=12.,azimuth=135.,elevation=-35.,lookat=[0.,0.,2.5]),
                  'drone-1':dict(width=512,height=720,distance=1.2,azimuth=135.,elevation=-10.),
                  'drone-2':dict(width=512,height=720,distance=1.2,azimuth=135.,elevation=-10.)})


def camera_profile():
    return deepcopy(_PROFILE)


def validate_camera(config):
    if config!=_PROFILE:
        raise ValueError('Exact declared dual-camera-v1 profile required')
    return deepcopy(config)


def verify_frame_coverage(frames,config,*,terminal_tick):
    validate_camera(config)
    if type(terminal_tick) is not int or terminal_tick<1 or len(frames)!=terminal_tick+1:
        raise ValueError('Complete reset-through-terminal frame sequence required')
    for i,row in enumerate(frames):
        if row['index']!=i or abs(row['time_s']-i/100)>1e-9:
            raise ValueError('Every native timestamp must appear exactly once')
        if set(row['views'])!=set(config['views']):
            raise ValueError('All three intended panels required')
        for name,view in row['views'].items():
            c=config['views'][name]
            boxes=view['assembly_bounds_px']
            if len(boxes)!=(2 if name=='overview' else 1):
                raise ValueError('Complete assemblies required in each panel')
            if name=='overview':
                boxes=boxes+[view['goal_bounds_px']]
            for box in boxes:
                b=np.asarray(box,dtype=float)
                m=config['border_fraction']
                if (b.shape!=(4,) or not np.isfinite(b).all()
                        or not (c['width']*m<=b[0]<b[2]<=c['width']*(1-m)
                                and c['height']*m<=b[1]<b[3]<=c['height']*(1-m))):
                    raise ValueError('Projected geometry violates declared 5% border')
    return dict(every_frame_verified=True,frames=len(frames),fps=100,
                physics_duration_s=terminal_tick/100,displayed_duration_s=(terminal_tick+1)/100,
                final_terminal_display_s=.01,playback_rate=1.,all_three_panels=True,border_fraction=.05)


class DualNativeRenderer:
    """Three native panels; projection checks include all assembly geom bounds."""
    def __init__(self,fixture,goals,config=None):
        self.fixture=fixture
        self.config=validate_camera(config or camera_profile())
        self.goals=np.asarray(goals,dtype=float)
        if self.goals.shape!=(10,3) or not np.isfinite(self.goals).all():
            raise ValueError('Ten finite destinations required')
        mj=fixture.mj
        from .dual_coordination_audit import NativePointAuditor
        self.model=NativePointAuditor(fixture.model_path,fixture.runtime()).model
        model=self.model
        model.vis.global_.offwidth=max(c['width'] for c in self.config['views'].values())
        model.vis.global_.offheight=720
        model.vis.headlight.ambient[:]=[.6,.6,.6]
        model.vis.headlight.diffuse[:]=[.75,.75,.75]
        # Display-only clone: distinguish the two visual meshes from the floor.
        for geom in range(model.ngeom):
            name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,geom) or ''
            if model.geom_type[geom]==mj.mjtGeom.mjGEOM_MESH:
                model.geom_rgba[geom]=[.3,.7,1.,1.] if name.startswith('drone-1/') else [1.,.65,.3,1.]
            elif name.endswith('/vehicle_collision') or name.endswith('-proxy'):
                model.geom_rgba[geom,3]=0.
        self.data=mj.MjData(model)
        self.ids=[]
        for d in (1,2):
            ids=[i for i in range(model.ngeom)
                 if (mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,i) or '').startswith(f'drone-{d}/')]
            if not ids:
                raise ValueError('Both complete named assemblies required')
            self.ids.append(ids)
        self.renderers={}
        self.cameras={}
        for name,c in self.config['views'].items():
            self.renderers[name]=mj.Renderer(model,height=c['height'],width=c['width'])
            camera=mj.MjvCamera()
            camera.type=mj.mjtCamera.mjCAMERA_FREE
            camera.distance,camera.azimuth,camera.elevation=c['distance'],c['azimuth'],c['elevation']
            self.cameras[name]=camera

    def render(self,record,*,render_pixels=True):
        from .payload_mujoco_overview_media import project_spheres
        if type(render_pixels) is not bool:
            raise ValueError('Explicit render_pixels boolean required')
        mj,model=self.fixture.mj,self.model
        actual=record['actual']
        tick=record['tick']
        if abs(actual['world_time']-tick/100)>1e-7:
            raise ValueError('Native frame clock differs')
        self.data.qpos[:],self.data.qvel[:]=actual['qpos'],actual['qvel']
        self.data.ctrl[:],self.data.time=actual['ctrl'],actual['world_time']
        mj.mj_forward(model,self.data)
        views={}
        pixels=[]
        for name,camera in self.cameras.items():
            camera.lookat[:]=self.config['views']['overview']['lookat'] if name=='overview' else np.mean(
                [actual['drones'][int(name[-1])-1][key] for key in ('drone_xyz','package_xyz')],axis=0)
            renderer=self.renderers[name]
            renderer.update_scene(self.data,camera=camera)
            # Destination markers belong to the overview. A marker at the
            # reached destination would otherwise occlude the entire drone in
            # its detail panel while still passing a geometric border check.
            for i,goal in enumerate(self.goals if name=='overview' else []):
                if renderer.scene.ngeom>=renderer.scene.maxgeom:
                    raise ValueError('Scene cannot show all destinations')
                geom=renderer.scene.geoms[renderer.scene.ngeom]
                owner=record['mission']['goals'][i]['owner']
                done=record['mission']['goals'][i]['status']=='DONE'
                color=[.3,.9,.4,1.] if done else ([.2,.6,1.,1.] if owner==0 else [1.,.55,.2,1.] if owner==1 else [.8,.8,.8,1.])
                mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_SPHERE,np.array([.08,.08,.08]),goal,np.eye(3).ravel(),np.array(color,dtype=np.float32))
                geom.label=chr(65+i)
                renderer.scene.ngeom+=1
            c=self.config['views'][name]
            kw=dict(width=c['width'],height=c['height'],fovy=model.vis.global_.fovy)
            selected=self.ids if name=='overview' else [self.ids[int(name[-1])-1]]
            bounds=[project_spheres(renderer.scene.camera,self.data.geom_xpos[ids],model.geom_rbound[ids],**kw) for ids in selected]
            views[name]=dict(assembly_bounds_px=bounds,goal_bounds_px=project_spheres(
                renderer.scene.camera,self.goals,np.full(10,.08),**kw) if name=='overview' else None)
            if render_pixels:
                pixels.append(renderer.render())
        receipt=dict(index=tick,time_s=tick/100,views=views)
        # Reuse the exact per-frame gate without pretending this is a traversal.
        check=dict(receipt,index=0,time_s=0.)
        verify_frame_coverage([check,dict(check,index=1,time_s=.01)],self.config,terminal_tick=1)
        return np.concatenate(pixels,axis=1) if render_pixels else None,receipt

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()


def compose_japanese_hud(native,record,*,font,method,config=None):
    from PIL import Image,ImageDraw,ImageFont
    config=validate_camera(config or camera_profile())
    native=np.asarray(native)
    width=sum(c['width'] for c in config['views'].values())
    if native.shape!=(720,width,3) or native.dtype!=np.uint8 or not Path(font).is_file():
        raise ValueError('Exact three native RGB panels and Japanese font required')
    image=Image.new('RGB',(width,720+config['hud_height']),'#101d2d')
    image.paste(Image.fromarray(native),(0,0))
    draw=ImageDraw.Draw(image)
    face=ImageFont.truetype(str(font),26)
    lines=hud_lines(record,method=method)
    for i,line in enumerate(lines):
        bounds=draw.textbbox((24,732+44*i),line,font=face)
        if bounds[0]<0 or bounds[2]>width-24 or bounds[3]>image.height-12:
            raise ValueError('Japanese HUD text would be clipped')
        draw.text((24,732+44*i),line,font=face,fill='#edf3fc')
    output=np.asarray(image).copy()
    if not np.array_equal(output[:720],native):
        raise ValueError('HUD modified native pixels')
    return output


def hud_lines(record,*,method):
    """Display zero-based ownership as human IDs and retain observed events."""
    mission=record['mission']
    action=decode_action(record['action']) if record.get('action') is not None else (WAIT,WAIT,mission['priority'])
    def target(i):
        return '待機' if i==WAIT or i is None else chr(65+i)
    summary='  '.join(chr(65+i)+':'+('完了' if g['status']=='DONE' else '未担当' if g['owner'] is None else '機'+str(g['owner']+1)) for i,g in enumerate(mission['goals']))
    audit=record.get('online_audit',{})
    minimum=audit.get('minimum_clearance_m',record.get('audit',{}).get('min_clearance_m'))
    distance='未計測' if minimum is None or not math.isfinite(minimum) else f'{minimum:.4f} m'
    held=record.get('supervisor',{}).get('held',[False,False])
    execution='  '.join('機'+str(i+1)+('安全停止' if held[i] else '待機' if mission['targets'][i]==WAIT else '担当へ追従') for i in range(2))
    event=record.get('event') or (mission.get('event_receipts',[]) or [None])[-1]
    if event is not None:
        event=event.get('event',event)
    event_text='なし' if not event else '優先度変更' if event['kind']=='priority' else '担当機の一時待機'
    return [f'MuJoCo 2機・各機吊り荷 | {method} | 全編1倍速 | 経過 {record["tick"]/100:.2f} 秒',
           f'提案: 機1→{target(action[0])}  機2→{target(action[1])}  優先: 機{action[2]}',
           f'担当: 機1→{target(mission["targets"][0])}  機2→{target(mission["targets"][1])} | 実行: {execution}',
           summary,
           f'実測最小機体間距離: {distance} | イベント: {event_text}',
           'シミュレーション／飛行制御NNの学習なし／物理受入・人手確認は別判定']
