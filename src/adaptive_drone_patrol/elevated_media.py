"""Explicit elevated camera reader and conservative whole-geometry projections."""
import json
from pathlib import Path
import numpy as np
from .payload_solver_media import NativeRenderer
from .elevated_layout import PROFILE


def load_elevated_camera(path):
    c=json.loads(Path(path).read_text())
    if (c.get('schema')!='payload-elevated-camera-v1' or c.get('profile')!=PROFILE
        or (c['fps'],c['raw_width'],c['raw_height'],c['hud_height'])!=(100,1792,720,480)
        or (c['overview']['width'],c['overview']['height'],c['detail']['width'],c['detail']['height'])!=(1280,720,512,720)
        or not .85<=c['detail']['distance']<=1.05):
        raise ValueError('Explicit elevated100Hz camera/distance required')
    for view in ('overview','detail'):
        v=c[view]; xyz=v['lookat' if view=='overview' else 'lookat_offset']
        if (np.asarray(xyz).shape!=(3,) or not np.isfinite(xyz).all()
            or not np.isfinite([v['distance'],v['azimuth'],v['elevation']]).all() or v['distance']<=0):
            raise ValueError('Finite fixed camera required')
    return c


def projected_bounds(camera,positions,radii,*,width,height,fovy):
    p=np.asarray(positions,float); r=np.asarray(radii,float)
    eye=np.mean([v.pos for v in camera],axis=0)
    forward=np.mean([v.forward for v in camera],axis=0); up=np.mean([v.up for v in camera],axis=0)
    if (p.ndim!=2 or p.shape[1]!=3 or r.shape!=(len(p),) or not len(p)
        or not np.isfinite(p).all() or not np.isfinite(r).all() or np.any(r<0)
        or not np.isfinite([*eye,*forward,*up,fovy]).all() or not 0<fovy<180
        or np.linalg.norm(forward)<1e-12 or np.linalg.norm(up)<1e-12):
        raise ValueError('Invalid geometry/camera projection')
    forward=forward/np.linalg.norm(forward); right=np.cross(forward,up)
    if np.linalg.norm(right)<1e-12:raise ValueError('Degenerate camera basis')
    right/=np.linalg.norm(right); up=np.cross(right,forward)
    scale=height/(2*np.tan(np.deg2rad(fovy)/2)); boxes=[]
    for center,radius in zip(p,r):
        delta=center-eye; depth=delta@forward
        near=max(float(getattr(v,'frustum_near',0.)) for v in camera)
        if depth-radius<=near: raise ValueError('Behind-camera/near-plane physical projection')
        # Exact conservative extrema of the projected sphere. The off-axis
        # center moves under perspective; simply adding r/(depth-r) at the
        # projected center is not conservative for all off-axis placements.
        def extent(component):
            v=float(delta@component); den=depth*depth-radius*radius
            spread=radius*np.sqrt(depth*depth+v*v-radius*radius)
            return (v*depth-spread)/den,(v*depth+spread)/den
        x0,x1=extent(right); y0,y1=extent(up)
        boxes.append([width/2+scale*x0,height/2-scale*y1,width/2+scale*x1,height/2-scale*y0])
    a=np.array(boxes)
    return [float(a[:,0].min()),float(a[:,1].min()),float(a[:,2].max()),float(a[:,3].max())]


def framing_report(rows,*,frames):
    if type(frames) is not int or frames<1 or len(rows)!=frames:raise ValueError('Complete native-frame coverage required')
    for i,r in enumerate(rows):
        if r['step']!=i or abs(r['time_s']-i/100)>1e-7:raise ValueError('Native PTS coverage differs')
        for key,w,margin in [('detail',512,12),('overview',1280,12),('goals',1280,40)]:
            b=np.asarray(r[key],float)
            if b.shape!=(4,) or not np.isfinite(b).all() or not (margin<=b[0]<b[2]<=w-margin and margin<=b[1]<b[3]<=720-margin):
                raise ValueError(key+' clips complete geometry')
        if not np.allclose(r['goals'],rows[0]['goals'],atol=1e-8,rtol=0):raise ValueError('Overview/goals must be fixed')
        if r['phase'] not in ('travel','settling'):raise ValueError('Classify actual dwell intervals')
        if not np.isfinite([r['drone_width'],r['old_drone_width']]).all() or min(r['drone_width'],r['old_drone_width'])<=0:raise ValueError('Finite positive subject widths required')
    def stats(subset):
        if not subset:raise ValueError('Both travel and settling coverage required')
        a=np.array([r['detail'] for r in subset]); widths=a[:,2]-a[:,0]; heights=a[:,3]-a[:,1]
        margins=np.stack([a[:,0],a[:,1],512-a[:,2],720-a[:,3]],axis=1)
        ratio=float(np.median([r['drone_width'] for r in subset])/np.median([r['old_drone_width'] for r in subset]))
        return dict(frames=len(subset),median_width_ratio=ratio,assembly_width_px=np.percentile(widths,[0,5,50,95,100]).tolist(),
                    assembly_height_px=np.percentile(heights,[0,5,50,95,100]).tolist(),minimum_margins_px=margins.min(axis=0).tolist())
    result=stats(rows)
    for phase in ('travel','settling'):result[phase]=stats([r for r in rows if r['phase']==phase])
    if result['median_width_ratio']<1.2:raise ValueError('Less than20% median enlargement')
    return result


class ElevatedRenderer(NativeRenderer):
    def __init__(self,model_path,config):
        super().__init__(model_path,config,cpu_only=True)
        self.bounds=[]
    def goal_label(self,index,goal,priority,style):
        return f"{style['symbol']} {chr(65+index)} {goal[2]:.2f}m p{priority}"
    def render(self,raw,goals,priorities,completed,target_id):
        frame=super().render(raw,goals,priorities,completed,target_id)
        # rbound encloses every mesh vertex, rotor, box corner and rod point,
        # including hidden collision volumes. Orientation cannot escape it.
        ids=[i for i in range(self.model.ngeom) if self.model.geom_bodyid[i]!=0]
        kw=dict(height=720,fovy=self.model.vis.global_.fovy)
        views={}
        for view,width in [('overview',1280),('detail',512)]:
            views[view]=projected_bounds(self.renderers[view].scene.camera,self.data.geom_xpos[ids],self.model.geom_rbound[ids],width=width,**kw)
        views['goals']=projected_bounds(self.renderers['overview'].scene.camera,goals,np.full(len(goals),self.config['marker_radius_m']),width=1280,**kw)
        views.update(time_s=raw['time_s'])
        self.bounds.append(views)
        return frame


def motion_label(vz,valid):
    if not np.isfinite(vz) or type(valid) is not bool:raise ValueError('Finite actual velocity and predicate required')
    if valid:return '静定判定中'
    return '上昇' if vz>.05 else '下降' if vz<-.05 else '水平移動・静定待ち'


def elevated_hud(raw,state,native,case,config,font,binding,*,goals,dwell_s,valid,final=False):
    from PIL import Image,ImageDraw,ImageFont
    from .payload_solver_media import compose_hud,_hud_font
    image=Image.fromarray(compose_hud(raw,state,native,case,config,font,binding['routing_sha256']))
    draw=ImageDraw.Draw(image);h=config['raw_height'];face=_hud_font(str(font),20)
    # Reuse the existing caption rectangles, keeping complete native pixels.
    draw.rectangle((845,h+54,1791,h+89),fill='#101d2d')
    target_z=state['target'][2];motion=motion_label(float(native['drone_velocity'][2]),bool(valid))
    draw.text((850,h+55),f"目標 {target_z:.2f}m｜機体 {native['drone_xyz'][2]:.2f}m｜荷物 {native['package_xyz'][2]:.2f}m｜{motion}",font=face,fill='#e9f3fb')
    for index,priority in enumerate(state['priorities']):
        x,y=22+(index%5)*350,h+140+(index//5)*102
        draw.rectangle((x+12,y+5,x+288,y+41),fill='#20374d')
        draw.text((x+15,y+8),f"{chr(65+index)} 優先{priority} / 高度{goals[index][2]:.2f}m",font=face,fill='white')
    draw.rectangle((0,h+390,1791,h+479),fill='#101d2d')
    status='最終機械受入済・人手確認待ち' if final else '開発実証・正式受入前'
    if binding['provisional']:status+='（暫定B）'
    draw.text((22,h+395),f"{status}｜同時静定 {dwell_s:.2f}/1.00秒｜routing {binding['routing_model_name']} SHA {binding['routing_sha256'][:16]}",font=face,fill='#ffe59b')
    draw.text((22,h+432),'MuJoCo3.15.0 float64 RK4 s16｜C-local-lqr-v1｜飛行NN未使用｜payload-settling-video-v1｜完全1倍速',font=face,fill='#c2dced')
    result=np.asarray(image).copy()
    if not np.array_equal(result[:h],raw):raise ValueError('HUD changed native pixels')
    return result


class CameraProbe:
    """The same MuJoCo scene camera as native rendering, without rasterization."""
    def __init__(self,model_path,config):
        import mujoco as mj
        self.mj=mj;self.model=mj.MjModel.from_xml_path(str(model_path));self.data=mj.MjData(self.model)
        self.scene=mj.MjvScene(self.model,maxgeom=4096);self.options=mj.MjvOption();self.config=config
        self.body_ids=[i for i in range(self.model.ngeom) if self.model.geom_bodyid[i]!=0]
        self.drone_ids=[i for i in self.body_ids if mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_BODY,self.model.geom_bodyid[i]) in ['base_link',*[f'prop{j}_link' for j in range(4)]]]
    def bounds(self,native,goals):
        mj=self.mj;self.data.qpos[:]=native['qpos'];self.data.qvel[:]=native['qvel'];self.data.time=native['time_s']
        mj.set_mjcb_control(None);mj.mj_forward(self.model,self.data)
        result={};drone_widths=[]
        configurations=[('overview',self.config['overview']),('detail',self.config['detail']),('old',dict(self.config['detail'],distance=1.25))]
        for name,c in configurations:
            cam=mj.MjvCamera();mj.mjv_defaultCamera(cam)
            cam.lookat[:]=c['lookat'] if name=='overview' else np.array(native['drone_xyz'])+c['lookat_offset']
            for key in ('distance','azimuth','elevation'):setattr(cam,key,c[key])
            mj.mjv_updateScene(self.model,self.data,self.options,None,cam,mj.mjtCatBit.mjCAT_ALL,self.scene)
            kw=dict(width=c['width'],height=c['height'],fovy=self.model.vis.global_.fovy)
            if name!='old':result[name]=projected_bounds(self.scene.camera,self.data.geom_xpos[self.body_ids],self.model.geom_rbound[self.body_ids],**kw)
            if name=='overview':result['goals']=projected_bounds(self.scene.camera,goals,np.full(len(goals),self.config['marker_radius_m']),**kw)
            else:
                b=projected_bounds(self.scene.camera,self.data.geom_xpos[self.drone_ids],self.model.geom_rbound[self.drone_ids],**kw);drone_widths.append(b[2]-b[0])
        result.update(drone_width=drone_widths[0],old_drone_width=drone_widths[1],time_s=native['time_s'])
        return result


def require_recording_case(case):
    if (case.get('scenario') not in ('normal','emergency','cost_shift') or case.get('suite')!='basic'
        or case.get('execution_seed')!=1 or case.get('case_id')!=f"basic-{case.get('scenario')}-1"):
        raise ValueError('Only the three predeclared basic seed1 recordings are allowed')
    return True
