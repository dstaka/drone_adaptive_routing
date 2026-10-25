"""Provisional video admission/labels; never substitutes for final acceptance."""

import numpy as np
from collections import OrderedDict
from functools import lru_cache
import json

_CANVASES=OrderedDict()


@lru_cache(maxsize=12)
def _font(path,size):
    from PIL import ImageFont
    return ImageFont.truetype(path,size)


def require_development_capture(summary,proof):
    if (summary.get('status')!='SUCCEEDED' or summary.get('completed_count')!=10 or
        not 0<=summary.get('wall_residence_s',600)<600 or
        proof.get('success') is not True or proof.get('physical_acceptance') is not True or
        proof.get('completed_count')!=10 or proof.get('steps')!=summary.get('steps')):
        raise ValueError('Complete fresh independently checked development mission required')
    return dict(adoption_status='DEVELOPMENT_NOT_FINAL',
        label='開発実証・動画優先静定基準 / 頑健性評価前',human_review='PENDING_FRESH_CONTINUOUS1X')


def compare_recomposition(raw,hud,rerendered,recomposed,*,raw_height):
    """Every software-rendered native/caption pixel must reconstruct exactly."""
    if (not np.array_equal(hud[:raw_height],raw) or
        not np.array_equal(hud[raw_height:],recomposed[raw_height:])):
        raise ValueError('Decoded HUD/caption differs from independent recomposition')
    difference=np.abs(raw.astype(np.int16)-rerendered.astype(np.int16))
    pixels=int(np.count_nonzero(np.any(difference,axis=2)))
    if pixels:
        raise ValueError('Independent software native raster differs')
    return dict(raster_changed_pixels=pixels,maximum_channel_difference=int(difference.max()))


def compose_development_hud(raw,state,native,payload,config,font,controller_hash):
    from PIL import Image,ImageDraw,ImageFont
    from .payload_solver_media import compose_hud

    base=compose_hud(raw,state,native,payload,config,font,controller_hash)
    image=Image.fromarray(base)
    draw=ImageDraw.Draw(image)
    h,w=config['raw_height'],config['raw_width']
    draw.rectangle((0,h+390,w,h+config['hud_height']),fill='#101d2d')
    face=ImageFont.truetype(str(font),23)
    small=ImageFont.truetype(str(font),20)
    draw.text((22,h+397),'開発実証・動画優先静定基準 / 頑健性評価前',font=face,fill='#ffe59b')
    draw.text((22,h+432),
        '制御器 C-local-LQR / 飛行NN未使用 | MuJoCo 3.15 / float64 / RK4 | payload-settling-video-v1 | 完全1倍速',
        font=small,fill='#c2dced')
    result=np.asarray(image).copy()
    if not np.array_equal(result[:h],raw):
        raise ValueError('Development labels changed native raw pixels')
    return result


def compose_development_hud_cached(raw,state,native,payload,config,font,controller_hash):
    """Cache only static captions; reconstruct changing telemetry every frame."""
    from PIL import Image,ImageDraw
    from .payload_contract import Payload,payload_metrics
    from .payload_solver_mujoco import FREE_MASS_KG
    from .payload_solver_trace import observation_sample
    key=json.dumps([state,payload,config,str(font),controller_hash],sort_keys=True)
    if key not in _CANVASES:
        _CANVASES[key]=compose_development_hud(raw,state,native,payload,config,font,controller_hash)
        if len(_CANVASES)>32:
            _CANVASES.popitem(last=False)
    _CANVASES.move_to_end(key)
    canvas=Image.fromarray(_CANVASES[key])
    canvas.paste(Image.fromarray(raw),(0,0))
    draw=ImageDraw.Draw(canvas)
    h,w=config['raw_height'],config['raw_width']
    # NotoJP's 28px title extends through row50. Never erase its bottom ink.
    draw.rectangle((0,h+54,w,h+87),fill='#101d2d')
    draw.rectangle((840,h+88,w,h+123),fill='#101d2d')
    face,small=_font(str(font),23),_font(str(font),20)
    cube=Payload(FREE_MASS_KG,payload['mass_ratio'],payload['side_m'])
    metrics=payload_metrics(observation_sample(native,cube.side_m),state['target'],cube)
    draw.text((22,h+55),f"物理時刻 {native['time_s']:.2f} 秒   目標 {state['selected'] or '-'}   完了 {len(state['visited'])}/10",font=face,fill='#e9f3fb')
    draw.text((850,h+55),f"高度：機体 {native['drone_xyz'][2]:.3f} m / 荷物 {native['package_xyz'][2]:.3f} m",font=face,fill='#e9f3fb')
    draw.text((850,h+91),f"揺れ {metrics['swing_angle_deg']:.2f} 度 / 荷物spin {metrics['package_spin_deg_s']:.2f} 度/秒",font=small,fill='#e9f3fb')
    result=np.asarray(canvas).copy()
    if not np.array_equal(result[:h],raw):
        raise ValueError('Cached HUD changed native raw pixels')
    return result
