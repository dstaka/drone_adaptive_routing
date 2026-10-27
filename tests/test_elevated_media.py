import copy,json
from types import SimpleNamespace
import numpy as np
import pytest
from adaptive_drone_patrol.elevated_media import load_elevated_camera, projected_bounds, framing_report


def camera():
    return [SimpleNamespace(pos=np.array([0.,0.,0.]), forward=np.array([0.,0.,-1.]),up=np.array([0.,1.,0.]))]*2


def test_new_reader_preserves_legacy_strict_rejection(tmp_path):
    from adaptive_drone_patrol.payload_mujoco_overview_media import load_overview_camera
    c=load_elevated_camera('configs/payload_mujoco_elevated_camera.json')
    assert c['detail']['distance']==.9
    with pytest.raises(ValueError): load_overview_camera('configs/payload_mujoco_elevated_camera.json')
    c['detail']['distance']=1.25
    p=tmp_path/'camera.json'; p.write_text(json.dumps(c))
    with pytest.raises(ValueError): load_elevated_camera(p)


def test_projection_includes_surface_extent_and_rejects_invalid():
    b=projected_bounds(camera(),[[0.,0.,-1.]],[.1],width=512,height=720,fovy=90.)
    assert b==pytest.approx([219.81863865066836,323.8186386506684,292.1813613493316,396.1813613493316])
    for pos in ([0,0,0],[0,0,.2],[float('nan'),0,-1]):
        with pytest.raises(ValueError): projected_bounds(camera(),[pos],[.1],width=512,height=720,fovy=90.)


def test_full_frame_framing_separate_travel_settling_and_enlargement():
    rows=[]
    for i in range(4):
        rows.append(dict(step=i,time_s=i/100,phase='travel' if i%2==0 else 'settling',
                         detail=[50,60,450,650],overview=[20,20,1250,690],goals=[50,50,1200,650],
                         drone_width=150.,old_drone_width=120.))
    report=framing_report(rows,frames=4)
    assert report['median_width_ratio']==1.25
    assert report['travel']['median_width_ratio']==1.25
    assert report['settling']['median_width_ratio']==1.25
    bad=copy.deepcopy(rows); bad[-1]['detail'][0]=11.9
    with pytest.raises(ValueError,match='detail'): framing_report(bad,frames=4)
    bad=copy.deepcopy(rows); bad[-1]['goals'][1]=51.
    with pytest.raises(ValueError,match='fixed'): framing_report(bad,frames=4)
    with pytest.raises(ValueError): framing_report(rows[:-1],frames=4)
    bad=copy.deepcopy(rows)
    for r in bad:r['drone_width']=140.
    with pytest.raises(ValueError,match='20%'): framing_report(bad,frames=4)


def test_japanese_motion_state_uses_actual_velocity_and_dwell():
    from adaptive_drone_patrol.elevated_media import motion_label
    assert motion_label(.2,False)=='上昇'
    assert motion_label(-.2,False)=='下降'
    assert motion_label(0.,True)=='静定判定中'
    assert motion_label(.01,False)=='水平移動・静定待ち'
    with pytest.raises(ValueError):motion_label(float('nan'),False)


def test_final_recording_cannot_substitute_seed_or_heldout_success():
    from adaptive_drone_patrol.elevated_media import require_recording_case
    case=dict(case_id='basic-normal-1',suite='basic',scenario='normal',execution_seed=1)
    assert require_recording_case(case)
    for bad in [dict(case,execution_seed=2),dict(case,case_id='heldout-normal-0-0',suite='heldout'),dict(case,case_id='basic-normal-2')]:
        with pytest.raises(ValueError):require_recording_case(bad)
