"""Three-panel framing and 1x timeline are independent of physical acceptance."""
import copy
import pytest


def test_camera_profile_is_new_and_requires_both_detail_panels():
    from adaptive_drone_patrol.dual_coordination_media import camera_profile,validate_camera
    config=camera_profile()
    assert set(config['views'])=={'overview','drone-1','drone-2'}
    assert config['border_fraction']==.05
    assert config['fps']==100
    assert all(config['views'][name]['distance']!=.9 for name in ('drone-1','drone-2'))
    bad=copy.deepcopy(config)
    del bad['views']['drone-2']
    with pytest.raises(ValueError):
        validate_camera(bad)


def test_each_frame_must_cover_both_complete_assemblies_and_all_goals():
    from adaptive_drone_patrol.dual_coordination_media import camera_profile,verify_frame_coverage
    config=camera_profile()
    views={name:dict(assembly_bounds_px=[[100.,100.,200.,200.]]*(2 if name=='overview' else 1),
                     goal_bounds_px=[100.,100.,300.,300.] if name=='overview' else None)
           for name in config['views']}
    frames=[dict(index=i,time_s=i/100,views=copy.deepcopy(views)) for i in range(3)]
    assert verify_frame_coverage(frames,config,terminal_tick=2)['every_frame_verified']
    frames[1]['views']['drone-2']['assembly_bounds_px'][0][0]=0
    with pytest.raises(ValueError,match='border'):
        verify_frame_coverage(frames,config,terminal_tick=2)
    frames[1]['views']['drone-2']['assembly_bounds_px'][0][0]=100
    frames[1]['time_s']=.025
    with pytest.raises(ValueError,match='timestamp'):
        verify_frame_coverage(frames,config,terminal_tick=2)
    with pytest.raises(ValueError,match='Complete'):
        verify_frame_coverage(frames[:2],config,terminal_tick=2)


def test_native_three_panel_renderer_keeps_source_model_and_both_loads(tmp_path):
    import json
    import os
    from pathlib import Path
    import numpy as np
    if os.environ.get('DUAL_RENDER')!='1':
        pytest.skip('Native EGL camera smoke is an explicit development check')
    pytest.importorskip('mujoco')
    from adaptive_drone_patrol.dual_coordination_media import DualNativeRenderer
    from adaptive_drone_patrol.dual_coordination_raw import read_raw
    from adaptive_drone_patrol.dual_payload_mujoco import DualMuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload
    from adaptive_drone_patrol.payload_solver_mujoco import FREE_MASS_KG
    source=Path(os.environ['DUAL_RENDER_PILOT'])
    metadata=json.loads((source/'environment.json').read_text())
    f=DualMuJoCoFixture(os.environ['DUAL_CF2X'],tmp_path/'fixture',
                       [Payload(FREE_MASS_KG,load['mass_ratio'],load['side_m']) for load in metadata['case']['loads']])
    f.reset(metadata['case']['initials'])
    before=f.model.vis.global_.offwidth,f.data.qpos.copy()
    renderer=DualNativeRenderer(f,metadata['case']['goals'])
    receipts=[]
    try:
        for segment in sorted(source.glob('replay-*.draw')):
            rows=list(read_raw(segment))
            for row in (rows[0],rows[-1]):
                frame,receipt=renderer.render(row)
                assert frame.shape==(720,2304,3) and frame.dtype==np.uint8
                assert set(receipt['views'])=={'overview','drone-1','drone-2'}
                receipts.append(receipt)
                if os.environ.get('DUAL_RENDER_OUTPUT'):
                    from PIL import Image
                    output=Path(os.environ['DUAL_RENDER_OUTPUT'])
                    output.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(frame).save(output/f'frame-{row["tick"]:05d}.png')
                    (output/'framing-smoke.json').write_text(json.dumps(receipts,indent=2))
        assert f.model.vis.global_.offwidth==before[0]
        np.testing.assert_array_equal(f.data.qpos,before[1])
    finally:
        renderer.close()


def test_hud_displays_zero_based_owners_as_ids_and_persists_observed_event():
    from adaptive_drone_patrol.dual_coordination_media import hud_lines
    from adaptive_drone_patrol.dual_coordination_schema import encode_action,WAIT
    row=dict(tick=100,action=encode_action(WAIT,1,2),mission=dict(priority=2,targets=[WAIT,1],
             goals=[dict(owner=i%2,status='ASSIGNED') for i in range(10)],
             event_receipts=[dict(event=dict(kind='priority',goal=0,priority=100))]),
             supervisor=dict(held=[True,False]),online_audit=dict(minimum_clearance_m=.35))
    lines=hud_lines(row,method='N-full')
    assert '機1→待機' in lines[1] and '機2→B' in lines[1]
    assert 'A:機1' in lines[3] and 'B:機2' in lines[3]
    assert '機1安全停止' in lines[2] and '機2担当へ追従' in lines[2]
    assert '優先度変更' in lines[4] and '0.3500 m' in lines[4]
