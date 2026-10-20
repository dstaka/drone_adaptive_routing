"""Display state follows encoded frame event sequence, including same-step events."""
from adaptive_drone_patrol.analytical import demo_fixture


def test_same_step_event_sequence_controls_priority_display():
    from adaptive_drone_patrol.annotations import display_state
    s = demo_fixture(40004)
    events = [
        {'event_seq':1, 'simulation_step':0, 'type':'SCENARIO_START'},
        {'event_seq':2, 'simulation_step':0, 'type':'TARGET_SELECTED','target_id':'B'},
        {'event_seq':3, 'simulation_step':100, 'type':'WAYPOINT_REACHED','target_id':'B'},
        {'event_seq':4, 'simulation_step':100, 'type':'PRIORITY_UPDATED',
         'update': {'kind':'priority','checkpoint_id':'C','old':s.priorities[2],'new':5},
         'marker_state': {'priorities':[s.priorities[i] if i!=2 else 5 for i in range(10)],
                          'visited':['B'],'selected':None}},
    ]
    before = display_state(s, events, 3)
    after = display_state(s, events, 4)
    assert before['priorities'][2] == s.priorities[2]
    assert after['priorities'][2] == 5
    assert before['priorities'][:2] == after['priorities'][:2]
    assert after['visited'] == ['B'] and after['alert'] == 'C'
    assert display_state(s, events, 1)['selected'] is None


def test_detail_crop_uses_unannotated_same_frame():
    from adaptive_drone_patrol.annotations import same_frame_detail
    from PIL import Image
    image=Image.new('RGB',(1280,720),'#112233')
    # A physical vehicle pixel stays in the exact center of the same-frame crop.
    image.putpixel((640,360),(255,0,0))
    crop=same_frame_detail(image,[640,360])
    assert crop.size==(240,180)
    assert crop.getpixel((120,90))[0]>100
    assert image.getpixel((24,450))==(17,34,51)


def test_wide_scene_detail_can_zoom_real_pixels_without_using_another_time():
    from adaptive_drone_patrol.annotations import same_frame_detail
    from PIL import Image,ImageDraw
    import numpy as np
    image=Image.new('RGB',(1280,720),'blue')
    ImageDraw.Draw(image).rectangle((669,359,672,362),fill='lime')
    image.putpixel((640,360),(255,0,0))
    wide=same_frame_detail(image,[640,360])
    close=same_frame_detail(image,[640,360],[48,36])
    assert np.asarray(wide)[:,:,1].max()>100
    assert np.asarray(close)[:,:,1].max()==0
    assert close.getpixel((120,90))[0]>100
