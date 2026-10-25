"""Frame schedules must preserve every physical control interval exactly once."""

import copy
import pytest


def native_rows():
    return [
        dict(kind="RESET", step=0, episode_id=1, actual=dict(time_s=0.0, qpos=[1, 0, 0, 0])),
        dict(kind="RETARGET", step=0, episode_id=1, actual=dict(time_s=0.0, qpos=[1, 0, 0, 0])),
        dict(kind="STEP", step=1, episode_id=1, actual=dict(time_s=0.01, qpos=[1, 0.1, 0, 0])),
        dict(kind="RETARGET", step=1, episode_id=1, actual=dict(time_s=0.01, qpos=[1, 0.1, 0, 0])),
        dict(kind="STEP", step=2, episode_id=1, actual=dict(time_s=0.02, qpos=[1, 0.2, 0, 0])),
    ]


def test_final_event_and_retargets_do_not_duplicate_frames_or_skip_intervals():
    from adaptive_drone_patrol.payload_solver_media import native_frame_rows

    frames = list(native_frame_rows(native_rows()))
    assert [(r["step"], r["actual"]["time_s"]) for r in frames] == [(0, 0), (1, 0.01), (2, 0.02)]


@pytest.mark.parametrize("change", ["retarget_pose", "missing", "clock", "reset"])
def test_nonphysical_video_time_or_hidden_reset_is_rejected(change):
    from adaptive_drone_patrol.payload_solver_media import native_frame_rows

    rows = copy.deepcopy(native_rows())
    if change == "retarget_pose":
        rows[1]["actual"]["qpos"][0] += 0.1
    elif change == "missing":
        rows.pop(2)
        rows.pop(2)
    elif change == "clock":
        rows[-1]["actual"]["time_s"] = 0.025
    else:
        rows[-1]["kind"] = "RESET"
    with pytest.raises(ValueError):
        list(native_frame_rows(rows))


def test_hud_appends_below_raw_without_modifying_any_original_pixel():
    import numpy as np
    import matplotlib
    from pathlib import Path
    from adaptive_drone_patrol.payload_solver_media import compose_hud, load_camera

    raw = np.arange(720 * 1792 * 3, dtype=np.uint8).reshape(720, 1792, 3)
    state = dict(
        scenario_id="normal",
        selected="A",
        visited=["B"],
        priorities=[81, 42, 43, 64, 23, 84, 6, 98, 22, 56],
        event_text=None,
        target=[0, 0, 1],
    )
    native = dict(
        time_s=0.0,
        drone_xyz=[0, 0, 1],
        package_xyz=[0, 0, 0.6575],
        drone_velocity=[0, 0, 0],
        package_velocity=[0, 0, 0],
        drone_quaternion_wxyz=[1, 0, 0, 0],
        package_quaternion_wxyz=[1, 0, 0, 0],
        drone_angular_velocity=[0, 0, 0],
        package_angular_velocity=[0, 0, 0],
    )
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    result = compose_hud(
        raw, state, native, dict(side_m=0.045, mass_ratio=0.2), load_camera(), font, "a" * 64
    )
    assert result.shape == (1200, 1792, 3)
    assert np.array_equal(result[:720], raw)
    assert result[720:].std() > 1


def test_hud_does_not_show_future_emergency_or_invent_completed_goals():
    from adaptive_drone_patrol.payload_solver_media import mission_display_state
    from adaptive_drone_patrol.scenario import Scenario
    from adaptive_drone_patrol.util import load_config

    scenario=Scenario.from_dict(load_config('configs/demo_scenarios_dispersed/emergency.yaml'))
    events=[dict(type='TARGET_SELECTED',simulation_step=0,target_id='A'),
        dict(type='SHARED_SETTLED',simulation_step=125,target_id='A'),
        dict(type='PRIORITY_UPDATED',simulation_step=125,
             update=dict(checkpoint_id='C',old=43,value=100))]
    before=mission_display_state(scenario,events,124)
    after=mission_display_state(scenario,events,125)
    assert before['priorities'][2]==43 and before['visited']==[]
    assert before['event_text'] is None
    assert after['priorities'][2]==100 and after['visited']==['A']
    assert '43' in after['event_text'] and '100' in after['event_text']


def test_priority_fill_is_inherited_palette_and_independent_of_visit_state():
    from adaptive_drone_patrol.payload_solver_media import goal_display_style
    from adaptive_drone_patrol.scene import priority_color

    styles = [goal_display_style(43, visited=v, selected=s) for v,s in [(False,False),(False,True),(True,False)]]
    assert {s['fill'] for s in styles} == {priority_color(43,100)}
    assert len({s['symbol'] for s in styles}) == 3
    assert goal_display_style(100)['fill'] == priority_color(100,100)
    assert goal_display_style(100)['fill'] != styles[0]['fill']
