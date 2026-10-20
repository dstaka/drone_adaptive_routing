import copy
import numpy as np
import pytest
from adaptive_drone_patrol.video_timing import capture_schedule, flight_samples, sample_index


def fixture(start=0):
    telemetry = [{"simulation_step": str(i), "sim_time_s": str(start + i * 0.01)} for i in range(1, 6)]
    frames = [
        dict(frame_index=i, simulation_step=s, sim_time_s=start + s * 0.01, pts_s=i * 0.02, event_seq=q)
        for i, (s, q) in enumerate([(0, 1), (2, 1), (4, 1), (5, 2), (5, 3)])
    ]
    events = [
        dict(event_seq=1, simulation_step=0, sim_time_s=start, type="SCENARIO_START"),
        dict(event_seq=2, simulation_step=5, sim_time_s=start + 0.05, type="WAYPOINT_REACHED"),
        dict(event_seq=3, simulation_step=5, sim_time_s=start + 0.05, type="MISSION_FINISHED"),
    ]
    return frames, telemetry, events


def test_forced_terminal_duplicate_is_explicit_and_no_fictitious_coverage():
    schedule = capture_schedule(*fixture(2))
    assert schedule["dt"] == pytest.approx(0.01)
    assert schedule["end"] == pytest.approx(2.05)
    assert schedule["indices"] == [0, 1, 2, 4]
    assert schedule["discarded"] == [3]
    assert schedule["terminal_packet_s"] == pytest.approx(0.01)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "backward", "wrong_time", "gap"])
def test_invalid_capture_records_fail(mutation):
    f, t, e = copy.deepcopy(fixture())
    if mutation == "missing":
        f.pop(1)
    if mutation == "duplicate":
        f[-1]["event_seq"] = f[-2]["event_seq"]
    if mutation == "backward":
        f[2]["simulation_step"] = 1
    if mutation == "wrong_time":
        t[2]["sim_time_s"] = ".031"
    if mutation == "gap":
        f[1]["simulation_step"] = 4
    with pytest.raises(ValueError):
        capture_schedule(f, t, e)


def test_50_to_30_and_partial_final_sample_use_unit_slope():
    points = flight_samples(2, 2.05, 30)
    assert len(points) == 2
    assert points[-1] == pytest.approx(2.05)
    regular = flight_samples(2, 3, 30)
    assert np.allclose(regular[:-1], 2 + np.arange(29) / 30)
    assert regular[-1] == 3
    assert sample_index([2, 2.02, 2.04, 2.05], points[-1]) == 3
