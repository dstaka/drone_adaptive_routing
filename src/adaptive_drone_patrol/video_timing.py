"""Physical capture validation and endpoint-aware unit-slope sampling."""

import bisect
import math
from collections import Counter
import numpy as np


def sample_index(times, timestamp):
    return max(0, bisect.bisect_right(times, timestamp + 1e-9) - 1)


def capture_schedule(frames, telemetry, events):
    if len(frames) < 2 or len(telemetry) < 2:
        raise ValueError("Missing capture or physics timing")
    times = [float(r["sim_time_s"]) for r in telemetry]
    steps = [int(r["simulation_step"]) for r in telemetry]
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError("Missing or duplicate physics steps")
    dt = (times[-1] - times[0]) / (steps[-1] - steps[0])
    origin = times[0] - dt * steps[0]
    if (
        dt <= 0
        or not np.isfinite(times).all()
        or any(abs(t - origin - s * dt) > 1e-8 for t, s in zip(times, steps))
    ):
        raise ValueError("Nonuniform actual timestep")
    if [int(f["frame_index"]) for f in frames] != list(range(len(frames))):
        raise ValueError("Missing or duplicate capture records")
    terminal = events[-1]
    if terminal["type"] != "MISSION_FINISHED" or terminal["simulation_step"] != steps[-1]:
        raise ValueError("Missing terminal physical coverage")
    indices, discarded = [], []
    pts = [float(f["pts_s"]) for f in frames]
    if not np.isfinite(pts).all() or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Nonmonotonic encoded capture index")
    for i, f in enumerate(frames):
        s, t, q = int(f["simulation_step"]), float(f["sim_time_s"]), int(f["event_seq"])
        if not math.isfinite(t) or s < 0 or s > steps[-1] or abs(t - origin - s * dt) > 1e-8:
            raise ValueError("Frame/physical step time mismatch")
        if q < 1 or q > len(events) or int(events[q - 1]["simulation_step"]) > s:
            raise ValueError("Frame/event reference mismatch")
        if indices:
            previous = frames[indices[-1]]
            diff = s - int(previous["simulation_step"])
            if diff < 0 or q < int(previous["event_seq"]):
                raise ValueError("Nonmonotonic capture timing")
            if diff == 0:
                if q <= int(previous["event_seq"]) or not any(
                    int(e["simulation_step"]) == s for e in events[int(previous["event_seq"]) : q]
                ):
                    raise ValueError("Unexplained duplicate capture")
                discarded.append(indices.pop())
        indices.append(i)
    unique_steps = [int(frames[i]["simulation_step"]) for i in indices]
    differences = [b - a for a, b in zip(unique_steps, unique_steps[1:])]
    cadence = Counter(differences).most_common(1)[0][0]
    if unique_steps[0] != 0 or unique_steps[-1] != steps[-1] or any(d > cadence for d in differences):
        raise ValueError("Missing capture cadence or terminal coverage")
    return {
        "dt": dt,
        "origin": origin,
        "start": origin,
        "end": float(terminal["sim_time_s"]),
        "indices": indices,
        "discarded": discarded,
        "cadence_steps": cadence,
        "terminal_packet_s": dt,
        "boundary_convention": "[start,end); final encoded sample represents end point only",
    }


def flight_samples(start, end, fps):
    """Use unit-slope grid; replace its last sample with the terminal point.

    The last sample is at most one output quantum ahead of the grid. It exposes
    completion without claiming an extra physical interval after mission end.
    """
    if fps not in [25, 30] or not end > start:
        raise ValueError("Invalid flight interval or output FPS")
    count = math.ceil((end - start) * fps - 1e-9)
    points = [start + i / fps for i in range(count)]
    points[-1] = end
    return points
