"""Physical arrival acceptance; logical inspections require continuous dwell."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class FlightSample:
    xyz: list
    velocity: list
    done: bool = False
    reset: bool = False
    tilt_deg: float = 0.


class ArrivalGate:
    def __init__(self, dt, radius=.20, max_speed=.30, hold_s=.20, service_s=.50,
                 floor=.1, max_tilt_deg=180):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in [dt, radius, max_speed, hold_s, service_s, max_tilt_deg]) or service_s < hold_s or not math.isfinite(floor):
            raise ValueError('Invalid arrival timing')
        self.dt, self.radius, self.max_speed = dt, radius, max_speed
        self.hold_steps = math.ceil(hold_s / dt - 1e-9)
        self.service_steps = math.ceil(service_s / dt - 1e-9)
        self.floor, self.max_tilt = floor, max_tilt_deg
        self.steps = 0

    def update(self, sample, target):
        values = np.asarray([*sample.xyz, *sample.velocity, sample.tilt_deg], dtype=float)
        if not np.isfinite(values).all() or sample.done or sample.reset:
            raise RuntimeError('Flight termination/reset/nonfinite state')
        if sample.xyz[2] < self.floor or abs(sample.tilt_deg) > self.max_tilt:
            raise RuntimeError('Flight safety violation')
        distance = np.linalg.norm(np.asarray(sample.xyz) - target)
        if distance > self.radius or np.linalg.norm(sample.velocity) > self.max_speed:
            self.steps = 0
            return []
        self.steps += 1
        events = []
        if self.steps == self.hold_steps:
            events.append('ARRIVAL_CONFIRMED')
        if self.steps == self.service_steps:
            events.append('WAYPOINT_REACHED')
        return events


def quaternion_tilt_degrees(quaternion):
    """Return maximum absolute roll/pitch from the actual world-frame quaternion."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise RuntimeError('Invalid physical orientation')
    w,x,y,z = q / np.linalg.norm(q)
    roll = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch = math.asin(float(np.clip(2*(w*y-z*x),-1,1)))
    return math.degrees(max(abs(roll),abs(pitch)))
