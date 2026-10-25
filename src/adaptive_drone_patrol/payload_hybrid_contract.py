"""Revision 2 contracts. Historical readers retain their original semantics."""

from dataclasses import dataclass, asdict
import hashlib
import json
import math


@dataclass(frozen=True)
class SettlingProfile:
    id: str
    limits: tuple
    dt_s: float = .01
    dwell_s: float = 1.
    leg_s: float = 30.
    mission_wall_s: float = 600.
    aggregate_simulated_deadline: None = None

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def valid(self, metrics, *, contact=False):
        return bool(not contact and all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= b
            for k, b in self.limits)
            and metrics['package_floor_m'] > .10 and metrics['drone_floor_m'] > .10
            and abs(metrics['drone_roll_deg']) <= 60 and abs(metrics['drone_pitch_deg']) <= 60
            and 0 <= metrics['constraint_residual_m'] <= 2e-5)


VIDEO_PROFILE = SettlingProfile('payload-settling-video-v1', (
    ('drone_error_m', .20), ('drone_speed_m_s', .30), ('package_error_m', .20),
    ('package_speed_m_s', .15), ('swing_angle_deg', 5.), ('swing_rate_deg_s', 15.),
    ('package_spin_deg_s', 30.)))
STRICT_PROFILE = SettlingProfile('payload-settling-strict-shadow-v1', (
    ('drone_error_m', .20), ('drone_speed_m_s', .30), ('package_error_m', .20),
    ('package_speed_m_s', .10), ('swing_angle_deg', 5.), ('swing_rate_deg_s', 10.),
    ('package_spin_deg_s', 10.)))


class MissionClock:
    """Wall origin is captured before reset; references cannot reset a leg."""

    def __init__(self, *, reset_wall_s):
        if not math.isfinite(reset_wall_s):
            raise ValueError('Finite reset clock required')
        self.reset_wall_s = self.last_wall_s = float(reset_wall_s)
        self.leg_start_s = 0.
        self.command_key = None
        self.last_sim_s = 0.

    def command(self, sim_s, target_id, target):
        point = tuple(float(x) for x in target)
        if len(point) != 3 or not all(math.isfinite(x) for x in point):
            raise ValueError('Finite target xyz required')
        if not math.isfinite(sim_s) or sim_s < self.last_sim_s:
            raise ValueError('Invalid simulation clock')
        key = (target_id, point)
        if self.command_key is not None:
            if key == self.command_key:
                return False
            if point == self.command_key[1]:
                raise ValueError('Stationary target relabeling cannot restart a leg')
        self.command_key = key
        self.leg_start_s = self.last_sim_s = float(sim_s)
        return True

    def check(self, sim_s, wall_s, *, complete=False, unsafe=False):
        if (not all(math.isfinite(x) for x in (sim_s, wall_s)) or
                sim_s < self.last_sim_s or wall_s < self.last_wall_s):
            raise ValueError('Nonmonotonic or invalid mission clock')
        self.last_sim_s, self.last_wall_s = float(sim_s), float(wall_s)
        if unsafe:
            return 'SAFETY_FAILURE'
        if wall_s - self.reset_wall_s >= VIDEO_PROFILE.mission_wall_s:
            return 'WALL_TIMEOUT'
        elapsed = sim_s - self.leg_start_s
        if complete and elapsed <= VIDEO_PROFILE.leg_s + 1e-9:
            return 'COMPLETE'
        if elapsed >= VIDEO_PROFILE.leg_s - 1e-9:
            return 'LEG_TIMEOUT'
        return 'CONTINUE'


def acceptance_counts(branch):
    if branch not in ('controller', 'nn'):
        raise ValueError('Explicit selected branch required')
    methods = 1 if branch == 'controller' else 2
    return dict(validation=12, transitions=120, basic=9, endpoints=18,
                heldout=100, missions=127*methods, methods=methods)
