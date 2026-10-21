"""CPU-only payload contracts; these helpers do not certify physical flight."""

from dataclasses import dataclass
import math
import numpy as np

SIZES_M = (.030, .045, .060)
LINK_M = .30
DRONE_ATTACHMENT = np.array([0., 0., -.020])
OBSERVATION_CONTRACT = {
    'id': 'payload29-ideal-world-v1', 'size': 29, 'actions': 4,
    'legacy_prefix': 'hover17-relative-world-v1',
    'fields': ['legacy17', 'package_minus_drone_world_xyz_m',
               'package_minus_drone_world_velocity_m_s', 'suspension_world_unit_direction',
               'suspension_world_angular_rate_rad_s'],
    'scales': [1/.30, 1/3, 1, 1/math.pi], 'clip': [-1, 1],
    'quaternion': 'wxyz; body to world', 'history': 'legacy last four motor actions only',
    'hidden_parameters': [], 'source': 'ideal simulator rigid-body state',
}


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError('Expected a finite real number')
    if not math.isfinite(value):
        raise ValueError('Expected a finite real number')
    return float(value)


@dataclass(frozen=True)
class Payload:
    reference_mass_kg: float
    mass_ratio: float
    side_m: float

    def __post_init__(self):
        mass, ratio, side = map(_number, (self.reference_mass_kg, self.mass_ratio, self.side_m))
        if mass <= 0 or not .10 <= ratio <= .30 or side not in SIZES_M:
            raise ValueError('Payload must use positive original drone mass and the declared mass/size domain')

    @property
    def mass(self):
        return self.reference_mass_kg * self.mass_ratio

    @property
    def inertia(self):
        return [self.mass * self.side_m**2 / 6] * 3


class PayloadSampler:
    """Draw on every selected reset using independent per-environment streams.

    Geometry/mass application is the physical adapter's responsibility. A draw
    is not proof that the solver actually applied any of these properties.
    """
    def __init__(self, reference_mass_kg, seed, split, randomized=True):
        Payload(reference_mass_kg, .2, .045)
        if split not in ('training', 'validation'):
            raise ValueError('Evaluation must use its frozen case manifest, not a sampler')
        if type(seed) is not int or seed < 0:
            raise ValueError('Seed must be a nonnegative integer')
        self.reference_mass_kg, self.seed, self.split = reference_mass_kg, seed, split
        self.randomized = randomized
        self.current, self.draw_indices, self._streams = {}, {}, {}

    def reset(self, env_ids):
        ids = list(env_ids)
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Reset indices must be unique nonnegative integers')
        result = {}
        for i in ids:
            if i not in self._streams:
                tag = 710 if self.split == 'training' else 720
                self._streams[i] = np.random.default_rng(np.random.SeedSequence([self.seed, tag, i]))
            rng = self._streams[i]
            side = float(rng.choice(SIZES_M)) if self.randomized else .045
            ratio = float(rng.uniform(.1, .3)) if self.randomized else .2
            self.current[i] = result[i] = Payload(self.reference_mass_kg, ratio, side)
            self.draw_indices[i] = self.draw_indices.get(i, 0) + 1
        return result


def _vector(value, size=3):
    v = np.asarray(value, dtype=float)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError('Invalid finite vector shape')
    return v


def rotation(quaternion_wxyz):
    """Body-to-world rotation; reject invalid measurements rather than repair them."""
    q = _vector(quaternion_wxyz, 4)
    if abs(np.linalg.norm(q) - 1) > 1e-5:
        raise ValueError('Quaternion must be unit wxyz')
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _kinematics(sample, side_m):
    d, p = _vector(sample['drone_xyz']), _vector(sample['package_xyz'])
    vd, vp = _vector(sample['drone_velocity']), _vector(sample['package_velocity'])
    wd, wp = _vector(sample['drone_angular_velocity']), _vector(sample['package_angular_velocity'])
    rd, rp = rotation(sample['drone_quaternion_wxyz']), rotation(sample['package_quaternion_wxyz'])
    od, op = rd @ DRONE_ATTACHMENT, rp @ np.array([0., 0., side_m/2])
    r = p + op - d - od
    length = np.linalg.norm(r)
    if length < 1e-8:
        raise ValueError('Degenerate suspension endpoints')
    relative_v = vp + np.cross(wp, op) - vd - np.cross(wd, od)
    angular_rate = np.cross(r, relative_v) / length**2
    return d, p, vd, vp, wd, wp, rd, rp, r, length, angular_rate


def payload_metrics(sample, target, payload):
    """World-frame SI metrics from measured centers, poses and angular velocities.

    Attachment offset is provisional until the pinned physical asset is probed.
    This function computes predicates only; contact, solver constraints, resets
    and timeouts must also be checked by an eventual physical runner/checker.
    """
    d, p, vd, vp, wd, wp, rd, rp, r, length, rate = _kinematics(sample, payload.side_m)
    goal = _vector(target)
    nominal = goal + DRONE_ATTACHMENT - np.array([0, 0, LINK_M + payload.side_m/2])
    angle = math.degrees(math.acos(float(np.clip(-r[2]/length, -1, 1))))
    return {
        'drone_error_m': float(np.linalg.norm(d-goal)), 'drone_speed_m_s': float(np.linalg.norm(vd)),
        'package_error_m': float(np.linalg.norm(p-nominal)), 'package_speed_m_s': float(np.linalg.norm(vp)),
        'swing_angle_deg': angle, 'swing_rate_deg_s': float(np.degrees(np.linalg.norm(rate))),
        'package_spin_deg_s': float(np.degrees(np.linalg.norm(wp))),
        'constraint_residual_m': float(abs(length-LINK_M)),
        'package_floor_m': float(p[2] - payload.side_m/2 * np.abs(rp[2]).sum()),
        'drone_floor_m': float(d[2]),
        'drone_tilt_deg': math.degrees(math.acos(float(np.clip(rd[2, 2], -1, 1)))),
        'drone_roll_deg': math.degrees(math.atan2(rd[2, 1], rd[2, 2])),
        'drone_pitch_deg': math.degrees(math.asin(float(np.clip(-rd[2, 0], -1, 1)))),
    }


def settled(metrics):
    limits = {'drone_error_m': .20, 'drone_speed_m_s': .30,
              'package_error_m': .20, 'package_speed_m_s': .10,
              'swing_angle_deg': 5., 'swing_rate_deg_s': 10., 'package_spin_deg_s': 10.}
    return (all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= bound for k, bound in limits.items())
            and metrics['package_floor_m'] > .10 and metrics['drone_floor_m'] > .10
            and abs(metrics['drone_roll_deg']) <= 60. and abs(metrics['drone_pitch_deg']) <= 60.)


class SharedSettling:
    """Count observed consecutive control-step intervals, never an instant."""
    def __init__(self, dt=.01):
        self.dt = _number(dt)
        if self.dt <= 0:
            raise ValueError('dt must be positive')
        self.previous_time, self.previous_valid, self.target_id = None, False, None
        self.intervals = 0

    def update(self, time_s, valid, target_id=None):
        t = _number(time_s)
        if type(valid) is not bool:
            raise ValueError('Validity must be a boolean')
        if self.previous_time is not None and not math.isclose(t-self.previous_time, self.dt, abs_tol=1e-8):
            raise ValueError('Samples must be consecutive control steps')
        same_target = target_id == self.target_id
        if not valid or not self.previous_valid or not same_target:
            self.intervals = 0
        else:
            self.intervals += 1
        self.previous_time, self.previous_valid, self.target_id = t, valid, target_id
        return self.intervals * self.dt >= 1. - 1e-10

    @property
    def dwell_s(self):
        return self.intervals * self.dt


def actor_observation(legacy17, sample):
    """Append ideal measured body/attachment motion, without hidden parameters.

    The physical adapter must measure both attachment points and velocities in
    world coordinates, including angular cross-offset terms. This helper never
    reads episode mass, dimensions, geometry identifiers or random seeds.
    """
    old = _vector(legacy17, 17)
    d, p = _vector(sample['drone_xyz']), _vector(sample['package_xyz'])
    r = _vector(sample['attachment_b_world']) - _vector(sample['attachment_a_world'])
    length = np.linalg.norm(r)
    if length < 1e-8:
        raise ValueError('Degenerate observed suspension direction')
    v = _vector(sample['package_velocity']) - _vector(sample['drone_velocity'])
    va = _vector(sample['attachment_b_velocity']) - _vector(sample['attachment_a_velocity'])
    extra = np.concatenate([(p-d)/.30, v/3, r/length, np.cross(r, va)/length**2/math.pi])
    return np.concatenate([old, np.clip(extra, -1, 1)])


def require_checkpoint_contract(embedded, actual):
    if embedded != OBSERVATION_CONTRACT or actual != OBSERVATION_CONTRACT:
        raise ValueError('Payload checkpoint observation semantics mismatch')


def training_outside_envelope(sample,target):
    """Retain the adopted parent's target-relative training termination envelope."""
    delta=np.abs(_vector(sample['drone_xyz'])-_vector(target))
    return bool(np.any(delta>np.array([6.,4.,3.])))


def numerical_excursion(sample,reward):
    """Abort development on unusable numerics; this is not an acceptance limit."""
    if not math.isfinite(float(reward)) or abs(reward)>1e6:
        return True
    for body in ('drone','package'):
        for field in ('xyz','velocity','angular_velocity','quaternion_wxyz'):
            values=np.asarray(sample[f'{body}_{field}'],dtype=float)
            if not np.isfinite(values).all():
                return True
        if np.linalg.norm(sample[f'{body}_velocity'])>1e4:
            return True
    return False


def require_physical_admission(receipt, declaration, thresholds_sha256):
    """Reject obsolete short probes after a known operating-domain failure."""
    if declaration.get('operating_domain_admission') != 'admitted':
        raise ValueError('Payload operating-domain admission is blocked; new physical validation is required')
    if receipt.get('thresholds_sha256') != thresholds_sha256:
        raise ValueError('Physical thresholds differ from the measured receipt')
    if receipt.get('status') != 'PASS' or not receipt.get('physical_training_admitted'):
        raise ValueError('Physical preflight did not admit training')
