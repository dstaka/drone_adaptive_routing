"""Immutable C2 polynomial references, with conservative derivative certificates.

These are DEVELOPMENT motion limits, not physics-qualified stopping or tracking
envelopes. Actual state is measured at each transition and retained as tracking
error; the caller must qualify/inflate its safety envelopes with that error.
No method writes the simulator state or snaps a moving reference to measured pose.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


def _vector(value):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError('Expected a finite three-dimensional vector')
    return tuple(float(x) for x in array)


def _time(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Time must be finite')
    return value


@dataclass(frozen=True)
class KinematicState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float] | None = None

    def __post_init__(self):
        for name in ('position', 'velocity', 'acceleration'):
            value = getattr(self, name)
            if name != 'acceleration' or value is not None:
                object.__setattr__(self, name, _vector(value))


@dataclass(frozen=True)
class ReferenceLimits:
    max_speed: float = 1.2
    max_acceleration: float = .8
    max_jerk: float = 2.
    max_duration: float = 120.
    max_attempts: int = 64

    def __post_init__(self):
        for name in ('max_speed', 'max_acceleration', 'max_jerk', 'max_duration'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Limits must be finite and positive')
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 256:
            raise ValueError('Search attempts must be an integer in [1,256]')


@dataclass(frozen=True)
class TrackingError:
    position: float
    velocity: float
    acceleration: float | None = None

    def __post_init__(self):
        for name in ('position', 'velocity', 'acceleration'):
            value = getattr(self, name)
            if name == 'acceleration' and value is None:
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError('Tracking error must be finite and nonnegative')


@dataclass(frozen=True)
class DerivativeBounds:
    speed: float
    acceleration: float
    jerk: float


class ReferenceInfeasible(ValueError):
    """No certified segment was found within the fixed search budget."""


def nominal_duration(start, goal, limits=ReferenceLimits()):
    """Common B-dynamic rest-to-rest duration, preserving SmoothReference defaults.

    The historical rounded acceleration coefficient 5.774 is conservative for
    the exact 10/sqrt(3) maximum. Moving starts require retarget certification.
    """
    length = float(np.linalg.norm(np.subtract(_vector(goal), _vector(start))))
    return max(3., 1.875*length/limits.max_speed,
               math.sqrt(5.774*length/limits.max_acceleration),
               (60*length/limits.max_jerk)**(1/3))


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _derivative_intervals(low, high, duration):
    """Outward-rounded Bernstein derivative controls of the stored polynomial."""
    degree = low.shape[-2]-1
    # Every subtraction/multiply/divide is individually rounded outward.
    lo = _down(_down(_down(low[..., 1:, :]-high[..., :-1, :])*degree)/duration)
    hi = _up(_up(_up(high[..., 1:, :]-low[..., :-1, :])*degree)/duration)
    return lo, hi


def _split_intervals(low, high):
    count, degree = low.shape[:2]
    result_low = np.empty((2*count,degree,3))
    result_high = np.empty_like(result_low)
    result_low[:count,0],result_high[:count,0] = low[:,0],high[:,0]
    result_low[count:,-1],result_high[count:,-1] = low[:,-1],high[:,-1]
    for index in range(1,degree):
        low = _down(_down(low[:, :-1]+low[:, 1:])*.5)
        high = _up(_up(high[:, :-1]+high[:, 1:])*.5)
        result_low[:count,index],result_high[:count,index] = low[:,0],high[:,0]
        result_low[count:,-index-1],result_high[count:,-index-1] = low[:,-1],high[:,-1]
    return result_low,result_high


def _norm_upper(low, high, depth=6):
    # A vector Bezier curve lies in its control-point convex hull. Subdivision
    # tightens that enclosure; no time sample or polynomial-root guess is used.
    low, high = low[None], high[None]
    for _ in range(depth):
        low, high = _split_intervals(low, high)
    magnitudes = np.maximum(np.abs(low), np.abs(high))
    squares = _up(magnitudes*magnitudes)
    summed = _up(_up(squares[..., 0]+squares[..., 1])+squares[..., 2])
    return float(np.max(_up(np.sqrt(summed))))


def _bounds(points, duration, depth=6):
    if duration == 0:
        return DerivativeBounds(0., 0., 0.)
    low = high = np.asarray(points)
    values = []
    for _ in range(3):
        low, high = _derivative_intervals(low, high, duration)
        values.append(_norm_upper(low, high, depth))
    return DerivativeBounds(*values)


def _evaluate(points, u):
    points = points.copy()
    while len(points) > 1:
        points = (1-u)*points[:-1]+u*points[1:]
    return points[0].copy()


@dataclass(frozen=True, init=False)
class ContinuousReference:
    control_points: tuple[tuple[float, float, float], ...]
    start_time: float
    duration: float
    limits: ReferenceLimits
    tracking_error: TrackingError | None
    kind: str
    _certificate: DerivativeBounds

    def __init__(self, position, *, start_time=0., limits=ReferenceLimits()):
        point = _vector(position)
        self._initialize((point,)*6, _time(start_time), 0., limits, None, 'hold')

    def _initialize(self, points, start, duration, limits, tracking, kind):
        for name, value in (('control_points', tuple(_vector(p) for p in points)),
                            ('start_time', _time(start)), ('duration', _time(duration)),
                            ('limits', limits), ('tracking_error', tracking), ('kind', kind)):
            object.__setattr__(self, name, value)
        certificate = _bounds(self.control_points, duration)
        values = (certificate.speed, certificate.acceleration, certificate.jerk)
        caps = (limits.max_speed, limits.max_acceleration, limits.max_jerk)
        # A coarse Bernstein hull may reject a feasible near-limit curve.
        # Spend the larger finite subdivision budget only on ambiguous curves.
        # The1.001 trigger controls computation, never the admissible limits:
        # every returned transition still needs each outward upper bound<=cap.
        if any(value > cap for value,cap in zip(values,caps)) and all(
                value <= cap*1.001 for value,cap in zip(values,caps)):
            certificate = _bounds(self.control_points, duration, depth=10)
        object.__setattr__(self, '_certificate', certificate)

    @property
    def end_time(self):
        return self.start_time+self.duration

    @property
    def target(self):
        return self.control_points[-1]

    def bounds(self):
        """Upper bounds valid on the whole segment, including its endpoints."""
        return self._certificate

    def sample(self, time):
        time = _time(time)
        if time < self.start_time:
            raise ValueError('A reference cannot be sampled before its start time')
        if self.duration == 0 or time >= self.end_time:
            return np.array(self.target), np.zeros(3), np.zeros(3)
        u = (time-self.start_time)/self.duration
        points = np.array(self.control_points)
        result = []
        for _ in range(3):
            result.append(_evaluate(points, u))
            points = (len(points)-1)*np.diff(points, axis=0)/self.duration
        return tuple(result)

    def positions(self, times):
        """Batched positions with the scalar sampler's exact de Casteljau order.

        No derivative arrays are constructed. Each returned row is independent;
        the immutable reference and its derivative certificate remain unchanged.
        """
        times = np.asarray(times,dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < self.start_time):
            raise ValueError('Finite one-dimensional times at or after segment start required')
        result = np.broadcast_to(self.target,(len(times),3)).copy()
        active = times < self.end_time
        if self.duration and np.any(active):
            u = ((times[active]-self.start_time)/self.duration)[:,None,None]
            points = np.broadcast_to(self.control_points,(int(active.sum()),6,3))
            while points.shape[1] > 1:
                points = (1-u)*points[:,:-1]+u*points[:,1:]
            result[active] = points[:,0]
        return result

    def _tracking(self, time, actual_state):
        if not isinstance(actual_state, KinematicState):
            raise ValueError('A measured KinematicState is required')
        p, v, a = self.sample(time)
        acceleration = (None if actual_state.acceleration is None else
                        float(np.linalg.norm(a-actual_state.acceleration)))
        return TrackingError(float(np.linalg.norm(p-actual_state.position)),
                             float(np.linalg.norm(v-actual_state.velocity)), acceleration)

    def _transition(self, goal, time, actual_state, brake):
        time = _time(time)
        p, v, a = self.sample(time)
        tracking = self._tracking(time, actual_state)
        if brake and np.array_equal(v, np.zeros(3)) and np.array_equal(a, np.zeros(3)):
            result = object.__new__(type(self))
            result._initialize((tuple(p),)*6, time, 0., self.limits, tracking, 'hold')
            return result
        duration = (max(.2, 2*np.linalg.norm(v)/self.limits.max_acceleration,
                        math.sqrt(6*np.linalg.norm(v)/self.limits.max_jerk),
                        4*np.linalg.norm(a)/self.limits.max_jerk)
                    if brake else nominal_duration(p, goal, self.limits))
        # Tiny slack admits representable certificates at exactly nominal peaks.
        duration *= 1+1e-10
        next_expanded_duration = duration*1.2
        shrinking = False
        shorter_attempts = 0
        shrink_finished = False
        for _ in range(self.limits.max_attempts):
            if not math.isfinite(duration) or duration > self.limits.max_duration:
                break
            target = p+v*duration/2+a*duration**2/12 if brake else np.asarray(goal)
            points = [p, p+v*duration/5, p+2*v*duration/5+a*duration**2/20,
                      target, target, target]
            result = object.__new__(type(self))
            result._initialize(points, time, duration, self.limits, tracking, 'brake' if brake else 'target')
            bound = result.bounds()
            if (bound.speed <= self.limits.max_speed and bound.acceleration <= self.limits.max_acceleration
                    and bound.jerk <= self.limits.max_jerk):
                return result
            # With nonzero initial acceleration, a longer stop can INCREASE
            # peak speed. Search the same stop family in the shorter direction
            # when speed is the sole failing limit. This consumes the same fixed
            # total attempt budget and never changes any derivative limit.
            speed_only = (bound.speed > self.limits.max_speed
                          and bound.acceleration <= self.limits.max_acceleration
                          and bound.jerk <= self.limits.max_jerk)
            if brake and speed_only and not shrink_finished and shorter_attempts < 8 and duration/1.2 >= .2:
                if not shrinking:
                    next_expanded_duration = duration*1.2
                shrinking = True
                shorter_attempts += 1
                duration /= 1.2
            elif shrinking:
                shrink_finished = True
                shrinking = False
                duration = next_expanded_duration
            else:
                duration *= 1.2
        raise ReferenceInfeasible('Bounded search found no admissible reference; retain current reference and fail closed')

    def retarget(self, goal, time, *, actual_state):
        """Return a stopped-endpoint segment beginning at the existing p/v/a."""
        return self._transition(_vector(goal), time, actual_state, False)

    def brake(self, time, *, actual_state):
        """Return a bounded stop with endpoint p+v*T/2+a*T²/12.

        Actual-state error is retained for the supervisor's separately qualified
        envelope. This reference certificate alone does not certify physical stop.
        """
        return self._transition(None, time, actual_state, True)

    def to_dict(self):
        return dict(schema='dual-continuous-reference-v1', control_points=[list(p) for p in self.control_points],
                    start_time=self.start_time, duration=self.duration, limits=asdict(self.limits),
                    tracking_error=None if self.tracking_error is None else asdict(self.tracking_error), kind=self.kind)

    @classmethod
    def from_dict(cls, state):
        if state['schema'] != 'dual-continuous-reference-v1':
            raise ValueError('Unknown reference schema')
        points = tuple(_vector(p) for p in state['control_points'])
        duration, start = _time(state['duration']), _time(state['start_time'])
        limits = ReferenceLimits(**state['limits'])
        if (len(points) != 6 or duration < 0 or duration > limits.max_duration or
                points[-1] != points[-2] or points[-1] != points[-3] or
                (duration == 0 and any(p != points[0] for p in points))):
            raise ValueError('Invalid reference segment')
        result = object.__new__(cls)
        tracking = None if state['tracking_error'] is None else TrackingError(**state['tracking_error'])
        result._initialize(points, start, duration, limits, tracking, state['kind'])
        bound = result.bounds()
        if (bound.speed > limits.max_speed or bound.acceleration > limits.max_acceleration or
                bound.jerk > limits.max_jerk):
            raise ValueError('Restored reference exceeds motion bounds')
        return result
