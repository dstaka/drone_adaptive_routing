"""Measured-state local feedback. No simulator, load property or seed inputs.

Ideal sensors: body-to-world quaternions and world angular velocities of drone,
link and box, plus drone position/velocity. Units are SI; no normalization or
clipping of sensor state. Sampling/update interval .01s with zero sensor delay.
Gains are designed once on the nominal model, never on the case's true load.
"""

import numpy as np

from .payload_contract import rotation
from .payload_ball_bank import _quaternion

SENSOR_FIELDS = ('drone_xyz', 'drone_velocity', *(
    body + suffix for body in ('drone', 'rod', 'package')
    for suffix in ('_quaternion_wxyz', '_angular_velocity')))


def _rotvec(matrix):
    q = _quaternion(matrix)
    length = np.linalg.norm(q[1:])
    return 2*q[1:] if length < 1e-10 else 2*np.arctan2(length, q[0])*q[1:]/length


def measured_state(sample, target):
    s = {k: np.asarray(sample[k], float) for k in SENSOR_FIELDS}
    if not all(np.isfinite(v).all() for v in s.values()):
        raise ValueError('Nonfinite sensor')
    rd, rr, rp = (rotation(s[b+'_quaternion_wxyz']) for b in ('drone', 'rod', 'package'))
    wd, wr, wp = (s[b+'_angular_velocity'] for b in ('drone', 'rod', 'package'))
    q = np.r_[s['drone_xyz']-np.asarray(target), _rotvec(rd),
              _rotvec(rd.T @ rr)[:2], _rotvec(rr.T @ rp)[:2]]
    v = np.r_[s['drone_velocity'], rd.T @ wd, (rr.T @ (wr-wd))[:2], (rp.T @ (wp-wr))[:2]]
    result = np.r_[q, v]
    if result.shape != (20,):
        raise ValueError('Sensor shape mismatch')
    return result


class Controller:
    """Discrete feedback plus bounded altitude integral with anti-windup.

    Optional H layer: world acceleration correction <=.5m/s² per axis; converted
    to a target displacement using fixed nominal position stiffness 1.2s⁻².
    Every action passes through the original immediate four-motor mapping.
    """

    def __init__(self, gain, hover_action, *, vibration_feedback=True):
        self.gain = np.asarray(gain, float).copy()
        self.hover_action = np.asarray(hover_action, float).copy()
        if self.gain.shape != (4,20) or self.hover_action.shape != (4,):
            raise ValueError('Fixed nominal controller tensors required')
        if not np.isfinite(self.gain).all() or not np.isfinite(self.hover_action).all():
            raise ValueError('Finite controller tensors required')
        self.vibration_feedback = vibration_feedback
        self.reset()

    def reset(self):
        self.tick = None
        self.integral = 0.
        self.last_audit = None

    def action(self, sensors, target, *, tick, correction=None, reference_velocity=(0,0,0)):
        if type(tick) is not int or tick < 0 or (self.tick is not None and tick != self.tick+1):
            raise ValueError('One update per consecutive control tick required')
        delta = np.zeros(3) if correction is None else np.asarray(correction, float)
        if delta.shape != (3,) or not np.isfinite(delta).all() or np.any(np.abs(delta) > .5):
            raise ValueError('Bounded world acceleration correction required')
        x = measured_state(sensors, np.asarray(target) + delta/1.2)
        x[10:13] -= reference_velocity
        if not self.vibration_feedback:
            x[[6,7,8,9,16,17,18,19]] = 0
        proposed = float(np.clip(self.integral - .01*.04*x[2], -.12, .12))
        before = self.hover_action - self.gain @ x + self.integral
        raw = self.hover_action - self.gain @ x + proposed
        # Freeze integration when saturated action would be pushed farther out.
        if np.all(np.abs(raw) <= 1) or np.max(np.abs(raw)) < np.max(np.abs(before)):
            self.integral = proposed
        else:
            raw = before
        result = np.clip(raw, -1., 1.)
        self.last_audit = dict(state=x.tolist(), correction_world_m_s2=delta.tolist(),
            before_motor_clip=raw.tolist(), motor_action=result.tolist(),
            saturated=bool(np.any(raw != result)), altitude_integral=self.integral,
            vibration_feedback=self.vibration_feedback)
        self.tick = tick
        return result
