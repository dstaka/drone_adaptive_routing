"""Measured planar rod/cube Lagrangian replay; no simulator or admission path.

Angles are absolute downward body axes towards world +X, omega_Y=-angle_rate.
T=.5*A*u²+.5*D*v²+C*cos(theta-phi)*u*v; V=G*(1-cos(theta))+H*(1-cos(phi)).
This constrained world-XZ diagnostic cannot establish a general 3D engine defect.
"""
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _array(value, shape=None):
    if type(value) not in (list, tuple, np.ndarray):
        raise ValueError('Expected finite real array')
    result = np.asarray(value)
    if (result.dtype.kind not in 'fiu' or not np.isfinite(result).all() or
            (shape is not None and result.shape != shape)):
        raise ValueError('Invalid finite real array shape')
    return result.astype(float)


def _number(value):
    try:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Expected finite real scalar')
    except OverflowError as exc:
        raise ValueError('Unrepresentable real scalar') from exc
    return value


def _finite_math(function):
    @wraps(function)
    def checked(*args, **kwargs):
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                result = function(*args, **kwargs)
            except FloatingPointError as exc:
                raise ValueError('Nonfinite derived mechanics') from exc
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite derived mechanics')
        return result
    return checked


@dataclass(frozen=True)
class PlanarModel:
    rod_mass_kg: float
    cube_mass_kg: float
    rod_inertia_y: float
    cube_inertia_y: float
    side_m: float

    def __post_init__(self):
        if any(_number(v) <= 0 for v in vars(self).values()) or self.side_m not in (.03, .045, .06):
            raise ValueError('Positive measured masses/inertias and declared cube side required')
        a, d, c, g, h = self.coefficients
        if not all(math.isfinite(x) and x > 0 for x in (a, d, c, g, h, a*d-c*c)):
            raise ValueError('Unrepresentable or singular planar mechanics')

    @property
    def coefficients(self):
        mr, mp, a = self.rod_mass_kg, self.cube_mass_kg, self.side_m/2
        return (mr*.15**2+self.rod_inertia_y+mp*.3**2, mp*a*a+self.cube_inertia_y,
                mp*.3*a, 9.81*(mr*.15+mp*.3), 9.81*mp*a)

    def _state(self, state):
        s = _array(state)
        if s.ndim not in (1, 2) or s.shape[-1] != 4:
            raise ValueError('Expected theta,phi,theta_dot,phi_dot')
        return s.T

    @_finite_math
    def acceleration(self, state):
        t, p, u, v = self._state(state)
        a, d, c, g, h = self.coefficients
        coupling, sine = c*np.cos(t-p), c*np.sin(t-p)
        f, k = -sine*v*v-g*np.sin(t), sine*u*u-h*np.sin(p)
        det = a*d-coupling**2
        return np.stack([(d*f-coupling*k)/det, (a*k-coupling*f)/det], axis=-1)

    @_finite_math
    def energy(self, state):
        t, p, u, v = self._state(state)
        a, d, c, g, h = self.coefficients
        return .5*a*u*u+.5*d*v*v+c*np.cos(t-p)*u*v+g*(1-np.cos(t))+h*(1-np.cos(p))

    @_finite_math
    def step(self, state, dt_s):
        if _number(dt_s) <= 0:
            raise ValueError('Positive integration dt required')
        s = _array(state)
        velocity = s[..., 2:]+dt_s*self.acceleration(s)
        return np.concatenate([s[..., :2]+dt_s*velocity, velocity], axis=-1)


def _model(info, substeps):
    if type(substeps) is not int or not 1 <= substeps <= 256:
        raise ValueError('Explicit integer substeps in [1,256] required')
    case, settings, rt = info['case'], info['settings'], info['runtime']
    if (info['fixture'] != 'anchor' or case['substeps'] != substeps or
            case['sample_dt_s'] != .01 or settings['control_dt_s'] != .01 or
            case['initial_direction_rad'] != 0 or case['reset_during_run'] is not False or
            not np.array_equal(_array(settings['gravity_m_s2'], (3,)), [0, 0, -9.81])):
        raise ValueError('Expected fixed passive XZ case and exact .01s control interval')
    for key, expected in [('dt', .01), ('substep_dt', .01/substeps), ('_substep_dt', .01/substeps)]:
        observed = rt['actual_solver_settings'][key]
        if type(observed) not in (str, float, int) or not math.isclose(float(observed), expected, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError('Actual solver dt differs from explicit replay substeps')
    if not math.isclose(_number(settings['rigid_dt_s']), .01/substeps, rel_tol=1e-10, abs_tol=1e-14):
        raise ValueError('Declared rigid dt differs')
    if any(np.any(_array(rt[key]) != 0) for key in ('damping_after', 'armature')):
        raise ValueError('Passive damping/armature outside conservative model')
    values = []
    for name, com in [('suspension_rod', [0, 0, -.15]), ('payload', [0, 0, 0])]:
        indices = [i for i, link in enumerate(rt['links']) if link['name'] == name]
        if len(indices) != 1:
            raise ValueError('Unique measured rod and cube required')
        i = indices[0]
        inertia = _array(rt['inertia_kg_m2'][i], (3, 3))
        if not np.allclose(inertia, np.diag(np.diag(inertia)), atol=1e-12, rtol=0) or np.min(np.diag(inertia)) <= 0:
            raise ValueError('Positive diagonal body COM inertia required')
        for key in ('links_inertial_quaternion_wxyz', 'links_frame_offset_quaternion_wxyz'):
            if not np.allclose(_array(rt[key][i], (4,)), [1, 0, 0, 0], atol=1e-7, rtol=0):
                raise ValueError('Unsupported measured inertia/body frame')
        if not np.allclose(_array(rt['links_com_body_m'][i], (3,)), com, atol=1e-7, rtol=0):
            raise ValueError('Measured COM outside planar model')
        values.append((_number(rt['links'][i]['mass_kg']), float(inertia[1, 1])))
    return PlanarModel(values[0][0], values[1][0], values[0][1], values[1][1], case['side_m'])


def diagnose(probe, *, substeps):
    """Restart a local velocity-first map from every observed state, no free replay."""
    probe = Path(probe)
    files = {name: (probe/name).read_bytes() for name in ('runtime-initial.json', 'samples.jsonl')}
    try:
        info = json.loads(files['runtime-initial.json'])
        model = _model(info, substeps)
        rows = [json.loads(line) for line in files['samples.jsonl'].splitlines()]
        duration = _number(info['case']['duration_s'])
        if duration <= 0 or len(rows) < 3 or not math.isclose(duration, (len(rows)-1)*.01, abs_tol=1e-10, rel_tol=0):
            raise ValueError('Complete .01s initial/intermediate/terminal trace required')
        states, positions, velocities = [], [], []
        anchor = _array(rows[0]['drone_xyz'], (3,))+[0, 0, -.02]
        for i, row in enumerate(rows):
            if type(row['step']) is not int or row['step'] != i or not math.isclose(_number(row['time_s']), i*.01, abs_tol=1e-10, rel_tol=0):
                raise ValueError('Exact .01s sample timing required')
            if np.any(_array(row['rpm'], (4,)) != 0) or row['contact'] is not False:
                raise ValueError('Nonpassive input or contact')
            for key in ('rotor_force_world_n', 'rotor_torque_world_nm'):
                if key in row and np.any(_array(row[key], (4, 3)) != 0):
                    raise ValueError('Nonzero recorded motor force/torque')
            if any(key in row and _number(row[key]) != 0 for key in ('motor_power_w', 'motor_work_j')):
                raise ValueError('Nonzero recorded motor work/power')
            angle, rate = [], []
            for body in ('drone', 'rod', 'package'):
                q = _array(row[f'{body}_quaternion_wxyz'], (4,))
                xyz, velocity = _array(row[f'{body}_xyz'], (3,)), _array(row[f'{body}_velocity'], (3,))
                omega = _array(row[f'{body}_angular_velocity'], (3,))
                if abs(np.linalg.norm(q)-1) > 1e-5 or max(abs(q[1]), abs(q[3]), abs(omega[0]), abs(omega[2]), abs(xyz[1]-anchor[1]), abs(velocity[1])) > 1e-7:
                    raise ValueError('Nonplanar or invalid measured state')
                if body in ('drone', 'rod') and (np.linalg.norm(velocity) > 1e-7 or np.linalg.norm(xyz-(anchor+([0, 0, .02] if body == 'drone' else [0, 0, 0]))) > 1e-6):
                    raise ValueError('Moving anchor/rod origin outside fixed model')
                if body == 'drone' and (np.linalg.norm(omega) > 1e-7 or abs(q[2]) > 1e-7):
                    raise ValueError('Moving/tilted anchor outside fixed model')
                if body != 'drone':
                    angle.append(-2*math.atan2(q[2], q[0]))
                    rate.append(-omega[1])
            states.append(angle+rate)
            positions.append(row['package_xyz'])
            velocities.append(row['package_velocity'])
        states = np.asarray(states)
        states[:, :2] = np.unwrap(states[:, :2], axis=0)
        if np.max(abs(states[0, :2]-math.radians(_number(info['case']['initial_swing_deg'])))) > 1e-5 or np.max(abs(states[0, 2:])) > 1e-7:
            raise ValueError('Measured initial state differs from declared displacement/rest')
        t, p, u, v = states.T
        a = model.side_m/2
        xyz = anchor+np.array([.3*np.sin(t)+a*np.sin(p), np.zeros(len(t)), -.3*np.cos(t)-a*np.cos(p)]).T
        vel = np.array([.3*np.cos(t)*u+a*np.cos(p)*v, np.zeros(len(t)), .3*np.sin(t)*u+a*np.sin(p)*v]).T
        predicted = states[:-1].copy()
        for _ in range(substeps):
            predicted = model.step(predicted, .01/substeps)
        energy = model.energy(states)
        increments = model.energy(predicted)-energy[:-1]
        errors = increments-np.diff(energy)
        state_error = predicted-states[1:]
        state_error[:, :2] = (state_error[:, :2]+np.pi)%(2*np.pi)-np.pi
        return dict(schema='payload-planar-local-replay-v1', status='CPU_PLANAR_DIAGNOSTIC_ONLY', physical_training_admitted=False,
                    sample_count=len(rows), substeps=substeps, actual_integration_dt_s=.01/substeps,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    inputs={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in files.items()},
                    measured_model=vars(model), mechanical_energy_j=energy.tolist(),
                    kinematic_closure=dict(maximum_cube_position_error_m=float(np.linalg.norm(xyz-positions, axis=1).max()),
                                           maximum_cube_velocity_error_m_s=float(np.linalg.norm(vel-velocities, axis=1).max())),
                    local_replay=dict(maximum_state_error=float(abs(state_error).max()), rms_state_error=np.sqrt(np.mean(state_error**2, axis=0)).tolist(),
                                      maximum_energy_increment_error_j=float(abs(errors).max()), rms_energy_increment_error_j=float(np.sqrt(np.mean(errors**2))),
                                      observed_total_energy_change_j=float(energy[-1]-energy[0]), predicted_summed_local_energy_change_j=float(increments.sum())),
                    interpretation_ja=['各観測状態から速度先行更新を再始動した平面内の整合性診断。3次元の原因特定、物理資格、学習許可を示さない。',
                                       '0.01秒差分の加速度は複数substepを平均するため、残差を瞬間的な未知の力と解釈できない。',
                                       '入力検査は記録RPMと存在する推定wrench/workを対象とし、実solver外力の未観測を補わない。'])
    except (KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError('Invalid measured planar input') from exc
