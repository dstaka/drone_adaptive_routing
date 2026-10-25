"""Offline nominal local design; true case parameters never reach runtime C."""

import numpy as np

from .payload_contract import Payload
from .payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG, HOVER_RPM

INDICES = np.array([0,1,2,3,4,5,6,7,9,10])


def design_nominal(original, output, *, recipe='C-local-lqr-v1'):
    from scipy.linalg import expm, solve_discrete_are

    if recipe not in ('C-local-lqr-v1', 'C-fast-damping-v2', 'C-fast-authority-v3'):
        raise ValueError('Predeclared controller hypothesis required')
    f = MuJoCoFixture(original, output, Payload(FREE_MASS_KG,.2,.045), substeps=16)
    q0 = f.data.qpos.copy()
    hover = np.full(4, (np.sqrt((FREE_MASS_KG+.0001+FREE_MASS_KG*.2)/FREE_MASS_KG)-1)/.8)
    eps = 1e-5

    def acceleration(q, v, u):
        f.data.qpos[:] = q
        f.data.qvel[:] = v
        f.rpm = (1+.8*u)*HOVER_RPM
        f.mj.mj_forward(f.model, f.data)
        f._control(f.model, f.data)
        f.mj.mj_forward(f.model, f.data)
        return f.data.qacc[INDICES].copy()

    a = np.zeros((20,20))
    a[:10,10:] = np.eye(10)
    b = np.zeros((20,4))
    for j in range(20):
        values = []
        for sign in (-1,1):
            q, v = q0.copy(), np.zeros(12)
            if j < 10:
                tangent = np.zeros(12)
                tangent[INDICES[j]] = sign*eps
                f.mj.mj_integratePos(f.model, q, tangent, 1.)
            else:
                v[INDICES[j-10]] = sign*eps
            values.append(acceleration(q,v,hover))
        a[10:,j] = (values[1]-values[0])/(2*eps)
    for j in range(4):
        delta = np.zeros(4)
        delta[j] = eps
        b[10:,j] = (acceleration(q0,np.zeros(12),hover+delta)-
                    acceleration(q0,np.zeros(12),hover-delta))/(2*eps)
    augmented = np.zeros((24,24))
    augmented[:20,:20], augmented[:20,20:] = a,b
    discrete = expm(augmented*.01)
    ad, bd = discrete[:20,:20], discrete[:20,20:]
    weights = np.array([8,8,12, .5,.5,1, .1,.1, .5,.5,
                        3,3,4, .03,.03,.05, .004,.004, .004,.004])
    if recipe == 'C-fast-damping-v2':
        weights[[8,9,18,19]] *= 10
    if recipe == 'C-fast-authority-v3':
        weights[[8,9,18,19]] *= 100
    q, r = np.diag(weights), np.eye(4)*.5
    p = solve_discrete_are(ad,bd,q,r)
    gain = np.linalg.solve(r+bd.T@p@bd, bd.T@p@ad)
    return dict(recipe=recipe, gain=gain.tolist(), hover_action=hover.tolist(),
        continuous_a=a.tolist(), continuous_b=b.tolist(), weights=weights.tolist(),
        spectral_radius=float(np.max(np.abs(np.linalg.eigvals(ad-bd@gain)))),
        design_load=dict(side_m=.045,mass_ratio=.2),
        scope='Offline nominal design only; excluded axial ball coordinates are unactuated symmetry modes')
