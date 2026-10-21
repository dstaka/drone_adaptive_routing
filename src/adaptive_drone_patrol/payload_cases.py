"""Reproducible paired evaluation inputs, independent of training/validation."""

from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
from .payload_contract import SIZES_M, LINK_M, DRONE_ATTACHMENT, rotation
from .util import sha256, json_write

SCENARIOS = ('normal', 'emergency', 'cost_shift')
CASE_SEED = 731021


def initial_state(rng, side_m, endpoint_direction=None):
    roll, pitch = np.radians(rng.uniform(-2, 2, 2))
    cr, sr, cp, sp = math.cos(roll/2), math.sin(roll/2), math.cos(pitch/2), math.sin(pitch/2)
    q = [cr*cp, sr*cp, cr*sp, -sr*sp]
    angle = float(rng.uniform(0, 10)) if endpoint_direction is None else 10.
    direction = float(rng.uniform(0, 2*math.pi)) if endpoint_direction is None else endpoint_direction
    theta = math.radians(angle)
    r = LINK_M * np.array([math.sin(theta)*math.cos(direction), math.sin(theta)*math.sin(direction),
                          -math.cos(theta)])
    drone = np.array([*rng.uniform(-.02, .02, 2), 1.])
    anchor = drone + rotation(q) @ DRONE_ATTACHMENT
    package = anchor + r - np.array([0, 0, side_m/2])
    return dict(drone_xyz=drone.tolist(), drone_quaternion_wxyz=q, package_xyz=package.tolist(),
                package_quaternion_wxyz=[1., 0., 0., 0.],
                drone_velocity=[0., 0., 0.], drone_angular_velocity=[0., 0., 0.],
                package_velocity=[0., 0., 0.], package_angular_velocity=[0., 0., 0.],
                swing_angle_deg=angle, swing_direction_rad=direction)


def build_cases():
    cases = []
    for scenario in SCENARIOS:
        for seed in (1, 2, 3):
            rng = np.random.default_rng(np.random.SeedSequence([CASE_SEED, 730, seed]))
            cases.append(dict(case_id=f'basic-{scenario}-{seed}', suite='basic', scenario=scenario,
                              execution_seed=seed, mass_ratio=.2, side_m=.045,
                              initial=initial_state(rng, .045)))
    for si, scenario in enumerate(SCENARIOS):
        for zi, size in enumerate(SIZES_M):
            for mi, mass in enumerate((.1, .3)):
                seed = 74000 + si*6 + zi*2 + mi
                rng = np.random.default_rng(np.random.SeedSequence([CASE_SEED, 740, seed]))
                direction = 2*math.pi*(si*6 + zi*2 + mi)/18
                cases.append(dict(case_id=f'endpoint-{scenario}-{zi}-{mi}', suite='endpoints',
                                  scenario=scenario, execution_seed=seed, mass_ratio=mass, side_m=size,
                                  initial=initial_state(rng, size, direction)))
    for si, scenario in enumerate(SCENARIOS):
        for zi, size in enumerate(SIZES_M):
            n = 12 if si == zi == 0 else 11
            for j in range(n):
                seed = 75000 + si*1000 + zi*100 + j
                # Mass and initial state consume separate keyed streams.
                mass_rng = np.random.default_rng(np.random.SeedSequence([CASE_SEED, 750, seed]))
                initial_rng = np.random.default_rng(np.random.SeedSequence([CASE_SEED, 751, seed]))
                cases.append(dict(case_id=f'heldout-{scenario}-{zi}-{j}', suite='heldout', scenario=scenario,
                                  execution_seed=seed, mass_ratio=float(mass_rng.uniform(.1, .3)), side_m=size,
                                  initial=initial_state(initial_rng, size)))
    assert Counter(c['suite'] for c in cases) == dict(basic=9, endpoints=18, heldout=100)
    return cases


def _scenario_contract(root):
    import yaml
    result = {'priority_seed': 40029, 'layout_seed': 51000, 'layout_attempt': 3353, 'files': {}}
    for name in SCENARIOS:
        path = Path(root)/f'configs/demo_scenarios_dispersed/{name}.yaml'
        d = yaml.safe_load(path.read_text())
        expected = [81, 42, 43, 64, 23, 84, 6, 98, 22, 56]
        if d['priorities'] != expected or d['coords'][0] != [0, 0, 1]:
            raise ValueError('Inherited scenario contract changed')
        result['files'][name] = dict(path=str(path.relative_to(root)), sha256=sha256(path))
    return result


def write_case_manifest(path, root):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    data = dict(schema='payload-cases-v1', generator='numpy-PCG64-keyed-v1', seed=CASE_SEED,
                scenario_contract=_scenario_contract(Path(root)),
                pairing='Use these exact physical cases for both policies; never redraw at execution',
                attachment_status='PROVISIONAL: underside offset requires pinned physical asset probe',
                drone_attachment_body_m=DRONE_ATTACHMENT.tolist(), link_m=LINK_M,
                recording_cases=[f'basic-{s}-1' for s in SCENARIOS],
                results='NOT_RUN', cases=build_cases())
    json_write(path, data)
    return sha256(path)


def verify_case_manifest(path, expected_hash, root):
    if sha256(path) != expected_hash:
        raise ValueError('Case manifest hash mismatch')
    data = json.loads(Path(path).read_text())
    if data['schema'] != 'payload-cases-v1' or data['cases'] != build_cases():
        raise ValueError('Cases differ from predeclared generator')
    if data['scenario_contract'] != _scenario_contract(Path(root)):
        raise ValueError('Actual scenario contents differ from manifest')
    return data
