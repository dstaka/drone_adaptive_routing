"""Predeclared development inputs, disjoint from formal/video fixtures."""

import math
import numpy as np

from .payload_cases import initial_state
from .payload_contract import DRONE_ATTACHMENT, LINK_M, rotation

OFFSETS = ((.5,0,0),(-.5,0,0),(0,.5,0),(0,-.5,0),(0,0,.5))


def development_cases():
    result = []
    for gate, count in (('C0',10),('C1',10),('C2',30)):
        for i in range(count):
            seed = 82624000 + len(result)
            size = (.030,.045,.060)[i//10] if gate == 'C2' else .045
            ratio = (.1,.3)[i//5%2] if gate == 'C2' else .2
            rng = np.random.default_rng(np.random.SeedSequence([seed, 8262]))
            initial = initial_state(rng,size)
            angle = 5. if gate == 'C0' and i < 5 else 10.
            azimuth = float(rng.uniform(0,2*math.pi))
            theta = math.radians(angle)
            direction = np.array([math.sin(theta)*math.cos(azimuth),
                                  math.sin(theta)*math.sin(azimuth), -math.cos(theta)])
            initial.update(swing_angle_deg=angle,swing_direction_rad=azimuth)
            initial['package_xyz'] = (np.asarray(initial['drone_xyz'])+
                rotation(initial['drone_quaternion_wxyz'])@DRONE_ATTACHMENT+
                LINK_M*direction-[0,0,size/2]).tolist()
            targets = [[0.,0.,1.]]
            if gate != 'C0':
                offset = OFFSETS[i//2] if gate == 'C1' else OFFSETS[i%5]
                targets += [(np.array([0.,0.,1.])+offset).tolist(), [0.,0.,1.]]
            result.append(dict(case_id=f'{gate}-{i:02d}',gate=gate,seed=seed,
                side_m=size,mass_ratio=ratio,initial=initial,targets=targets))
    return result


def validate_cases(cases):
    """Check generator while executing frozen input bytes across NumPy builds.

    BLAS/NumPy roundoff in derived initial body geometry may differ below 1e-14m;
    targets, seed allocation and all noninitial fields must be exactly equal.
    The consumer always executes the original manifest, never this redraw.
    """
    expected = development_cases()
    if len(cases)!=len(expected):
        raise ValueError('Wrong development denominator')
    for supplied, generated in zip(cases,expected):
        if {k:v for k,v in supplied.items() if k!='initial'}!={k:v for k,v in generated.items() if k!='initial'}:
            raise ValueError('Development command allocation differs')
        if set(supplied['initial'])!=set(generated['initial']) or any(
            not np.allclose(supplied['initial'][k],generated['initial'][k],atol=1e-14,rtol=0)
            for k in generated['initial']):
            raise ValueError('Development initial draw differs')
    return True


def validate_gate_result(gate, outcomes):
    cases = [c for c in development_cases() if c['gate']==gate]
    if not cases or len(outcomes) != len(cases):
        return False
    by_id = {r['case_id']:r for r in outcomes}
    if len(by_id) != len(cases) or set(by_id) != {c['case_id'] for c in cases}:
        return False
    if not all(r['safe'] for r in outcomes):
        return False
    def passed(c):
        r = by_id[c['case_id']]
        return (r['success'] and r['completed_targets']==len(c['targets'])
                and (gate=='C0' or r['max_displacement_m'] >= .10-1e-8))
    # .50m targets with .20m tolerance imply >=.10m between actual stop poses;
    # the checker also reconstructs each stop neighborhood and ordered commands.
    if gate == 'C0':
        return all(passed(c) for c in cases)
    if gate == 'C1':
        return sum(passed(c) for c in cases) >= 9
    return all(sum(passed(c) for c in cases if c['side_m']==s and c['mass_ratio']==m)>=4
               for s in (.03,.045,.06) for m in (.1,.3))
