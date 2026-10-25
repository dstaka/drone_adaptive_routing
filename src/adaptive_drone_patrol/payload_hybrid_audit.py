"""Independent raw command/settling reconstruction; saved verdicts are not input.

This reader uses native body getters and separately implements endpoint velocity,
oriented clearance and interval predicates. It does not import the live runtime.
"""

import gzip
import json
import math
from pathlib import Path

import numpy as np

from .payload_contract import rotation
from .util import sha256

LIMITS = dict(drone_error_m=.20, drone_speed_m_s=.30, package_error_m=.20,
              package_speed_m_s=.15, swing_angle_deg=5., swing_rate_deg_s=15., package_spin_deg_s=30.)


def independent_metrics(s, target, side):
    rd, rp = (rotation(s[b+'_quaternion_wxyz']) for b in ('drone','package'))
    d,p,vd,vp,wd,wp = (np.asarray(s[k],float) for k in (
        'drone_xyz','package_xyz','drone_velocity','package_velocity',
        'drone_angular_velocity','package_angular_velocity'))
    a, b = rd@np.array([0,0,-.02]), rp@np.array([0,0,side/2])
    link = p+b-d-a
    length = float(np.linalg.norm(link))
    if not np.isfinite(length) or length<1e-8:
        raise ValueError('Invalid native attachment state')
    relative = vp+np.cross(wp,b)-vd-np.cross(wd,a)
    nominal = np.asarray(target)+[0,0,-.02-.30-side/2]
    result = dict(drone_error_m=float(np.linalg.norm(d-target)),
        drone_speed_m_s=float(np.linalg.norm(vd)),package_error_m=float(np.linalg.norm(p-nominal)),
        package_speed_m_s=float(np.linalg.norm(vp)),
        swing_angle_deg=float(np.degrees(np.arctan2(np.linalg.norm(link[:2]),-link[2]))),
        swing_rate_deg_s=float(np.degrees(np.linalg.norm(np.cross(link,relative))/length**2)),
        package_spin_deg_s=float(np.degrees(np.linalg.norm(wp))),
        constraint_residual_m=abs(length-.30),
        drone_floor_m=float(d[2]-.06*np.sqrt(max(0.,1-rd[2,2]**2))-.0125*abs(rd[2,2])),
        package_floor_m=float(p[2]-side/2*np.abs(rp[2]).sum()),
        drone_roll_deg=float(np.degrees(np.arctan2(rd[2,1],rd[2,2]))),
        drone_pitch_deg=float(np.degrees(np.arcsin(np.clip(-rd[2,0],-1,1)))))
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite reconstructed predicates')
    return result


def interval_update(state, valid):
    previous, count = state
    return bool(valid), count+1 if valid and previous else 0


def _valid(m, contact, strict=False):
    limits = dict(LIMITS)
    if strict:
        limits.update(package_speed_m_s=.10,swing_rate_deg_s=10.,package_spin_deg_s=10.)
    return bool(not contact and all(0<=m[k]<=b for k,b in limits.items())
        and m['drone_floor_m']>.1 and m['package_floor_m']>.1
        and abs(m['drone_roll_deg'])<=60 and abs(m['drone_pitch_deg'])<=60
        and m['constraint_residual_m']<=2e-5)


def audit_case(output, *, expected_case=None):
    root = Path(output)
    case = json.loads((root/'case.json').read_text())
    if expected_case is not None and case!=expected_case:
        raise ValueError('Case/command stream differs from predeclaration')
    saved = json.loads((root/'result.json').read_text())
    metadata = json.loads((root/'metadata.json').read_text())
    from .payload_hybrid_contract import VIDEO_PROFILE
    if metadata['profile']!=VIDEO_PROFILE.id or metadata['profile_sha256']!=VIDEO_PROFILE.sha256:
        raise ValueError('Profile binding differs')
    for filename,digest in metadata['source_sha256'].items():
        if sha256(Path(__file__).parent/filename)!=digest:
            raise ValueError('Executed runtime source differs from audit source tree')
    with gzip.open(root/'control.jsonl.gz','rt') as stream:
        control = [json.loads(line) for line in stream]
    if not control or control[-1].get('kind')!='TERMINAL':
        raise ValueError('Missing terminal evidence/flush')
    terminal = control.pop()['outcome']
    if terminal['completed_targets']!=saved['completed_targets'] or terminal['success']!=saved['success']:
        raise ValueError('Terminal evidence differs')
    live, shadow = (False,0),(False,0)
    current_leg, previous_step, steps, resets = 0,-1,0,0
    completed, stops, strict_stops = [],[],[]
    previous_pose = None
    previous_sample = None
    last_wall = 0.
    all_safe = True
    max_displacement, minimum_floor, peak_tilt, max_margin_error = 0.,math.inf,0.,0.
    with gzip.open(root/'runtime/raw.jsonl.gz','rt') as stream:
        for line in stream:
            native = json.loads(line)
            s = native['actual']
            step = native['step']
            if native['kind']=='RESET':
                resets += 1
                if resets!=1 or step!=0 or previous_step!=-1:
                    raise ValueError('Undeclared reset')
                previous_step = 0
                initial_pose = np.asarray(s['drone_xyz'])
                for body in ('drone','package'):
                    for field in ('xyz','quaternion_wxyz','velocity','angular_velocity'):
                        if not np.allclose(s[body+'_'+field],case['initial'][body+'_'+field],rtol=0,atol=1e-9):
                            raise ValueError('Native initial state differs')
            elif native['kind']=='RETARGET':
                if step!=previous_step or len(completed)!=current_leg+1:
                    raise ValueError('Uncompleted leg or clock reset')
                current_leg += 1
                if current_leg>=len(case['targets']):
                    raise ValueError('Excess target command')
                if previous_sample is None or s!=previous_sample:
                    raise ValueError('Retarget changed physical state')
                live,shadow=(False,0),(False,0)
            elif native['kind']=='STEP':
                if step!=previous_step+1 or steps>=len(control):
                    raise ValueError('Missing/reordered native samples')
                record = control[steps]
                if record['step']!=step or record['leg']!=current_leg or record['target']!=case['targets'][current_leg]:
                    raise ValueError('Actual command stream differs')
                if record['reset_elapsed_wall_s']<last_wall:
                    raise ValueError('Nonmonotonic wall clock')
                last_wall = record['reset_elapsed_wall_s']
                rpm = (1+.8*np.asarray(record['control']['motor_action']))*14468.429183500699
                if not np.allclose(s['rpm'],rpm,rtol=0,atol=1e-9):
                    raise ValueError('Current motor input mismatch')
                steps += 1
                previous_step = step
            else:
                raise ValueError('Unexpected native event')
            if native['target']!=case['targets'][current_leg] or native['target_id']!=str(current_leg):
                raise ValueError('Native target stream differs')
            if abs(s['time_s']-step*.01)>1e-7 or any(s['warning_counts']) or np.any(s['actual_xfrc_applied']):
                raise ValueError('Time discontinuity/backend reset/direct external body force')
            m = independent_metrics(s,case['targets'][current_leg],case['side_m'])
            contact = s['contact']
            if type(contact) is not bool:
                raise ValueError('Missing native contact state')
            all_safe &= bool(not contact and m['drone_floor_m']>.1 and m['package_floor_m']>.1
                and abs(m['drone_roll_deg'])<=60 and abs(m['drone_pitch_deg'])<=60
                and m['constraint_residual_m']<=2e-5)
            live = interval_update(live,_valid(m,contact))
            shadow = interval_update(shadow,_valid(m,contact,True))
            minimum_floor = min(minimum_floor,m['drone_floor_m'],m['package_floor_m'])
            peak_tilt = max(peak_tilt,abs(m['drone_roll_deg']),abs(m['drone_pitch_deg']))
            max_displacement = max(max_displacement,float(np.linalg.norm(np.asarray(s['drone_xyz'])-initial_pose)))
            if previous_pose is not None and native['kind']=='STEP':
                bound = .01*max(np.linalg.norm(s['drone_velocity']),np.linalg.norm(previous_sample['drone_velocity']))+.005
                if np.linalg.norm(np.asarray(s['drone_xyz'])-previous_pose)>bound:
                    raise ValueError('Physical pose discontinuity')
            if native['kind']=='STEP':
                for k in LIMITS:
                    error = abs(m[k]-record['metrics'][k])
                    max_margin_error = max(max_margin_error,error)
                    if error>1e-6:
                        raise ValueError('Saved predicate disagrees with native raw')
                if abs(live[1]*.01-record['dwell_s'])>1e-8 or abs(shadow[1]*.01-record['strict_dwell_s'])>1e-8:
                    raise ValueError('Saved interval count differs')
                if record['verdict']=='COMPLETE':
                    if live[1]<100 or len(completed)!=current_leg:
                        raise ValueError('False shared completion')
                    leg_start = 0 if not stops else stops[-1]['step']
                    if step-leg_start>3000 or last_wall>=600:
                        raise ValueError('Late completion')
                    completed.append(current_leg)
                    stops.append(dict(step=step,pose=s['drone_xyz'],metrics=m))
                    strict_stops.append(dict(leg=current_leg,simultaneous_strict_at_stop=shadow[1]>=100))
            previous_pose = np.asarray(s['drone_xyz'])
            previous_sample = s
    if resets!=1 or steps!=len(control) or steps!=saved['steps']:
        raise ValueError('Incomplete native/control evidence')
    success = (len(completed)==len(case['targets']) and all_safe and
               last_wall<600 and saved['terminal_enqueue_wall_s']<600)
    if saved['success']!=success or saved['completed_targets']!=len(completed):
        raise ValueError('Saved mission result differs from reconstruction')
    return dict(schema='hybrid-independent-control-audit-v1',audit_passed=True,
        physical_success=success,completed_targets=len(completed),steps=steps,stops=stops,
        safe=all_safe,
        minimum_floor_m=minimum_floor,peak_absolute_roll_pitch_deg=peak_tilt,
        max_displacement_m=max_displacement,maximum_metric_residual=max_margin_error,
        strict_shadow= strict_stops,raw_sha256=sha256(root/'runtime/raw.jsonl.gz'),
        control_sha256=sha256(root/'control.jsonl.gz'),checker_sha256=sha256(__file__),
        scope='Raw command, motor, physical-continuity and predicate reconstruction; no robustness or learning claim')
