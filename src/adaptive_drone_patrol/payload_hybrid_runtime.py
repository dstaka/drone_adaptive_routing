"""Revision 2 diagnostics on an explicitly rebound reference, without learning."""

import copy
import gzip
import json
from pathlib import Path
import time

import numpy as np

from .payload_contract import SharedSettling, Payload
from .payload_hybrid_contract import VIDEO_PROFILE, STRICT_PROFILE, MissionClock
from .payload_hybrid_controller import Controller, SENSOR_FIELDS
from .payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG
from .payload_solver_runtime import SolverRuntime
from .payload_solver_qualification import physical_errors, current_solver_binding
from .jobs import write_json
from .util import sha256


class SmoothReference:
    """Quintic position path, max speed1.2, acceleration.8, jerk2 in SI."""

    def __init__(self, start, target):
        self.start, self.target = np.asarray(start,float), np.asarray(target,float)
        self.delta = self.target-self.start
        length = np.linalg.norm(self.delta)
        self.duration = max(3.,1.875*length/1.2, np.sqrt(5.774*length/.8), (60*length/2)**(1/3))

    def sample(self, elapsed):
        u = np.clip(elapsed/self.duration,0,1)
        s = 10*u**3-15*u**4+6*u**5
        v = (30*u**2-60*u**3+30*u**4)/self.duration
        return self.start+s*self.delta, v*self.delta


def diagnostic_summary(times, angular_world, actions):
    t, y, actions = np.asarray(times), np.asarray(angular_world), np.asarray(actions)
    result = dict(dominant_hz=None, envelope_log_decay_per_s=None,
                  phase_delay='UNIDENTIFIABLE_WITHOUT_EXCITATION',
                  motor_saturation_s=float(np.sum(np.any(np.abs(actions)>=1-1e-12,axis=1))*.01))
    if len(t)<200 or np.max(np.abs(y))<1e-6:
        return result
    signal = y[:,np.argmax(np.std(y,axis=0))]
    frequencies = np.fft.rfftfreq(len(signal),.01)
    spectrum = np.abs(np.fft.rfft((signal-signal.mean())*np.hanning(len(signal))))
    spectrum[frequencies<.1] = 0
    result['dominant_hz'] = float(frequencies[np.argmax(spectrum)])
    windows = [(float(t[i:i+100].mean()),float(np.sqrt(np.mean(signal[i:i+100]**2))))
               for i in range(0,len(t)-99,100)]
    fitted = [(a,b) for a,b in windows if b>1e-6]
    result['amplitude_envelope_rms_rad_s'] = windows
    if len(fitted)>=4:
        result['envelope_log_decay_per_s'] = float(np.polyfit(
            [p[0] for p in fitted],np.log([p[1] for p in fitted]),1)[0])
    return result


def run_case(*, original, output, case, design, rebound_binding, vibration_feedback=True,
             substeps=16, clock=time.monotonic):
    """Complete predeclared C0/C1/C2 command stream. Never admits learning.

    Caller must have independently reconstructed historical reference raw and
    rebound identical actual binding. A refined timestep is a labeled diagnostic,
    not inherited full qualification for the modified runtime.
    """
    if current_solver_binding() != rebound_binding:
        raise ValueError('Actual reference source/build differs from rebound qualification')
    root = Path(output)
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'case.json',case)
    write_json(root/'design.json',design)
    metadata = dict(schema='payload-hybrid-control-diagnostic-v1',
        profile=VIDEO_PROFILE.id,profile_sha256=VIDEO_PROFILE.sha256,
        backend='mujoco-cpu-float64-RK4',substeps=substeps,
        reference_binding=rebound_binding,vibration_feedback=vibration_feedback,
        scope='DEVELOPMENT_NOT_ACCEPTANCE',learning_admitted=False,
        source_sha256={p:sha256(Path(__file__).parent/p) for p in (
            'payload_hybrid_runtime.py','payload_hybrid_controller.py','payload_hybrid_contract.py')})
    write_json(root/'metadata.json',metadata)
    f = MuJoCoFixture(original,root/'fixture',Payload(FREE_MASS_KG,case['mass_ratio'],case['side_m']),substeps=substeps)
    if physical_errors(f.runtime(),dict(side_m=case['side_m'],mass_ratio=case['mass_ratio'],substeps=substeps),fixed=False):
        raise ValueError('Actual case model/properties differ')
    write_json(root/'runtime-properties.json',f.runtime())
    core = SolverRuntime(f,root/'runtime')
    controller = Controller(design['gain'],design['hover_action'],vibration_feedback=vibration_feedback)
    started = clock()  # immediately before physical reset, after compilation/setup
    mission_clock = MissionClock(reset_wall_s=started)
    outcome = dict(case_id=case['case_id'],success=False,safe=True,completed_targets=0,
        max_displacement_m=0.,stops=[],failure=None,strict_shadow=[],longest_joint_valid_s=0.)
    times, spins, actions = [],[],[]
    try:
        core.reset_case(initial=case['initial'],target=case['targets'][0],target_id='0')
        initial_pose = np.array(core.current_sample['drone_xyz'])
        with gzip.open(root/'control.jsonl.gz','wt') as log:
            for leg,target in enumerate(case['targets']):
                sim_start = core.step_count*.01
                mission_clock.command(sim_start,str(leg),target)
                if leg:
                    core.set_target(target,target_id=str(leg))
                reference = SmoothReference(core.current_sample['drone_xyz'],target)
                live, shadow = SharedSettling(), SharedSettling()
                from .payload_contract import payload_metrics
                initial_metrics = payload_metrics(core.current_sample,target,core.payload)
                initial_metrics['drone_floor_m'] = core.current_sample['vehicle_collision_aabbs']['vehicle_collision'][0][2]
                live.update(sim_start,VIDEO_PROFILE.valid(initial_metrics,contact=core.current_sample['contacts']),str(leg))
                shadow.update(sim_start,STRICT_PROFILE.valid(initial_metrics,contact=core.current_sample['contacts']),str(leg))
                longest_shadow, shadow_complete = 0.,False
                for _ in range(3000):
                    ref, vel = reference.sample(core.step_count*.01-sim_start)
                    sensors = {k:core.current_sample[k] for k in SENSOR_FIELDS}
                    action = controller.action(sensors,ref,tick=core.step_count,reference_velocity=vel)
                    row = core.step(action)
                    complete = live.update(row['time_s'],VIDEO_PROFILE.valid(row['metrics'],contact=row['contacts']),str(leg))
                    strict = shadow.update(row['time_s'],STRICT_PROFILE.valid(row['metrics'],contact=row['contacts']),str(leg))
                    shadow_complete |= strict
                    longest_shadow = max(longest_shadow,shadow.dwell_s)
                    wall = clock()
                    verdict = mission_clock.check(row['time_s'],wall,complete=complete,unsafe=row['unsafe'])
                    record = dict(step=core.step_count,time_s=row['time_s'],leg=leg,target=target,
                        reference_position=ref.tolist(),reference_velocity=vel.tolist(),
                        control=copy.deepcopy(controller.last_audit),metrics=row['metrics'],
                        dwell_s=live.dwell_s,strict_dwell_s=shadow.dwell_s,verdict=verdict,
                        reset_elapsed_wall_s=wall-started)
                    log.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                    times.append(row['time_s'])
                    spins.append(row['package_angular_velocity'])
                    actions.append(action)
                    outcome['longest_joint_valid_s'] = max(outcome['longest_joint_valid_s'],live.dwell_s)
                    outcome['max_displacement_m'] = max(outcome['max_displacement_m'],float(np.linalg.norm(np.asarray(row['drone_xyz'])-initial_pose)))
                    if verdict=='COMPLETE':
                        outcome['completed_targets'] += 1
                        outcome['stops'].append(dict(leg=leg,time_s=row['time_s'],pose=row['drone_xyz'],metrics=row['metrics']))
                        break
                    if verdict!='CONTINUE':
                        outcome['failure'] = verdict
                        outcome['safe'] = verdict!='SAFETY_FAILURE'
                        break
                outcome['strict_shadow'].append(dict(leg=leg,passed=shadow_complete,longest_dwell_s=longest_shadow))
                if outcome['failure']:
                    break
            outcome['success'] = outcome['completed_targets']==len(case['targets']) and not outcome['failure']
            outcome['terminal_enqueue_wall_s'] = clock()-started
            if outcome['terminal_enqueue_wall_s']>=600:
                outcome.update(success=False,failure='WALL_TIMEOUT')
            log.write(json.dumps(dict(kind='TERMINAL',outcome=outcome),separators=(',',':'))+'\n')
    except BaseException as exc:
        outcome.update(success=False,safe=False,failure=repr(exc))
        raise
    finally:
        core.close()
        outcome.update(steps=core.step_count,simulated_s=core.step_count*.01,
            final_flush_wall_s=clock()-started,
            diagnosis=diagnostic_summary(times,spins,actions) if times else {},
            raw_sha256=sha256(root/'runtime/raw.jsonl.gz'))
        write_json(root/'result.json',outcome)
    return outcome
