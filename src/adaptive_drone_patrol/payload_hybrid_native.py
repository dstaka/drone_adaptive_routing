"""Reconstruct measured-controller inputs/actions against every native getter."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np

from .payload_hybrid_controller import Controller, SENSOR_FIELDS
from .payload_hybrid_runtime import SmoothReference
from .payload_solver_trace import match_native_sample
from .payload_solver_qualification import physical_errors
from .payload_solver_validation import verify_initial_sample, measured_predicates
from .util import sha256


def verify_native_kinematics(model,data,actual):
    """Read-only forward kinematics of raw joints; no simulation step or input."""
    import mujoco as mj
    data.qpos[:]=actual['qpos']
    data.qvel[:]=actual['qvel']
    mj.set_mjcb_control(None)
    mj.mj_forward(model,data)
    maximum=0.
    for label,name in [('drone','base_link'),('rod','suspension_rod'),('package','payload')]:
        body=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)
        jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mj.mj_jacBody(model,data,jp,jr,body)
        for field,expected in [('xyz',data.xpos[body]),('velocity',jp@data.qvel),
                               ('angular_velocity',jr@data.qvel)]:
            error=float(np.max(np.abs(expected-np.asarray(actual[label+'_'+field]))))
            maximum=max(maximum,error)
        q=np.asarray(actual[label+'_quaternion_wxyz'])
        maximum=max(maximum,float(min(np.linalg.norm(q-data.xquat[body]),np.linalg.norm(q+data.xquat[body]))))
    if maximum>1e-8:
        raise ValueError('Native body getter differs from joint kinematics')
    return maximum


def verify_hybrid_native(rows, native_path, properties_path, checkpoint, expected_hash,
                         case, binding, control_path, admission):
    admission.require_current()
    if admission.binding!=binding or sha256(checkpoint)!=expected_hash:
        raise ValueError('Frozen controller/reference binding differs')
    design=json.loads(Path(checkpoint).read_text())
    controller=Controller(design['gain'],design['hover_action'])
    properties=json.loads(Path(properties_path).read_text())
    import mujoco as mj
    model_path=Path(properties_path).parent/'fixture/model.xml'
    if sha256(model_path)!=properties['model_xml_sha256']:
        raise ValueError('Native physical model bytes changed')
    model=mj.MjModel.from_xml_path(str(model_path))
    data=mj.MjData(model)
    if physical_errors(properties,dict(side_m=case['side_m'],mass_ratio=case['mass_ratio'],substeps=16),fixed=False):
        raise ValueError('Actual native properties differ')
    with gzip.open(native_path,'rt') as stream:
        native=[json.loads(line) for line in stream]
    with gzip.open(control_path,'rt') as stream:
        controls=[json.loads(line) for line in stream]
    if len(native)!=len(rows) or len(controls)!=sum(r['kind']=='STEP' for r in rows):
        raise ValueError('Incomplete native/controller records')
    previous,reference=None,None
    history=np.zeros(4)
    command_start=0.
    index=0
    shadow=(False,0)
    strict_targets=[]
    current_strict=False
    old_target=None
    longest_strict=0
    max_action_residual=0.
    maximum_kinematic_residual=0.
    for row,record in zip(rows,native):
        expected_kind={'INITIAL':'RESET','RETARGET':'RETARGET','STEP':'STEP'}[row['kind']]
        if record['kind']!=expected_kind or record['episode_id']!=1 or record['step']!=row['step']:
            raise ValueError('Native command/time/reset differs')
        actual=record['actual']
        match_native_sample(row['sample'],actual,properties)
        maximum_kinematic_residual=max(maximum_kinematic_residual,verify_native_kinematics(model,data,actual))
        if expected_kind=='RESET':
            verify_initial_sample(actual,case['initial'])
            if record['step']!=0 or np.any(record['actions']) or np.any(actual['rpm']):
                raise ValueError('Initial state/history differs')
        else:
            if record['target_id']!=row['target_index'] or record['target']!=row['target']:
                raise ValueError('Actual commanded target differs')
            if expected_kind=='RETARGET':
                if actual!=previous or not np.array_equal(record['actions'],history):
                    raise ValueError('Target change reset physical state or action history')
                if reference is not None:
                    strict_targets.append(dict(target=old_target,passed=current_strict,
                        longest_dwell_s=longest_strict*.01))
                reference=SmoothReference(actual['drone_xyz'],record['target'])
                command_start=record['step']*.01
                old_target=record['target_id']
                shadow=(False,0)
                current_strict=False
                longest_strict=0
            else:
                c=controls[index]
                ref,vel=reference.sample((row['step']-1)*.01-command_start)
                sensor={k:copy.deepcopy(previous[k]) for k in SENSOR_FIELDS}
                action=controller.action(sensor,ref,tick=row['step']-1,reference_velocity=vel)
                residual=float(np.max(np.abs(action-record['actions'])))
                max_action_residual=max(max_action_residual,residual)
                if residual>1e-10 or c['step']!=row['step'] or c['target_id']!=record['target_id']:
                    raise ValueError('Frozen measured-controller action differs')
                if not np.allclose(c['state'],controller.last_audit['state'],rtol=0,atol=1e-10):
                    raise ValueError('Runtime input included undeclared information/history')
                if not np.allclose(actual['rpm'],(1+.8*action)*14468.429183500699,rtol=0,atol=1e-8):
                    raise ValueError('Current original motor action differs')
                strict=measured_predicates(row['sample'],record['target'],case['side_m'])['valid']
                shadow=(strict,shadow[1]+1 if strict and shadow[0] else 0)
                longest_strict=max(longest_strict,shadow[1])
                current_strict |= shadow[1]>=100
                history=action.copy()
                index+=1
        if not np.array_equal(row['sample']['actions'],record['actions']):
            raise ValueError('Applied action history differs from native evidence')
        previous=actual
    if reference is not None:
        strict_targets.append(dict(target=old_target,passed=current_strict,longest_dwell_s=longest_strict*.01))
    return dict(native_raw_sha256=sha256(native_path),actual_properties_sha256=sha256(properties_path),
        controller_sha256=expected_hash,checkpoint_sha256=expected_hash,records=len(native),
        controller_outputs_reconstructed=True,maximum_action_residual=max_action_residual,
        maximum_kinematic_residual=maximum_kinematic_residual,
        strict_shadow=strict_targets,scope='Strict shadow uses actual video-first event stream; no counterfactual strict mission claim')
