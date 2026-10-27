"""Independent same-input native/RK replay and oriented clearance at every stage."""
import gzip,json,tempfile
from pathlib import Path
import numpy as np
from .payload_hybrid_duration import ObservedFixture
from .payload_hybrid_duration_audit import _stages,check_native_equal
from .payload_solver_work import pack_stages,total_mechanical_energy
from .payload_solver_work_check import reconstruct_batch_work
from .payload_ball_bank import ball_joint_configuration
from .payload_contract import Payload
from .payload_solver_mujoco import FREE_MASS_KG
from .payload_hybrid_audit import independent_metrics
from .util import sha256


def audit_numeric(trial,*,original):
    root=Path(trial);case=json.loads((root/'case.json').read_text())
    with gzip.open(root/'flight/runtime/raw.jsonl.gz','rt') as stream: rows=[json.loads(line) for line in stream]
    steps=sum(r['kind']=='STEP' for r in rows)
    stages=_stages(root/'flight/stages',steps)
    maximum_native=maximum_stage=maximum_work=maximum_constraint=0.
    motor=loss=bound=0.; minima={}; peak_swing=peak_spin=peak_overshoot=peak_reference_error=0.; stage_count=0
    with gzip.open(root/'flight/control.jsonl.gz','rt') as stream:controls={r['step']:r for r in (json.loads(l) for l in stream)}
    leg_start_z=float(rows[0]['actual']['drone_xyz'][2])
    target=[0.,0.,1.]; leg='S'; previous=None
    def retain_floor(name,value,time_s,where):
        if name not in minima or value<minima[name]['m']:minima[name]=dict(m=float(value),time_s=float(time_s),leg=leg,where=where)
    with tempfile.TemporaryDirectory(prefix='elevated-independent-replay-') as temp:
        f=ObservedFixture(original,Path(temp)/'fixture',Payload(FREE_MASS_KG,case['mass_ratio'],case['side_m']),substeps=16)
        props=f.runtime(); f.reset(position=case['initial']['drone_xyz'],quaternion=case['initial']['drone_quaternion_wxyz'])
        upper,lower=ball_joint_configuration(case['initial'],case['side_m'])
        mj=f.mj
        for name,values in [('suspension_ball',upper),('payload_ball',lower)]:
            j=mj.mj_name2id(f.model,mj.mjtObj.mjOBJ_JOINT,name);start=f.model.jnt_qposadr[j]; f.data.qpos[start:start+4]=values
        mj.mj_forward(f.model,f.data)
        kin=mj.MjData(f.model); drone=mj.mj_name2id(f.model,mj.mjtObj.mjOBJ_BODY,'base_link'); package=mj.mj_name2id(f.model,mj.mjtObj.mjOBJ_BODY,'payload')
        energy0=total_mechanical_energy(rows[0]['actual'],props,case['side_m'])
        for record in rows:
            actual=record['actual']; step=record['step']
            if record['kind']=='RETARGET':
                target=record['target'];leg=record['target_id'];leg_start_z=float(actual['drone_xyz'][2]);continue
            if record['kind']=='RESET':
                maximum_native=max(maximum_native,check_native_equal(f.raw_state(),actual))
            elif record['kind']=='STEP':
                stage,acc=next(stages)
                f.step_rpm(actual['rpm'])
                maximum_native=max(maximum_native,check_native_equal(f.raw_state(),actual))
                expected=pack_stages(f.stage_rows); expected_acc=np.asarray([r['qacc_before_solve'] for r in f.stage_rows])
                delta=max(float(np.max(abs(stage-expected))),float(np.max(abs(acc-expected_acc))))
                if delta>1e-10:raise ValueError('Actual native RK evidence differs from same-input replay')
                maximum_stage=max(maximum_stage,delta)
                work=reconstruct_batch_work(stage[None],substeps=16,first_step=step-1,total_mass_kg=sum(v['mass_kg'] for v in props['links']))
                motor+=float(work['motor_work_j'][0]);loss+=float(work['dissipative_work_j'][0]);bound+=float(work['endpoint_absolute_work_bound_j'][0])
                energy=total_mechanical_energy(actual,props,case['side_m']);residual=energy-energy0-motor-loss
                maximum_work=max(maximum_work,abs(residual))
                if abs(residual)>.02 or energy-energy0>.02+1.1*bound:raise ValueError('Inherited numerical work envelope failed')
                for vector in stage:
                    kin.qpos[:]=vector[1:16];kin.qvel[:]=vector[16:28];kin.time=vector[0]
                    mj.mj_forward(f.model,kin)
                    rd=kin.xmat[drone].reshape(3,3);rp=kin.xmat[package].reshape(3,3)
                    d=kin.xpos[drone];p=kin.xpos[package]
                    df=d[2]-.06*np.sqrt(max(0.,1-rd[2,2]**2))-.0125*abs(rd[2,2])
                    pf=p[2]-case['side_m']/2*np.abs(rp[2]).sum()
                    residual_c=abs(np.linalg.norm(p+rp@np.array([0,0,case['side_m']/2])-d-rd@np.array([0,0,-.02]))-.30)
                    maximum_constraint=max(maximum_constraint,float(residual_c))
                    roll=np.degrees(np.arctan2(rd[2,1],rd[2,2]));pitch=np.degrees(np.arcsin(np.clip(-rd[2,0],-1,1)))
                    retain_floor('package',pf,kin.time,'RK_STAGE');retain_floor('drone',df,kin.time,'RK_STAGE')
                    if min(df,pf)<=.10 or residual_c>2e-5 or kin.ncon or max(abs(roll),abs(pitch))>60:
                        raise ValueError(f'Actual substep safety failure at {kin.time}: {df}, {pf}, {residual_c}')
                    stage_count+=1
            else:raise ValueError('Unexpected native record')
            measured=independent_metrics(actual,target,case['side_m'])
            retain_floor('package',measured['package_floor_m'],actual['time_s'],'NATIVE')
            retain_floor('drone',measured['drone_floor_m'],actual['time_s'],'NATIVE')
            peak_swing=max(peak_swing,measured['swing_angle_deg']);peak_spin=max(peak_spin,measured['package_spin_deg_s'])
            z=actual['drone_xyz'][2]
            peak_overshoot=max(peak_overshoot,z-max(leg_start_z,target[2]),min(leg_start_z,target[2])-z)
            if step in controls:peak_reference_error=max(peak_reference_error,abs(z-controls[step]['reference_position'][2]))
            if actual['contact'] or min(measured['drone_floor_m'],measured['package_floor_m'])<=.10:raise ValueError('Native/reset floor/contact failure')
            previous=actual
    if next(stages,None) is not None or stage_count!=steps*64:raise ValueError('Missing or extra native stages')
    flush=json.loads((root/'final-flush.json').read_text())
    if not flush['flushed'] or flush['elapsed_wall_s']>=600:raise ValueError('Reset-through-final-flush wall deadline failed')
    return dict(schema='elevated-native-stage-independent-audit-v1',passed=True,steps=steps,stage_count=stage_count,
                maximum_native_replay_error=maximum_native,maximum_stage_replay_error=maximum_stage,
                maximum_signed_work_residual_j=maximum_work,maximum_constraint_residual_m=maximum_constraint,
                minimum_clearances=minima,max_swing_deg=peak_swing,max_box_spin_deg_s=peak_spin,
                max_z_overshoot_beyond_leg_endpoint_envelope_m=peak_overshoot,
                max_z_error_from_held_controller_reference_m=peak_reference_error,
                auditor_sha256=sha256(__file__),
                raw_sha256=sha256(root/'flight/runtime/raw.jsonl.gz'),stage_index_sha256=sha256(root/'flight/stages/stages-index.json'),
                flush_under600=flush['flushed'] and flush['elapsed_wall_s']<600)
