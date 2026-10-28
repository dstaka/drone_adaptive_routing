"""Explicit native C++ audit adapter; use requires separate physical qualification."""
from copy import deepcopy
import math

from .dual_coordination_audit import OnlineNativeAudit
from .dual_coordination_native_audit import CompiledAuditShadow
from .util import sha256


def compiled_report(audit,metrics,*,self_pairs,library_hash):
    steps=audit['control_steps']
    coverage=(audit['rk_points']==64*steps and audit['native_intervals']==16*steps
              and audit['body_intervals']==16*steps and audit['point_evaluations']==1+80*steps)
    reconstructed=(steps>0 and coverage and audit['certified_intervals']==16*steps
                   and not metrics['reconstruction_failures'])
    safe=bool(coverage and audit['safe'] and not metrics['violations'] and not metrics['reconstruction_failures'])
    return dict(safe=safe,retained_points_safe=audit['safety_flags']==0,
        continuous_safety_qualified=safe and reconstructed,full_physical_acceptance_qualified=False,
        qualification_scope='retained-points-and-cross-self-floor-wash-attitude-constraint-reconstruction',
        proof_gaps=[] if coverage else ['Incomplete compiled internal-stage/native interval coverage'],
        limitations=['qualification is for saved native states and declared linear-centers-shortest-rotation reconstruction; unsaved true continuous physics is not proved'],
        errors=[],control_steps=steps,rk_states=audit['rk_points'],native_poses=1+audit['native_intervals'],
        reconstructed_intervals=audit['native_intervals'],body_reconstructed_intervals=audit['body_intervals'],
        reconstructed_path_safe=reconstructed,reconstruction_failures=deepcopy(metrics['reconstruction_failures']),
        audited_points=audit['point_evaluations'],violation_count=len(metrics['violations']),
        violations=deepcopy(metrics['violations']),minimum_clearance_m=audit['minimum_clearance_m'],
        floor_minima_m=metrics['floor_minima_m'],max_swing_deg=metrics['max_swing_deg'],
        max_spin_deg_s=metrics['max_spin_deg_s'],max_roll_pitch_deg=metrics['max_roll_pitch_deg'],
        reconstructed_clearance_lower_bound_m=metrics['sweep_lower'],
        reconstructed_self_clearance_lower_bound_m=metrics['self_lower'],
        reconstructed_constraint_upper_bound_m=metrics['constraint_upper'],
        reconstructed_tilt_upper_bound_deg=metrics['tilt_upper'],self_collision_pairs=[list(pair) for pair in self_pairs],
        python_fallback_intervals=audit['python_fallback_intervals'],permanent_raw_required=False,
        reconstruction_contract='linear-centers-shortest-rotation-v1',evidence_mode='compiled-native-with-python-near-interval-fallback',
        compiled_library_sha256=library_hash)


class CompiledOnlineNativeAudit(OnlineNativeAudit):
    """Same native integrator/commands; a separate clone audits then publishes state.

    The file-bound experiment capability must qualify this adapter before its use
    for production. Detailed stage-file export remains the Python path's job.
    """
    def __init__(self,fixture,library):
        self.fixture=fixture
        self.library_hash=sha256(library)
        self.shadow=CompiledAuditShadow.from_fixture(fixture,library)
        point=self.shadow.inspect()
        self.metrics=dict(floor_minima_m=list(point['floor_minima_m']),max_swing_deg=point['max_swing_deg'],
            max_spin_deg_s=point['max_spin_deg_s'],max_roll_pitch_deg=point['max_roll_pitch_deg'],
            constraint_upper=point['constraint_max_m'],tilt_upper=point['max_roll_pitch_deg'],
            sweep_lower=None,self_lower=None,violations=[],reconstruction_failures=[])
        if point['flags']:
            self.metrics['violations'].append(dict(location='reset',native_flags=point['flags'],point=point))
        self.audit=deepcopy(self.shadow.snapshot()['audit'])

    def advance_compiled(self,fixture,rpm):
        if fixture is not self.fixture:
            raise ValueError('One exact native fixture required')
        result=self.shadow.step_audited(rpm,fast_path=True)
        self.shadow.copy_to_fixture(fixture)
        self.audit=deepcopy(result['cumulative_audit'])
        for index,point in enumerate(result['rk_points']+result['boundaries']):
            self.metrics['floor_minima_m']=[min(a,b) for a,b in zip(self.metrics['floor_minima_m'],point['floor_minima_m'])]
            for key in ('max_swing_deg','max_spin_deg_s','max_roll_pitch_deg'):
                self.metrics[key]=max(self.metrics[key],point[key])
            if point['flags']:
                self.metrics['violations'].append(dict(location=f'control:{self.audit["control_steps"]}/point:{index}',
                                                      native_flags=point['flags'],point=deepcopy(point)))
        for index,interval in enumerate(result['interval_audit']['reports']):
            sweep,body=interval['sweep'],interval['body']
            for key,field in (('sweep_lower','lower_clearance_m'),('self_lower','self_lower_clearance_m')):
                value=sweep.get(field)
                if value is not None:
                    current=self.metrics[key]
                    self.metrics[key]=value if current is None else min(current,value)
            self.metrics['constraint_upper']=max(self.metrics['constraint_upper'],body['constraint_upper_bound_m'])
            if body.get('tilt_upper_bound_deg') is not None:
                self.metrics['tilt_upper']=max(self.metrics['tilt_upper'],body['tilt_upper_bound_deg'])
            if sweep['status']!='certified' or body['status']!='certified':
                self.metrics['reconstruction_failures'].append(dict(control_step=self.audit['control_steps'],interval=index,**deepcopy(interval)))
        return self.report()

    def report(self):
        return compiled_report(self.audit,self.metrics,self_pairs=self.shadow.self_pairs,library_hash=self.library_hash)

    def to_dict(self):
        return dict(schema='dual-compiled-online-audit-v1',library_hash=self.library_hash,
                    shadow=self.shadow.snapshot(),metrics=deepcopy(self.metrics))

    def restore(self,state):
        if state.get('schema')!='dual-compiled-online-audit-v1' or state.get('library_hash')!=self.library_hash:
            raise ValueError('Matching complete compiled audit checkpoint required')
        metrics=deepcopy(state['metrics'])
        for name in ('floor_minima_m','max_swing_deg','max_spin_deg_s','max_roll_pitch_deg','constraint_upper','tilt_upper'):
            values=metrics[name] if isinstance(metrics[name],list) else [metrics[name]]
            if not values or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
                raise ValueError('Finite complete compiled audit metrics required')
        for name in ('violations','reconstruction_failures'):
            if not isinstance(metrics[name],list):
                raise ValueError('Complete failure history required')
        self.shadow.restore(state['shadow'])
        import numpy as np
        mj=self.fixture.mj
        signature=int(mj.mjtState.mjSTATE_INTEGRATION)
        restored=np.empty(mj.mj_stateSize(self.fixture.model,signature))
        mj.mj_getState(self.fixture.model,self.fixture.data,restored,signature)
        if not np.array_equal(restored,state['shadow']['state']):
            raise ValueError('Compiled audit/native environment checkpoint state differs')
        self.metrics=metrics
        self.audit=deepcopy(state['shadow']['audit'])

    def close(self):
        self.shadow.close()
