"""Native two-drone DEVELOPMENT environment; no production qualification token.

One step commits one 5 Hz joint action and advances at most twenty 100 Hz
control ticks in one shared native world. Sensor observations have the named
11/66/11 policy layout and fixed scales below. Hidden case events, sampled mass,
and box dimensions are never adapter inputs. Actual measured payload poses may
naturally depend on physical properties; that is sensor information, not a
latent parameter feature.

Native solver state uses mjSTATE_INTEGRATION (including solver warmstart), as
specified at https://mujoco.readthedocs.io/en/latest/programming/simulation.html .
Compact exact input/output replay is always captured; detailed internal raw is
optional. Compact mode audits live native state without serializing internal
stage/substep arrays; detailed mode independently replays transient rows.

Checkpointing closes the current capture segment; continuation starts another RESET
segment bound to its predecessor hashes. A segment RESET is a resume marker,
not a simulated reset. Complete raw segment chains still require an independent
joint audit. This adapter and its supervisor are development-only and cannot
admit production learning, validation, or held-out evaluation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .dual_coordination_mission import MissionState
from .dual_coordination_audit import OnlineNativeAudit
from .dual_coordination_policy import GOAL_FEATURES, DRONE_FEATURES, GLOBAL_FEATURES
from .dual_coordination_progress import atomic_json
from .dual_coordination_raw import RawWriter
from .dual_coordination_reference import ContinuousReference, KinematicState, ReferenceInfeasible
from .dual_coordination_safety import ReferenceSupervisor, SafetyLedger, audit_geometry
from .dual_coordination_schema import WAIT, decode_action, validate_case
from .dual_payload_mujoco import DualMuJoCoFixture
from .payload_contract import Payload, payload_metrics
from .payload_hybrid_controller import Controller, measured_state
from .payload_solver_mujoco import FREE_MASS_KG, HOVER_RPM
from .util import sha256

OBSERVATION_SCALES = dict(position_m=5.,velocity_m_s=3.,angular_velocity_rad_s=math.pi,
                          link_angle_rad=math.pi,link_rate_rad_s=math.pi,acceleration_m_s2=3.,
                          commitment_ticks=100.,leg_ticks=3000.,dwell_ticks=100.,team_ticks=30000.,
                          priority=100.,relative_position_m=10.,relative_velocity_m_s=6.,clearance_m=10.)


def _array(value, shape):
    result = np.asarray(value,dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Finite measured array with declared shape required')
    return result


def build_observation(sample, mission, goals, references, prediction):
    """Return unbatched float32 policy arrays, using only causal measured state.

    Angles/rates are the rod's rotation relative to the drone and the measured
    rod-minus-drone angular velocity, resolved in the rod frame, as in the
    retained controller. No inferred mass, sampled size, or future event enters.
    Terminal observations may occur between policy boundaries and are never an
    invitation for another decision. No clipping conceals out-of-scale state.
    """
    if mission.tick % 20 and mission.terminal_reason is None:
        raise ValueError('Policy observations require a 5 Hz boundary')
    if len(sample['drones']) != 2 or len(references) != 2:
        raise ValueError('Two measured drones and references required')
    if [row.get('drone_id') for row in sample['drones']] != [1,2]:
        raise ValueError('Ordered distinct public drone IDs 1/2 required')
    if not math.isclose(float(sample['world_time']),mission.tick*.01,abs_tol=1e-7):
        raise ValueError('Physical and mission clocks disagree')
    coordinates = _array(goals,(10,3))
    goal_rows = []
    statuses = ('UNASSIGNED','ASSIGNED','SERVICING','DONE')
    for goal,p,priority in zip(mission.goals,coordinates,mission.priorities):
        goal_rows.append([*(p/5),priority/100,*[float(goal.status == s) for s in statuses],
                          *[float(goal.owner == owner) for owner in (None,0,1)]])
    last_targets = (WAIT,WAIT) if mission.last_action is None else decode_action(mission.last_action)[:2]
    drone_rows = []
    for drone,(row,reference) in enumerate(zip(sample['drones'],references)):
        state = measured_state(row,(0,0,0))
        p,v,a = reference.sample(mission.tick*.01)
        vector = []
        for key,shape,scale in (
            ('drone_xyz',(3,),5.),('drone_velocity',(3,),3.),
            ('drone_quaternion_wxyz',(4,),1.),('drone_angular_velocity',(3,),math.pi),
            ('package_xyz',(3,),5.),('package_quaternion_wxyz',(4,),1.),
            ('package_velocity',(3,),3.),('package_angular_velocity',(3,),math.pi),
        ):
            vector.extend(_array(row[key],shape)/scale)
        vector.extend(state[6:8]/math.pi)
        vector.extend(state[16:18]/math.pi)
        vector.extend(p/5)
        vector.extend(v/3)
        vector.extend(a/3)
        vector.extend(float(mission.targets[drone] == target) for target in range(11))
        vector.extend(float(last_targets[drone] == target) for target in range(11))
        commit,leg = mission.commitment_start_ticks[drone],mission.leg_start_ticks[drone]
        target = mission.targets[drone]
        dwell = 0 if target == WAIT else mission.goals[target].dwell_ticks
        vector.extend((0 if commit is None else (mission.tick-commit)/100,
                       0 if leg is None else (mission.tick-leg)/3000,dwell/100,
                       float(mission.available[drone]),float(mission.safety_held[drone])))
        drone_rows.append(vector)
    delta_p = np.subtract(sample['drones'][1]['drone_xyz'],sample['drones'][0]['drone_xyz'])
    delta_v = np.subtract(sample['drones'][1]['drone_velocity'],sample['drones'][0]['drone_velocity'])
    global_row = [mission.tick/30000,float(mission.priority == 1),float(mission.priority == 2),
                  *(delta_p/10),*(delta_v/6),float(prediction['clearance_m'])/10,
                  float(prediction['conflict'])]
    result = dict(goals=np.asarray(goal_rows,dtype=np.float32),drones=np.asarray(drone_rows,dtype=np.float32),
                  global_features=np.asarray(global_row,dtype=np.float32))
    for key,shape in (('goals',(10,len(GOAL_FEATURES))),('drones',(2,len(DRONE_FEATURES))),
                      ('global_features',(len(GLOBAL_FEATURES),))):
        _array(result[key],shape)
    return result


def compact_replay_record(record):
    """Retain exact selected controls/states/references/events and audit summaries.

    Internal solver/stage arrays and redundant force diagnostics are transient.
    This is replay evidence, not a claim of independent offline stage auditing.
    """
    result = {key:deepcopy(value) for key,value in record.items()
              if key not in ('stages','substeps','geometry','service_metrics','final_metrics','actual','audit')}
    result['actual'] = {key:deepcopy(value) for key,value in record['actual'].items()
                        if key in ('schema','world_time','drones','qpos','qvel','rpm','ctrl','warning_counts','contacts')}
    if 'audit' in record:
        result['audit'] = {key:deepcopy(value) for key,value in record['audit'].items() if key != 'metrics'}
    result['evidence_mode'] = 'COMPACT_EXACT_REPLAY_WITH_ONLINE_AUDIT'
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _settled(metrics):
    limits = dict(drone_error_m=.20,package_error_m=.20,drone_speed_m_s=.30,package_speed_m_s=.15,
                  swing_angle_deg=5.,swing_rate_deg_s=15.,package_spin_deg_s=30.)
    return all(math.isfinite(metrics[key]) and 0 <= metrics[key] <= value for key,value in limits.items())


def step_native_with_audit(fixture, rpm, auditor, *, detailed=False, location='control'):
    """Advance exactly one native 100 Hz interval and audit all internal states.

    This low-level helper has no mission clock, so qualification drivers can
    exercise longer runs without bypassing a mission's 300 s deadline. Each
    auditor instance uses either live or detailed replay mode throughout its
    lifetime. MuJoCo's process-global callback requires isolated worker processes.
    Returned qualification is limited to the auditor's explicit scope.
    """
    if type(detailed) is not bool or not isinstance(auditor,OnlineNativeAudit):
        raise ValueError('An OnlineNativeAudit and boolean detailed mode are required')
    from .dual_coordination_compiled_audit import CompiledOnlineNativeAudit
    if type(auditor) is CompiledOnlineNativeAudit:
        if detailed:
            raise ValueError('Compiled adapter cannot claim detailed stage-file export')
        return auditor.advance_compiled(fixture,rpm)
    if detailed:
        fixture.step_rpm(rpm,observe_stages=True)
        return auditor.observe_fixture_step(fixture)
    fixture.set_rpm(rpm)
    fixture.stage_rows,fixture.substep_rows = [],[]
    mj,model,data = fixture.mj,fixture.model,fixture.data
    auditor.observe_native_state(model,data,phase='native',location=f'{location}/start')
    stage_index = 0

    def observe_stage(model, data):
        nonlocal stage_index
        auditor.observe_native_state(model,data,phase='rk',location=f'{location}/rk:{stage_index}')
        stage_index += 1

    mj.set_mjcb_control(observe_stage)
    try:
        for index in range(fixture.substeps):
            mj.mj_step(model,data)
            mj.set_mjcb_control(None)
            mj.mj_forward(model,data)
            auditor.observe_native_state(model,data,phase='native',location=f'{location}/substep:{index+1}')
            mj.set_mjcb_control(observe_stage)
    finally:
        mj.set_mjcb_control(None)
    mj.mj_forward(model,data)
    return auditor.report()


def _admit_environment(case, design, original, mode, admission, retain_replay, online_auditor,native_audit_library=None):
    if type(retain_replay) is not bool:
        raise ValueError('Explicit replay retention boolean required')
    if mode == 'development':
        if case['split'] != 15 or admission is not None:
            raise ValueError('Only declared development cases are admitted without a verified capability')
        if not retain_replay:
            raise ValueError('Development replay cannot be disabled')
        return None
    from .dual_coordination_campaign import VerifiedLearningLock, VerifiedHeldout
    learning = mode in ('training','validation') and type(admission) is VerifiedLearningLock
    heldout = mode in ('routing-heldout','physical-heldout') and type(admission) is VerifiedHeldout
    if not (learning or heldout):
        raise ValueError('A file-verified capability for the requested learning/held-out mode is required')
    admission.require_current()
    admission.verify_case(case,mode)
    if admission.selected_backend not in ('native-cpu','native-cuda'):
        raise ValueError('The native environment cannot substitute for a qualified GPU physics backend')
    if online_auditor is not None:
        raise ValueError('Production admission requires the frozen built-in native auditor')
    qualification=getattr(admission,'freeze',admission).receipt['qualification']
    compiled=qualification.get('audit_backend','python-native')=='compiled-native'
    if compiled != (native_audit_library is not None):
        raise ValueError('Native audit implementation must match the qualified frozen backend')
    if native_audit_library is not None:
        library=Path(native_audit_library).resolve()
        if qualification.get('audit_library_sha256')!=sha256(library):
            raise ValueError('Compiled library differs from the qualified artifact')
        if not any(row['bytes']==library.stat().st_size and row['sha256']==sha256(library)
                   for row in admission.binding.files['audit']):
            raise ValueError('Compiled audit library must be file-bound to qualification')
    if not retain_replay and mode != 'training':
        raise ValueError('Validation replay is required')
    root,files=admission.binding.root,admission.binding.files
    source_by_name={Path(row['path']).name:row['sha256'] for row in files['source']}
    for name in ('dual_coordination_env.py','dual_coordination_mission.py','dual_coordination_cases.py',
                 'dual_coordination_schema.py','dual_coordination_audit.py','dual_coordination_reference.py',
                 'dual_coordination_safety.py','dual_payload_mujoco.py','payload_hybrid_controller.py',
                 'payload_contract.py','payload_solver_mujoco.py','payload_ball_asset.py'):
        if source_by_name.get(name) != sha256(Path(__file__).with_name(name)):
            raise ValueError('Runtime physical/mission source differs from verified learning inputs: '+name)
    controllers=files['controller']
    if len(controllers)!=1 or json.loads((root/controllers[0]['path']).read_text())!=design:
        raise ValueError('Controller differs from verified learning inputs')
    import xml.etree.ElementTree as ET
    original=Path(original).resolve()
    required=[original]+[(original.parent/mesh.attrib['filename']).resolve()
                         for mesh in ET.parse(original).findall('.//mesh') if 'filename' in mesh.attrib]
    bound_assets={(Path(row['path']).name,row['sha256']) for row in files['assets']}
    if any((path.name,sha256(path)) not in bound_assets for path in required):
        raise ValueError('Physical assets differ from verified learning inputs')
    return admission.digest


def qualified_audit_library(admission):
    """Resolve the exact independently qualified library, never an environment variable."""
    from .dual_coordination_campaign import VerifiedLearningLock,VerifiedHeldout
    if type(admission) not in (VerifiedLearningLock,VerifiedHeldout):
        raise ValueError('File-verified runtime admission required')
    qualification=getattr(admission,'freeze',admission).receipt['qualification']
    backend=qualification.get('audit_backend','python-native')
    if backend=='python-native':
        return None
    if backend!='compiled-native':
        raise ValueError('Unknown qualified native audit implementation')
    rows=[r for r in admission.binding.files['audit'] if r['sha256']==qualification.get('audit_library_sha256')]
    if len(rows)!=1:
        raise ValueError('Exactly one bound qualified compiled library required')
    return admission.binding.root/rows[0]['path']


class _StepTimings:
    """Disjoint diagnostic wall phases; never reads or adjusts the mission clock.

    Exception exits close the active phase too. Disabled collection makes no
    diagnostic clock calls. This transient diagnostic is not simulation state.
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.result = None
        self.phase = 'controller_reference_supervisor'
        self.seconds = dict(controller_reference_supervisor=0.,physics_audit=0.,
                            mask_observation=0.,record_io=0.)

    def __enter__(self):
        if self.enabled:
            self.started = self.previous = time.perf_counter()
        return self

    def switch(self, phase):
        if self.enabled:
            now = time.perf_counter()
            self.seconds[self.phase] += max(0.,now-self.previous)
            self.previous = now
            self.phase = phase

    def __exit__(self, *_):
        if self.enabled:
            self.switch(self.phase)
            # Floating summation can exceed the telescoping interval by one ulp.
            total = max(0.,self.previous-self.started,sum(self.seconds.values()))
            self.result = dict(self.seconds,total_step=total)


class DualCoordinationEnv:
    """A single new development mission; construct a new instance to reset.

    ``step`` returns (observation, reward, terminated, info). All physical/deadline
    and infrastructure failures are terminal without bootstrap. Compact exact
    replay is retained for development/validation; training may retain complete
    recovery state instead. File-verified qualification/benchmark/input capabilities
    admit split10/20. Exact post-freeze held-out capabilities admit split30/40.
    """

    def __init__(self, original, output, case, design, record_raw=False, *, mode='development', online_auditor=None, admission=None, retain_replay=True, replay_stride_ticks=1,native_audit_library=None,no_transfer=False,collect_timings=False):
        validate_case(case)
        if type(collect_timings) is not bool:
            raise ValueError('collect_timings must be a boolean')
        self.collect_timings = collect_timings
        self.last_step_timings = None
        if type(no_transfer) is not bool or (no_transfer and mode not in ('development','validation','routing-heldout')):
            raise ValueError('No-transfer ablation is an explicit development/validation/routing-heldout boolean only')
        if mode!='development' and native_audit_library is None:
            native_audit_library=qualified_audit_library(admission)
        if native_audit_library is not None and (record_raw or online_auditor is not None):
            raise ValueError('Compiled built-in audit requires compact mode without a custom auditor')
        if type(replay_stride_ticks) is not int or replay_stride_ticks not in (1,20):
            raise ValueError('Replay stride must be exactly 1 or 20 control ticks')
        if record_raw and replay_stride_ticks != 1:
            raise ValueError('detailed raw requires full-rate replay')
        self.replay_stride_ticks = replay_stride_ticks
        self.admission_digest = _admit_environment(case,design,original,mode,admission,retain_replay,online_auditor,native_audit_library)
        self.native_audit_library=Path(native_audit_library).resolve() if native_audit_library is not None else None
        self.no_transfer=no_transfer
        self.admission = admission
        self.retain_replay = retain_replay
        self.scope = {'development':'DEVELOPMENT_ONLY','training':'FINITE_TRAINING',
                      'validation':'PROSPECTIVE_VALIDATION','routing-heldout':'FROZEN_ROUTING_HELDOUT',
                      'physical-heldout':'FROZEN_PHYSICAL_HELDOUT'}[mode]
        self.production_admitted = mode != 'development'
        if online_auditor is not None and not callable(online_auditor):
            raise ValueError('Online auditor must be a callable, never a qualification boolean')
        self.online_auditor = online_auditor
        self._default_online_auditor = online_auditor is None
        self.online_audit_summary = dict(control_ticks=0,internal_stages=0,internal_substeps=0,
                                         full_physical_acceptance_qualified=False,failures=[])
        if type(record_raw) is not bool:
            raise ValueError('record_raw must be a boolean')
        self.case = deepcopy(case)
        self.design = deepcopy(design)
        self.controllers = [Controller(design['gain'],design['hover_action']) for _ in range(2)]
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=False)
        self.record_raw = record_raw
        self.raw_segments = []
        self.replay_segments = []
        self._raw = None
        self._replay = None
        self._raw_start_tick = None
        self._raw_sequence = 0
        self.closed = False
        self.mode = mode
        compile_start = time.monotonic()
        self.fixture = DualMuJoCoFixture(original,self.output/'fixture',
            [Payload(FREE_MASS_KG,load['mass_ratio'],load['side_m']) for load in case['loads']])
        self.compilation_wall_seconds = time.monotonic()-compile_start
        self._clock_start = time.monotonic()
        self._elapsed_offset = 0.
        self.mission_started_at = datetime.now(timezone.utc).isoformat()
        self.fixture.reset(case['initials'])
        if self._default_online_auditor:
            self.online_auditor = self._make_auditor()
        self.mission = MissionState(case['priorities'],no_transfer=no_transfer)
        self.supervisor = ReferenceSupervisor()
        self.ledger = SafetyLedger()
        self._event_applied = False
        self._reference_goals = [None,None]
        samples = self.fixture.sample()['drones']
        self.desired_references = [ContinuousReference(row['drone_xyz']) for row in samples]
        self.executed_references = list(self.desired_references)
        self.last_held = (False,False)
        self.last_supervisor_reason = 'clear'
        self.last_prediction = self._prediction(self.desired_references,samples)
        self._last_action_mask = self.mission.legal_mask().tolist()
        self.rng = np.random.default_rng(np.random.SeedSequence([10272701,case['split'],case['index'],991]))
        self.binding = dict(admission_sha256=self.admission_digest, mode=mode,no_transfer=no_transfer,
                            collect_timings=collect_timings,
                            audit_library_sha256=sha256(self.native_audit_library) if self.native_audit_library else None,
                            case_sha256=_digest(case),controller_sha256=_digest(design),
                            model_sha256=sha256(self.fixture.model_path),
                            profile_sha256=_digest(dict(scales=OBSERVATION_SCALES,
                                source_sha256={name:sha256(Path(__file__).with_name(name)) for name in (
                                    'dual_coordination_env.py','dual_coordination_mission.py',
                                    'dual_coordination_audit.py','dual_coordination_reference.py',
                                    'dual_coordination_safety.py','dual_payload_mujoco.py',
                                    'payload_hybrid_controller.py')},profile='dual-payload-v1')))
        self.ledger.observe(audit_geometry(self.fixture))
        if not self.ledger.safe:
            self._terminate('safety:initial_state')
        self._last_finite_observation = build_observation(self.fixture.sample(),self.mission,case['goals'],
                                                         self.executed_references,self.last_prediction)
        atomic_json(self.output/'environment.json',dict(schema='dual-development-environment-v1',
            scope=self.scope,production_admitted=self.production_admitted,mode=mode,no_transfer=no_transfer,
            collect_timings=collect_timings,
            audit_backend='compiled-native' if self.native_audit_library else 'python-native',
            retain_replay=retain_replay,replay_stride_ticks=replay_stride_ticks,case=case,design=design,
            binding=self.binding,observation_scales=OBSERVATION_SCALES,
            compilation_wall_seconds=self.compilation_wall_seconds,record_raw=record_raw,
            runtime=self.fixture.runtime()))

    def _make_auditor(self):
        if self.native_audit_library is not None:
            from .dual_coordination_compiled_audit import CompiledOnlineNativeAudit
            return CompiledOnlineNativeAudit(self.fixture,self.native_audit_library)
        return OnlineNativeAudit(self.fixture)

    def _elapsed(self):
        return self._elapsed_offset+time.monotonic()-self._clock_start

    @property
    def terminated(self):
        return self.mission.terminal_reason is not None

    def legal_mask(self):
        return self.mission.legal_mask()

    def observation(self):
        sample = self.fixture.sample()
        self.last_prediction = self._prediction(self.desired_references,sample['drones'])
        value = build_observation(sample,self.mission,self.case['goals'],
                                  self.executed_references,self.last_prediction)
        self._last_finite_observation = {key:array.copy() for key,array in value.items()}
        return value

    def _states(self, rows):
        return [KinematicState(row['drone_xyz'],row['drone_velocity']) for row in rows]

    def _prediction(self, references, rows):
        now = self.mission.tick*.01
        radii = [self.supervisor.assembly_radius+self.supervisor.tracking_inflation+
                 np.linalg.norm(reference.sample(now)[0]-np.array(row['drone_xyz']))
                 for reference,row in zip(references,rows)]
        conflict,clearance = self.supervisor.predict(references,now,2.,radii)
        return dict(clearance_m=clearance,conflict=conflict)

    def _predicates(self, sample):
        service,final = [],[]
        service_metrics,final_metrics = [],[]
        for drone,(row,payload,reference) in enumerate(zip(sample['drones'],self.fixture.payloads,
                                                          self.executed_references)):
            target = self.mission.targets[drone]
            sm = None if target == WAIT else payload_metrics(row,self.case['goals'][target],payload)
            fm = payload_metrics(row,reference.target,payload)
            service.append(sm is not None and _settled(sm))
            final.append(_settled(fm))
            service_metrics.append(sm)
            final_metrics.append(fm)
        return tuple(service),tuple(final),service_metrics,final_metrics

    def _change_reference(self, drone, target, rows):
        now = self.mission.tick*.01
        reference = self.executed_references[drone]
        actual = KinematicState(rows[drone]['drone_xyz'],rows[drone]['drone_velocity'])
        if target == WAIT:
            new = reference.brake(now,actual_state=actual)
            self._reference_goals[drone] = None
        else:
            new = reference.retarget(self.case['goals'][target],now,actual_state=actual)
            self._reference_goals[drone] = target
        self.desired_references[drone] = new
        self.executed_references[drone] = new

    def _terminate(self, reason):
        # Wall/raw infrastructure failures can occur without another simulated
        # interval (including after a successful final interval's raw flush).
        if self.mission.terminal_reason in (None,'success'):
            self.mission.terminal_reason = reason
            self.mission.team_completion_tick = None
            self.mission.reward_components['failure'] -= 1000

    def _ensure_raw(self):
        if not self.retain_replay or self._replay is not None:
            return
        sequence = self._raw_sequence
        self._raw_sequence += 1
        self._raw_start_tick = self.mission.tick
        reset = dict(kind='RESET',tick=self.mission.tick,actual=self.fixture.sample(),
                     geometry=self.fixture.geometry_state(),mission=self._mission_row(),
                     resume_predecessors=deepcopy(self.replay_segments),**self._raw_identity())
        self._replay = RawWriter(self.output/f'replay-{sequence:04d}.draw')
        self._replay.append(compact_replay_record(reset))
        if self.record_raw:
            self._raw = RawWriter(self.output/f'native-{sequence:04d}.draw')
            self._raw.append(dict(reset,resume_predecessors=deepcopy(self.raw_segments)))

    def _raw_identity(self):
        return dict(case_id=self.case['case_id'],model_hash=self.binding['model_sha256'],
                    profile_hash=self.binding['profile_sha256'],replay_stride_ticks=self.replay_stride_ticks,
                    recording_profile='ALL_100HZ_TICKS' if self.replay_stride_ticks==1 else 'DECISION_FIRST_AND_LAST_TICK_PLUS_TERMINAL')

    def _mission_row(self):
        return dict(tick=self.mission.tick,targets=list(self.mission.targets),
                    goals=[asdict(goal) for goal in self.mission.goals],
                    priority=self.mission.priority,priorities=list(self.mission.priorities),
                    available=list(self.mission.available),leg_start_ticks=list(self.mission.leg_start_ticks),
                    commitment_start_ticks=list(self.mission.commitment_start_ticks),
                    final_dwell_ticks=list(self.mission.final_dwell_ticks),
                    terminal_reason=self.mission.terminal_reason,event_receipts=deepcopy(self.mission.event_receipts))

    def _finish_raw_segment(self):
        for attribute,segments in (('_raw',self.raw_segments),('_replay',self.replay_segments)):
            writer = getattr(self,attribute)
            if writer is not None:
                writer.close()
                segments.append(dict(path=str(writer.path),sha256=sha256(writer.path),
                                     bytes=writer.path.stat().st_size,start_tick=self._raw_start_tick,
                                     end_tick=self.mission.tick,records=writer.frame_count))
                setattr(self,attribute,None)
        if self._elapsed() > 600:
            self._terminate('infrastructure:wall_timeout')

    def _stream_audit(self, native_report=None):
        """Record live audit coverage, or run a supplied transient-row callback.

        Custom auditors must provide their own checkpoint methods for recovery.
        No callback or scoped audit report grants production admission.
        """
        if native_report is None:
            native_report = self.online_auditor(fixture=self.fixture,
                stages=self.fixture.stage_rows,substeps=self.fixture.substep_rows)
        if not isinstance(native_report,dict) or type(native_report.get('safe')) is not bool:
            raise ValueError('Online auditor must return boolean safe and serializable summary')
        json.dumps(native_report,allow_nan=False)
        self.online_audit_summary['control_ticks'] += 1
        self.online_audit_summary['internal_stages'] += 64
        self.online_audit_summary['internal_substeps'] += 16
        self.online_audit_summary['latest'] = deepcopy(native_report)
        if not native_report['safe']:
            self.online_audit_summary['failures'].append(dict(tick=self.mission.tick+1,
                                                             summary=deepcopy(native_report)))
        return native_report

    def step(self, action):
        """Advance one decision; optional seconds include terminal flush and observation.

        Raised exceptions remain raised; their completed diagnostic is available
        as ``last_step_timings``. Diagnostics never enter replay or policy inputs.
        """
        timing = _StepTimings(self.collect_timings)
        try:
            with timing:
                result = self._step(action,timing)
        finally:
            self.last_step_timings = timing.result
        if self.collect_timings:
            result[3]['timings'] = dict(timing.result)
        return result

    def _step(self, action, timing):
        if self.closed or self.terminated:
            raise ValueError('Cannot step a closed or terminal development mission')
        # Validate before creating raw output, advancing time, or changing refs.
        decoded = decode_action(action)
        timing.switch('mask_observation')
        mask = self.mission.legal_mask()
        if not mask[action] or self.mission.tick % 20:
            raise ValueError('Illegal joint action or non-decision boundary')
        timing.switch('controller_reference_supervisor')
        before_reward = dict(self.mission.reward_components)
        start_tick = self.mission.tick
        held_seconds = 0.
        timing.switch('record_io')
        self._ensure_raw()
        timing.switch('controller_reference_supervisor')
        if self._elapsed() > 600:
            self._terminate('infrastructure:wall_timeout')
        else:
            old_targets = self.mission.targets
            action_receipt = self.mission.apply(action,self.mission.tick)
            self._last_action_mask = mask.tolist()
            try:
                rows = self.fixture.sample()['drones']
                for drone,(old,new) in enumerate(zip(old_targets,decoded[:2])):
                    if old != new:
                        self._change_reference(drone,new,rows)
                if self.mission.tick == 0:
                    service,final,_,_ = self._predicates(self.fixture.sample())
                    self.mission.seed_settling(settling=service,final_settling=final)
                for _ in range(20):
                    if self.terminated:
                        break
                    if self._elapsed() > 600:
                        self._terminate('infrastructure:wall_timeout')
                        break
                    tick = self.mission.tick
                    now = tick*.01
                    sample = self.fixture.sample()
                    states = self._states(sample['drones'])
                    # Resume a held intention from its actual executed p/v/a,
                    # never jump back onto the old moving desired trajectory.
                    for drone,held in enumerate(self.last_held):
                        if held:
                            goal = self._reference_goals[drone]
                            if goal is not None and self.mission.available[drone]:
                                self.desired_references[drone] = self.executed_references[drone].retarget(
                                    self.case['goals'][goal],now,actual_state=states[drone])
                            else:
                                self.desired_references[drone] = self.executed_references[drone]
                    requested = list(self.desired_references)
                    decision = self.supervisor.decide(requested,states,now,priority=self.mission.priority)
                    self.last_prediction = self._prediction(requested,sample['drones'])
                    self.executed_references = list(decision.references)
                    self.last_held = tuple(decision.held)
                    self.last_supervisor_reason = decision.reason
                    if decision.failure is not None:
                        self._terminate('supervisor:'+decision.failure)
                        break
                    values = [reference.sample(now) for reference in self.executed_references]
                    motor = np.array([controller.action(row,p,tick=tick,reference_velocity=v)
                                      for controller,row,(p,v,a) in zip(self.controllers,sample['drones'],values)])
                    timing.switch('physics_audit')
                    if isinstance(self.online_auditor,OnlineNativeAudit):
                        native_report = step_native_with_audit(self.fixture,(1+.8*motor)*HOVER_RPM,
                            self.online_auditor,detailed=self.record_raw,location=f'step:{tick+1}')
                    else:
                        self.fixture.step_rpm((1+.8*motor)*HOVER_RPM,observe_stages=True)
                        native_report = None
                    sample = self.fixture.sample()
                    audit = audit_geometry(self.fixture)
                    self.ledger.observe(audit)
                    reason = None if self.ledger.safe else 'safety:observed_violation'
                    stream_audit = self._stream_audit(native_report)
                    if not stream_audit['safe']:
                        reason = 'safety:online_internal_audit'
                    service,final,service_metrics,final_metrics = self._predicates(sample)
                    timing.switch('controller_reference_supervisor')
                    event = None
                    if not self._event_applied and self.case['event'] is not None and self.case['event']['tick'] == tick+1:
                        event = deepcopy(self.case['event'])
                        self._event_applied = True
                    receipt = self.mission.advance(tick+1,settling=service,final_settling=final,
                        held=decision.held,safety_interrupted=decision.held,event=event,failure_reason=reason)
                    held_seconds += .01*sum(decision.held)
                    save_replay = (self.replay_stride_ticks==1 or tick==start_tick
                                   or (tick+1)%20==0 or self.terminated)
                    if self._replay is not None and save_replay:
                        timing.switch('record_io')
                        record = dict(kind='STEP',tick=tick+1,actual=sample,controls=[c.last_audit for c in self.controllers],
                            action=action,action_tick=start_tick,action_receipt=action_receipt if tick == start_tick else None,
                            structural_mask=self._last_action_mask,mission=self._mission_row(),
                            reference=[dict(p=p.tolist(),v=v.tolist(),a=a.tolist()) for p,v,a in values],
                            requested_references=[r.to_dict() for r in requested],
                            executed_references=[r.to_dict() for r in self.executed_references],
                            supervisor=dict(reason=decision.reason,held=list(decision.held),horizon_s=decision.horizon_s,
                                            predicted_clearance_m=decision.predicted_clearance_m),
                            event=event,reward=receipt,service_metrics=service_metrics,final_metrics=final_metrics,
                            stages=self.fixture.stage_rows,substeps=self.fixture.substep_rows,audit=audit,
                            online_audit=stream_audit,**self._raw_identity())
                        self._replay.append(compact_replay_record(record))
                        if self._raw is not None:
                            self._raw.append(record)
                    timing.switch('controller_reference_supervisor')
                    # Event receipt is immediate for the 100 Hz operational stop;
                    # no policy inference/action occurs between 5 Hz boundaries.
                    if event is not None and event['kind'] == 'availability-delay' and not self.terminated:
                        self._change_reference(event['drone_id']-1,WAIT,sample['drones'])
            except ReferenceInfeasible as error:
                timing.switch('controller_reference_supervisor')
                self._terminate('reference:'+str(error))
            except (ValueError,FloatingPointError) as error:
                timing.switch('controller_reference_supervisor')
                self._terminate('numerical_or_contract:'+str(error))
        if self.terminated:
            timing.switch('record_io')
            self._finish_raw_segment()
        timing.switch('controller_reference_supervisor')
        if self._elapsed() > 600:
            self._terminate('infrastructure:wall_timeout')
            timing.switch('record_io')
            self._finish_raw_segment()
        timing.switch('controller_reference_supervisor')
        components = {key:value-before_reward[key] for key,value in self.mission.reward_components.items()}
        timing.switch('mask_observation')
        try:
            observation = self.observation()
            terminal_observation_source = 'current_physical_state'
        except ValueError:
            if not self.terminated:
                raise
            observation = {key:value.copy() for key,value in self._last_finite_observation.items()}
            terminal_observation_source = 'last_finite_observation_no_bootstrap'
        timing.switch('controller_reference_supervisor')
        info = dict(scope=self.scope,production_admitted=self.production_admitted,mode=self.mode,tick=self.mission.tick,
                    elapsed_simulated_s=(self.mission.tick-start_tick)*.01,
                    reward_components=components,terminal_reason=self.mission.terminal_reason,
                    supervisor_held_drone_seconds=held_seconds,supervisor_reason=self.last_supervisor_reason,
                    wall_elapsed_s=self._elapsed(),terminal_observation_source=terminal_observation_source,
                    events=deepcopy(self.mission.event_receipts),raw_segments=deepcopy(self.raw_segments),
                    replay_segments=deepcopy(self.replay_segments),online_audit=deepcopy(self.online_audit_summary),
                    formal_physics_acceptance=False)
        if self.terminated:
            timing.switch('record_io')
            self._write_result()
        return observation,sum(components.values()),self.terminated,info

    def _write_result(self):
        atomic_json(self.output/'result.json',dict(schema='dual-development-result-v1' if self.mode=='development' else 'dual-mission-result-v1',scope=self.scope,
            production_admitted=self.production_admitted,mode=self.mode,retain_replay=self.retain_replay,replay_stride_ticks=self.replay_stride_ticks,collect_timings=self.collect_timings,formal_physics_acceptance=False,terminal_reason=self.mission.terminal_reason,
            simulated_seconds=self.mission.tick*.01,wall_elapsed_seconds=self._elapsed(),
            completed_counts=self.mission.completed_counts,reward_components=self.mission.reward_components,
            raw_segments=self.raw_segments,replay_segments=self.replay_segments,binding=self.binding,
            final_native_state=dict(world_time=float(self.fixture.data.time),qpos=self.fixture.data.qpos.tolist(),
                                    qvel=self.fixture.data.qvel.tolist(),ctrl=self.fixture.data.ctrl.tolist()),
            mission=self.mission.to_dict(),online_audit=self.online_audit_summary,
            wall_deadline_exceeded=self._elapsed()>600))

    def close(self):
        if not self.closed:
            self._finish_raw_segment()
            self._write_result()
            self.closed = True
            if self.native_audit_library is not None:
                self.online_auditor.close()

    def to_dict(self):
        """Complete mission/backend/controller/reference state; closes a raw segment."""
        if self.mission.tick % 20 and not self.terminated:
            raise ValueError('Checkpoint only at completed joint decision boundaries')
        if not callable(getattr(self.online_auditor,'to_dict',None)) or not callable(getattr(self.online_auditor,'restore',None)):
            raise ValueError('Checkpoint requires a stateful auditable recovery protocol')
        self._finish_raw_segment()
        mj = self.fixture.mj
        signature = int(mj.mjtState.mjSTATE_INTEGRATION)
        integration = np.empty(mj.mj_stateSize(self.fixture.model,signature))
        mj.mj_getState(self.fixture.model,self.fixture.data,integration,signature)
        return dict(schema='dual-development-env-state-v1' if self.mode=='development' else 'dual-env-state-v1',scope=self.scope,mode=self.mode,retain_replay=self.retain_replay,replay_stride_ticks=self.replay_stride_ticks,collect_timings=self.collect_timings,binding=deepcopy(self.binding),
            mission=self.mission.to_dict(),case=deepcopy(self.case),
            backend=dict(signature=signature,integration=integration.tolist(),qpos=self.fixture.data.qpos.tolist(),
                         qvel=self.fixture.data.qvel.tolist(),ctrl=self.fixture.data.ctrl.tolist(),rpm=self.fixture.rpm.tolist()),
            controllers=[dict(integral=c.integral,tick=c.tick,last_audit=deepcopy(c.last_audit)) for c in self.controllers],
            desired_references=[r.to_dict() for r in self.desired_references],
            executed_references=[r.to_dict() for r in self.executed_references],
            reference_goals=list(self._reference_goals),last_held=list(self.last_held),
            last_prediction=deepcopy(self.last_prediction),last_supervisor_reason=self.last_supervisor_reason,
            event_applied=self._event_applied,last_action_mask=list(self._last_action_mask),
            ledger=asdict(self.ledger),rng=deepcopy(self.rng.bit_generator.state),
            last_finite_observation={key:value.tolist() for key,value in self._last_finite_observation.items()},
            mission_started_at=self.mission_started_at,saved_at=datetime.now(timezone.utc).isoformat(),
            elapsed_wall_s=self._elapsed(),raw_segments=deepcopy(self.raw_segments),
            replay_segments=deepcopy(self.replay_segments),online_audit=deepcopy(self.online_audit_summary),
            online_auditor_state=deepcopy(self.online_auditor.to_dict()),
            default_online_auditor=self._default_online_auditor,record_raw=self.record_raw)

    def restore(self, snapshot):
        """Restore into a fresh matching world; resume downtime consumes wall budget."""
        if self.closed or self.mission.last_action is not None or self.mission.tick != 0 or self._replay is not None:
            raise ValueError('Restore requires a fresh unused environment')
        if self.admission is not None:
            self.admission.require_current()
        expected_schema='dual-development-env-state-v1' if self.mode=='development' else 'dual-env-state-v1'
        if (not isinstance(snapshot,dict) or snapshot.get('schema') != expected_schema
                or snapshot.get('scope') != self.scope or snapshot.get('mode','development') != self.mode
                or snapshot.get('retain_replay',True) != self.retain_replay or snapshot.get('binding') != self.binding
                or snapshot.get('replay_stride_ticks',1) != self.replay_stride_ticks
                or snapshot.get('collect_timings') is not self.collect_timings
                or snapshot.get('case') != self.case or snapshot.get('record_raw') != self.record_raw
                or snapshot.get('default_online_auditor') != self._default_online_auditor):
            raise ValueError('Exact matching development checkpoint and recording mode required')
        # Reject nonfinite checkpoint data before touching native state.
        json.dumps(snapshot,allow_nan=False)
        mission = MissionState.from_dict(snapshot['mission'])
        if mission.no_transfer!=self.no_transfer:
            raise ValueError('Recovered transfer ablation differs from environment binding')
        desired = [ContinuousReference.from_dict(row) for row in snapshot['desired_references']]
        executed = [ContinuousReference.from_dict(row) for row in snapshot['executed_references']]
        if len(desired) != 2 or len(executed) != 2 or len(snapshot['controllers']) != 2:
            raise ValueError('Two complete reference/controller states required')
        mj,model,data = self.fixture.mj,self.fixture.model,self.fixture.data
        signature = int(mj.mjtState.mjSTATE_INTEGRATION)
        if snapshot['backend']['signature'] != signature:
            raise ValueError('Native integration state signature mismatch')
        integration = _array(snapshot['backend']['integration'],(mj.mj_stateSize(model,signature),))
        for key,size in (('qpos',model.nq),('qvel',model.nv),('ctrl',model.nu)):
            _array(snapshot['backend'][key],(size,))
        rpm = _array(snapshot['backend']['rpm'],(2,4))
        for row in snapshot['controllers']:
            if (not math.isfinite(row['integral']) or abs(row['integral']) > .12
                    or row['tick'] != (mission.tick-1 if mission.tick else None)):
                raise ValueError('Controller clock/integral mismatch')
        elapsed = snapshot['elapsed_wall_s']
        saved = datetime.fromisoformat(snapshot['saved_at'])
        started = datetime.fromisoformat(snapshot['mission_started_at'])
        if saved.tzinfo is None or started.tzinfo is None or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Measured resume wall clock required')
        for segment in snapshot['raw_segments']+snapshot['replay_segments']:
            path = Path(segment['path'])
            if not path.is_file() or path.stat().st_size != segment['bytes'] or sha256(path) != segment['sha256']:
                raise ValueError('Prior raw segment missing or changed')
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(snapshot['rng'])
        # Validate native input consistency in an isolated data buffer first;
        # rejected checkpoints must not partially mutate the fresh live world.
        candidate = mj.MjData(model)
        mj.mj_setState(model,candidate,integration,signature)
        for key in ('qpos','qvel','ctrl'):
            if not np.array_equal(getattr(candidate,key),snapshot['backend'][key]):
                raise ValueError('Redundant checkpoint backend fields disagree')
        if not math.isclose(candidate.time,mission.tick*.01,abs_tol=1e-7):
            raise ValueError('Restored native and mission clocks differ')
        # Derived data is recomputed, then integration inputs are reinstalled so
        # any warmstart update by mj_forward cannot change the next step.
        mj.mj_setState(model,data,integration,signature)
        mj.mj_forward(model,data)
        mj.mj_setState(model,data,integration,signature)
        self.fixture.rpm = rpm.copy()
        if self._default_online_auditor:
            if self.native_audit_library is not None:
                self.online_auditor.close()
            self.online_auditor = self._make_auditor()
        self.online_auditor.restore(snapshot['online_auditor_state'])
        self.fixture.stage_rows,self.fixture.substep_rows = [],[]
        self.mission,self.desired_references,self.executed_references = mission,desired,executed
        for controller,row in zip(self.controllers,snapshot['controllers']):
            controller.integral,controller.tick = row['integral'],row['tick']
            controller.last_audit = deepcopy(row['last_audit'])
        self._reference_goals = list(snapshot['reference_goals'])
        self.last_held = tuple(snapshot['last_held'])
        self.last_prediction = deepcopy(snapshot['last_prediction'])
        self.last_supervisor_reason = snapshot['last_supervisor_reason']
        self._event_applied = snapshot['event_applied']
        self._last_action_mask = list(snapshot['last_action_mask'])
        self.ledger = SafetyLedger(**deepcopy(snapshot['ledger']))
        self.rng = rng
        self._last_finite_observation = {key:np.asarray(value,dtype=np.float32)
                                         for key,value in snapshot['last_finite_observation'].items()}
        self.raw_segments = deepcopy(snapshot['raw_segments'])
        self.replay_segments = deepcopy(snapshot['replay_segments'])
        self.online_audit_summary = deepcopy(snapshot['online_audit'])
        self.mission_started_at = snapshot['mission_started_at']
        self._elapsed_offset = elapsed+max(0.,(datetime.now(timezone.utc)-saved).total_seconds())
        self._clock_start = time.monotonic()
        if self._elapsed() > 600:
            self._terminate('infrastructure:wall_timeout')
        return self.observation()
