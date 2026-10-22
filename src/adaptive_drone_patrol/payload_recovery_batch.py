"""Experimental three-size batched backend; no physical or learning admission.

Each size owns ONE Genesis scene with N real environments. Logical environment
row i stays row i in every size scene. Only episode boundaries may select another
size. Unselected rows still advance physically, with zero RPM, and are observable
but never count as transitions. Effective batch geometry/isolation, runtime reset
forces and N=1 equivalence require actual GPU qualification before any learning.

The orchestration uses an injectable NumPy-compatible array backend and bucket
builder for CPU contract tests. The real adapter uses batched device tensors in
Genesis, but snapshots transfer complete arrays to CPU: no throughput claim is
made, and device-side guards/chunked telemetry remain future optimization work.
The preparation adapter supports at most128 substeps. A base128 candidate needs a
half-dt256 comparison which this adapter cannot run; therefore this is not a
complete selected-batch62-condition qualification runner.
"""
from collections import deque
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

from .payload_ball_asset import write_ball_mjcf
from .payload_ball_bank import ball_joint_configuration
from .payload_contract import Payload, SIZES_M, rotation
from .payload_recovery_sampling import EpisodeLoads, EpisodeSchedule
from .util import sha256

REFERENCE_MASS = .02700047753751278


class BatchQuarantined(RuntimeError):
    """The complete current rollout is unusable; preserve evidence and stop."""


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


class GenesisSizeBucket:
    """Thin adapter using pinned 1.4.3 public batched setters and motor kernel."""
    def __init__(self, scene, entity, ground, gs, authored_mass, asset):
        import torch
        self.scene, self.entity, self.ground, self.gs = scene, entity, ground, gs
        self.authored_mass, self.asset = authored_mass, asset
        self.torch = torch
        self.package = entity.get_link('payload')
        self.payload_local = self.package.idx-entity.link_start
        names = ('base_link','prop0_link','prop1_link','prop2_link','prop3_link','suspension_rod','payload')
        if [link.name for link in entity.links] != list(names) or (entity.n_qs,entity.n_dofs) != (15,12):
            raise ValueError('Inherited loaded body ordering and DOFs required')
        self.rotors = self.tensor([entity.get_link(f'prop{i}_link').idx for i in range(4)], integer=True)
        self.spins = self.tensor([-1,1,-1,1])

    def tensor(self, value, *, integer=False):
        return self.torch.as_tensor(value,device=self.gs.device,
                                    dtype=self.gs.tc_int if integer else self.gs.tc_float)

    def reset_rows(self, rows, draws, initials):
        """Write selected rows, with independently specified actual load properties."""
        e, ids = self.entity, list(rows)
        masses = np.array([draw.payload.mass for draw in draws])
        inertia = np.array([np.eye(3)*draw.payload.inertia[0] for draw in draws])[:,None]
        e.set_links_mass(self.tensor(masses[:,None]),links_idx_local=[self.payload_local],envs_idx=ids)
        e.set_links_inertia(self.tensor(inertia),links_idx_local=[self.payload_local],envs_idx=ids)
        e.set_pos(self.tensor([s['drone_xyz'] for s in initials]),envs_idx=ids,zero_velocity=True,relative=True)
        e.set_quat(self.tensor([s['drone_quaternion_wxyz'] for s in initials]),envs_idx=ids,
                   zero_velocity=True,relative=True)
        q = e.get_qpos(envs_idx=ids).clone()
        coordinates = [ball_joint_configuration(s,draw.payload.side_m) for draw,s in zip(draws,initials)]
        for name, column in [('suspension_ball',0),('payload_ball',1)]:
            start = e.get_joint(name).q_start-e.q_start
            q[:,start:start+4] = self.tensor([value[column] for value in coordinates])
        e.set_qpos(q,envs_idx=ids,zero_velocity=True)
        damping = np.zeros((len(ids),e.n_dofs))
        roots = [joint for joint in e.joints if joint.n_qs == 7 and joint.n_dofs == 6]
        if len(roots) != 1:
            raise ValueError('Exactly one original free vehicle root required')
        start = roots[0].dof_start-e.dof_start
        damping[:,start+3:start+6] = ((self.authored_mass+masses+.0001)*1e-5)[:,None]
        e.set_dofs_damping(self.tensor(damping),envs_idx=ids)

    def set_rpm(self, rpm):
        self.entity.solver.set_drone_rpm(self.rotors,3.16e-10,7.94e-12,self.tensor(rpm),self.spins,False)

    def step(self):
        self.scene.step()

    def snapshot(self):
        """Read every active and inactive physical slot, including held force fields."""
        from genesis.utils.misc import qd_to_torch
        e = self.entity
        state = e.solver.dyn_state.links
        contacts = e.get_contacts(with_entity=self.ground)
        result = dict(qpos=_numpy(e.get_qpos()),velocity=_numpy(e.get_dofs_velocity()),
                      xyz=_numpy(e.get_links_pos()),quaternion=_numpy(e.get_links_quat()),
                      linear_velocity=_numpy(e.get_links_vel()),angular_velocity=_numpy(e.get_links_ang()),
                      mass=_numpy(e.get_links_mass()),inertia=_numpy(e.get_links_inertia()),
                      package_aabb=_numpy(self.package.get_AABB()),
                      contact=_numpy(contacts['valid_mask']).any(axis=-1),
                      force=_numpy(qd_to_torch(state.cfrc_applied_vel,copy=True)).swapaxes(0,1),
                      torque=_numpy(qd_to_torch(state.cfrc_applied_ang,copy=True)).swapaxes(0,1))
        # A generated mesh/hash is private provenance; no geometry verification claim.
        if sha256(self.asset['path']) != self.asset['sha256']:
            raise ValueError('Generated physical asset bytes changed')
        return result


def build_size_bucket(output, side, n_envs, substeps, *, genesis_module=None):
    """Build one real N-row MJCF scene; caller owns gs.init(precision='32')."""
    import torch
    if genesis_module is None:
        import genesis as genesis_module
    gs = genesis_module
    if gs.tc_float != torch.float32:
        raise ValueError('This unqualified candidate requires caller-initialized precision32')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    original = Path(gs.__file__).parent/'assets/urdf/drones/cf2x.urdf'
    authored = sum(float(m.get('value')) for m in ET.parse(original).findall('.//inertial/mass'))
    asset = write_ball_mjcf(original,output/'candidate.xml',Payload(REFERENCE_MASS,.2,side))
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.01),
                     rigid_options=gs.options.RigidOptions(dt=.01/substeps,batch_links_info=True,
                         batch_dofs_info=True,enable_collision=True,enable_joint_limit=True,
                         integrator=gs.integrator.approximate_implicitfast,
                         constraint_solver=gs.constraint_solver.Newton),show_viewer=False)
    ground = scene.add_entity(gs.morphs.Plane())
    entity = scene.add_entity(gs.morphs.MJCF(file=str((output/'candidate.xml').resolve()),
                              pos=(0,0,1),default_armature=0.,align=True,collision=True))
    scene.build(n_envs=n_envs)
    bucket = GenesisSizeBucket(scene,entity,ground,gs,authored,
                               dict(path=output/'candidate.xml',sha256=asset['generated_asset_sha256']))
    masses = _numpy(entity.get_links_mass())
    if masses.shape != (n_envs,7) or not np.allclose(masses[:,:5].sum(axis=1),REFERENCE_MASS,atol=1e-8,rtol=0):
        raise ValueError('Every physical row must preserve the original free vehicle mass')
    # Initial inactive rows retain nominal load and declared vehicle damping.
    damping = np.zeros((n_envs,12))
    damping[:,3:6] = (authored+.2*REFERENCE_MASS+.0001)*1e-5
    entity.set_dofs_damping(bucket.tensor(damping),envs_idx=list(range(n_envs)))
    return bucket


class RecoveryBatch:
    """Raw physical batching preparation, without actor, reward or optimizer API."""
    training_admitted = False
    effective_geometry_verified = False
    optimizer_transitions = 0

    def __init__(self, output, n_envs, *, substeps=2, scene_builder=build_size_bucket,
                 array_backend=np, sink=None):
        if type(n_envs) is not int or n_envs < 1 or type(substeps) is not int or not 1 <= substeps <= 128:
            raise ValueError('Positive N and declared finite substeps in[1,128] required')
        if not callable(sink):
            raise ValueError('A complete ordered raw evidence sink is required')
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.n_envs, self.substeps, self.arrays, self.sink = n_envs,substeps,array_backend,sink
        self.buckets = [scene_builder(self.output/f'load_{side:.3f}',side,n_envs,substeps) for side in SIZES_M]
        self.loads, self.schedule = EpisodeLoads(REFERENCE_MASS),EpisodeSchedule(REFERENCE_MASS)
        self.bucket_for_env = array_backend.full(n_envs,-1,dtype=int)
        self.needs_reset = array_backend.ones(n_envs,dtype=bool)
        self.actions = array_backend.zeros((n_envs,4))
        self.previous_actions = array_backend.zeros((n_envs,4))
        self.targets = array_backend.zeros((n_envs,3))
        self.episode_steps = array_backend.zeros(n_envs,dtype=int)
        self.leg_steps = array_backend.zeros(n_envs,dtype=int)
        self.dwell_intervals = array_backend.zeros(n_envs,dtype=int)
        self.recent = deque(maxlen=128)
        self.rollout_quarantined = False
        self.generated_valid = self.generated_invalid = 0
        self.vector_steps = 0

    def _open(self):
        if self.rollout_quarantined:
            raise BatchQuarantined('Complete rollout quarantined; create a new declared candidate')

    def _ids(self, values):
        ids = list(values)
        if (not ids or any(type(i) is not int or not 0 <= i < self.n_envs for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Unique in-range logical environment rows required')
        return ids

    def masks(self):
        active = np.array([(self.bucket_for_env == i)&(~self.needs_reset) for i in range(3)],dtype=bool)
        return active,np.ones((3,self.n_envs),dtype=bool)

    def _snapshots(self):
        return [bucket.snapshot() for bucket in self.buckets]

    def _capture_after_failure(self, record, *, field='raw_after'):
        # A raising step may have advanced. Capture every readable bucket without
        # treating an exception as proof of either completion or no advancement.
        snapshots, errors = [], []
        for index,bucket in enumerate(self.buckets):
            try:
                snapshots.append(bucket.snapshot())
            except Exception as exc:
                snapshots.append(None)
                errors.append(dict(bucket=index,error=repr(exc)))
        record[field] = snapshots
        record['capture_errors'] = errors

    def _emit(self, record):
        self.recent.append(record)
        try:
            self.sink(record)
        except Exception as exc:
            self.rollout_quarantined = True
            raise BatchQuarantined('Raw evidence sink failed; rollout quarantined') from exc

    def _stop(self, record, reason):
        self.rollout_quarantined = True
        record['quarantined'] = True
        record['reason'] = reason
        self._emit(record)
        raise BatchQuarantined(reason)

    def mark_done(self, env_ids):
        self._open()
        ids = self._ids(env_ids)
        if np.any(self.bucket_for_env[ids] < 0) or np.any(self.needs_reset[ids]):
            raise ValueError('Only currently active episodes can finish')
        self.needs_reset[ids] = True

    def retarget(self, targets):
        self._open()
        ids = self._ids(targets)
        values = np.asarray([targets[i] for i in ids],dtype=float)
        if values.shape != (len(ids),3) or not np.isfinite(values).all() or np.any(self.needs_reset[ids]):
            raise ValueError('Finite per-row commands for active episodes required')
        self.targets[ids] = values
        self.leg_steps[ids] = self.dwell_intervals[ids] = 0
        self._emit(dict(kind='retarget',env_ids=ids,targets=values.copy(),history_preserved=True))

    def reset(self, assignments, *, initials, targets, randomized):
        self._open()
        ids = self._ids(assignments)
        if (set(initials) != set(ids) or set(targets) != set(ids)
                or np.any(~self.needs_reset[ids])):
            raise ValueError('Geometry may change only at selected episode boundaries')
        values = np.asarray([targets[i] for i in ids],dtype=float)
        if values.shape != (len(ids),3) or not np.isfinite(values).all():
            raise ValueError('Finite full per-row targets required')
        draws = {i:self.loads.draw(assignments[i],randomized=randomized) for i in ids}
        for i in ids:
            ball_joint_configuration(initials[i],draws[i].payload.side_m)
        self.schedule.reset(assignments,randomized=randomized)
        record = dict(kind='reset',env_ids=ids,episode_draws={i:draws[i].audit_record() for i in ids})
        self._capture_after_failure(record,field='raw_before')
        if record['capture_errors']:
            self._stop(record,'Current reset state observation incomplete before setters')
        before = record['raw_before']
        try:
            for index,(side,bucket) in enumerate(zip(SIZES_M,self.buckets)):
                selected = [i for i in ids if draws[i].payload.side_m == side]
                if selected:
                    bucket.reset_rows(selected,[draws[i] for i in selected],[initials[i] for i in selected])
                    self.bucket_for_env[selected] = index
            after = self._snapshots()
            record['raw_after'] = after
            others = [i for i in range(self.n_envs) if i not in ids]
            for old,new in zip(before,after):
                self._check_shapes(old)
                self._check_shapes(new)
                if set(old) != set(new) or any(not np.array_equal(old[k][others],new[k][others]) for k in old):
                    raise ValueError('Selective reset changed another physical row')
                for key in ('force','torque'):
                    if np.any(new[key][ids] != 0):
                        raise ValueError('Reset has nonzero observed held external wrench')
            for i in ids:
                self._reset_properties(after[int(self.bucket_for_env[i])],i,draws[i].payload,initials[i])
        except Exception as exc:
            self._capture_after_failure(record)
            self._stop(record,f'Reset/property/isolation evidence failed: {exc}')
        self.needs_reset[ids] = False
        self.actions[ids] = self.previous_actions[ids] = 0
        self.targets[ids] = values
        self.episode_steps[ids] = self.leg_steps[ids] = self.dwell_intervals[ids] = 0
        record['active_mask'],record['physical_mask'] = self.masks()
        self._emit(record)

    def _check_shapes(self, raw, *, numerical_guard=True):
        shapes = dict(qpos=(self.n_envs,15),velocity=(self.n_envs,12),xyz=(self.n_envs,7,3),
                      quaternion=(self.n_envs,7,4),linear_velocity=(self.n_envs,7,3),
                      angular_velocity=(self.n_envs,7,3),package_aabb=(self.n_envs,2,3),
                      mass=(self.n_envs,7),inertia=(self.n_envs,7,3,3),contact=(self.n_envs,),
                      force=(self.n_envs,8,3),torque=(self.n_envs,8,3))
        if set(raw) != set(shapes):
            raise ValueError('Every physical state/property/wrench field is required')
        for key,shape in shapes.items():
            if np.asarray(raw[key]).shape != shape or not np.isfinite(raw[key]).all():
                raise ValueError(f'Invalid actual full batched {key}')
        if np.asarray(raw['contact']).dtype != np.bool_:
            raise ValueError('Actual per-environment boolean contact observations required')
        # Inactive rows may contact the floor, but cannot silently retain
        # numerical excursions that could invalidate shared batch computation.
        quaternions = np.asarray(raw['quaternion'])
        generalized_quaternions = np.asarray(raw['qpos'])[:,[3,4,5,6,7,8,9,10,11,12,13,14]].reshape(-1,3,4)
        bad = (np.max(np.abs(np.linalg.norm(quaternions,axis=-1)-1),axis=1) > 1e-4
               ) | (np.max(np.abs(np.linalg.norm(generalized_quaternions,axis=-1)-1),axis=1) > 1e-4
               ) | (np.max(np.linalg.norm(raw['linear_velocity'],axis=-1),axis=1) > 20
               ) | (np.max(np.linalg.norm(raw['angular_velocity'],axis=-1),axis=1) > 200
               ) | (np.max(np.abs(raw['xyz']),axis=(1,2)) > 20
               ) | (np.max(np.abs(raw['qpos'][:,:3]),axis=1) > 20)
        rows = np.flatnonzero(bad).tolist()
        if rows and numerical_guard:
            raise ValueError(f'Numerical excursion in physical rows {rows}')
        return rows

    def _reset_properties(self, raw, row, payload, initial):
        self._check_shapes(raw)
        mass, inertia = np.asarray(raw['mass']),np.asarray(raw['inertia'])
        if (mass.shape != (self.n_envs,7) or inertia.shape != (self.n_envs,7,3,3)
                or not np.isfinite(mass[row]).all() or not np.isfinite(inertia[row]).all()
                or not math.isclose(float(mass[row,:5].sum()),REFERENCE_MASS,abs_tol=1e-8,rel_tol=0)
                or not math.isclose(float(mass[row,6]),payload.mass,rel_tol=1e-5)
                or not np.allclose(inertia[row,6],np.eye(3)*payload.inertia[0],atol=1e-15,rtol=1e-5)
                or np.asarray(raw['velocity']).shape != (self.n_envs,12) or np.any(raw['velocity'][row] != 0)):
            raise ValueError('Effective reset mass/inertia/generalized velocity differs')
        for name, index in [('drone',0),('package',6)]:
            if (not np.allclose(raw['xyz'][row,index],initial[name+'_xyz'],atol=1e-6,rtol=0)
                    or not np.allclose(rotation(raw['quaternion'][row,index]),
                                       rotation(initial[name+'_quaternion_wxyz']),atol=1e-6,rtol=0)):
                raise ValueError('Actual reset pose differs')
        center = raw['xyz'][row,6]
        extent = np.abs(rotation(raw['quaternion'][row,6])) @ np.full(3,payload.side_m/2)
        if np.asarray(raw['package_aabb']).shape != (self.n_envs,2,3) or not np.allclose(
                raw['package_aabb'][row],[center-extent,center+extent],atol=1e-5,rtol=0):
            raise ValueError('Effective cube collision extent differs')

    def _invalid_rows(self, snapshots, active):
        bad = set()
        for index,raw in enumerate(snapshots):
            numerical = self._check_shapes(raw,numerical_guard=False)
            inactive = [row for row in numerical if not active[index,row]]
            if inactive:
                raise ValueError(f'Numerical excursion in inactive physical bucket {index}, rows {inactive}')
            bad.update(numerical)
            for row in np.flatnonzero(active[index]):
                q = raw['quaternion'][row]
                rd,rp = rotation(q[0]),rotation(q[6])
                side = SIZES_M[index]
                delta = raw['xyz'][row,6]+rp @ [0,0,side/2]-(raw['xyz'][row,0]+rd @ [0,0,-.02])
                floor = raw['xyz'][row,0,2]-.0125*abs(rd[2,2])-.06*np.linalg.norm(rd[2,:2])
                roll = math.degrees(math.atan2(rd[2,1],rd[2,2]))
                pitch = math.degrees(math.asin(float(np.clip(-rd[2,0],-1,1))))
                if (np.max(np.abs(np.linalg.norm(q,axis=-1)-1)) > 1e-4
                        or np.max(np.linalg.norm(raw['linear_velocity'][row],axis=-1)) > 20
                        or np.max(np.linalg.norm(raw['angular_velocity'][row],axis=-1)) > 200
                        or abs(np.linalg.norm(delta)-.3) > 2e-5 or floor <= .1
                        or raw['package_aabb'][row,0,2] <= .1 or abs(roll) > 60 or abs(pitch) > 60
                        or raw['contact'][row]):
                    bad.add(int(row))
        return sorted(bad)

    def step(self, actions):
        self._open()
        values = np.asarray(actions,dtype=float)
        if values.shape != (self.n_envs,4):
            raise ValueError('One full [N,4] action tensor required')
        active,physical = self.masks()
        selected = active.any(axis=0)
        if not selected.any():
            raise ValueError('At least one assigned active episode required')
        record = dict(kind='step',vector_step=self.vector_steps,raw_actions=values.copy(),
                      active_mask=active.copy(),physical_mask=physical.copy(),bucket_step_status=['NOT_ATTEMPTED']*3)
        self._capture_after_failure(record,field='raw_before')
        if record['capture_errors']:
            self._stop(record,'Current physical state observation incomplete before motor input')
        if not np.isfinite(values).all():
            self._stop(record,'Nonfinite raw action before clipping')
        try:
            invalid = self._invalid_rows(record['raw_before'],active)
            if invalid:
                record['invalid_logical_rows'] = invalid
                self._stop(record,'Invalid physical sample before next motor input')
            rpm = (1+.8*np.clip(values,-1,1))*14468.429183500699
            record['rpm'] = []
            for index,(bucket,mask) in enumerate(zip(self.buckets,active)):
                commands = np.where(mask[:,None],rpm,0.)
                record['rpm'].append(commands.copy())
                bucket.set_rpm(commands)
                record['bucket_step_status'][index] = 'UNKNOWN'
                bucket.step()
                record['bucket_step_status'][index] = 'COMPLETED'
            record['raw_after'] = self._snapshots()
            invalid = self._invalid_rows(record['raw_after'],active)
            record['invalid_logical_rows'] = invalid
        except BatchQuarantined:
            raise
        except Exception as exc:
            self._capture_after_failure(record)
            self._stop(record,f'Batch integration/evidence failed: {exc}')
        count = int(selected.sum())
        self.vector_steps += 1
        self.generated_invalid += len(invalid)
        self.generated_valid += count-len(invalid)
        record['generated_transitions'] = count
        if invalid:
            self._stop(record,'Invalid raw physical sample; complete rollout quarantined')
        self.previous_actions[selected] = self.actions[selected]
        self.actions[selected] = np.clip(values[selected],-1,1)
        self.episode_steps[selected] += 1
        self.leg_steps[selected] += 1
        self.schedule.advance(np.flatnonzero(selected).tolist())
        self._emit(record)
        return record
