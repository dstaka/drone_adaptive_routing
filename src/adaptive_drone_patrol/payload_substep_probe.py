"""Read-only native-ball integration-path evidence; no forces or state setters."""
import json

METHOD = 'constant-world-cfrc+semiimplicit-root-path-v1'


class SubstepRecorder:
    """Wrap one solver instance, preserving exactly one original substep call.

    Only the articulated 15q/12DOF fixture is supported. The pinned standalone
    free-body midpoint path overwrites vel_next and is deliberately excluded.
    All reads use qpos/solver fields, avoiding Cartesian getter refreshes.
    """
    def __init__(self, solver, entity, stream, *, capture=None):
        if entity.n_qs != 15 or entity.n_dofs != 12:
            raise ValueError('Substep work requires the audited articulated fixture')
        self.h = float(solver.substep_dt)
        if self.h not in (.005, .0025):
            raise ValueError('Undeclared physics substep')
        self.solver, self.entity, self.stream = solver, entity, stream
        self.capture = capture or self._capture
        self.control, self.index = None, 0
        self.expected = round(.01/self.h)
        self.original, self.wrapper = None, None
        self.records = 0

    def _capture(self, post=False):
        from genesis.utils.misc import qd_to_torch
        state = self.solver.dyn_state
        base = self.entity.get_link('base_link').idx
        result = dict(
            qpos=self.entity.get_qpos()[0].detach().cpu().tolist(),
            negative_external_force_field=qd_to_torch(state.links.cfrc_applied_vel, copy=True).cpu().tolist(),
            negative_external_torque_about_root_com_field=qd_to_torch(state.links.cfrc_applied_ang, copy=True).cpu().tolist(),
            root_com=qd_to_torch(state.links.root_COM, copy=True)[base, 0].cpu().tolist(),
        )
        if post:
            start = self.entity.dof_start
            result['dofs_velocity_next'] = qd_to_torch(state.dofs.vel_next, copy=True)[start:start+12, 0].cpu().tolist()
        return result

    def begin_control(self, step):
        if type(step) is not int or step < 0 or (self.control is not None and
                (step != self.control+1 or self.index != self.expected)):
            raise ValueError('Missing, duplicate or incomplete control interval')
        self.control, self.index = step, 0

    def __enter__(self):
        if self.original is not None:
            raise RuntimeError('Recorder already installed')
        self.original = self.solver.substep
        def wrapped(f):
            if self.control is None or self.index >= self.expected:
                raise RuntimeError('Undeclared or extra physics substep')
            before = self.capture(post=False)
            error = None
            try:
                return self.original(f)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                after = self.capture(post=True)
                record = dict(control_step=self.control, substep=self.index,
                    time_s=(self.control*self.expected+self.index)*self.h, dt_s=self.h,
                    before=before, after=after)
                if 'qpos' in before:
                    record.update(qpos_before=before['qpos'], qpos_after=after['qpos'],
                        dofs_velocity_next=after['dofs_velocity_next'],
                        root_com_before=before['root_com'], root_com_after=after['root_com'],
                        negative_external_force_field=before['negative_external_force_field'],
                        negative_external_torque_about_root_com_field=before['negative_external_torque_about_root_com_field'],
                        negative_external_force_field_after=after['negative_external_force_field'],
                        negative_external_torque_about_root_com_field_after=after['negative_external_torque_about_root_com_field'])
                    # Keep each measurement once; test-injected captures retain their raw dictionaries.
                    del record['before'], record['after']
                if error:
                    record['error'] = error
                self.stream.write(json.dumps(record, allow_nan=True)+'\n')
                self.stream.flush()
                self.index += 1
                self.records += 1
        self.wrapper = wrapped
        self.solver.substep = wrapped
        return self

    def __exit__(self, *_):
        if self.solver.substep is not self.wrapper:
            raise RuntimeError('Recorder hook was unexpectedly replaced')
        self.solver.substep = self.original
        return False
