"""Payload-specific continuous missions and actual-boundary routing evidence.

CPU doubles may exercise this state machine with explicit cpu-fixture provenance.
Physical execution requires the complete recovery62 admission gate. A successful
runner result is execution evidence, never independent operational acceptance.
"""

import copy
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np

from .analytical import (content_hash, event_comparison, explain_decision, restore,
                         rollout, snapshot)
from .baselines import exact_remaining, remaining_cost
from .objective import RoutingState, schema_for
from .payload_contract import SharedSettling, payload_metrics, rotation, settled
from .payload_recovery_runtime import require_recovery_runtime_admission
from .physical_evidence import physical_explanation
from .util import sha256

MISSION_CONTRACT = 'payload-native-recovery-mission-v1'
PRIORITIES = [81, 42, 43, 64, 23, 84, 6, 98, 22, 56]
DT = .01
LEG_SECONDS, MISSION_SECONDS, WALL_SECONDS = 30., 120., 600.


class MissionFailure(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=True)
        stream.write('\n')
        stream.flush()


def _line(stream, value):
    stream.write(json.dumps(value, separators=(',', ':'), allow_nan=True)+'\n')
    stream.flush()


def _validate_scenario(scenario):
    if (scenario.schema_version != 3 or scenario.seed != 40029 or
            scenario.checkpoint_ids != list('ABCDEFGHIJ') or scenario.priorities != PRIORITIES or
            scenario.alpha != 1 or scenario.beta != .05 or scenario.return_to_base or
            scenario.nominal_speed_mps != 1 or scenario.service_duration_s != [.5]*10 or
            not np.array_equal(scenario.multipliers, np.ones((11, 11)))):
        raise ValueError('Payload mission requires the frozen integer100 ten-target routing contract')
    events = dict(normal=None, emergency=dict(kind='priority', after=1, checkpoint='C', value=100),
                  cost_shift=dict(kind='cost', after=1, edges=[['F', 'A']], value=5))
    if scenario.scenario_id not in events or scenario.event != events[scenario.scenario_id]:
        raise ValueError('Payload mission event differs from the exact frozen scenario')
    scenario.validate_physical()


def _vector(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise MissionFailure('INVALID_RAW_BODY_STATE')
    return value


def _measures(sample, target, payload):
    """Reconstruct every settling/safety predicate instead of trusting complete."""
    for body in ('drone', 'package'):
        _vector(sample[body+'_xyz'], 3)
        rotation(sample[body+'_quaternion_wxyz'])
        if (np.linalg.norm(_vector(sample[body+'_velocity'], 3)) > 20 or
                np.linalg.norm(_vector(sample[body+'_angular_velocity'], 3)) > 200):
            raise MissionFailure('RAW_SPEED_GUARD')
    metrics = payload_metrics(sample, target, payload)
    collision = sample.get('vehicle_collision_aabbs')
    if not isinstance(collision, dict) or not collision:
        raise MissionFailure('MISSING_VEHICLE_COLLISION_GEOMETRY')
    floors = []
    for value in collision.values():
        bounds = np.asarray(value, float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
            raise MissionFailure('INVALID_VEHICLE_COLLISION_GEOMETRY')
        floors.append(float(bounds[0, 2]))
    metrics['drone_floor_m'] = min(floors)
    if type(sample.get('contacts')) is not bool:
        raise MissionFailure('CONTACT_STATE_UNAVAILABLE')
    if sample.get('done') is not False or sample.get('reset') is not False:
        raise MissionFailure('HIDDEN_RESET_OR_TERMINAL')
    unsafe = (sample['contacts'] or metrics['drone_floor_m'] <= .1 or metrics['package_floor_m'] <= .1 or
              abs(metrics['drone_roll_deg']) > 60 or abs(metrics['drone_pitch_deg']) > 60 or
              metrics['constraint_residual_m'] > 2e-5)
    return metrics, bool(unsafe)


def _continuity(previous, sample):
    for body in ('drone', 'package'):
        a, b = np.asarray(previous[body+'_xyz']), np.asarray(sample[body+'_xyz'])
        speed = max(np.linalg.norm(previous[body+'_velocity']), np.linalg.norm(sample[body+'_velocity']))
        if np.linalg.norm(b-a) > DT*speed+.005:
            raise MissionFailure('PHYSICAL_DISCONTINUITY')
        qa, qb = [np.asarray(s[body+'_quaternion_wxyz'], float) for s in (previous, sample)]
        dot = abs(float(np.dot(qa, qb)/(np.linalg.norm(qa)*np.linalg.norm(qb))))
        turn = 2*np.arccos(np.clip(dot, 0., 1.))
        spin = max(np.linalg.norm(previous[body+'_angular_velocity']), np.linalg.norm(sample[body+'_angular_velocity']))
        if turn > DT*spin+.005:
            raise MissionFailure('ORIENTATION_DISCONTINUITY')


def explain_payload_decision(decision, chooser):
    """Keep adopted routing semantics separate from measured-pose cost analysis."""
    state = restore(decision['boundary_snapshot'])
    if decision['evidence_kind'] == 'physical':
        value = physical_explanation(decision, chooser)
    else:
        value = explain_decision(state, chooser, decision['policy_hash'], decision['run_id'], decision['decision_id'])
        value.update(evidence_kind='cpu-fixture', actual_reference={key:copy.deepcopy(decision[key]) for key in
            ('simulation_step', 'event_seq', 'physical_pose', 'boundary_hash')})
    pose = state.copy()
    pose.scenario.coords[pose.current] = copy.deepcopy(decision['physical_pose'])
    # The actual actor retains its adopted nominal-coordinate input semantics.
    # Only this labelled counterfactual objective starts at the measured pose.
    order = rollout(state, chooser)
    value['pose_cost_analysis'] = dict(
        scope='Frozen remaining tasks from measured decision pose; nominal speed/service, no future events; '
              'policy suffix uses adopted routing inputs; no physical-time/energy optimality claim',
        actual_pose=copy.deepcopy(decision['physical_pose']),
        snapshot=snapshot(pose), actual_selected_action=decision['selected_action'],
        policy_suffix=remaining_cost(pose, order), exact=exact_remaining(pose),
        forced_first=[dict(action=int(a), **exact_remaining(pose, first=int(a))) for a in np.flatnonzero(pose.mask())])
    return value


def _latest_sample(flight):
    runtime = getattr(flight, 'runtime', None)
    if getattr(runtime, 'last_row', None) is not None:
        return copy.deepcopy(runtime.last_row)
    try:
        return copy.deepcopy(flight.sample())
    except Exception as exc:
        return {'capture_error':type(exc).__name__}


def require_solver_mission_admission(flight, evidence):
    """Accept only a raw-reconstructed process capability on the solver adapter."""
    from .payload_solver_flight import SolverAdmission, SolverFlight

    capability = evidence.get("solver_admission")
    if (
        not isinstance(flight, SolverFlight)
        or not isinstance(capability, SolverAdmission)
        or getattr(flight, "admission", None) is not capability
        or evidence.get("actual_binding") != capability.binding
    ):
        raise ValueError("Actual solver admission capability and matching flight are required")
    capability.require_current()
    return capability.proof


def execute_payload_mission(flight, planner, scenario, output, metadata, *,
                            clock=time.monotonic, recorder=None, physical_evidence=None):
    """Run one explicit case, preserving failures and all ten shared completions.

    The caller resets its predeclared case exactly once before entry. This runner
    never resets, resamples, clips raw telemetry or substitutes a probe controller.
    ``recorder`` may be an existing FrameRecorder with an already-built camera;
    camera construction, HUD encoding and all-frame verification remain separate.
    """
    kind = metadata.get('evidence_kind')
    if kind not in ('cpu-fixture', 'physical'):
        raise ValueError('Explicit physical or cpu-fixture evidence provenance is required')
    if kind == 'physical':
        evidence = physical_evidence or {}
        from .payload_solver_flight import SolverFlight
        if isinstance(flight, SolverFlight):
            require_solver_mission_admission(flight, evidence)
        else:
            require_recovery_runtime_admission(evidence.get('receipt', {}), evidence.get('actual_binding', {}),
                                              evidence.get('locations', {}))
    _validate_scenario(scenario)
    if flight.dt != DT or getattr(flight.runtime, 'step_count', None) != 0:
        raise ValueError('Start a fresh explicit case at the required 0.01-second interval')
    if not isinstance(metadata.get('routing_model_hash'), str) or len(metadata['routing_model_hash']) != 64:
        raise ValueError('Actual routing-model hash is required')
    started = float(clock())
    if not np.isfinite(started):
        raise ValueError('A finite monotonic clock is required')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root/'scenario.json', scenario.to_dict())
    _write_json(root/'metadata.json', dict(schema=MISSION_CONTRACT, **copy.deepcopy(metadata)))
    state = RoutingState(scenario)
    run_id, step, seq, leg_start = root.name, 0, 0, 0
    completed, intended, decisions, legs = [], [], [], []
    sample = copy.deepcopy(flight.sample())
    initial = copy.deepcopy(sample)
    _write_json(root/'initial_state.json', initial)
    payload = flight.runtime.payload
    _write_json(root/'payload.json', dict(reference_mass_kg=payload.reference_mass_kg,
        mass_ratio=payload.mass_ratio, side_m=payload.side_m, mass_kg=payload.mass, inertia_kg_m2=payload.inertia,
        properties_scope='Declared case; actual reset properties retained in native runtime trace'))
    status, cause, failure_code, failure_row = 'FAILED', None, None, None
    wall_elapsed, distance = 0., 0.
    paths = {name:root/name for name in ('samples.jsonl', 'events.jsonl', 'decisions.jsonl', 'predicates.jsonl')}
    streams = {name:path.open('x') for name, path in paths.items()}

    def elapsed():
        value = float(clock())-started
        if not np.isfinite(value) or value < wall_elapsed-1e-9:
            raise MissionFailure('INVALID_MONOTONIC_CLOCK')
        return value

    def event(event_type, **fields):
        nonlocal seq
        seq += 1
        row = dict(type=event_type, event_seq=seq, run_id=run_id, simulation_step=step,
                   sim_time_s=step*DT, physical_pose=copy.deepcopy(sample.get('drone_xyz')),
                   completed_order=completed.copy(), evidence_kind=kind, **fields)
        _line(streams['events.jsonl'], row)
        return row

    def check_deadlines(before_step=False):
        nonlocal wall_elapsed
        wall_elapsed = elapsed()
        if wall_elapsed >= WALL_SECONDS:
            raise MissionFailure('WALL_TIMEOUT')
        if step*DT > MISSION_SECONDS or (before_step and step*DT >= MISSION_SECONDS):
            raise MissionFailure('MISSION_TIMEOUT')
        if (step-leg_start)*DT > LEG_SECONDS or (before_step and (step-leg_start)*DT >= LEG_SECONDS):
            raise MissionFailure('LEG_TIMEOUT')

    old_handler = None
    if threading.current_thread() is threading.main_thread():
        def interrupted(signum, _frame):
            raise MissionFailure(f'SIGNAL_{signum}')
        old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        _line(streams['samples.jsonl'], dict(kind='INITIAL', simulation_step=0, sim_time_s=0., sample=initial))
        event('MISSION_STARTED')
        if recorder:
            recorder.capture(0, force=True, event_seq=seq)
        while not state.terminal:
            leg_start = step
            check_deadlines(before_step=True)
            action = planner.predict(state.copy())
            if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or
                    action not in range(10) or not state.mask()[action]):
                raise MissionFailure('INVALID_ROUTING_ACTION')
            action = int(action)
            target_id, target = scenario.checkpoint_ids[action], state.scenario.coords[action+1]
            boundary, departure = snapshot(state), state.departure(action)
            intended.append(target_id)
            decision = event('TARGET_SELECTED', target_id=target_id, selected_action=action)
            decision.update(decision_id=f'decision-{len(decisions)}', boundary_snapshot=boundary,
                boundary_hash=content_hash(boundary), selected_action=action, policy_hash=metadata['routing_model_hash'],
                target=copy.deepcopy(target), visited=[scenario.checkpoint_ids[i] for i in range(10) if state.visited[i]],
                eligible=[scenario.checkpoint_ids[i] for i in range(10) if state.mask()[i]],
                feature_vector=state.observation().tolist(), mask=state.mask().tolist(),
                observation_schema=schema_for(state.scenario)['id'])
            decisions.append(decision)
            _line(streams['decisions.jsonl'], decision)
            # Retarget updates commands/history-dependent observations only.
            flight.set_target(target, target_id=target_id)
            gate = SharedSettling(DT)
            gate.update(step*DT, False, target_id)
            event('LEG_STARTED', target_id=target_id)
            if recorder:
                recorder.capture(step, force=True, event_seq=seq)
            while True:
                check_deadlines(before_step=True)
                previous = sample
                try:
                    sample = copy.deepcopy(flight.advance())
                except BaseException:
                    current = _latest_sample(flight)
                    failure_row = dict(kind='ADVANCE_EXCEPTION', simulation_step=None,
                        attempted_step=step+1, last_confirmed_step=step, sim_time_s=None, sample=current)
                    _line(streams['samples.jsonl'], failure_row)
                    raise
                step += 1
                # Raw first, before rotation, finite guards, reward-like decisions
                # or independent settling. NaN survives in the failure record.
                raw = dict(kind='RAW_STEP', simulation_step=step, sim_time_s=step*DT,
                           target_id=target_id, target=copy.deepcopy(target), sample=copy.deepcopy(sample),
                           shared_complete=False)
                failure_row = raw
                _line(streams['samples.jsonl'], raw)
                if (sample.get('step') != step or sample.get('time_s') != step*DT or
                        not np.array_equal(sample.get('effective_target'), target)):
                    raise MissionFailure('SAMPLE_COMMAND_OR_TIME_MISMATCH')
                metrics, unsafe = _measures(sample, target, payload)
                _continuity(previous, sample)
                check_deadlines()
                if unsafe:
                    raise MissionFailure('PHYSICAL_SAFETY_FAILURE')
                complete = gate.update(step*DT, bool(settled(metrics)), target_id)
                distance += float(np.linalg.norm(np.asarray(sample['drone_xyz'])-previous['drone_xyz']))
                # Keep a distinct derived record, so offending raw never depends
                # on successful metrics or a rewrite of prior evidence bytes.
                _line(streams['predicates.jsonl'], dict(type='SAMPLE_PREDICATES', simulation_step=step,
                    sim_time_s=step*DT, target_id=target_id, metrics=metrics,
                    valid_intervals=gate.intervals, dwell_s=gate.dwell_s, shared_complete=complete))
                if recorder:
                    recorder.capture(step, force=complete, event_seq=seq)
                if complete:
                    state.complete(action, departure)
                    completed.append(target_id)
                    legs.append(dict(target_id=target_id, start_step=leg_start, end_step=step,
                        valid_intervals=gate.intervals, dwell_s=gate.dwell_s, terminal_metrics=metrics,
                        nominal_objective=departure))
                    event('SHARED_SETTLED', target_id=target_id, valid_intervals=gate.intervals,
                          dwell_s=gate.dwell_s)
                    if len(completed) == 1 and state.scenario.event:
                        before = state.copy()
                        comparison = event_comparison(before, planner.predict)
                        counterfactual_action = int(planner.predict(before.copy()))
                        update = state.apply_event()
                        if update is None or update['skipped']:
                            event('SCENARIO_EVENT_REJECTED', update=update, comparison=comparison)
                            raise MissionFailure('DECLARED_EVENT_NOT_APPLICABLE')
                        actual_action = int(planner.predict(state.copy()))
                        pose = state.copy()
                        pose.scenario.coords[pose.current] = copy.deepcopy(sample['drone_xyz'])
                        retained = remaining_cost(pose, comparison['no_event_order'])
                        replanned = remaining_cost(pose, comparison['post_event_order'])
                        pose_comparison = dict(
                            actual_pose=copy.deepcopy(sample['drone_xyz']),
                            scope='Same actual pose and post-event remaining tasks; adopted policy suffixes; '
                                  'nominal speed/service, no future events or physical optimality claim',
                            retained_order_post_event=retained, replanned_order_post_event=replanned,
                            improvement=retained['cost']-replanned['cost'], exact=exact_remaining(pose),
                            forced_first=[dict(action=int(a), **exact_remaining(pose, first=int(a)))
                                          for a in np.flatnonzero(pose.mask())])
                        event('PRIORITY_UPDATED' if update['kind'] == 'priority' else 'COST_UPDATED',
                            update=update, comparison=comparison, actual_action=actual_action,
                            pose_cost_comparison=pose_comparison,
                            counterfactual_action=counterfactual_action,
                            actual_features=state.observation().tolist(), counterfactual_features=before.observation().tolist(),
                            mask=state.mask().tolist())
                    break
        status = 'SUCCEEDED'
        failure_row = None
    except BaseException as exc:
        failure_code = exc.code if isinstance(exc, MissionFailure) else type(exc).__name__
        cause = f'{type(exc).__name__}: {exc}'
        if failure_row is None:
            failure_row = dict(kind='TERMINAL', simulation_step=step, sim_time_s=step*DT,
                               sample=_latest_sample(flight))
        _write_json(root/'failure-sample.json', failure_row)
    finally:
        capture_errors = []
        if recorder:
            try:
                recorder.capture(step, force=True, event_seq=seq+1)
            except Exception as exc:
                capture_errors.append(repr(exc))
            try:
                recorder.close()
            except Exception as exc:
                capture_errors.append(repr(exc))
            if capture_errors:
                if status == 'SUCCEEDED':
                    failure_code, cause = 'CAPTURE_FINALIZATION_FAILED', '; '.join(capture_errors)
                status = 'FAILED'
        if status != 'SUCCEEDED' and not (root/'failure-sample.json').exists():
            _write_json(root/'failure-sample.json', dict(kind='TERMINAL', simulation_step=step,
                        sim_time_s=step*DT, sample=_latest_sample(flight)))
        event('MISSION_FINISHED' if status == 'SUCCEEDED' else 'MISSION_FAILED',
              termination_code=failure_code, cause=cause, capture_errors=capture_errors)
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)
        for stream in streams.values():
            stream.close()
    # Exact cost work is offline. Its time does not become additional simulated
    # flight or silently change any earlier decision, event or physical state.
    explanation_started = time.monotonic()
    explanation_error = None
    with (root/'decision_explanations.jsonl').open('x') as stream:
        for decision in decisions:
            try:
                _line(stream, explain_payload_decision(decision, planner.predict))
            except Exception as exc:
                explanation_error = f'{type(exc).__name__}: {exc}'
                status, failure_code, cause = 'FAILED', 'EXPLANATION_RECONSTRUCTION_FAILED', explanation_error
                if not (root/'failure-sample.json').exists():
                    _write_json(root/'failure-sample.json', dict(kind='TERMINAL', simulation_step=step,
                        sim_time_s=step*DT, sample=_latest_sample(flight)))
                seq += 1
                with (root/'events.jsonl').open('a') as event_stream:
                    _line(event_stream, dict(type='MISSION_FAILED', event_seq=seq, run_id=run_id,
                        simulation_step=step, sim_time_s=step*DT, completed_order=completed.copy(),
                        physical_pose=copy.deepcopy(sample.get('drone_xyz')), evidence_kind=kind,
                        termination_code=failure_code, cause=cause, failure_stage='OFFLINE_EXPLANATION'))
                break
    summary = dict(schema=MISSION_CONTRACT, status=status, termination_code=failure_code, termination_cause=cause,
        run_id=run_id, scenario_id=scenario.scenario_id, evidence_kind=kind,
        operational_acceptance='NOT_ESTABLISHED_CPU_FIXTURE' if kind == 'cpu-fixture' else 'REQUIRES_INDEPENDENT_REPLAY',
        completed_order=completed, intended_order=intended, completed_count=len(completed), num_checkpoints=10,
        steps=step, sim_time_s=step*DT, wall_residence_s=wall_elapsed, measured_distance_m=distance,
        explanation_wall_s=time.monotonic()-explanation_started, explanation_error=explanation_error,
        legs=legs, nominal_objective_cost=state.cost,
        telemetry_sha256=sha256(root/'samples.jsonl'), events_sha256=sha256(root/'events.jsonl'),
        predicates_sha256=sha256(root/'predicates.jsonl'),
        decisions_sha256=sha256(root/'decisions.jsonl'), explanations_sha256=sha256(root/'decision_explanations.jsonl'),
        scenario_sha256=sha256(root/'scenario.json'), metadata_sha256=sha256(root/'metadata.json'),
        payload_sha256=sha256(root/'payload.json'), initial_state_sha256=sha256(root/'initial_state.json'),
        failure_sample_sha256=sha256(root/'failure-sample.json') if (root/'failure-sample.json').is_file() else None,
        source_sha256=sha256(__file__))
    _write_json(root/'mission_summary.json', summary)
    return summary
