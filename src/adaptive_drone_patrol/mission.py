"""Same-run physical telemetry and arrival-boundary learned patrol execution."""

import copy
import csv
import json
import time
import signal
import threading
from datetime import datetime, timezone
import numpy as np
from .objective import RoutingState, schema_for
from .patrol_runner import FlightSample, ArrivalGate
from .analytical import snapshot as boundary_snapshot, content_hash, event_comparison
from .util import unique_dir, json_write
from .recorder import FrameRecorder


def execute_mission(flight, planner, scenario, out, metadata, record=True, limits=None):
    limits = limits or {}
    if scenario.return_to_base:
        raise ValueError('Physical return_to_base is unsupported; use false for patrols')
    if any(not np.isfinite(limits.get(key, default)) or limits.get(key, default) <= 0
           for key, default in [('max_leg_s', 30), ('max_mission_s', 120), ('max_wall_s', 600)]):
        raise ValueError('Mission timeouts must be positive and finite')
    ArrivalGate(flight.dt, radius=limits.get('arrival_radius_m', .2),
                max_speed=limits.get('arrival_speed_mps', .3), hold_s=limits.get('arrival_hold_s', .2),
                service_s=limits.get('inspection_dwell_s', .5), floor=limits.get('floor_m', .1),
                max_tilt_deg=limits.get('max_tilt_deg', 60))
    intro_s = limits.get('intro_s', 0.)
    if not np.isfinite(intro_s) or intro_s < 0:
        raise ValueError('Intro duration must be finite and nonnegative')
    scenario.validate_physical(limits.get('arrival_radius_m', .2))
    out = unique_dir(out)
    state = RoutingState(scenario)
    ids, n = scenario.checkpoint_ids, scenario.num_checkpoints
    nodes = ["S"] + ids
    initial_sample = flight.sample() if hasattr(flight, 'sample') else {
        'xyz':scenario.coords[0], 'velocity':[0.,0.,0.], 'quaternion_wxyz':[1.,0.,0.,0.]}
    physical_pose = initial_sample['xyz']
    run_id = out.name
    json_write(out / 'scenario.json', scenario.to_dict())
    json_write(out / 'run_metadata.json', {'schema_version': 1, 'run_id': run_id,
               'scenario_id': scenario.scenario_id, 'seed': scenario.seed,
               'started_utc': datetime.now(timezone.utc).isoformat(), **metadata})
    events_file = (out / 'events.jsonl').open('x')
    decisions_file = (out / 'decisions.jsonl').open('x')
    telemetry_file = (out / 'telemetry.csv').open('x', newline='')
    seq, step = 0, 0
    start = time.monotonic()
    dt = flight.dt
    completed, intended, legs = [], [], []
    trajectory_distance = 0.
    previous_xyz = np.asarray(physical_pose)
    frame_recorder = FrameRecorder(flight.cam, out / 'raw.mp4', out / 'frames.csv', dt) if record else None
    columns = ['run_id', 'simulation_step', 'sim_time_s', 'wall_elapsed_s', 'x', 'y', 'z',
               'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'speed_mps', 'target_id',
               'target_x', 'target_y', 'target_z', 'distance_to_goal_m', 'actions', 'dwell_s',
               'runner_state', 'reset', 'done', 'safety_failure', 'frame_index', 'tilt_deg', 'relative_observation_absmax']
    writer = csv.DictWriter(telemetry_file, fieldnames=columns)
    writer.writeheader()

    def event(kind, **fields):
        nonlocal seq
        seq += 1
        item = {'event_seq': seq, 'type': kind, 'run_id': run_id, 'simulation_step': step,
                'sim_time_s': step * dt, 'model_time_s': state.elapsed,
                'marker_state': {'visited': [ids[i] for i in range(n) if state.visited[i]],
                                 'selected': intended[-1] if intended and intended[-1] not in completed else None,
                                 'priorities': state.scenario.priorities.copy()},
                **copy.deepcopy(fields)}
        events_file.write(json.dumps(item, allow_nan=False) + '\n')
        events_file.flush()
        return item

    status, cause = 'FAILED', None
    def shutdown(signum, _frame):
        raise RuntimeError(f'Mission interrupted by signal {signum}')
    old_handler = signal.signal(signal.SIGTERM, shutdown) if threading.current_thread() is threading.main_thread() else None
    event('SCENARIO_START', scenario_id=scenario.scenario_id)
    json_write(out / 'initial_state.json', {'simulation_step': 0, 'sim_time_s': 0., 'xyz': list(physical_pose),
               'velocity':initial_sample['velocity'], 'quaternion_wxyz':initial_sample['quaternion_wxyz']})
    if getattr(flight, 'scene_manifest', None):
        json_write(out / 'scene_manifest.json', {**flight.scene_manifest,
                   'verified_runtime_positions': flight.rendered_marker_positions})
    try:
        if frame_recorder:
            frame_recorder.capture(0, force=True, event_seq=seq)
        if intro_s:
            flight.set_target(scenario.coords[0])
            event('INTRO_STARTED', duration_s=intro_s)
            safety = ArrivalGate(dt, floor=limits.get('floor_m', .1),
                                 max_tilt_deg=limits.get('max_tilt_deg', 60))
            for _ in range(int(np.ceil(intro_s/dt))):
                result = flight.advance()
                step += 1
                physical_pose = result['xyz']
                if not np.allclose(result['effective_target'], scenario.coords[0], atol=1e-6):
                    raise RuntimeError('Intro effective target mismatch')
                safety_error = None
                try:
                    safety.update(FlightSample(result['xyz'], result['velocity'], result['done'],
                                               result['reset'], result['tilt_deg']), scenario.coords[0])
                except RuntimeError as exc:
                    safety_error = exc
                xyz = np.asarray(physical_pose)
                trajectory_distance += float(np.linalg.norm(xyz-previous_xyz))
                previous_xyz = xyz
                frame = frame_recorder.capture(step, event_seq=seq) if frame_recorder else None
                row = {'run_id':run_id, 'simulation_step':step, 'sim_time_s':step*dt,
                       'wall_elapsed_s':time.monotonic()-start, 'target_id':'S', 'dwell_s':0.,
                       'runner_state':'INTRO', 'reset':result['reset'], 'done':result['done'], 'safety_failure':bool(safety_error),
                       'frame_index':frame if frame is not None else '', 'actions':json.dumps(result['actions']),
                       'speed_mps':float(np.linalg.norm(result['velocity'])),
                       'distance_to_goal_m':float(np.linalg.norm(xyz-scenario.coords[0])),
                       'tilt_deg':result['tilt_deg'],
                       'relative_observation_absmax':result.get('relative_observation_absmax','')}
                for keys,values in [(['x','y','z'],physical_pose), (['vx','vy','vz'],result['velocity']),
                                    (['qw','qx','qy','qz'],result['quaternion_wxyz']),
                                    (['target_x','target_y','target_z'],scenario.coords[0])]:
                    row.update(zip(keys,values))
                writer.writerow(row)
                if safety_error:
                    raise safety_error
            event('INTRO_FINISHED', actual_xyz=physical_pose)
        while not state.terminal:
            a = planner.predict(state)
            if isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer)) or a not in range(n) or not state.mask()[a]:
                raise RuntimeError('Invalid learned routing action')
            target = np.asarray(scenario.coords[a + 1])
            snapshot = state.departure(a)
            intended.append(ids[a])
            decision = event('TARGET_SELECTED', action=a, target_id=ids[a], target=target.tolist())
            frozen_boundary = boundary_snapshot(state)
            decision.update(decision_id=f'decision-{len(intended)-1}', boundary_snapshot=frozen_boundary,
                            boundary_hash=content_hash(frozen_boundary), physical_pose=list(physical_pose),
                            checkpoint_ids=ids, evidence_kind=metadata.get('evidence_kind', 'mock'),
                            decision_index=len(intended) - 1, current_node=nodes[state.current],
                            visited=[ids[i] for i in range(n) if state.visited[i]],
                            eligible=[ids[i] for i in range(n) if not state.visited[i]],
                            feature_vector=state.observation().tolist(), observation_schema=schema_for(scenario)["id"],
                            mask=state.mask().tolist(), priorities=state.scenario.priorities.copy(),
                            cost_snapshot=copy.deepcopy(state.scenario.multipliers),
                            selected_action=a, policy_hash=metadata.get('routing_model_hash'),
                            inference_mode='deterministic', event_source='trained_policy')
            decisions_file.write(json.dumps(decision) + '\n')
            decisions_file.flush()
            flight.set_target(target.tolist())
            event('LEG_STARTED', target_id=ids[a], departure_snapshot=snapshot)
            if frame_recorder:
                frame_recorder.capture(step, force=True, event_seq=seq)
            leg_start = step
            gate = ArrivalGate(dt, radius=limits.get('arrival_radius_m', .2),
                               max_speed=limits.get('arrival_speed_mps', .3),
                               hold_s=limits.get('arrival_hold_s', .2),
                               service_s=limits.get('inspection_dwell_s', .5),
                               floor=limits.get('floor_m', .1), max_tilt_deg=limits.get('max_tilt_deg', 60))
            arrival_time = None
            errors, speeds, tilts, altitudes, dwell_errors, dwell_speeds = [], [], [], [], [], []
            while True:
                if ((step - leg_start) * dt >= limits.get('max_leg_s', 30)
                        or step * dt >= limits.get('max_mission_s', 120)
                        or time.monotonic() - start >= limits.get('max_wall_s', 600)):
                    raise RuntimeError('Flight timeout')
                result = flight.advance()
                step += 1
                if not np.allclose(result['effective_target'], target, atol=1e-6):
                    raise RuntimeError('Effective target mismatch')
                sample = FlightSample(result['xyz'], result['velocity'], result['done'], result['reset'],
                                      result['tilt_deg'])
                if sample.reset or sample.done:
                    event('RESET_DETECTED', pose=sample.xyz, done=sample.done)
                flags = None
                safety_error = None
                try:
                    flags = gate.update(sample, target)
                except RuntimeError as exc:
                    safety_error = exc
                xyz = np.asarray(sample.xyz)
                physical_pose = sample.xyz
                errors.append(float(np.linalg.norm(xyz - target)))
                speeds.append(float(np.linalg.norm(sample.velocity)))
                tilts.append(float(sample.tilt_deg))
                altitudes.append(float(xyz[2]))
                if gate.steps:
                    dwell_errors.append(errors[-1])
                    dwell_speeds.append(speeds[-1])
                else:
                    dwell_errors, dwell_speeds = [], []
                if previous_xyz is not None and not sample.reset and not sample.done and np.isfinite(xyz).all():
                    trajectory_distance += float(np.linalg.norm(xyz - previous_xyz))
                previous_xyz = xyz.copy() if not sample.reset else None
                frame = frame_recorder.capture(step, force=bool(flags and 'WAYPOINT_REACHED' in flags), event_seq=seq) if frame_recorder else None
                runner_state = 'INSPECT' if gate.steps >= gate.hold_steps else ('ARRIVAL_HOLD' if gate.steps else 'FLY')
                row = dict(zip(columns[:4], [run_id, step, step * dt, time.monotonic() - start]))
                row.update(dict(zip(['x', 'y', 'z'], sample.xyz)))
                row.update(dict(zip(['qw', 'qx', 'qy', 'qz'], result['quaternion_wxyz'])))
                row.update(dict(zip(['vx', 'vy', 'vz'], sample.velocity)))
                row.update(speed_mps=float(np.linalg.norm(sample.velocity)), target_id=ids[a],
                           target_x=float(target[0]), target_y=float(target[1]), target_z=float(target[2]),
                           distance_to_goal_m=float(np.linalg.norm(xyz - target)), actions=json.dumps(result['actions']),
                           dwell_s=gate.steps * dt, runner_state=runner_state, reset=sample.reset, done=sample.done,
                           tilt_deg=sample.tilt_deg, relative_observation_absmax=result.get('relative_observation_absmax', ''),
                           safety_failure=bool(safety_error), frame_index=frame if frame is not None else '')
                writer.writerow(row)
                if safety_error:
                    raise safety_error
                for flag in flags:
                    event(flag, target_id=ids[a], actual_xyz=sample.xyz,
                          residual_m=row['distance_to_goal_m'], speed_mps=row['speed_mps'])
                    if flag == 'ARRIVAL_CONFIRMED':
                        arrival_time = step * dt
                if 'WAYPOINT_REACHED' in flags:
                    state.complete(a, snapshot)
                    completed.append(ids[a])
                    legs.append({'target_id': ids[a], 'departure_sim_s': leg_start * dt,
                                 'arrival_sim_s': arrival_time, 'service_sim_s': step * dt,
                                 'arrival_residual_m': row['distance_to_goal_m'],
                                 'tracking_max_m': max(errors), 'tracking_p95_m': float(np.percentile(errors, 95)),
                                 'max_speed_mps': max(speeds), 'max_tilt_deg': max(tilts),
                                 'min_altitude_m': min(altitudes), 'max_altitude_m': max(altitudes),
                                 'dwell_max_residual_m': max(dwell_errors),
                                 'dwell_max_speed_mps': max(dwell_speeds), 'dwell_s': gate.steps * dt, **snapshot})
                    if state.scenario.event and state.visited.sum() == 1:
                        before = state.copy()
                        comparison = event_comparison(before, planner.predict)
                        hypothetical_action = planner.predict(before)
                        update = state.apply_event()
                        actual_action = planner.predict(state.copy())
                        event('PRIORITY_UPDATED' if update['kind'] == 'priority' else 'COST_UPDATED', update=update,
                              actual_action=actual_action, counterfactual_action=hypothetical_action, comparison=comparison,
                              actual_features=state.observation().tolist(), counterfactual_features=before.observation().tolist(),
                              mask=state.mask().tolist(), evidence_type='planner_counterfactual')
                        if actual_action == hypothetical_action:
                            event('NO_POLICY_CHANGE')
                    break
        status = 'SUCCEEDED'
    except BaseException as exc:
        cause = f'{type(exc).__name__}: {exc}'
        event('LEG_FAILED', cause=cause, completed_order=completed)
    finally:
        if frame_recorder:
            try:
                frame_recorder.capture(step, force=True, event_seq=seq+1)
            except Exception as exc:
                status, cause = 'FAILED', f'Final capture failed: {exc}'
            try:
                frame_recorder.close()
            except Exception as exc:
                status, cause = 'FAILED', f'Recording finalization failed: {exc}'
        event('MISSION_FINISHED' if status == 'SUCCEEDED' else 'MISSION_FAILED',
              cause=cause, completed_order=completed)
        telemetry_file.close()
        events_file.close()
        decisions_file.close()
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)
    result = {'schema_version': 1, 'run_id': run_id, 'scenario_id': scenario.scenario_id, 'seed': scenario.seed,
              'status': status, 'termination_cause': cause, 'completed_order': completed, 'intended_order': intended,
              'checkpoint_ids': ids, 'num_checkpoints': n, 'evidence_kind': metadata.get('evidence_kind', 'mock'),
              'failure_count': int(status != 'SUCCEEDED'),
              'completed_count': len(completed), 'measured_duration_s': step * dt,
              'measured_distance_m': trajectory_distance, 'wall_elapsed_s': time.monotonic() - start,
              'analytical_cost': state.cost, 'priority_completion_penalty': state.penalty,
              'nominal_distance_m': state.distance, 'legs': legs, 'events': state.events, **metadata}
    json_write(out / 'mission_summary.json', result)
    return result
