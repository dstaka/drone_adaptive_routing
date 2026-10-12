#!/usr/bin/env python
"""Measure one/two/sequential external-goal arrival without intermediate resets."""
import argparse
import json
import subprocess
import sys
import numpy as np
from adaptive_drone_patrol.util import load_config, unique_dir, json_write


def trial(runtime, limits, out, seed, goal_count):
    from adaptive_drone_patrol.flight_agent import FlightSession
    from adaptive_drone_patrol.patrol_runner import ArrivalGate, FlightSample
    from adaptive_drone_patrol.recorder import FrameRecorder
    out = unique_dir(out)
    flight = FlightSession(runtime, seed, camera=True)
    goals = [[.6, 0, 1], [-.3, .519615242, 1], [-.3, -.519615242, 1]][:goal_count]
    recorder = FrameRecorder(flight.cam, out / 'raw.mp4', out / 'frames.csv', flight.dt)
    rows, arrivals, step = [], [], 0
    status, cause = 'FAILED', None
    try:
        for goal in goals:
            flight.set_target(goal)
            gate = ArrivalGate(flight.dt, max_tilt_deg=limits['max_tilt_deg'])
            leg_start = step
            for _ in range(int(limits['max_leg_s'] / flight.dt)):
                result = flight.advance()
                step += 1
                row = {'simulation_step': step, 'sim_time_s': step * flight.dt, 'goal': goal, **result}
                rows.append(row)
                recorder.capture(step)
                events = gate.update(FlightSample(result['xyz'], result['velocity'], result['done'], result['reset'],
                                                  result['tilt_deg']), goal)
                if 'WAYPOINT_REACHED' in events:
                    arrivals.append({'goal': goal, 'sim_time_s': step * flight.dt,
                                     'duration_s': (step - leg_start) * flight.dt,
                                     'residual_m': float(np.linalg.norm(np.asarray(result['xyz']) - goal))})
                    break
            else:
                raise RuntimeError('External goal timeout')
        status = 'SUCCEEDED'
    except Exception as exc:
        cause = str(exc)
    finally:
        recorder.close()
    json_write(out / 'telemetry.json', rows)
    result = {'status': status, 'seed': seed, 'goal_count': goal_count, 'arrivals': arrivals,
              'termination_cause': cause, 'completed_count': len(arrivals),
              'flight_checkpoint_hash': runtime['flight_checkpoint_hash']}
    json_write(out / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--execution', required=True)
    p.add_argument('--mode', choices=['sequential-targets'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--goal-count', type=int, choices=[1, 2, 3], default=3)
    a = p.parse_args()
    runtime, limits = load_config(a.execution), load_config(a.config)
    if a.worker:
        r = trial(runtime, limits, a.out, a.seed, a.goal_count)
        print(json.dumps(r), flush=True)
        raise SystemExit(0 if r['status'] == 'SUCCEEDED' else 2)
    out = unique_dir(a.out)
    results = []
    for goals, seed in [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)]:
        dest = out / f'goals_{goals}_seed_{seed}'
        cmd = [sys.executable, '-u', __file__, '--config', a.config, '--execution', a.execution,
               '--mode', a.mode, '--out', str(dest), '--worker', '--seed', str(seed), '--goal-count', str(goals)]
        with (out / f'{dest.name}.log').open('w') as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
        r = json.loads((dest / 'summary.json').read_text()) if (dest / 'summary.json').exists() else {
            'status': 'FAILED', 'completed_count': 0, 'goal_count': goals, 'seed': seed,
            'termination_cause': f'Worker exit {code}'}
        results.append(r)
        json_write(out / 'all_trials.json', results)
        print(json.dumps(r), flush=True)
        if r['status'] != 'SUCCEEDED':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
