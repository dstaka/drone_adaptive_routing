#!/usr/bin/env python
"""Run frozen paired scenarios, keeping every attempt and physical failure."""
import argparse
import json
import subprocess
import sys
from adaptive_drone_patrol.util import load_config, unique_dir, json_write, sha256
from adaptive_drone_patrol.scenario import Scenario


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--mode', choices=['planning-only', 'genesis'], required=True)
    p.add_argument('--record', action='store_true')
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--out', required=True)
    p.add_argument('--scenario')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.worker:
        command = [cfg['planner_python'], '-u', '-m', 'adaptive_drone_patrol.planner_ipc',
                   '--model', cfg['routing_model'], '--hash', cfg['routing_model_hash']]
        checked = subprocess.run(command + ['--validate-scenario', a.scenario], capture_output=True, text=True)
        if checked.returncode:
            raise ValueError('Routing preflight failed; retrain for the required contract: ' + checked.stderr)
        contract = json.loads(checked.stdout)['schema']
        s = Scenario.from_dict(load_config(a.scenario))
        if s.num_checkpoints != 3:
            raise ValueError('Ten-point physical integration is pending PR 2; use analytical replay for this PR')
        from adaptive_drone_patrol.flight_agent import FlightSession
        from adaptive_drone_patrol.mission import execute_mission
        from adaptive_drone_patrol.planner_ipc import PlannerClient
        s = Scenario.from_dict(load_config(a.scenario))
        flight = FlightSession(cfg, a.seed, camera=a.record)
        command = [cfg['planner_python'], '-u', '-m', 'adaptive_drone_patrol.planner_ipc',
                   '--model', cfg['routing_model'], '--hash', cfg['routing_model_hash']]
        with PlannerClient(command, cfg['routing_model_hash'], schema=contract) as planner:
            result = execute_mission(flight, planner, s, a.out,
                       {'flight_checkpoint_hash': cfg['flight_checkpoint_hash'],
                        'routing_model_hash': cfg['routing_model_hash'], 'flight_config_hash': cfg['flight_cfg_hash'],
                        'execution_seed': a.seed, 'hover_source_hash': cfg['hover_source_hash'],
                        'flight_acceptance_config_hash': sha256(cfg['flight_eval_config']),
                        'source_revision': cfg.get('source_revision', 'uncommitted-reviewed-source'),
                        'environment_reference': cfg.get('environment_reference', 'requirements/flight-linux.lock.txt')}, record=a.record, limits=load_config(cfg['flight_eval_config']))
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'SUCCEEDED' else 2)
    if a.trials <= 0:
        p.error('--trials must be positive')
    out = unique_dir(a.out)
    results = []
    for scenario_path in cfg['scenario_files']:
        scenario = Scenario.from_dict(load_config(scenario_path))
        for trial in range(a.trials):
            dest = out / f'{scenario.scenario_id}_seed_{trial + 1}'
            if a.mode == 'planning-only':
                from adaptive_drone_patrol.benchmark import evaluate_scenario
                from adaptive_drone_patrol.routing_agent import RoutingAgent
                agent = RoutingAgent(cfg['routing_model'], cfg['routing_model_hash'])
                dest.mkdir()
                result = evaluate_scenario(scenario, agent.predict)
                result.update(status='PLANNING_ONLY', evidence_kind='analytical', physical_completion=None)
                json_write(dest / 'planning.json', result)
                results.append(result)
            else:
                cmd = [sys.executable, '-u', __file__, '--config', a.config, '--mode', 'genesis', '--worker',
                       '--scenario', scenario_path, '--seed', str(trial + 1), '--out', str(dest)]
                if a.record:
                    cmd.append('--record')
                with (out / f'{dest.name}.log').open('w') as log:
                    try:
                        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                summary_path = dest / 'mission_summary.json'
                result = json.loads(summary_path.read_text()) if summary_path.exists() else {
                    'run_id': dest.name, 'scenario_id': scenario.scenario_id, 'status': 'FAILED',
                    'completed_count': 0, 'termination_cause': f'Worker exit {code}; see log'}
                result['worker_exit_code'] = code
                results.append(result)
                print(json.dumps({'run_id': dest.name, 'status': result['status'],
                                  'completed_count': result.get('completed_count')}), flush=True)
            json_write(out / 'all_attempts.json', results)
    if a.mode == 'genesis' and any(not any(r['status'] == 'SUCCEEDED' and r['scenario_id'] == s
                                         for r in results) for s in ['normal', 'emergency', 'cost_shift']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
