"""Failure-path regressions discovered by the independent final review."""
import json
import signal
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
from adaptive_drone_patrol import evidence, jobs, video
from adaptive_drone_patrol.patrol_runner import ArrivalGate
from adaptive_drone_patrol.mission import execute_mission
from adaptive_drone_patrol.scenario import Scenario
from test_patrol import FakeFlight, FixedPlanner, mission_scenario


def test_interrupted_supervisor_reaps_child_before_releasing_ledger(tmp_path, monkeypatch):
    proc = Mock(pid=987654)
    proc.wait.side_effect = [KeyboardInterrupt(), -signal.SIGTERM]
    proc.poll.side_effect = [None, -signal.SIGTERM]
    monkeypatch.setattr(jobs.subprocess, 'Popen', Mock(return_value=proc))
    kills = Mock()
    monkeypatch.setattr(jobs.os, 'killpg', kills)
    with pytest.raises(KeyboardInterrupt):
        jobs.run_job(['worker'], tmp_path / 'job', tmp_path / 'ledger.json', 1)
    assert kills.call_args_list[0].args == (proc.pid, signal.SIGTERM)
    assert kills.call_args_list[-1].args == (proc.pid, signal.SIGKILL)
    assert proc.wait.call_count == 2
    ledger = json.loads((tmp_path / 'ledger.json').read_text())
    assert not ledger.get('active_job') and ledger['used_s'] > 0


@pytest.mark.parametrize('field,value', [('radius', np.nan), ('max_speed', np.nan),
                                        ('hold_s', 0), ('service_s', np.inf), ('floor', np.nan)])
def test_nonfinite_or_nonpositive_acceptance_is_rejected(field, value):
    with pytest.raises(ValueError):
        ArrivalGate(.01, **{field: value})


@pytest.mark.parametrize('change', [{'event': {'kind': 'priority', 'after': 1, 'checkpoint': 1.0, 'value': 5}},
    {'event': {'kind': 'cost', 'after': 1, 'edges': [[1.0, 2]], 'value': 5}},
    {'return_to_base': np.nan}, {'event': []}, {'event': {'kind': 'priority', 'after': True, 'checkpoint': 1, 'value': 5}}])
def test_invalid_scenario_types_are_rejected(change):
    data = mission_scenario().to_dict()
    data.update(change)
    with pytest.raises(ValueError):
        Scenario.from_dict(data)


def test_physical_return_is_explicitly_rejected_before_execution(tmp_path):
    scenario = mission_scenario()
    scenario.return_to_base = True
    with pytest.raises(ValueError, match='return'):
        execute_mission(FakeFlight(), FixedPlanner(), scenario, tmp_path / 'return', {}, record=False)
    assert not (tmp_path / 'return').exists()


def test_geometry_uses_the_configured_acceptance_radius(tmp_path):
    with pytest.raises(ValueError, match='separation|radius'):
        execute_mission(FakeFlight(), FixedPlanner(), mission_scenario(), tmp_path / 'wide', {},
                        record=False, limits={'arrival_radius_m': .6})


def test_recording_failure_emits_failed_terminal_event(tmp_path, monkeypatch):
    recorder = Mock()
    recorder.capture.return_value = None
    recorder.close.side_effect = RuntimeError('encoder failed')
    monkeypatch.setattr('adaptive_drone_patrol.mission.FrameRecorder', Mock(return_value=recorder))
    flight = FakeFlight()
    flight.cam = None
    result = execute_mission(flight, FixedPlanner(), mission_scenario(), tmp_path / 'run', {}, record=True)
    events = [json.loads(x) for x in (tmp_path / 'run' / 'events.jsonl').read_text().splitlines()]
    assert result['status'] == 'FAILED'
    assert events[-1]['type'] == 'MISSION_FAILED'
    assert not any(e['type'] == 'MISSION_FINISHED' for e in events)


def test_successful_run_requires_matching_terminal_and_telemetry(tmp_path):
    run = tmp_path / 'run'
    execute_mission(FakeFlight(), FixedPlanner(), mission_scenario(), run, {}, record=False)
    assert evidence.validate_run(run)['status'] == 'SUCCEEDED'
    events = [json.loads(x) for x in (run / 'events.jsonl').read_text().splitlines()]
    events[-1]['type'] = 'MISSION_FAILED'
    (run / 'events.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in events))
    with pytest.raises(ValueError, match='terminal|status'):
        evidence.validate_run(run)
    events[-1]['type'] = 'MISSION_FINISHED'
    (run / 'events.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in events))
    (run / 'telemetry.csv').unlink()
    with pytest.raises(ValueError, match='telemetry'):
        evidence.validate_run(run)


def test_strict_delivery_rejects_a_valid_but_partial_manifest(tmp_path):
    with pytest.raises(ValueError, match='required|deliver'):
        evidence.verify_manifest({'artifacts': []}, tmp_path, require_local_artifacts=True)


def test_hud_state_uses_capture_step_during_slow_nonuniform_playback():
    telemetry = [{'simulation_step': '410', 'sim_time_s': '4.10'},
                 {'simulation_step': '411', 'sim_time_s': '4.11'},
                 {'simulation_step': '413', 'sim_time_s': '4.13'}]
    assert video.capture_sample_index(telemetry, {'simulation_step': '410', 'sim_time_s': '4.10'}) == 0
    assert video.capture_sample_index(telemetry, {'simulation_step': '411', 'sim_time_s': '4.11'}) == 1
    with pytest.raises(ValueError):
        video.capture_sample_index(telemetry, {'simulation_step': '412', 'sim_time_s': '4.12'})


def test_video_collision_rejects_complete_output_set_before_reading_inputs(tmp_path):
    (tmp_path / 'timeline.json').write_text('preserve original mapping')
    with pytest.raises(FileExistsError):
        video.compose(tmp_path / 'absent', tmp_path / 'submission.mp4', tmp_path / 'font.ttf')
    assert (tmp_path / 'timeline.json').read_text() == 'preserve original mapping'


def test_transfer_excludes_all_private_execution_variants(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('transfer', Path(__file__).parents[1] / 'scripts/transfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    Path('configs').mkdir()
    for name in ['execution.yaml', 'execution_new.yaml', 'execution_trained.yaml', 'execution.example.yaml']:
        Path('configs', name).write_text('metadata')
    assert [str(p) for p in module.collect_sources()] == ['configs/execution.example.yaml']


def test_rounded_routing_rollouts_respect_actual_step_cap(tmp_path, monkeypatch):
    from adaptive_drone_patrol.routing_agent import train_one
    monkeypatch.setattr('adaptive_drone_patrol.routing_agent.torch.set_num_threads', Mock(side_effect=RuntimeError('must reject before training')))
    with pytest.raises(ValueError, match='rounded|actual'):
        train_one(42, tmp_path / 'oversize', static_steps=100000, event_steps=100000)
    assert not (tmp_path / 'oversize_run').exists()


def test_unreaped_child_keeps_ledger_blocked(tmp_path, monkeypatch):
    proc = Mock(pid=987654)
    proc.wait.side_effect = KeyboardInterrupt()
    monkeypatch.setattr(jobs.subprocess, 'Popen', Mock(return_value=proc))
    monkeypatch.setattr(jobs.os, 'killpg', Mock(side_effect=PermissionError('cleanup denied')))
    with pytest.raises(KeyboardInterrupt):
        jobs.run_job(['worker'], tmp_path / 'job', tmp_path / 'ledger.json', 1)
    ledger = json.loads((tmp_path / 'ledger.json').read_text())
    assert ledger['active_job']['requires_reconciliation']
    with pytest.raises(RuntimeError, match='Unreconciled'):
        jobs.run_job(['worker'], tmp_path / 'second', tmp_path / 'ledger.json', 1)


def test_worker_sigterm_finalizes_failed_evidence(tmp_path):
    import os
    class InterruptedFlight(FakeFlight):
        def advance(self):
            os.kill(os.getpid(), signal.SIGTERM)
            return super().advance()
    result = execute_mission(InterruptedFlight(), FixedPlanner(), mission_scenario(), tmp_path / 'run', {}, record=False)
    assert result['status'] == 'FAILED' and result['completed_count'] == 0
    assert 'signal' in result['termination_cause']
    assert json.loads((tmp_path / 'run/events.jsonl').read_text().splitlines()[-1])['type'] == 'MISSION_FAILED'


def test_real_supervisor_sigterm_stops_owned_child_and_accounts_cleanup(tmp_path):
    import os
    import subprocess
    import sys
    import time
    child_pid_file = tmp_path / 'child.pid'
    child_code = f'from pathlib import Path; import os,time; Path({str(child_pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)'
    command = [sys.executable, '-m', 'adaptive_drone_patrol.jobs', '--out', str(tmp_path / 'job'),
               '--ledger', str(tmp_path / 'ledger.json'), '--timeout', '50', '--', sys.executable, '-c', child_code]
    supervisor = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert child_pid_file.exists(), 'Owned child did not start'
        child_pid = int(child_pid_file.read_text())
        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=5) != 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        data = json.loads((tmp_path / 'ledger.json').read_text())
        assert not data.get('active_job') and data['used_s'] > 0
        assert data['jobs'][-1]['status'] == 'FAILED'
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=5)
