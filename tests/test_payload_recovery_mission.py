"""Mission logic on explicit CPU boundary doubles, never physical flight proof."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from adaptive_drone_patrol.payload_contract import Payload
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.util import load_config


def scenario(kind='normal'):
    return Scenario.from_dict(load_config(f'configs/demo_scenarios_dispersed/{kind}.yaml'))


class Planner:
    def predict(self, state):
        eligible = np.flatnonzero(state.mask()).tolist()
        if state.scenario.priorities[2] == 100 and 2 in eligible:
            return 2
        return eligible[0]


class CpuFlight:
    """Move only through advance; set_target never changes body/history state."""
    dt = .01

    def __init__(self, *, cruise=.05, pause=0, offset=.03, override=None):
        self.position = np.array([offset, 0., 1.])
        self.target = np.array([0., 0., 1.])
        self.runtime = SimpleNamespace(payload=Payload(.02700047753751278, .2, .045), step_count=0)
        self.steps, self.target_steps, self.resets = 0, 0, 1
        self.targets, self.history_at_target = [], []
        self.actions = [.2, -.1, .3, -.2]
        self.cruise, self.pause, self.override = cruise, pause, override
        self.stop_short = 0.
        self.current = self._state(np.zeros(3))

    def _state(self, velocity):
        d = self.position
        p = d+[0., 0., -.3425]
        return dict(drone_xyz=d.tolist(), package_xyz=p.tolist(),
            drone_quaternion_wxyz=[1., 0., 0., 0.], package_quaternion_wxyz=[1., 0., 0., 0.],
            drone_velocity=velocity.tolist(), package_velocity=velocity.tolist(),
            drone_angular_velocity=[0., 0., 0.], package_angular_velocity=[0., 0., 0.],
            attachment_a_world=(d+[0., 0., -.02]).tolist(), attachment_b_world=(p+[0., 0., .0225]).tolist(),
            attachment_a_velocity=velocity.tolist(), attachment_b_velocity=velocity.tolist(),
            vehicle_collision_aabbs={'base_link':[(d-.04).tolist(), (d+.04).tolist()]},
            contacts=False, done=False, reset=False, actions=self.actions.copy(),
            effective_target=self.target.tolist(), complete=True, shared_settled=True,
            step=self.steps, time_s=self.steps*.01)

    def sample(self):
        return copy.deepcopy(self.current)

    def set_target(self, target, target_id=None):
        self.target = np.asarray(target, float)
        self.targets.append(target_id)
        self.history_at_target.append(self.actions.copy())
        self.target_steps = 0

    def advance(self):
        self.steps += 1
        self.target_steps += 1
        delta = self.target+[self.stop_short, 0., 0.]-self.position
        movement = delta*min(1., self.cruise/max(1e-12, np.linalg.norm(delta)))
        self.position += movement
        velocity = movement/.01
        self.current = self._state(velocity)
        if self.target_steps <= self.pause and np.linalg.norm(movement) < 1e-12:
            self.current['package_velocity'] = [.11, 0., 0.]
            self.current['attachment_b_velocity'] = [.11, 0., 0.]
        if self.override:
            self.override(self)
        return copy.deepcopy(self.current)


def rows(root, name):
    return [json.loads(line) for line in (root/name).read_text().splitlines()]


def run(tmp_path, flight=None, kind='normal', clock=None):
    from adaptive_drone_patrol.payload_recovery_mission import execute_payload_mission
    out = tmp_path/kind
    flight = flight or CpuFlight()
    kw = {} if clock is None else {'clock':clock}
    result = execute_payload_mission(flight, Planner(), scenario(kind), out,
        {'evidence_kind':'cpu-fixture', 'routing_model_hash':'a'*64}, **kw)
    return flight, out, result


def test_all_ten_use_actual_shared_intervals_without_history_or_body_resets(tmp_path):
    flight, out, result = run(tmp_path)
    assert result['status'] == 'SUCCEEDED'
    assert result['completed_order'] == list('ABCDEFGHIJ')
    assert result['completed_count'] == 10 and result['operational_acceptance'] == 'NOT_ESTABLISHED_CPU_FIXTURE'
    events = rows(out, 'events.jsonl')
    completions = [e for e in events if e['type'] == 'SHARED_SETTLED']
    assert len(completions) == 10
    starts = [0]+[e['simulation_step'] for e in completions[:-1]]
    assert all(e['simulation_step']-start >= 101 and e['valid_intervals'] >= 100
               for e, start in zip(completions, starts))
    assert flight.targets == list('ABCDEFGHIJ') and flight.resets == 1
    assert flight.history_at_target == [[.2, -.1, .3, -.2]]*10
    samples = rows(out, 'samples.jsonl')
    assert len(samples) == result['steps']+1
    assert samples[-1]['sample']['drone_xyz'] == flight.current['drone_xyz']
    assert result['telemetry_sha256'] and result['events_sha256']


@pytest.mark.parametrize('kind', ['emergency', 'cost_shift'])
def test_event_is_exactly_once_after_first_shared_completion_and_before_next_decision(tmp_path, kind):
    _, out, result = run(tmp_path, kind=kind)
    assert result['status'] == 'SUCCEEDED'
    events, decisions = rows(out, 'events.jsonl'), rows(out, 'decisions.jsonl')
    shared = next(e for e in events if e['type'] == 'SHARED_SETTLED')
    changes = [e for e in events if e['type'] in ('PRIORITY_UPDATED', 'COST_UPDATED')]
    assert len(changes) == 1
    event = changes[0]
    assert event['simulation_step'] == shared['simulation_step']
    assert shared['event_seq'] < event['event_seq'] < decisions[1]['event_seq']
    assert event['physical_pose'] == shared['physical_pose']
    before = decisions[0]['boundary_snapshot']['scenario']
    after = decisions[1]['boundary_snapshot']['scenario']
    if kind == 'emergency':
        assert before['priorities'][2] == 43 and after['priorities'][2] == 100
        assert decisions[1]['target_id'] == 'C'
        assert event['update']['old'] == 43 and event['update']['value'] == 100
    else:
        expected = copy.deepcopy(before['multipliers'])
        expected[6][1] = 5.
        assert after['multipliers'] == expected and after['multipliers'][1][6] == 1.
    assert decisions[1]['visited'] == ['A'] and 'A' not in decisions[1]['eligible']


def test_package_predicate_resets_shared_dwell_even_when_core_claims_completion(tmp_path):
    flight = CpuFlight(pause=150)
    _, out, result = run(tmp_path, flight=flight)
    assert result['status'] == 'SUCCEEDED'
    complete = next(e for e in rows(out, 'events.jsonl') if e['type'] == 'SHARED_SETTLED')
    assert complete['simulation_step'] >= 251
    raw = rows(out, 'samples.jsonl')
    assert raw[1]['sample']['complete'] is True and raw[1]['shared_complete'] is False


@pytest.mark.parametrize('fault', ['floor', 'nonfinite', 'reset', 'contacts', 'constraint'])
def test_failure_saves_offending_raw_terminal_without_resample(tmp_path, fault):
    def corrupt(flight):
        if flight.steps != 7:
            return
        if fault == 'floor':
            flight.current['vehicle_collision_aabbs']['base_link'][0][2] = .099
        elif fault == 'nonfinite':
            flight.current['package_velocity'][0] = float('nan')
        elif fault == 'reset':
            flight.current['reset'] = True
        elif fault == 'contacts':
            flight.current['contacts'] = True
        else:
            flight.current['package_xyz'][2] += .1
    flight, out, result = run(tmp_path, flight=CpuFlight(override=corrupt))
    assert result['status'] == 'FAILED' and result['steps'] == 7
    assert result['completed_count'] == 0 and flight.resets == 1 and len(flight.targets) == 1
    raw = rows(out, 'samples.jsonl')[-1]
    terminal = json.loads((out/'failure-sample.json').read_text())
    assert raw['simulation_step'] == 7 and terminal['simulation_step'] == 7
    assert terminal['sample']['drone_xyz'] == raw['sample']['drone_xyz']
    assert result['failure_sample_sha256']


def test_leg_timeout_is_independent_of_reported_core_completion(tmp_path):
    flight, out, result = run(tmp_path, flight=CpuFlight(pause=100000))
    assert result['status'] == 'FAILED' and result['termination_code'] == 'LEG_TIMEOUT'
    assert result['steps'] == 3000 and flight.resets == 1
    assert rows(out, 'samples.jsonl')[-1]['simulation_step'] == 3000


def test_mission_timeout_is_independent_of_leg_timeout(tmp_path):
    _, _, result = run(tmp_path, flight=CpuFlight(pause=1250))
    assert result['status'] == 'FAILED' and result['termination_code'] == 'MISSION_TIMEOUT'
    assert result['steps'] == 12000 and 1 <= result['completed_count'] < 10


def test_wall_timeout_uses_case_residence_and_retains_last_step(tmp_path):
    flight = CpuFlight()
    _, out, result = run(tmp_path, flight=flight, clock=lambda:flight.steps*10.)
    assert result['status'] == 'FAILED' and result['termination_code'] == 'WALL_TIMEOUT'
    assert result['wall_residence_s'] == 600.
    assert result['steps'] == 60
    assert rows(out, 'samples.jsonl')[-1]['simulation_step'] == 60


def test_decision_explanations_bind_real_pose_and_recompute_pose_cost_components(tmp_path):
    from adaptive_drone_patrol.baselines import remaining_cost, exact_remaining
    from adaptive_drone_patrol.analytical import restore
    _, out, result = run(tmp_path)
    assert result['status'] == 'SUCCEEDED'
    decision = rows(out, 'decisions.jsonl')[0]
    value = rows(out, 'decision_explanations.jsonl')[0]
    assert decision['physical_pose'] == [.03, 0., 1.]
    assert value['evidence_kind'] == 'cpu-fixture'
    assert value['pose_cost_analysis']['actual_pose'] == [.03, 0., 1.]
    state = restore(decision['boundary_snapshot'])
    state.scenario.coords[state.current] = decision['physical_pose']
    pose = value['pose_cost_analysis']
    assert pose['exact']['cost'] == pytest.approx(exact_remaining(state)['cost'])
    replay = remaining_cost(state, pose['policy_suffix']['order'])
    assert pose['policy_suffix']['cost'] == pytest.approx(replay['cost'])
    assert pose['policy_suffix']['cost'] == pytest.approx(
        pose['policy_suffix']['weighted_travel_cost']+pose['policy_suffix']['weighted_completion_penalty'])
    assert len(pose['forced_first']) == 10
    assert value['boundary_hash'] == decision['boundary_hash']


def test_physical_production_gate_rejects_before_creating_outputs_or_stepping(tmp_path):
    from adaptive_drone_patrol.payload_recovery_mission import execute_payload_mission
    flight = CpuFlight()
    with pytest.raises(ValueError, match='62'):
        execute_payload_mission(flight, Planner(), scenario(), tmp_path/'physical',
            {'evidence_kind':'physical', 'routing_model_hash':'a'*64},
            physical_evidence={'receipt':{}, 'actual_binding':{}, 'locations':{}})
    assert flight.steps == 0 and not (tmp_path/'physical').exists()


def test_events_have_one_ordered_sequence_and_predicate_records_cover_every_raw_step(tmp_path):
    _, out, result = run(tmp_path)
    events = rows(out, 'events.jsonl')
    assert [e['event_seq'] for e in events] == list(range(1, len(events)+1))
    predicates = rows(out, 'predicates.jsonl')
    assert [p['simulation_step'] for p in predicates] == list(range(1, result['steps']+1))
    assert sum(bool(p['shared_complete']) for p in predicates) == 10


@pytest.mark.parametrize('where', ['initial_capture', 'close'])
def test_capture_failures_preserve_terminal_sample_and_close_encoder(tmp_path, where):
    from adaptive_drone_patrol.payload_recovery_mission import execute_payload_mission
    class Recorder:
        closed = False
        def capture(self, step, force=False, event_seq=None):
            if where == 'initial_capture':
                raise RuntimeError('Camera read failed')
        def close(self):
            self.closed = True
            if where == 'close':
                raise RuntimeError('Encoder finalization failed')
    flight, recorder = CpuFlight(), Recorder()
    out = tmp_path/'capture-failure'
    result = execute_payload_mission(flight, Planner(), scenario(), out,
        {'evidence_kind':'cpu-fixture', 'routing_model_hash':'a'*64}, recorder=recorder)
    assert result['status'] == 'FAILED' and recorder.closed
    assert (out/'failure-sample.json').is_file() and result['failure_sample_sha256']
    assert rows(out, 'events.jsonl')[-1]['type'] == 'MISSION_FAILED'


def test_success_terminal_event_follows_last_shared_completion(tmp_path):
    _, out, result = run(tmp_path)
    events = rows(out, 'events.jsonl')
    assert events[-1]['type'] == 'MISSION_FINISHED'
    assert events[-1]['simulation_step'] == result['steps']
    assert events[-1]['completed_order'] == list('ABCDEFGHIJ')


def test_event_cost_comparison_reprices_both_suffixes_from_measured_boundary_pose(tmp_path):
    from adaptive_drone_patrol.analytical import restore
    from adaptive_drone_patrol.baselines import exact_remaining, remaining_cost
    flight = CpuFlight()
    # A fixed initial perturbation already makes the first decision nonnominal;
    # a steady permitted terminal offset makes the event boundary nonnominal too.
    flight.stop_short = .03
    _, out, result = run(tmp_path, flight=flight, kind='cost_shift')
    assert result['status'] == 'SUCCEEDED'
    event = next(e for e in rows(out, 'events.jsonl') if e['type'] == 'COST_UPDATED')
    comparison = event['pose_cost_comparison']
    assert comparison['actual_pose'] == event['physical_pose']
    following = rows(out, 'decisions.jsonl')[1]
    state = restore(following['boundary_snapshot'])
    state.scenario.coords[state.current] = event['physical_pose']
    assert comparison['exact']['cost'] == pytest.approx(exact_remaining(state)['cost'])
    for key in ('retained_order_post_event', 'replanned_order_post_event'):
        expected = remaining_cost(state, comparison[key]['order'])
        assert comparison[key]['cost'] == pytest.approx(expected['cost'])
    assert len(comparison['forced_first']) == 9


def test_emergency_c_already_completed_is_retained_as_failure_without_reroll(tmp_path):
    from adaptive_drone_patrol.payload_recovery_mission import execute_payload_mission
    class CFirst:
        def predict(self, state):
            return 2 if state.mask()[2] else int(np.flatnonzero(state.mask())[0])
    flight = CpuFlight()
    out = tmp_path/'c-first'
    result = execute_payload_mission(flight, CFirst(), scenario('emergency'), out,
        {'evidence_kind':'cpu-fixture', 'routing_model_hash':'a'*64})
    assert result['status'] == 'FAILED' and result['completed_order'] == ['C']
    assert flight.resets == 1 and flight.targets == ['C']
    events = rows(out, 'events.jsonl')
    rejected = next(e for e in events if e['type'] == 'SCENARIO_EVENT_REJECTED')
    assert rejected['update']['skipped'] is True
    assert not any(e['type'] == 'PRIORITY_UPDATED' for e in events)


def test_offline_explanation_failure_has_matching_failed_terminal_event_and_raw(tmp_path, monkeypatch):
    from adaptive_drone_patrol import payload_recovery_mission as mod
    def broken(*args):
        raise ValueError('Frozen suffix did not reproduce')
    monkeypatch.setattr(mod, 'explain_payload_decision', broken)
    _, out, result = run(tmp_path)
    assert result['status'] == 'FAILED' and result['termination_code'] == 'EXPLANATION_RECONSTRUCTION_FAILED'
    assert rows(out, 'events.jsonl')[-1]['type'] == 'MISSION_FAILED'
    assert result['failure_sample_sha256'] and (out/'failure-sample.json').is_file()
    assert json.loads((out/'failure-sample.json').read_text())['simulation_step'] == result['steps']
