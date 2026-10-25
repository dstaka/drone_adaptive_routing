"""Independent ten-stop replay and unchanged paired denominators."""

import json
import pytest
from test_payload_recovery_mission import run, Planner


def test_mission_replay_derives_all_ten_completions_from_samples(tmp_path):
    from adaptive_drone_patrol.payload_solver_acceptance import verify_mission

    _, root, _ = run(tmp_path)
    result = verify_mission(root, Planner())
    assert result["success"] and result["completed_count"] == 10
    assert not result["physical_acceptance"]


def test_summary_and_predicate_labels_cannot_invent_ten_completions(tmp_path):
    from adaptive_drone_patrol.payload_solver_acceptance import verify_mission
    from adaptive_drone_patrol.util import sha256

    _, root, _ = run(tmp_path)
    data = [json.loads(x) for x in (root / "samples.jsonl").read_text().splitlines()]
    initial = data[0]["sample"]
    for row in data[1:]:
        for key in initial:
            if key.startswith(("drone_", "package_", "attachment_")):
                row["sample"][key] = initial[key]
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    summary = json.loads((root / "mission_summary.json").read_text())
    summary["telemetry_sha256"] = sha256(root / "samples.jsonl")
    (root / "mission_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        verify_mission(root, Planner())


def test_paired_statistics_reject_missing_or_duplicate_frozen_cases():
    from adaptive_drone_patrol.payload_solver_acceptance import paired_statistics

    with pytest.raises(ValueError):
        paired_statistics([], [])


def test_unbound_payload_mass_cannot_escape_original_mission_binding(tmp_path):
    from adaptive_drone_patrol.payload_solver_acceptance import verify_mission

    _, root, _ = run(tmp_path)
    path = root / 'payload.json'
    data = json.loads(path.read_text())
    data['mass_ratio'] = .27
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='binding'):
        verify_mission(root, Planner())


def test_case_scenario_requires_full_canonical_configuration():
    from adaptive_drone_patrol.payload_solver_acceptance import verify_case_scenario
    from adaptive_drone_patrol.scenario import Scenario
    from adaptive_drone_patrol.util import load_config

    normal = Scenario.from_dict(load_config('configs/demo_scenarios_dispersed/normal.yaml'))
    assert verify_case_scenario(normal, dict(scenario='normal'))
    for name in ['emergency', 'cost_shift']:
        with pytest.raises(ValueError, match='scenario'):
            verify_case_scenario(normal, dict(scenario=name))


@pytest.fixture(scope='module')
def successful_event_missions(tmp_path_factory):
    root = tmp_path_factory.mktemp('event-source')
    return {kind: run(root, kind=kind)[1] for kind in ['normal', 'emergency', 'cost_shift']}


@pytest.mark.parametrize('kind,event_type,field,value', [
    ('normal', 'TARGET_SELECTED', 'target_id', 'J'),
    ('normal', 'TARGET_SELECTED', 'simulation_step', 1),
    ('normal', 'TARGET_SELECTED', 'sim_time_s', .01),
    ('normal', 'TARGET_SELECTED', 'physical_pose', [0., 0., 99.]),
    ('normal', 'MISSION_STARTED', 'simulation_step', 9000),
    ('normal', 'MISSION_STARTED', 'type', 'UNVERIFIED_DISPLAY_EVENT'),
    ('normal', 'MISSION_STARTED', 'completed_order', ['J']),
    ('normal', 'LEG_STARTED', 'target_id', 'J'),
    ('normal', 'SHARED_SETTLED', 'dwell_s', .5),
    ('normal', 'MISSION_FINISHED', 'cause', 'Fabricated terminal explanation'),
    ('emergency', 'PRIORITY_UPDATED', 'physical_pose', [0., 0., 99.]),
    ('emergency', 'PRIORITY_UPDATED', 'type', 'COST_UPDATED'),
    ('cost_shift', 'COST_UPDATED', 'actual_features', []),
])
def test_rehashed_display_event_cannot_disagree_with_actual_replay(
        tmp_path, successful_event_missions, kind, event_type, field, value):
    import shutil
    from adaptive_drone_patrol.payload_solver_acceptance import verify_mission
    from adaptive_drone_patrol.util import sha256

    root = tmp_path / kind
    shutil.copytree(successful_event_missions[kind], root)
    events = [json.loads(x) for x in (root / 'events.jsonl').read_text().splitlines()]
    event = next(e for e in events if e['type'] == event_type)
    event[field] = value
    (root / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    summary = json.loads((root / 'mission_summary.json').read_text())
    summary['events_sha256'] = sha256(root / 'events.jsonl')
    (root / 'mission_summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='[Ee]vent'):
        verify_mission(root, Planner())


def test_rehashed_actual_pose_cost_improvement_cannot_be_invented(
        tmp_path, successful_event_missions):
    import shutil
    from adaptive_drone_patrol.payload_solver_acceptance import verify_mission
    from adaptive_drone_patrol.util import sha256

    root = tmp_path / 'cost_shift'
    shutil.copytree(successful_event_missions['cost_shift'], root)
    events = [json.loads(x) for x in (root / 'events.jsonl').read_text().splitlines()]
    update = next(e for e in events if e['type'] == 'COST_UPDATED')
    update['pose_cost_comparison']['improvement'] += 10000
    (root / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    summary = json.loads((root / 'mission_summary.json').read_text())
    summary['events_sha256'] = sha256(root / 'events.jsonl')
    (root / 'mission_summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='[Ee]vent'):
        verify_mission(root, Planner())


@pytest.mark.parametrize('kind', ['emergency', 'cost_shift'])
def test_complete_real_event_fields_and_actual_pose_costs_reconstruct(
        successful_event_missions, kind):
    from adaptive_drone_patrol.payload_solver_acceptance import verify_mission

    proof = verify_mission(successful_event_missions[kind], Planner())
    assert proof['success'] and proof['completed_count'] == 10
    assert not proof['physical_acceptance']
