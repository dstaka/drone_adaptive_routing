"""Fail-closed evidence validation and explicit presentation/source-time mapping."""

import csv
import json
import math
from pathlib import Path
from .util import sha256


def map_presentation_time(segments, time_s):
    for s in segments:
        if s['presentation_start_s'] <= time_s < s['presentation_end_s']:
            source = s['source_start_s'] + (time_s - s['presentation_start_s']) * s['playback_rate']
            if source > s['source_end_s'] + 1e-6:
                raise ValueError('Timeline exceeds retained source interval')
            return s['run_id'], source
    raise ValueError('Presentation time has no source mapping')


def verify_manifest(manifest, root, require_local_artifacts=False):
    root = Path(root).resolve()
    required = {'README.md', 'spec.md', 'pyproject.toml', 'THIRD_PARTY_NOTICES.md',
                'requirements/flight-linux.lock.txt', 'requirements/routing-linux.lock.txt',
                'requirements/routing-local.lock.txt', 'outputs/routing_boundary/models.json',
                'outputs/flight/model_300.pt', 'outputs/flight/cfgs.pkl', 'outputs/gpu_ledger.json',
                'outputs/videos/submission.mp4', 'outputs/videos/timeline.json',
                'outputs/report/report.pdf', 'outputs/genesis/all_attempts.json', 'outputs/evaluation/flight/all_trials.json'}
    for split in ['static', 'dynamic']:
        required.update(f'outputs/evaluation/{split}/{name}' for name in
                        ['results.csv', 'summary.json', 'scenarios.json', 'scenario_results.json'])
    for scenario in ['normal', 'emergency', 'cost_shift']:
        for seed in [1, 2, 3]:
            required.update(f'outputs/genesis/{scenario}_seed_{seed}/{name}' for name in
                            ['mission_summary.json', 'run_metadata.json', 'scenario.json', 'raw.mp4',
                             'telemetry.csv', 'events.jsonl', 'decisions.jsonl', 'frames.csv'])
    for source_root in ['src', 'scripts', 'tests', 'configs', 'requirements', 'docs', 'third_party']:
        required.update(str(p.relative_to(root)) for p in (root / source_root).rglob('*')
                        if p.is_file() and '__pycache__' not in str(p) and p.suffix not in ['.pyc', '.lock']
                        and (not p.name.startswith('execution') or p.name == 'execution.example.yaml'))
    required.update(f'outputs/evaluation/figures/{name}.png' for name in
                    ['cost_comparison', 'scenario_order_comparison', 'event_replanning', 'real_trajectory'])
    required.update(f'outputs/videos/raw_{name}.mp4' for name in ['normal', 'emergency', 'cost_shift'])
    declared = {entry['path'] for entry in manifest['artifacts']}
    if len(declared) != len(manifest['artifacts']):
        raise ValueError('Duplicate manifest paths')
    if require_local_artifacts and not required <= declared:
        raise ValueError(f'Missing required deliverables: {sorted(required - declared)}')
    count = 0
    for entry in manifest['artifacts']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest path is outside delivery root')
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Artifact integrity failure: {entry["path"]}')
        count += 1
    if not count:
        raise ValueError('Empty manifest')
    if require_local_artifacts:
        attempts = json.loads((root / 'outputs/genesis/all_attempts.json').read_text())
        if len(attempts) != 9 or len({a['run_id'] for a in attempts}) != 9:
            raise ValueError('Delivery requires all nine physical attempts')
        for scenario in ['normal', 'emergency', 'cost_shift']:
            for seed in [1, 2, 3]:
                validate_run(root / f'outputs/genesis/{scenario}_seed_{seed}', require_media=True)
        timeline = json.loads((root / 'outputs/videos/timeline.json').read_text())
        if len(timeline['segments']) != 3 or not 60 <= timeline['duration_s'] <= 180:
            raise ValueError('Incomplete submission timeline')
        for split, episodes in [('static', 100), ('dynamic', 50)]:
            scenarios = json.loads((root / f'outputs/evaluation/{split}/scenarios.json').read_text())
            if len(scenarios) != episodes:
                raise ValueError('Incorrect held-out scenario count')
        models = json.loads((root / 'outputs/routing_boundary/models.json').read_text())
        if {m['seed'] for m in models['models']} != {42, 43, 44}:
            raise ValueError('Missing required routing training seeds')
        for model in models['models']:
            model_path = Path(model['path'])
            if not model_path.is_absolute():
                model_path = root / model_path
            if not model_path.is_relative_to(root) or sha256(model_path) != model['sha256']:
                raise ValueError('Routing checkpoint hash/path mismatch')
            if str(model_path.relative_to(root)) not in declared:
                raise ValueError('Routing checkpoint absent from manifest')
    return count


def validate_run(path, require_media=False):
    path = Path(path)
    summary = json.loads((path / 'mission_summary.json').read_text())
    events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
    if [e['event_seq'] for e in events] != list(range(1, len(events) + 1)):
        raise ValueError('Nonmonotonic event sequence')
    times = [e['sim_time_s'] for e in events]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('Event simulation time moved backwards')
    reached = [e['target_id'] for e in events if e['type'] == 'WAYPOINT_REACHED']
    if reached != summary['completed_order'] or len(set(reached)) != len(reached):
        raise ValueError('Completed visits do not match physical service events')
    if summary['status'] == 'SUCCEEDED' and set(reached) != {'A', 'B', 'C'}:
        raise ValueError('Successful mission lacks three physical service events')
    if not events or events[-1]['type'] not in ['MISSION_FINISHED', 'MISSION_FAILED']:
        raise ValueError('Missing terminal mission event')
    for e in events:
        if e['type'] in ['PRIORITY_UPDATED', 'COST_UPDATED']:
            if e['update']['boundary'] != 1 or not any(x['type'] == 'WAYPOINT_REACHED'
                        and x['event_seq'] < e['event_seq'] for x in events):
                raise ValueError('Event precedes physical inspection service')
    expected_terminal = {'SUCCEEDED': 'MISSION_FINISHED', 'FAILED': 'MISSION_FAILED'}.get(summary['status'])
    if expected_terminal is None or events[-1]['type'] != expected_terminal:
        raise ValueError('Terminal event contradicts summary status')
    if not (path / 'telemetry.csv').is_file():
        raise ValueError('Missing physical telemetry')
    rows = list(csv.DictReader((path / 'telemetry.csv').open()))
    if not rows:
        raise ValueError('Empty physical telemetry')
    run_id = summary['run_id']
    if any(r['run_id'] != run_id for r in rows) or any(e['run_id'] != run_id for e in events):
        raise ValueError('Cross-run evidence reference')
    steps = [int(r['simulation_step']) for r in rows]
    if steps != list(range(1, len(rows) + 1)):
        raise ValueError('Missing/nonmonotonic simulation steps')
    telemetry = dict(zip(steps, rows))
    if not (path / 'decisions.jsonl').is_file():
        raise ValueError('Missing decision references')
    decisions = [json.loads(line) for line in (path / 'decisions.jsonl').read_text().splitlines()]
    selected = [e for e in events if e['type'] == 'TARGET_SELECTED']
    if len(decisions) != len(selected):
        raise ValueError('Missing decision/event references')
    for i, (decision, selected_event) in enumerate(zip(decisions, selected)):
        if (decision['run_id'] != run_id or decision['event_seq'] != selected_event['event_seq']
                or decision['decision_index'] != i or decision['target_id'] != selected_event['target_id']
                or decision['simulation_step'] != selected_event['simulation_step']
                or not decision['mask'][decision['selected_action']]):
            raise ValueError('Invalid decision reference or masked action')
    if summary['status'] == 'SUCCEEDED':
        if summary['completed_count'] != 3 or summary['intended_order'] != reached:
            raise ValueError('Inconsistent successful visit count/order')
        if any(r[k] != 'False' for r in rows for k in ['done', 'reset', 'safety_failure']):
            raise ValueError('Successful flight includes a termination/safety failure')
        for e in events:
            if e['type'] != 'WAYPOINT_REACHED':
                continue
            end_step = e['simulation_step']
            row = telemetry[end_step]
            dwell = float(row['dwell_s'])
            dt = float(rows[0]['sim_time_s'])
            count = round(dwell / dt)
            service = [telemetry.get(k) for k in range(end_step - count + 1, end_step + 1)]
            if dwell < .5 - 1e-9 or any(r is None or r['target_id'] != e['target_id']
                  or not math.isfinite(float(r['distance_to_goal_m']))
                  or float(r['distance_to_goal_m']) > .2 + 1e-9
                  or not math.isfinite(float(r['speed_mps'])) or float(r['speed_mps']) > .3 + 1e-9 for r in service):
                raise ValueError('Physical service lacks continuous accepted telemetry')
    if require_media and (not (path / 'frames.csv').is_file() or not (path / 'raw.mp4').is_file()):
        raise ValueError('Missing same-run recording/frame map')
    if (path / 'frames.csv').exists():
        frames = list(csv.DictReader((path / 'frames.csv').open()))
        if not frames:
            raise ValueError('Empty frame map')
        for i, frame in enumerate(frames):
            if int(frame['frame_index']) != i or int(frame['simulation_step']) not in telemetry:
                raise ValueError('Frame reference cannot resolve to same-run telemetry')
            sample = telemetry[int(frame['simulation_step'])]
            if sample['frame_index'] != str(i) or float(sample['sim_time_s']) != float(frame['sim_time_s']):
                raise ValueError('Frame/telemetry timing mismatch')
    return summary
