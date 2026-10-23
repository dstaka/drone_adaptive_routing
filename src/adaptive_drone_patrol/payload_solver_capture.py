"""Complete accepted-case native replay with independently recomposed Japanese HUD."""

import gzip
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np

from .jobs import write_json
from .payload_solver_freeze import verify_file_snapshot
from .payload_solver_media import NativeRenderer, native_frame_rows, load_camera, compose_hud, mission_display_state
from .payload_solver_video import (PairedVideoWriter, decoded_rgb, rgb_sha,
    verify_video_pair, create_viewing_copies, verify_viewing_copies)
from .scenario import Scenario
from .util import sha256


def _fresh_capture_acceptance(freeze_path, trials, admission, planner):
    from .payload_solver_trials import recheck_paired_trials

    freeze_hash = sha256(freeze_path)
    freeze = json.loads(Path(freeze_path).read_text())
    proof = recheck_paired_trials(freeze_path=freeze_path, output=trials,
                                 admission=admission, planner=planner)
    if proof['trials'] != 254 or not proof['statistics']['randomized_acceptance']:
        raise ValueError('Fresh paired254 randomized acceptance required before capture')
    if sha256(freeze_path) != freeze_hash:
        raise ValueError('Actual acceptance freeze changed during capture admission')
    proof['capture_freeze_sha256'] = freeze_hash
    cases = json.loads(Path(freeze['cases']).read_text())['cases']
    return freeze, cases, proof


def _capture_bound_case(*, trial, case, freeze, output, font, trials, proof, freeze_path):
    from .payload_solver_trials import artifact_snapshot

    trial = Path(trial)
    if (trial.resolve() != (Path(trials) / 'randomized' / case['case_id']).resolve()
            or json.loads((trial / 'case.json').read_text()) != case):
        raise ValueError('Actual randomized trial/case binding required for capture')
    records = [r for r in proof['results'] if r['method'] == 'randomized'
               and r['case_id'] == case['case_id']]
    if len(records) != 1 or not records[0]['success'] or not records[0]['physical_acceptance']:
        raise ValueError('Fresh accepted randomized case proof required for capture')
    verified = records[0]
    accepted_artifacts = verified['artifact_sha256']
    if not {'flight/fixture/model.xml', 'flight/runtime-properties.json',
            'flight/runtime/raw.jsonl.gz', 'mission/mission_summary.json'} <= set(accepted_artifacts):
        raise ValueError('Complete accepted trial artifact binding required')
    for name, digest in accepted_artifacts.items():
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or sha256(trial / name) != digest):
            raise ValueError('Accepted trial artifact changed after fresh replay')
    if sha256(freeze_path) != proof['capture_freeze_sha256']:
        raise ValueError('Actual acceptance freeze changed before capture')
    metadata = json.loads((trial / 'mission/metadata.json').read_text())
    native = trial / 'flight/runtime/raw.jsonl.gz'
    if (metadata['method'] != 'randomized'
            or verified['native_proof']['checkpoint_sha256'] != freeze['models']['randomized']['sha256']
            or verified['native_proof']['native_raw_sha256'] != sha256(native)
            or verified['summary_sha256'] != sha256(trial / 'mission/mission_summary.json')):
        raise ValueError('Fresh capture model/native/mission proof differs')
    summary = json.loads((trial / 'mission/mission_summary.json').read_text())
    for name, key in [('samples.jsonl', 'telemetry_sha256'), ('events.jsonl', 'events_sha256'),
            ('decisions.jsonl', 'decisions_sha256'), ('predicates.jsonl', 'predicates_sha256'),
            ('decision_explanations.jsonl', 'explanations_sha256'), ('scenario.json', 'scenario_sha256'),
            ('metadata.json', 'metadata_sha256'), ('payload.json', 'payload_sha256'),
            ('initial_state.json', 'initial_state_sha256')]:
        if sha256(trial / 'mission' / name) != summary[key]:
            raise ValueError('Actual mission artifact changed after fresh capture replay')
    snapshot = artifact_snapshot(trial)
    if any(snapshot.get(name) != digest for name, digest in accepted_artifacts.items()):
        raise ValueError('Accepted trial artifact changed before capture snapshot')
    result = _render_case(trial=trial, case=case, freeze=freeze, output=output, font=font)
    final_snapshot = artifact_snapshot(trial)
    if (final_snapshot != snapshot
            or any(final_snapshot.get(name) != digest for name, digest in accepted_artifacts.items())):
        raise ValueError('Actual accepted trial changed during capture')
    verify_file_snapshot(freeze['artifact_sha256'])
    if sha256(freeze_path) != proof['capture_freeze_sha256']:
        raise ValueError('Actual acceptance freeze changed during capture')
    result['acceptance'] = dict(fresh_paired_trials_replayed=254, method='randomized',
                               freeze_sha256=proof['capture_freeze_sha256'],
                               independent_mission_proof=verified)
    write_json(Path(output) / 'all-frame-check.json', result)
    return result


def capture_case(*, trial, case, freeze, output, font,
                 freeze_path=None, trials=None, admission=None, planner=None):
    """Public single-case capture always reconstructs the complete acceptance."""
    if any(value is None for value in [freeze_path, trials, admission, planner]):
        raise ValueError('Fresh paired acceptance replay inputs required for capture')
    actual, cases, proof = _fresh_capture_acceptance(freeze_path, trials, admission, planner)
    if actual != freeze or case not in cases:
        raise ValueError('Frozen capture case/model binding differs')
    return _capture_bound_case(trial=trial, case=case, freeze=actual, output=output,
                              font=font, trials=trials, proof=proof, freeze_path=freeze_path)


def capture_all_cases(*, freeze_path, trials, admission, planner, font, output):
    """Recheck all254 once, then render only its three bound randomized cases."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve every capture attempt')
    freeze, cases, proof = _fresh_capture_acceptance(freeze_path, trials, admission, planner)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for scenario in ['normal', 'emergency', 'cost_shift']:
        case = next(c for c in cases if c['case_id'] == f'basic-{scenario}-1')
        results.append(_capture_bound_case(trial=Path(trials) / 'randomized' / case['case_id'],
            case=case, freeze=freeze, output=output / scenario, font=font, trials=trials,
            proof=proof, freeze_path=freeze_path))
    delivery = dict(freeze=str(freeze_path), scenarios=results,
                   human_review='PENDING_FRESH_CONTINUOUS1X_REVIEW_OF_FINAL_HASHES')
    write_json(output / 'delivery.json', delivery)
    return delivery


def review_windows(events, final_step):
    """Give concrete continuous1x intervals ending at the NEXT measured stop."""
    windows = []
    for event in events:
        if event['type'] not in ['PRIORITY_UPDATED', 'COST_UPDATED']:
            continue
        step = event['simulation_step']
        next_stop = next((e['simulation_step'] for e in events
            if e['type'] == 'SHARED_SETTLED' and e['simulation_step'] > step), None)
        if next_stop is None or next_stop > final_step:
            raise ValueError('Actual next shared stop required after each event')
        before = max(0, step - 100)
        windows.append(dict(event=event['type'], event_step=step, from_step=before,
            through_step=next_stop, from_s=before / 100, through_s=next_stop / 100))
    return windows


def _native_frames(path):
    with gzip.open(path, 'rt') as stream:
        yield from native_frame_rows(json.loads(line) for line in stream)


def _render_case(*, trial, case, freeze, output, font):
    """Render already bound native states; physics is never stepped by the camera."""
    trial, output = Path(trial), Path(output)
    verify_file_snapshot(freeze['artifact_sha256'])
    if str(font) not in freeze['camera_assets']:
        raise ValueError('Actual frozen Japanese font reference required')
    mission = trial / 'mission'
    summary = json.loads((mission / 'mission_summary.json').read_text())
    checked = json.loads((trial / 'independent-check.json').read_text())
    if (summary['status'] != 'SUCCEEDED' or summary['completed_count'] != 10
            or not checked['success'] or not checked['physical_acceptance']
            or case['case_id'] != f"basic-{case['scenario']}-1"
            or case['mass_ratio'] != .2 or case['side_m'] != .045):
        raise ValueError('Accepted complete basicseed1 nominal randomized case required')
    scenario = Scenario.from_dict(json.loads((mission / 'scenario.json').read_text()))
    events = [json.loads(x) for x in (mission / 'events.jsonl').read_text().splitlines()]
    native = trial / 'flight/runtime/raw.jsonl.gz'
    config = load_camera()
    checkpoint_sha = freeze['models']['randomized']['sha256']
    renderer = NativeRenderer(trial / 'flight/fixture/model.xml', config, cpu_only=True)
    try:
        with PairedVideoWriter(output, width=config['raw_width'], raw_height=config['raw_height'],
                hud_height=config['hud_height']) as writer:
            for frame in _native_frames(native):
                state = mission_display_state(scenario, events, frame['step'])
                raw = renderer.render(frame['actual'], scenario.coords[1:], state['priorities'],
                                      state['visited'], state['selected'])
                hud = compose_hud(raw, state, frame['actual'], case, config, font, checkpoint_sha)
                writer.append(frame['step'], raw, hud)
                if frame['step'] % 100 == 0:
                    print(json.dumps(dict(stage='capture', case=case['case_id'], step=frame['step'])), flush=True)
        proof = verify_video_pair(output)
        if proof['frames'] != summary['steps'] + 1:
            raise ValueError('Full accepted mission not present in video')
        # A separate traversal recomputes all visible poses, native times and
        # Japanese subtitles, then compares every lossless decoded HUD pixel.
        masters = [decoded_rgb(output / 'raw.mkv', config['raw_width'], config['raw_height']),
                   decoded_rgb(output / 'hud.mkv', config['raw_width'], config['raw_height'] + config['hud_height'])]
        frame_hashes = []
        try:
            for frame, raw, hud in zip_longest(_native_frames(native), *masters):
                if frame is None or raw is None or hud is None:
                    raise ValueError('Native/decoded full frame counts differ')
                state = mission_display_state(scenario, events, frame['step'])
                rerendered = renderer.render(frame['actual'], scenario.coords[1:], state['priorities'],
                                            state['visited'], state['selected'])
                recomposed = compose_hud(raw, state, frame['actual'], case, config, font, checkpoint_sha)
                if not np.array_equal(raw, rerendered) or not np.array_equal(hud, recomposed):
                    raise ValueError('Actual pose, Japanese subtitle or telemetry recomposition differs')
                frame_hashes.append(dict(step=frame['step'], native_time_s=frame['actual']['time_s'],
                    raw_rgb_sha256=rgb_sha(raw), hud_rgb_sha256=rgb_sha(hud)))
        finally:
            for stream in masters:
                stream.close()
        create_viewing_copies(output)
        viewing = verify_viewing_copies(output)
        if viewing['hud']['caption_regions_checked'] != 29:
            raise ValueError('Every frozen Japanese caption/telemetry region must be checked')
        windows = review_windows(events, summary['steps'])
        result = dict(case_id=case['case_id'], scenario=case['scenario'],
            checkpoint_sha256=checkpoint_sha, native_sha256=sha256(native),
            events_sha256=sha256(mission / 'events.jsonl'),
            mission_summary_sha256=sha256(mission / 'mission_summary.json'),
            camera_config_sha256=sha256('configs/payload_solver_camera.json'), font_sha256=sha256(font),
            renderer=renderer.renderer_string, renderer_device=renderer.device,
            lossless=proof, viewing=viewing, all_native_pose_hud_frames_recomposed=True,
            frame_checks=frame_hashes, review_windows=windows,
            human_review='PENDING_FRESH_CONTINUOUS1X_REVIEW_OF_FINAL_HASHES',
            source='Every0.01s native state of the accepted real mission, rendered offline without stepping physics',
            playback='Full0..N native instants, one normal final display period, no cuts/extra event frames')
        return result
    finally:
        renderer.close()
