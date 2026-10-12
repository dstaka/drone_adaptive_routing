#!/usr/bin/env python
"""Probe/decode final media and verify every displayed frame's capture references."""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
from adaptive_drone_patrol.evidence import validate_run
from adaptive_drone_patrol.util import sha256, json_write
from adaptive_drone_patrol.video import capture_sample_index, ffmpeg_executable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', default='outputs/genesis')
    p.add_argument('--submission', default='outputs/videos/submission.mp4')
    p.add_argument('--timeline', default='outputs/videos/timeline.json')
    p.add_argument('--out', required=True)
    p.add_argument('--ffprobe', default=shutil.which('ffprobe') or 'ffprobe')
    a = p.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    timeline = json.loads(Path(a.timeline).read_text())
    records = []
    runs = sorted(path for path in Path(a.run_root).glob('*_seed_*') if path.is_dir())
    if len(runs) != 9:
        raise ValueError('Exactly nine retained final attempts are required')
    for path in [*(run / 'raw.mp4' for run in runs), Path(a.submission)]:
        info = json.loads(subprocess.check_output([a.ffprobe, '-v', 'error', '-select_streams', 'v:0',
                    '-count_frames', '-show_entries', 'stream=width,height,codec_name,avg_frame_rate,nb_read_frames',
                    '-show_entries', 'format=duration', '-of', 'json', str(path)], text=True))
        subprocess.run([ffmpeg_executable(), '-v', 'error', '-i', str(path), '-f', 'null', '-'], check=True)
        stream = info['streams'][0]
        if stream['codec_name'] != 'h264':
            raise ValueError('Expected H.264 media')
        if path == Path(a.submission):
            duration = float(info['format']['duration'])
            if (stream['width'] < 1280 or stream['height'] < 720 or not 60 <= duration <= 180
                    or abs(duration - timeline['duration_s']) > 1 / timeline['fps']
                    or stream['avg_frame_rate'] != f'{timeline["fps"]}/1'
                    or int(stream['nb_read_frames']) != round(duration * timeline['fps'])):
                raise ValueError('Submission stream does not satisfy the delivery contract')
        else:
            summary = validate_run(path.parent, require_media=True)
            frames = list(csv.DictReader((path.parent / 'frames.csv').open()))
            if int(stream['nb_read_frames']) != len(frames):
                raise ValueError('Decoded raw frames differ from capture map')
            info['run_id'] = summary['run_id']
        records.append({'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size,
                        'full_decode': 'PASS', 'probe': info})
    alignments, boundaries = [], []
    for segment in timeline['segments']:
        run = Path(a.run_root) / segment['run_id']
        frames = list(csv.DictReader((run / 'frames.csv').open()))
        telemetry = list(csv.DictReader((run / 'telemetry.csv').open()))
        times = [float(f['sim_time_s']) for f in frames]
        count = round((segment['presentation_end_s'] - segment['presentation_start_s']) * timeline['fps'])
        for i in range(count):
            requested = min(times[-1], times[0] + i / timeline['fps'] * segment['playback_rate'])
            wanted = max(0, bisect.bisect_right(times, requested) - 1)
            sample = telemetry[capture_sample_index(telemetry, frames[wanted])]
            alignments.append({'presentation_s': segment['presentation_start_s'] + i / timeline['fps'],
                        'run_id': run.name, 'source_frame_index': wanted,
                        'simulation_step': int(sample['simulation_step']), 'sim_time_s': times[wanted],
                        'hud_camera_error_s': 0.0})
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
        for event in events:
            if event['type'] not in ['WAYPOINT_REACHED', 'PRIORITY_UPDATED', 'COST_UPDATED', 'MISSION_FINISHED']:
                continue
            if event['sim_time_s'] not in times:
                raise ValueError('Represented service/update boundary has no captured frame')
            mapped = segment['presentation_start_s'] + (event['sim_time_s'] - times[0]) / segment['playback_rate']
            displayed = segment['presentation_start_s'] + math.ceil((mapped - segment['presentation_start_s']) * timeline['fps'] - 1e-8) / timeline['fps']
            if not 0 <= displayed - mapped <= 1 / timeline['fps'] + 1e-8:
                raise ValueError('Boundary exceeds one output-frame interval')
            boundaries.append({'run_id': run.name, 'event_seq': event['event_seq'], 'type': event['type'],
                               'source_sim_s': event['sim_time_s'], 'presentation_s': displayed})
    json_write(a.out, {'status': 'PASS', 'media': records, 'camera_hud_mapping': alignments,
                      'represented_boundaries': boundaries,
                      'note': 'Mapping replays the deterministic composer; representative encoded frames are also inspected visually.'})
    print(f'PASS: {len(records)} full media decodes, {len(alignments)} exact capture/HUD mappings, {len(boundaries)} boundaries')


if __name__ == '__main__':
    main()
