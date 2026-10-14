"""Deterministic offline Japanese HUD composition from genuine camera frames."""

import bisect
import csv
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .evidence import validate_run
from .util import json_write


def ffmpeg_executable():
    available = shutil.which('ffmpeg')
    if available:
        return available
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def select_runs(root):
    root = Path(root)
    candidates = []
    for path in root.glob('*/mission_summary.json'):
        s = json.loads(path.read_text())
        if s['status'] == 'SUCCEEDED' and (path.parent / 'raw.mp4').is_file():
            candidates.append((path.parent, s))
    # Select the lowest paired execution seed with three complete comparable runs.
    for seed in sorted({s.get('execution_seed') for _, s in candidates}):
        selected = {s['scenario_id']: p for p, s in candidates if s.get('execution_seed') == seed}
        if all(name in selected for name in ['normal', 'emergency', 'cost_shift']):
            return [selected[name] for name in ['normal', 'emergency', 'cost_shift']]
    raise RuntimeError('No paired seed has three authentic complete patrols')


def capture_sample_index(telemetry, frame):
    for index, sample in enumerate(telemetry):
        if int(sample['simulation_step']) == int(frame['simulation_step']):
            if float(sample['sim_time_s']) != float(frame['sim_time_s']):
                raise ValueError('Camera frame/telemetry timestamp mismatch')
            return index
    raise ValueError('Camera frame has no same-step telemetry')


def compose(root, output, font_path, duration_s=120, fps=30):
    if not 60 <= duration_s <= 180 or fps not in [25, 30]:
        raise ValueError('Submission must be 60–180 seconds at 25/30 FPS')
    output = Path(output)
    timeline_path = output.parent / ('timeline.json' if output.name == 'submission.mp4' else output.stem + '.timeline.json')
    selection_path = output.with_suffix('.selection.json')
    reservation = output.with_suffix('.reservation')
    if any(p.exists() for p in [output, timeline_path, selection_path, reservation]):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation.touch(exist_ok=False)
    runs = select_runs(root)
    for run in runs:
        validate_run(run, require_media=True)
    def readable_font(size):
        result = ImageFont.truetype(str(font_path), size)
        try:
            result.set_variation_by_axes([400])
        except (OSError, AttributeError):
            pass
        return result
    font, title_font, small_font = readable_font(23), readable_font(36), readable_font(19)
    ffmpeg = ffmpeg_executable()
    intro_s, outro_s = 8., 10.
    scene_duration = (duration_s - intro_s - outro_s) / 3
    encoder = subprocess.Popen([ffmpeg, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', '1280x720', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
    timeline = {'schema_version': 1, 'duration_s': duration_s, 'fps': fps,
                'intro': [0, intro_s], 'outro': [duration_s - outro_s, duration_s], 'segments': []}

    def encode(image):
        encoder.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())

    def card(title, lines):
        img = Image.new('RGB', (1280, 720), '#102333')
        draw = ImageDraw.Draw(img)
        draw.text((70, 85), title, fill='#f3f8fa', font=title_font)
        for i, line in enumerate(lines):
            draw.text((70, 190 + i * 58), line, fill='#c7e0ed', font=font)
        return img

    try:
        intro = card('適応型ドローン巡回点検', ['A・B・C の3地点を実物理で巡回',
                     '上位：学習済み MaskablePPO が次の未訪問地点を選択',
                     '下位：学習済み PPO が4つのモータを制御',
                     '到達後に状況を更新し、その場で再計画',
                     'Genesis の実録画／判定は位置・速度・連続0.50秒の滞在'])
        for _ in range(round(intro_s * fps)):
            encode(intro)
        for index, run in enumerate(runs):
            validate_run(run)
            scenario = json.loads((run / 'scenario.json').read_text())
            telemetry = list(csv.DictReader((run / 'telemetry.csv').open()))
            frames = list(csv.DictReader((run / 'frames.csv').open()))
            if not frames:
                raise RuntimeError('No same-run frame timing evidence')
            frame_times = [float(r['sim_time_s']) for r in frames]
            events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
            source_start, source_end = frame_times[0], frame_times[-1]
            # The last captured frame remains visible for one source-frame interval.
            source_span = source_end - source_start + .02
            rate = source_span / scene_duration
            destination_start = intro_s + index * scene_duration
            timeline['segments'].append({'run_id': run.name, 'clip': str(run / 'raw.mp4'),
                'source_frame_start': 0, 'source_frame_end': len(frames) - 1,
                'source_start_s': source_start, 'source_end_s': source_end + .02,
                'presentation_start_s': destination_start, 'presentation_end_s': destination_start + scene_duration,
                'playback_rate': rate, 'disclosure': f'{rate:.3f}x playback; complete retained patrol, no cuts',
                'frame_map': str(run / 'frames.csv')})
            decoder = subprocess.Popen([ffmpeg, '-v', 'error', '-i', str(run / 'raw.mp4'),
                         '-vf', 'scale=768:576', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
            current_index, current_image = -1, None
            for output_index in range(round(scene_duration * fps)):
                sim_time = min(source_end, source_start + output_index / fps * rate)
                wanted = min(len(frames) - 1, max(0, bisect.bisect_right(frame_times, sim_time) - 1))
                while current_index < wanted:
                    data = decoder.stdout.read(768 * 576 * 3)
                    if len(data) != 768 * 576 * 3:
                        raise RuntimeError('Raw clip ended before its frame map')
                    current_image = Image.frombytes('RGB', (768, 576), data)
                    current_index += 1
                sim_time = frame_times[wanted]
                sample_index = capture_sample_index(telemetry, frames[wanted])
                sample = telemetry[sample_index]
                past = [e for e in events if e['sim_time_s'] <= frame_times[wanted] + 1e-9]
                visited = [e['target_id'] for e in past if e['type'] == 'WAYPOINT_REACHED']
                priorities = list(scenario['priorities'])
                multipliers = [list(row) for row in scenario['multipliers']]
                banner = ''
                logical_label = '辺倍率：全て1（更新前）'
                for e in past:
                    if e['type'] == 'PRIORITY_UPDATED':
                        update = e['update']
                        priorities[update['checkpoint']] = update['value']
                        banner = f'緊急更新：{"ABC"[update["checkpoint"]]} の優先度 → {update["value"]}'
                    elif e['type'] == 'COST_UPDATED':
                        for i, j in e['update']['edges']:
                            multipliers[i][j] = e['update']['value']
                        banner = '論理コスト更新：更新後の観測で次地点を再選択'
                        i, j = e['update']['edges'][0]
                        logical_label = f'倍率 {"SABC"[i]}→{"SABC"[j]}：{e["update"]["old"][0]:g}→{e["update"]["value"]:g}'
                img = Image.new('RGB', (1280, 720), '#102333')
                img.paste(current_image, (20, 83))
                draw = ImageDraw.Draw(img)
                titles = ['通常点検', '優先度の緊急更新', '論理移動コストの更新']
                draw.text((25, 20), f'{index + 1}/3  {titles[index]}', fill='#f3f8fa', font=title_font)
                draw.text((812, 80), f'実時間 {sim_time:.2f} 秒 ／訪問 {len(visited)}/3', fill='#f3f8fa', font=font)
                draw.text((812, 117), f'目標 {sample["target_id"]}  速度 {float(sample["speed_mps"]):.2f} m/s',
                          fill='#c7e0ed', font=font)
                draw.text((812, 154), '優先度 ' + '  '.join(f'{name}{value:g}' for name, value in zip('ABC', priorities)), fill='#c7e0ed', font=font)
                draw.text((812, 185), logical_label, fill='#ffbf69', font=small_font)
                draw.rectangle((820, 208, 1230, 536), fill='#1d3546', outline='#466375')
                def map_point(x, y):
                    return 1025 + float(x) * 190, 372 - float(y) * 190
                trajectory = [map_point(r['x'], r['y']) for r in telemetry[:sample_index + 1]]
                if len(trajectory) > 1:
                    draw.line(trajectory, fill='#58d5ce', width=3)
                for j, pos in enumerate(scenario['coords'][1:]):
                    x, y = map_point(pos[0], pos[1])
                    label = 'ABC'[j]
                    color = '#58d5ce' if label in visited else '#ffbf69'
                    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
                    draw.text((x + 12, y - 17), label, fill='white', font=font)
                x, y = map_point(sample['x'], sample['y'])
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill='white')
                draw.text((820, 546), '実軌跡（m）／次の未訪問地点のみ選択', fill='#c7e0ed', font=small_font)
                draw.text((812, 580), f'滞在 {float(sample["dwell_s"]):.2f}/0.50 秒', fill='#c7e0ed', font=font)
                draw.text((25, 669), f'Genesis実録画：{rate:.3f}倍速（観察用の低速再生）', fill='#ffbf69', font=small_font)
                if banner:
                    draw.text((25, 633), banner, fill='#ffbf69', font=font)
                draw.text((812, 621), '論理コストは障害物やエネルギーではない', fill='#c7e0ed', font=small_font)
                encode(img)
            decoder.stdout.close()
            decoder.terminate()
            decoder.wait()

        outro = card('独自性・限界・出典', ['優先度と論理移動コストに応じた学習済み巡回順序',
                    '実到達・滞在を確認してから更新／映像とログを同時記録',
                    '検証例は少数。障害物認識・センサ点検・実機安全性は未検証',
                    'Genesis v1.4.3 drone examples / Apache-2.0',
                    '提供ノートブックを継承。日本語フォント：Noto Sans JP / OFL'])
        for _ in range(round(outro_s * fps)):
            encode(outro)
    finally:
        encoder.stdin.close()
        if encoder.wait(timeout=120):
            raise RuntimeError('Submission encoding failed')
    json_write(timeline_path, timeline)
    json_write(selection_path, {'selected_runs': [r.name for r in runs],
               'reason': 'Lowest paired execution seed completing all scenarios'})
    subprocess.run([ffmpeg, '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
    reservation.unlink()
    return timeline
