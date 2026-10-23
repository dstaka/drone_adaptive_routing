"""Lossless raw/HUD masters with complete decoded RGB and100Hz timing checks."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .jobs import write_json
from .util import sha256
from .video import ffmpeg_executable


def rgb_sha(frame):
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


class PairedVideoWriter:
    """Preserve two complete lossless streams and a per-native-instant index."""

    def __init__(self, root, *, width, raw_height, hud_height):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.width, self.raw_height, self.hud_height = width, raw_height, hud_height
        self.step = 0
        self.closed = False
        self.frames = (self.root / 'frames.jsonl').open('x')
        self.logs, self.processes = [], []
        for name, height in [('raw', raw_height), ('hud', raw_height + hud_height)]:
            log = (self.root / (name + '-encoder.log')).open('x')
            self.logs.append(log)
            self.processes.append(subprocess.Popen([
                ffmpeg_executable(), '-nostdin', '-v', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '100',
                '-i', 'pipe:0', '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0',
                '-threads', '2', '-n', str(self.root / (name + '.mkv')),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log))

    def append(self, step, raw, hud):
        if self.closed or step != self.step:
            raise ValueError('Every video step must appear once in continuous order')
        raw, hud = np.asarray(raw), np.asarray(hud)
        if (raw.shape != (self.raw_height, self.width, 3)
                or hud.shape != (self.raw_height + self.hud_height, self.width, 3)
                or raw.dtype != np.uint8 or hud.dtype != np.uint8):
            raise ValueError('Exact full RGB dimensions required')
        if not np.array_equal(hud[:self.raw_height], raw):
            raise ValueError('HUD changed original raw pixels')
        for process, frame in zip(self.processes, [raw, hud]):
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames.write(json.dumps(dict(step=step, pts_s=step / 100,
            raw_rgb_sha256=rgb_sha(raw), hud_rgb_sha256=rgb_sha(hud))) + '\n')
        self.frames.flush()
        self.step += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        for process in self.processes:
            process.stdin.close()
        codes = [process.wait(timeout=120) for process in self.processes]
        for log in self.logs:
            log.close()
        self.frames.close()
        if codes != [0, 0] or self.step == 0:
            raise ValueError('Complete nonempty video encoding failed')
        write_json(self.root / 'encoding.json', dict(schema='payload-solver-video-pair-v1',
            width=self.width, raw_height=self.raw_height, hud_height=self.hud_height,
            frames=self.step, fps=100, codec='ffv1', pixel_format='bgr0',
            ffmpeg_sha256=sha256(ffmpeg_executable()),
            physical_last_frame_s=(self.step - 1) / 100,
            displayed_duration_s=self.step / 100,
            final_display='One normal display period, no extra physics step'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def decoded_rgb(path, width, height):
    """Decode every frame, rejecting partial output and decoder errors."""
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([ffmpeg_executable(), '-nostdin', '-v', 'error',
            '-i', str(path), '-map', '0:v:0', '-vsync', '0', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-threads', '2', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=log)
        try:
            size = width * height * 3
            while True:
                data = bytearray()
                while len(data) < size:
                    part = process.stdout.read(size - len(data))
                    if not part:
                        break
                    data.extend(part)
                if not data:
                    break
                if len(data) != size:
                    raise ValueError('Partial decoded RGB frame')
                yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            if process.wait(timeout=120) != 0:
                raise ValueError('Video decoder failed')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def decoded_pts(path):
    """Obtain all actual decoded timestamps without a separate ffprobe binary."""
    # FrameMD5 explicitly emits decoded frame timestamps in the output timebase.
    result = subprocess.run([ffmpeg_executable(), '-nostdin', '-v', 'error', '-copyts',
        '-i', str(path), '-map', '0:v:0', '-vsync', '0', '-f', 'framemd5', '-'],
        check=True, capture_output=True, text=True)
    base, pts = None, []
    for line in result.stdout.splitlines():
        if line.startswith('#tb 0:'):
            numerator, denominator = map(int, line.split(':', 1)[1].strip().split('/'))
            base = numerator / denominator
        elif line and not line.startswith('#'):
            values = [v.strip() for v in line.split(',')]
            if base is None or values[0] != '0':
                raise ValueError('Unexpected decoded video timebase/stream')
            pts.append(int(values[2]) * base)
    return pts


def verify_video_pair(root):
    """Independently decode BOTH complete masters; no sampled-frame approval."""
    from itertools import zip_longest

    root = Path(root)
    info = json.loads((root / 'encoding.json').read_text())
    if info['schema'] != 'payload-solver-video-pair-v1' or info['fps'] != 100:
        raise ValueError('Frozen1x100FPS encoding declaration required')
    rows = [json.loads(x) for x in (root / 'frames.jsonl').read_text().splitlines()]
    count = info['frames']
    if (count < 1 or len(rows) != count
            or [r['step'] for r in rows] != list(range(count))
            or [r['pts_s'] for r in rows] != [i / 100 for i in range(count)]):
        raise ValueError('Continuous complete native frame index required')
    w, h, extra = info['width'], info['raw_height'], info['hud_height']
    streams = [decoded_rgb(root / 'raw.mkv', w, h), decoded_rgb(root / 'hud.mkv', w, h + extra)]
    n = 0
    try:
        for row, raw, hud in zip_longest(rows, *streams):
            if row is None or raw is None or hud is None:
                raise ValueError('Decoded frame count differs')
            if rgb_sha(raw) != row['raw_rgb_sha256'] or rgb_sha(hud) != row['hud_rgb_sha256']:
                raise ValueError('Decoded RGB hash differs from original rendered frame')
            if not np.array_equal(raw, hud[:h]):
                raise ValueError('HUD master changed original raw pixels')
            n += 1
    finally:
        for stream in streams:
            stream.close()
    pts = [decoded_pts(root / name) for name in ['raw.mkv', 'hud.mkv']]
    expected = [i / 100 for i in range(count)]
    if any(len(p) != count or not np.allclose(p, expected, rtol=0, atol=1e-8) for p in pts):
        raise ValueError('Decoded PTS changed physical1x timing')
    return dict(frames=n, fps=100, last_physical_time_s=(n - 1) / 100,
        displayed_duration_s=n / 100, all_lossless_pixels_match=True,
        raw_viewport_equal_every_frame=True, pts=expected,
        video_sha256={name: sha256(root / name) for name in ['raw.mkv', 'hud.mkv']})


# Fixed before accepted capture. Chroma subsampling/quantization cannot preserve
# exact pixels; exact preservation is established separately on FFV1 masters.
VIEWING_ERROR = dict(maximum_frame_mean_absolute_rgb_error=6.0,
                     maximum_fraction_channels_over32=0.02)
CAPTION_ERROR = dict(maximum_region_luminance_mae=4.0,
                     maximum_foreground_tile_luminance_mae=4.0,
                     tile_pixels=24, foreground_luminance=150.0)
VIEWING_ENCODING = dict(codec='libx264',crf=0,pixel_format='yuv420p',fps=100)


def caption_regions():
    """Cover every visible Japanese caption and numeric field in the frozen HUD."""
    regions = dict(title=[22,730,1265,773], detail=[1295,730,1785,773],
        clock_target_progress=[22,775,845,811], altitude=[850,775,1780,811],
        load=[22,811,845,847], swing_spin=[850,811,1780,847],
        event=[22,1074,1780,1112], legend=[22,1117,1780,1151],
        model_hash=[22,1152,1780,1192])
    for index in range(10):
        x, y = 22 + (index % 5)*350, 860 + (index // 5)*102
        regions[f'goal_{index}_priority'] = [x+12,y+5,x+288,y+42]
        regions[f'goal_{index}_state'] = [x+12,y+41,x+288,y+79]
    return regions


def verify_caption_regions(original, viewing, regions):
    """Reject local glyph erasure even when whole-frame compression MAE passes."""
    original, viewing = np.asarray(original), np.asarray(viewing)
    if original.shape != viewing.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError('Full matching RGB caption frames required')
    maximum_region = maximum_tile = 0.0
    for name, (x0,y0,x1,y1) in regions.items():
        if not (0 <= x0 < x1 <= original.shape[1] and 0 <= y0 < y1 <= original.shape[0]):
            raise ValueError('Frozen caption region exceeds frame: ' + name)
        expected = original[y0:y1,x0:x1].astype(float) @ [0.2126,0.7152,0.0722]
        actual = viewing[y0:y1,x0:x1].astype(float) @ [0.2126,0.7152,0.0722]
        difference = np.abs(expected-actual)
        error = float(difference.mean())
        maximum_region = max(maximum_region,error)
        if error > CAPTION_ERROR['maximum_region_luminance_mae']:
            raise ValueError('Decoded caption luminance differs: ' + name)
        tile = CAPTION_ERROR['tile_pixels']
        for y in range(0,expected.shape[0],tile):
            for x in range(0,expected.shape[1],tile):
                mask = expected[y:y+tile,x:x+tile] > CAPTION_ERROR['foreground_luminance']
                if mask.any():
                    error = float(difference[y:y+tile,x:x+tile][mask].mean())
                    maximum_tile = max(maximum_tile,error)
                    if error > CAPTION_ERROR['maximum_foreground_tile_luminance_mae']:
                        raise ValueError('Decoded caption foreground glyph differs: ' + name)
    return dict(caption_regions_checked=len(regions), maximum_region_luminance_mae=maximum_region,
                maximum_foreground_tile_luminance_mae=maximum_tile)


def create_viewing_copies(root):
    """Encode ordinary players'1x copies while retaining the lossless masters."""
    root = Path(root)
    for name in ['raw', 'hud']:
        with (root / (name + '-viewing-encoder.log')).open('x') as log:
            subprocess.run([ffmpeg_executable(), '-nostdin', '-v', 'error',
                '-i', str(root / (name + '.mkv')), '-map', '0:v:0', '-an',
                '-vsync', '0', '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p',
                '-threads', '2', '-movflags', '+faststart', '-n', str(root / (name + '.mp4'))],
                check=True, stderr=log)
    write_json(root / 'viewing-encoding.json',dict(schema='payload-solver-viewing-encoding-v2',
        encoding=VIEWING_ENCODING,caption_error=CAPTION_ERROR,rgb_error=VIEWING_ERROR,
        ffmpeg_sha256=sha256(ffmpeg_executable())))


def verify_viewing_copies(root):
    """Decode every viewing pixel and PTS against both lossless masters."""
    from itertools import zip_longest

    root = Path(root)
    declaration = json.loads((root / 'viewing-encoding.json').read_text())
    if (declaration['encoding'] != VIEWING_ENCODING or declaration['caption_error'] != CAPTION_ERROR
            or declaration['rgb_error'] != VIEWING_ERROR
            or declaration['ffmpeg_sha256'] != sha256(ffmpeg_executable())):
        raise ValueError('Frozen viewing encoder/caption tolerances differ')
    info = json.loads((root / 'encoding.json').read_text())
    master = verify_video_pair(root)
    answer = {}
    for name, height in [('raw', info['raw_height']), ('hud', info['raw_height'] + info['hud_height'])]:
        pair = [decoded_rgb(root / (name + suffix), info['width'], height) for suffix in ['.mkv', '.mp4']]
        n, maximum_mean, maximum_fraction = 0, 0., 0.
        regions = caption_regions() if (name == 'hud' and (info['width'],info['raw_height'],info['hud_height'])
                                       == (1792,720,480)) else {}
        caption_proof = dict(caption_regions_checked=len(regions), maximum_region_luminance_mae=0.,
                             maximum_foreground_tile_luminance_mae=0.)
        try:
            for original, viewing in zip_longest(*pair):
                if original is None or viewing is None:
                    raise ValueError('Viewing copy added or removed a real frame')
                difference = np.abs(original.astype(np.int16) - viewing.astype(np.int16))
                mean, fraction = float(difference.mean()), float((difference > 32).mean())
                maximum_mean, maximum_fraction = max(maximum_mean, mean), max(maximum_fraction, fraction)
                if (mean > VIEWING_ERROR['maximum_frame_mean_absolute_rgb_error']
                        or fraction > VIEWING_ERROR['maximum_fraction_channels_over32']):
                    raise ValueError('Decoded viewing copy exceeds frozen RGB error tolerances')
                check = verify_caption_regions(original,viewing,regions)
                for key in ['maximum_region_luminance_mae','maximum_foreground_tile_luminance_mae']:
                    caption_proof[key] = max(caption_proof[key],check[key])
                n += 1
        finally:
            for stream in pair:
                stream.close()
        pts = decoded_pts(root / (name + '.mp4'))
        if len(pts) != master['frames'] or not np.allclose(pts, master['pts'], rtol=0, atol=1e-8):
            raise ValueError('Viewing PTS changed actual1x time')
        answer[name] = dict(frames=n, pts_s=pts,
            maximum_frame_mean_absolute_rgb_error=maximum_mean,
            maximum_fraction_channels_over32=maximum_fraction,
            **caption_proof,
            sha256=sha256(root / (name + '.mp4')))
    return answer
