"""Explicit Genesis RGB capture with exact simulation-step/frame associations."""

import csv
import subprocess
import numpy as np


class FrameRecorder:
    def __init__(self, cam, path, frames_path, dt, fps=50):
        self.cam = cam
        self.dt = dt
        self.interval = max(1, round(1 / (dt * fps)))
        self.fps = 1 / (dt * self.interval)
        self.frames = open(frames_path, 'w', newline='')
        self.writer = csv.DictWriter(self.frames, fieldnames=['frame_index', 'pts_s', 'simulation_step', 'sim_time_s', 'event_seq'])
        self.writer.writeheader()
        self.proc = None
        self.path = str(path)
        self.index = 0

    def capture(self, step, force=False, event_seq=None):
        if step % self.interval and not force:
            return None
        rgb = np.asarray(self.cam.render()[0])
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise RuntimeError(f'Unexpected renderer image: {rgb.shape}, {rgb.dtype}')
        h, w, _ = rgb.shape
        if self.proc is None:
            self.proc = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(self.fps), '-i', '-',
                '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', self.path],
                stdin=subprocess.PIPE)
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        index = self.index
        self.writer.writerow({'frame_index': index, 'pts_s': index / self.fps,
                              'simulation_step': step, 'sim_time_s': step * self.dt, 'event_seq': event_seq})
        self.frames.flush()
        self.index += 1
        return index

    def close(self):
        self.frames.close()
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait(timeout=30):
                raise RuntimeError('Video encoding failed')
