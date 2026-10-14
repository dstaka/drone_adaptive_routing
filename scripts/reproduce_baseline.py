#!/usr/bin/env python
"""Replay the frozen notebook policy and headless camera in a fresh process."""
import argparse
import inspect
import json
import subprocess
import torch
from adaptive_drone_patrol.flight_agent import load_flight
from adaptive_drone_patrol.util import load_config, unique_dir, json_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=400)
    a = p.parse_args()
    if a.steps <= 0:
        p.error('--steps must be positive')
    out = unique_dir(a.out)
    env, adapter, policy = load_flight(load_config(a.config), external=False)
    obs = env.reset()
    cam = env.cam
    start = inspect.signature(cam.start_recording).parameters
    stop = inspect.signature(cam.stop_recording).parameters
    new_api = 'save_to_filename' in start and 'fps' in start
    video = out / 'baseline.mp4'
    if new_api:
        cam.start_recording(save_to_filename=str(video), fps=50)
    elif 'save_to_filename' in stop:
        cam.start_recording()
    else:
        raise RuntimeError('Unverified recording API')
    resets = 0
    try:
        with torch.no_grad():
            for i in range(a.steps):
                obs, _, done, _ = env.step(policy(obs))
                resets += int(done[0].item())
                if not new_api and (i + 1) % 2 == 0:
                    cam.render()
    finally:
        if new_api:
            cam.stop_recording()
        else:
            cam.stop_recording(save_to_filename=str(video), fps=50)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
                        '-show_streams', '-of', 'json', str(video)]))
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
    json_write(out / 'result.json', {'status': 'PASS', 'steps': a.steps, 'dt': env.dt,
               'resets': resets, 'recording_api': 'start' if new_api else 'stop', 'probe': probe})


if __name__ == '__main__':
    main()
