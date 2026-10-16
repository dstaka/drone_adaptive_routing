#!/usr/bin/env python
"""Exercise long transitions/reversals and retain every actual physical sample."""
import argparse
import json
import numpy as np
from adaptive_drone_patrol.util import load_config, json_write, unique_dir
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.scene import scene_manifest, project
from adaptive_drone_patrol.patrol_runner import ArrivalGate, FlightSample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--without-markers', action='store_true')
    a = p.parse_args()
    cfg = load_config(a.config)
    s = Scenario.from_dict(load_config(cfg['scenario_files'][0]))
    manifest = scene_manifest(s)
    if not a.without_markers:
        cfg['scene_manifest'] = manifest
    from adaptive_drone_patrol.flight_agent import FlightSession
    out = unique_dir(a.out)
    f = FlightSession(cfg, seed=11, camera=not a.without_markers)
    targets = [s.coords[i] for i in [2, 10, 1, 4, 7, 10, 2, 2]]
    samples, legs, step = [], [], 0
    status, error = 'SUCCEEDED', None
    try:
        if not a.without_markers:
            from PIL import Image
            Image.fromarray(f.cam.render()[0]).save(out/'initial.png')
            json_write(out/'scene_manifest.json', manifest)
            actual = []
            # Verify independent pinhole implementation against runtime camera matrices.
            for marker in manifest['markers']:
                point = np.array([*marker['world_xyz'], 1.])
                camera_point = f.cam.extrinsics @ point
                pixels = f.cam.intrinsics @ camera_point[:3]
                xy = (pixels[:2] / pixels[2]).tolist()
                expected = project(marker['world_xyz'], manifest['camera'])
                if not np.allclose(xy, expected, atol=1e-4):
                    raise RuntimeError(f'Projection mismatch: {xy}, {expected}')
                actual.append({'id':marker['id'], 'projected_xy':xy, 'expected_xy':expected})
            json_write(out/'projection_check.json', actual)
        for i,target in enumerate(targets):
            f.set_target(target)
            gate = ArrivalGate(f.dt, max_tilt_deg=60)
            peak_obs, errors, speeds = 0., [], []
            for _ in range(3000):
                r = f.advance()
                step += 1
                residual = float(np.linalg.norm(np.array(r['xyz'])-target))
                speed = float(np.linalg.norm(r['velocity']))
                obs = float(f.adapter.current_observation()['policy'][0,:3].abs().max().item())
                peak_obs = max(peak_obs, obs)
                samples.append({'step':step,'target_index':i,'target':target, **r, 'relative_observation_absmax':obs})
                flags = gate.update(FlightSample(r['xyz'],r['velocity'],r['done'],r['reset'],r['tilt_deg']), target)
                errors.append(residual)
                speeds.append(speed)
                if 'WAYPOINT_REACHED' in flags:
                    legs.append({'target_index':i,'target':target,'end_step':step,
                                 'max_tracking_error_m':max(errors),'max_speed_mps':max(speeds),
                                 'peak_relative_observation':peak_obs,'dwell_s':gate.steps*f.dt})
                    if not a.without_markers:
                        from PIL import Image
                        Image.fromarray(f.cam.render()[0]).save(out/f'hover_{i}.png')
                    break
            else:
                raise RuntimeError(f'Transition timeout: {i}')
    except Exception as exc:
        status,error='FAILED',repr(exc)
    finally:
        with (out/'samples.jsonl').open('x') as stream:
            for r in samples:
                stream.write(json.dumps(r)+'\n')
        json_write(out/'summary.json',{'status':status,'error':error,'legs':legs,'steps':step,
                                     'evidence_kind':'physical','execution_seed':11,
                                     'markers':not a.without_markers})
    raise SystemExit(0 if status=='SUCCEEDED' else 2)


if __name__ == '__main__':
    main()
