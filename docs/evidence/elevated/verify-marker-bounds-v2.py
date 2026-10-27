#!/usr/bin/env python3
"""Post-freeze evidence only: enclose every decorative marker style.

Does not step physics, query a policy, or change any frozen experiment file.
The original framing reports used radius r even for boxes/cylinders. This
supplement supersedes only that goal-marker bound with sqrt(3)*r, preserving
the same prospective 40px criterion and the original reports unchanged.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from adaptive_drone_patrol.elevated_media import ElevatedRenderer, load_elevated_camera, projected_bounds
from adaptive_drone_patrol.payload_solver_media import native_frame_rows
from adaptive_drone_patrol.scenario import Scenario


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=digest(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-final', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run
    freeze_path = run / 'joint-freeze-v1.json'
    frozen = json.loads(freeze_path.read_text())
    camera_path = Path(frozen['binding']['camera_path'])
    camera = load_elevated_camera(camera_path)
    sources = [Path('src/adaptive_drone_patrol') / name for name in
               ('elevated_media.py', 'payload_solver_media.py')]
    layout_path = Path('configs/demo_scenarios_payload_elevated/layout.json')
    assert digest(layout_path) == frozen['binding']['layout_sha256']
    for path in [camera_path, layout_path, *sources]:
        assert digest(path) == frozen['files'][str(path)], path
    trials = [(p.parent, run / 'selected-framing-v1' / (p.parent.name + '.rows.jsonl.gz'), None)
              for p in sorted((run / 'selected-development-v1').glob('*/case.json'))]
    assert len(trials) == 15
    if args.include_final:
        for scenario in ('normal', 'emergency', 'cost_shift'):
            media = run / ('final-' + scenario + '-media-v1')
            proof = json.loads((media / 'video-proof.json').read_text())
            assert proof['whole_recording_recomposed']
            assert proof['freeze_sha256'] == digest(freeze_path)
            assert proof['binding'] == frozen['binding']
            assert proof['formal_independent_statistics_sha256'] == digest(run/'formal-v1/independent-statistics.json')
            for path in sources:
                assert proof['sources'][str(path)] == digest(path)
            trials.append((run / 'formal-v1' / ('basic-' + scenario + '-1'), media / 'framing.json', proof))
    results = []
    for trial, rows_path, proof in trials:
        inputs = [trial/'flight/runtime/raw.jsonl.gz', trial/'flight/fixture/model.xml',
                  trial/'mission/scenario.json']
        input_records = {str(path): record(path) for path in [*inputs, rows_path]}
        if proof is None:
            for path in [*inputs, rows_path]:
                assert input_records[str(path)]['sha256'] == frozen['files'][str(path)]
        else:
            for path in inputs:
                assert input_records[str(path)]['sha256'] == proof['trial_files'][str(path.relative_to(trial))]
        before = record(trial / 'flight/runtime/raw.jsonl.gz')
        scenario = Scenario.from_dict(json.loads((trial / 'mission/scenario.json').read_text()))
        goals = scenario.coords[1:]
        native_frames = 0
        with gzip.open(before['path'], 'rt') as stream:
            for i, frame in enumerate(native_frame_rows(json.loads(line) for line in stream)):
                assert frame['step'] == i
                if i == 0:
                    initial = frame['actual']
                native_frames += 1
        assert native_frames > 0
        renderer = ElevatedRenderer(trial / 'flight/fixture/model.xml', camera)
        try:
            # Display forward only. All visited yields the largest style (box).
            renderer.render(initial, goals, scenario.priorities, list('ABCDEFGHIJ'), 'A')
            scene = renderer.renderers['overview'].scene
            marker_geoms = list(scene.geoms)[scene.ngeom-10:scene.ngeom]
            for geom, goal in zip(marker_geoms, goals):
                assert np.allclose(geom.pos, goal, atol=1e-6, rtol=0)
                assert np.allclose(geom.size, camera['marker_radius_m'], atol=1e-7, rtol=0)
            cameras = [{key: np.asarray(getattr(c, key)).tolist() for key in
                        ('pos', 'forward', 'up', 'frustum_near', 'frustum_top', 'frustum_bottom')}
                       for c in scene.camera]
            # Include actual scene float conversion error in the enclosing sphere.
            radius = max(float(np.linalg.norm(geom.size) + np.linalg.norm(geom.pos-goal))
                         for geom, goal in zip(marker_geoms, goals))
            boxes = [projected_bounds(scene.camera, [goal], [radius], width=1280, height=720,
                                      fovy=renderer.model.vis.global_.fovy) for goal in goals]
            bounds = np.array([min(b[0] for b in boxes), min(b[1] for b in boxes),
                               max(b[2] for b in boxes), max(b[3] for b in boxes)])
            margins = [bounds[0], bounds[1], 1280-bounds[2], 720-bounds[3]]
            if rows_path.suffix == '.gz':
                with gzip.open(rows_path, 'rt') as stream:
                    rows = [json.loads(line) for line in stream]
            else:
                rows = json.loads(rows_path.read_text())
            assert len(rows) == native_frames
            if proof is not None:
                assert len(rows) == proof['framing']['frames']
            for i, row in enumerate(rows):
                assert row['step'] == i and abs(row['time_s']-i/100) < 1e-7
                assert np.allclose(row['goals'], renderer.bounds[0]['goals'], atol=1e-8, rtol=0)
            assert input_records == {str(path): record(path) for path in [*inputs, rows_path]}
            results.append(dict(case=trial.name, frames=len(rows), enclosing_radius_m=radius,
                                per_goal_bounds_px=boxes, union_bounds_px=bounds.tolist(),
                                margins_left_top_right_bottom_px=margins, passed=bool(min(margins) >= 40),
                                actual_camera=cameras, actual_marker_centers=[g.pos.tolist() for g in marker_geoms],
                                actual_marker_sizes=[g.size.tolist() for g in marker_geoms],
                                fovy_deg=float(renderer.model.vis.global_.fovy),
                                raw=before, framing_rows=record(rows_path),
                                model=record(trial / 'flight/fixture/model.xml'),
                                scenario=record(trial / 'mission/scenario.json')))
        finally:
            renderer.close()
    for path in [camera_path, layout_path, *sources]:
        assert digest(path) == frozen['files'][str(path)]
    output = dict(schema='elevated-marker-geometry-supplement-v2', passed=all(r['passed'] for r in results),
                  freeze=record(freeze_path), executed_supplement=record(__file__),
                  frozen_sources=[record(p) for p in sources], camera=record(camera_path), layout=record(layout_path),
                  new_physical_success_count=0, criterion_px=40, cases=results,
                  minimum_margins_left_top_right_bottom_px=np.min(
                      [r['margins_left_top_right_bottom_px'] for r in results], axis=0).tolist(),
                  rationale='All styles use size=[r,r,r] and fixed goal centers. Sphere radius r, '
                  'cylinder radius sqrt(2)*r, and box radius sqrt(3)*r are enclosed by sqrt(3)*r. '
                  'Frozen renderer recreates the same fixed overview camera every frame, independent '
                  'of vehicle pose; retained complete-frame goal bounds confirm its invariance. '
                  'This strengthens the original underestimated goal bound only; body bounds, '
                  'camera, flight, acceptance thresholds and original evidence are unchanged. '
                  'Text-label readability remains subject to human review.')
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(passed=output['passed'], cases=len(results),
                         minimum_margins=output['minimum_margins_left_top_right_bottom_px'])), flush=True)


if __name__ == '__main__':
    main()
