# Verified package and coordinate contracts

## Environments

- Local routing: Intel macOS, Python 3.11.14, Torch 2.2.2, NumPy 1.26.4, Gymnasium 1.0.0, SB3 2.4.1, SB3-Contrib 2.4.0. Python 3.13 was rejected because no Intel macOS Torch wheel was available. SB3 2.7.0 requires Torch >=2.3, which also has no usable Intel macOS wheel. No unsupported dependency was forced.
- Linux routing: independent CPU-only Torch 2.2.2 environment, Python 3.12.14; same Gymnasium/SB3/Contrib contract. It does not import Genesis and does not mutate flight packages. Both complete locks are retained.
- Linux flight: Python 3.12.14, Genesis 1.4.3, RSL-RL 5.5.1, Torch 2.14.0+cu132, NumPy 2.5.3. Complete dependency closure and exact index are recorded in its lock/discovery evidence. `pip check` passed.

## Flight source, state and policy

Pinned `HoverEnv.reset()` returns a TensorDict containing `policy`, shape (N,17); `step(actions)` returns observation/reward/done/extras. The project preserves this separate RSL-RL contract. `OnPolicyRunner.load` and `get_inference_policy` use the frozen checkpoint and original training configuration. Actor and critic each have 128/128 Tanh hidden layers; actor outputs four propeller controls. Observation normalizer is Identity in this trained configuration.

Observation offsets: relative target xyz [0:3] scaled by 1/3 and clipped to [-1,1]; quaternion [3:7] in wxyz order; body-frame linear velocity [7:10] scaled by 1/3; body-frame angular velocity [10:13] scaled by 1/pi; previous actions [13:17]. Targets, positions and direct `drone.get_vel()` telemetry are world-frame coordinates. One environment has zero environment-origin displacement. Source initial quaternion [1,0,0,0] and inspected Genesis quaternion functions establish wxyz. Physics dt is 0.01 s; rigid solver dt is 0.005 s. Motor RPM is `(1 + clip(action,-1,1)*0.8)*14468.429183500699`.

Command bounds: x/y [-1,1] m and z exactly 1 m. Original evaluation automatically resamples near a target and resets on termination; those behaviors invalidate naive patrol counting. The project subclass suppresses command resampling only after external target injection and captures termination without performing an automatic reset. `set_target` updates commands, marker, relative-position/reward history and observation, preserving pose and previous motor actions. Before each inference, effective goal and relative observation are asserted.

Physical acceptance frozen in `configs/flight_eval.yaml`: radius 0.20 m, speed 0.30 m/s, first arrival hold 20 steps, total service 50 consecutive steps; violations clear the hold. Source floor limit 0.1 m is retained. Evaluation uses a documented stricter 60-degree tilt limit than the upstream 180-degree values, fixed before trials. Source displacement limits 3/3/2 m are retained. Evaluation episode limit changes from 15 to 150 s so the mission watchdog (120 s) owns timeout. It does not change the actor architecture/reward/control convention. Per-leg limit 30 s, wall watchdog 600 s. Preflight stabilization is zero; initialization uses exactly one upstream reset.

## Rendering

Original source camera resolution is 640x480 (the notebook's unused VIDEO_RES does not alter it). `Camera.start_recording(save_to_filename=None,fps=None)` streams frames automatically every integer number of scene steps. Baseline replay uses this inspected API at 50 FPS, with `stop_recording()` in finally; recorded duration/frame count was independently verified.

Final patrols use the inspected `Camera.render()` RGB return and a streaming ffmpeg encoder to expose exact capture step/time. Every two 0.01-s steps is one 50-FPS frame; a physical final service event can force one additional boundary frame, whose actual time is listed explicitly. This creates a potentially nonuniform final interval, not an assumed constant simulation-time mapping. Frame map and telemetry belong to the same run. Raw RGB image shape/type are validated. Offline resizing to a 1280x720, 30-FPS canvas does not claim native 720p simulation rendering.

Official version-specific routing reference: https://sb3-contrib.readthedocs.io/en/v2.4.0/modules/ppo_mask.html . Masks are passed explicitly to deterministic inference. The Gymnasium checker samples unmasked actions; invalid actions leave modeled state unchanged and truncate at three. Unbounded Box warnings are expected because finite scenario coordinates/objective constants are validated without silently clipping unseen values.

## Fresh Linux installation

The platform locks now record PyPI plus the verified official PyTorch CPU/cu132 wheel indexes. `docs/wheel_sources.json` records the exact CPython 3.12 Linux x86_64 wheels, their upstream SHA-256 fragments, and the index checked on 2026-10-13. The inherited Torch packages had no `direct_url.json`; the entries establish available matching builds, without claiming to recover the image builder's original download transaction. The complete existing environment was frozen and `pip check` passed; a second clean full CUDA installation was not performed.

```bash
python3.12 -m venv .venv-routing
.venv-routing/bin/python -m pip install -r requirements/routing-linux.lock.txt
.venv-routing/bin/python -m pip install -e . --no-deps
.venv-routing/bin/python -m pip check

python3.12 -m venv .venv-flight
.venv-flight/bin/python -m pip install -r requirements/flight-linux.lock.txt
.venv-flight/bin/python -m pip install -e . --no-deps
.venv-flight/bin/python -m pip check
```

Physical patrol execution rejects `return_to_base=true` before creating a run. Optional analytic return accounting is tested in the routing environment; the required physical demonstrations use false. Physical acceptance and watchdog values must be finite, with positive radius, speed, hold, service and timeout values. Geometry validation uses the configured radius. Service must include the arrival hold.

Each video composition reserves its complete output set. The canonical `submission.mp4` has `timeline.json` and `submission.selection.json`; a different filename has `<stem>.timeline.json` and `<stem>.selection.json`. Existing video/timeline/selection/reservation paths are rejected. Input runs are never modified by composition. Failed partial outputs remain identifiable; choose a fresh output name for retry. Manifest creation likewise rejects an existing output.
