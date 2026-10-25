# Third-party notices and original contribution

## Genesis and official drone examples

- Project: Genesis-Embodied-AI/genesis-world, https://github.com/Genesis-Embodied-AI/genesis-world
- Exact reference: v1.4.3, commit `216a708e06124595521a9d36a51fae5393fd4ff8`.
- Reused files: `examples/drone/hover_env.py`, `hover_train.py`, `hover_eval.py`, retained unchanged under `third_party/genesis_drone/`.
- License: Apache License 2.0. The repository LICENSE supplies the same license text. The Genesis package provides its bundled Crazyflie cf2x URDF, meshes and plane/sphere assets; asset-specific upstream notices/licenses remain in the pinned installed distribution. No asset is reauthored or redistributed separately here.
- Adaptation: a checked in-memory GUI flag change; a subclass that prevents evaluation command resampling and automatic reset; refreshed external target-relative state. Original rewards, RPM mapping, actor architecture and learned PPO inference are preserved.

## Operator-provided notebook

`RL_drone_hovering.ipynb`, SHA-256 `4d993fa3cb9fcb53d77bfa6106b26864316a3c2b70e003e67fbbd6f18b8daf8d`, is a supplied reference and preserved unchanged. Its source-download, separate training/inference process, configuration load and camera-contract approach informed this project. Embedded video remains outside Git. No absent challenge image or historical result is presented as a new measurement.

## RL and simulation dependencies

- RSL-RL 5.5.1: https://github.com/leggedrobotics/rsl_rl, BSD-3-Clause; supplied by the working operator environment.
- Stable-Baselines3 2.4.1: https://github.com/DLR-RM/stable-baselines3, MIT.
- SB3-Contrib 2.4.0: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib, MIT. MaskablePPO and mask-aware deterministic inference are used.
- Gymnasium 1.0.0: https://github.com/Farama-Foundation/Gymnasium, MIT.
- PyTorch: https://github.com/pytorch/pytorch, BSD-style license; 2.2.2 CPU routing and 2.14.0+cu132 flight.
- NumPy (BSD-3-Clause), Matplotlib (PSF-derived license), Pillow (HPND), PyYAML (MIT), ReportLab (BSD), FontTools (MIT), imageio-ffmpeg (BSD-2-Clause). Exact resolved distributions are in platform-specific locks.
- FFmpeg: system/bundled build, LGPL/GPL terms depend on build configuration, with libx264 for H.264 encoding. Executables remain external, not committed.

## MuJoCo solver recovery

- Project: Google DeepMind MuJoCo, https://github.com/google-deepmind/mujoco . Version3.15.0, upstream commit `9ea3cdfcae93bf2cc4dc0e1a1627c5a39a1e06e5`.
- License: Apache License2.0. The retained upstream LICENSE has SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`; the upstream source archive has SHA-256 `1a4d9ab6135b63d07309e86d58931a71f8e55109409ad77e18579d07f96f62c1`.
- The installed engine is unmodified. Actual library, installed-file inventory, package RECORD, upstream source and dependency locks are retained separately from project implementation. No engine source or binary is committed here.
- The generated MJCF adapter preserves the inherited Crazyflie assets, motor mapping and articulated payload physics. It is project code, not copied MuJoCo engine implementation.
- New isolated solver environments use PyTorch2.14.0+cpu for flight learning and PyTorch2.2.2 CPU for routing. Historical CUDA locks remain historical; current solver locks are separate files under `requirements/`.

## Japanese font

Noto Sans JP from https://github.com/google/fonts/tree/main/ofl/notosansjp . `NotoSansJP[wght].ttf` and its `OFL.txt` were downloaded as retained assets. License: SIL Open Font License 1.1; both asset hashes are in the manifest. A static weight-400 instance is embedded/subset in the PDF. No restricted system font is copied or committed. Installed Hiragino can be an operator-local Pillow fallback but is not a redistributable project asset.

## Original work

The new contribution is analytical learned next-inspection selection under observable urgency/logical cost updates; a shared priority-weighted completion objective and exact static reference; masked three-checkpoint policy training; physically verified arrival/dwell boundaries; external goal observation refresh without teleport/reset; fail-closed persistent planner IPC; same-run camera/telemetry/event evidence; and deterministic Japanese video/report composition. No perception, obstacle avoidance, battery model, real sensor inspection or hardware safety certification is claimed.
