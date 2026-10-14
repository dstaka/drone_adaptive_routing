# Proven notebook baseline and newly executed reproduction

The supplied `RL_drone_hovering.ipynb` remains unchanged and excluded from Git because it embeds media. SHA-256: `4d993fa3cb9fcb53d77bfa6106b26864316a3c2b70e003e67fbbd6f18b8daf8d`. Challenge images were not present; the implementation uses the self-contained specification. No historical notebook output is relabeled as a new run.

The authorized instance was fresh: no `hover_env.py`, `cfgs.pkl` or trained checkpoint was found in the inspected project locations. Its original `/venv/main` contained Python 3.12.14, Torch 2.14.0+cu132, NumPy 2.5.3, RSL-RL 5.5.1 and Genesis 1.4.4. This original environment was preserved. A separate `.venv-flight` inheriting its site packages installed Genesis 1.4.3 and pinned NumPy 2.5.3. Its full final closure is in `requirements/flight-linux.lock.txt`; package consistency was checked.

The three official drone sources were obtained from `https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/v1.4.3/examples/drone/`. The tag resolves to commit `216a708e06124595521a9d36a51fae5393fd4ff8`. Original copies are retained in `third_party/genesis_drone/`; the remote copies are also frozen with checkpoint/config backups and hashes. These sources are Apache-2.0. The recording adapter verifies exactly one `GUI=True,` occurrence and changes it to `GUI=False,` only in the imported copy.

New GPU evidence:

1. Unique experiment `smoke_20261013`: 32 environments, 2 iterations, real checkpoints model_0/model_1, exit 0, 119.207 s including initial compilation. This verifies plumbing, not competence.
2. Smoke checkpoint save/load and camera replay: 400 steps at dt=0.01, H.264 640x480, 50 FPS, 200 frames, duration 4.000 s, full ffmpeg decode. Seven resets demonstrate why this clip is not competent patrol evidence. One frame was visually inspected.
3. Unique experiment `notebook_20261013`: 256 environments, 301 iterations, exit 0, 273.828 s. PPO architecture and upstream rewards unchanged. No scripted motor controller was introduced.
4. Frozen final checkpoint `model_300.pt`: SHA-256 `5d9d193a6fc4f6066c6fcb66a41e7ccc7203d450d2515c95a874deb3830d33d7`. Its separate 400-step recording replay had zero resets, duration 4 s and successful full decode; GPU phase 14.845 s.
5. One external goal, two consecutive goals, and three seeded three-goal tests all completed with unchanged pose/action history at target switches and continuous 0.50 s service. The three-goal fixture completed at 6.30 simulation seconds. All measured arrival residuals were 0.086–0.127 m.
6. Actual GPU pytest verified target-relative observation entries, no teleport, retained prior actions, resampling suppression, and real pre-reset termination capture. One integration test passed in 84.96 s; the durable phase lasted 88.859 s.

Remote GPU model is RTX 5060 Ti, 16311 MiB, driver 595.91.07, CUDA driver maximum 13.2. Observed GPU memory samples during phases were 360–460 MiB, with more than 15 GiB free. These are observed samples, not a guaranteed exhaustive allocator peak. The exact CUDA wheel/backend and environment paths are in discovery artifacts. No driver/system CUDA changes occurred.

Reproduction: use the preserved exact flight lock and source, a new experiment name, and `scripts/train_flight.py`; freeze explicit checkpoint/config/source paths with `scripts/doctor.py`. Camera replay must run in a fresh subprocess. Never use an existing upstream experiment name: its original trainer deletes colliding logs. The project wrapper rejects collisions before invoking it.

## Final correction validation

After the independent code review, the owned source snapshot was transferred by an explicit 69-file allowlist. Its archive SHA-256 is recorded in every final run and `docs/evidence/transfer.json`. A durable final job reran the real GPU contract test (1 passed) and all nine recorded patrols (9 succeeded). Total GPU wall accounting including this phase is 1,233.700360 s. The final evidence transfer contained 90 files whose byte sizes and SHA-256 matched locally. Previous recorded attempts remain preserved separately.
