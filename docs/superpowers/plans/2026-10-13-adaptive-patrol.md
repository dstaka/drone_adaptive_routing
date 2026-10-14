# Adaptive Drone Patrol Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task. The operator's spec explicitly authorizes autonomous execution without additional plan gates.

**Goal:** Execute three authentic adaptive A/B/C inspection patrols and deliver tested local artifacts and a commit.

**Architecture:** Shared pure scenario/objective state drives a CPU MaskablePPO planner. A preserved Genesis/RSL-RL flight policy receives externally refreshed goals. Physical arrival owns state transitions; evidence is recorded with the same execution.

**Tech Stack:** Python, Gymnasium, SB3-Contrib MaskablePPO, Genesis 1.4.3, RSL-RL 5.5.1, ffmpeg, pytest, Ruff, ReportLab.

**Spec:** `spec.md` (complete implementation contract).

## Global Constraints
- Exactly three checkpoints; no publication or push.
- GPU execution cap 8 hours, persisted ledger, one project GPU job at a time.
- Preserve reference notebook and existing remote environment; isolate changed dependencies.
- English source comments/docstrings; Japanese audience documentation/report/video.
- Frozen policies/geometry; 0.20 m radius, 0.30 m/s speed, 0.20 s arrival hold, total 0.50 s dwell.
- Three routing seeds 42/43/44, 100 static and 50 dynamic held-out cases per seed.
- Authentic footage, failure-inclusive results, no simulated substitute for missing physical evidence.

## Review Focus
- Output collision must fail before any overwrite (Tasks 1, 5).
- Nonfinite/invalid scenario input must fail validation (Task 2).
- Terminated flight/reset near a goal must never count as a visit (Task 4).
- Dead/malformed planner communication must fail the mission (Task 4).
- Missing or mismatched artifacts must fail delivery verification (Task 5).

### Task 1: Preservation and durable execution
**Files:** `scripts/doctor.py`, `scripts/run_remote_job.sh`, `src/adaptive_drone_patrol/jobs.py`, `tests/test_jobs.py`, `docs/proven-baseline.md`.
**Interfaces:** `run_job(command, output, ledger, timeout_s)` preserves unique outputs, exit status and cumulative GPU wall time. `sha256(path)` identifies source/models.
- [x] Write and observe failing budget/collision/deadline tests; implement and pass.
- [x] Discover remote packages/assets/checkpoints read-only; retain exact source and original snapshot.
- [x] Create separate flight environment for exact 1.4.3; 32-env/2-iteration smoke, then notebook 256-env/301-iteration training if no checkpoint.
- [x] Freeze hashes/lock; baseline subprocess camera replay, probe/decode/inspect.

### Task 2: Analytical foundations
**Files:** `scenario.py`, `objective.py`, `routing_env.py`, `baselines.py`, `tests/test_routing.py` under package/tests; `pyproject.toml`.
**Interfaces:** `Scenario`, `RoutingState`, `leg_cost(...)`, `RoutingEnv.reset/step/action_masks`, `choose_baseline(state, name)`.
- [x] Write failing hand-computed objective, event snapshot, invalid/mask/terminal, return and seeded schema tests.
- [x] Implement finite scenario validation, immutable departure accounting, fixed observation schema with no future event features, split generator, shared event transitions and static oracle.
- [x] Run pytest and Gymnasium checker; validate all planners against shared accounting.

### Task 3: Learned routing and comparisons
**Files:** `routing_agent.py`, `benchmark.py`, `scripts/{train_routing,eval_routing}.py`, `configs/routing_train.yaml`, `tests/test_agent.py`.
**Interfaces:** `RoutingAgent.predict(state)`, `train(config, seeds, out)`, `benchmark(models, split, episodes, out)`; JSON model selection and schema.
- [x] Write real short training/save/reload/mask tests; observe failure, implement and pass.
- [x] Train 50k static + 50k mixture per seed on CPU, validate and freeze selection before held-out evaluation.
- [x] Evaluate shared 100 static/50 dynamic inputs per seed; save CSV/JSON, paired differences, exact ratios, event counterfactuals and honest plots.
- [x] Search only demonstration split for both event action changes; freeze common initial state/geometry.

### Task 4: Physical integration
**Files:** `target_adapter.py`, `flight_agent.py`, `patrol_runner.py`, `recorder.py`, `scripts/{train_flight,eval_flight,run_demo_scenarios}.py`, flight/demo configs, `tests/test_patrol.py`, `tests/test_ipc.py`.
**Interfaces:** adapter `set_target/current_observation/step`, immutable `FlightSample`, `PatrolRunner`, persistent validated JSON planner.
- [x] Write failing dwell violation, reset/fall/NaN/mismatch/timeouts, event-order and IPC tests, then implement and pass.
- [x] Inspect pinned source; preserve architecture, disable automatic target changes, capture pre-reset state; verify external target observations and physical continuity.
- [x] Run external one-goal/two-goal and three seeded sequential trials; freeze verified controller envelope/configuration.
- [x] Run three trials per scenario with frozen policy pair; retain every result and same-run recording/telemetry/frame map, both counterfactual changes.

### Task 5: Evidence and local delivery
**Files:** `video.py`, `scripts/{replay_decisions,build_video,verify_delivery,transfer}.py`, `tests/test_evidence.py`, `README.md`, `docs/{compatibility,verification,report_ja}.md`, notices/locks/manifest.
**Interfaces:** verified frame-based timeline/replay, allowlisted transfer, hash/size manifest verification.
- [x] Write failing timeline/cut/manifest/log-reference tests; implement and pass.
- [x] Compose authentic Japanese H.264 60–180 s video, generate measured plots and 2–4 page Japanese PDF; inspect representative video frames and every PDF page.
- [x] Synchronize retained artifacts, compare hashes, run local CPU pytest/Ruff/compile/diff checks and fresh final review.
- [x] Commit explicit task-owned files locally, with an incomplete message if any mandatory gate remains unmet. Save ignored delivery receipt.

## Execution Ledger
- Initial checkout: main, initial commit 43d8bc7; only operator-provided spec/notebook untracked. No existing product code/tests.
- Ruling: Work in the explicitly designated authoritative local checkout; preserve supplied notebook unchanged and exclude embedded media from Git. Additional worktree not needed for this delivery contract.
- M0 access: SSH succeeded using existing trusted host entry; RTX 5060 Ti 16311 MiB, driver 595.91.07. Fresh instance has Genesis 1.4.4, Torch 2.14.0+cu132, RSL-RL 5.5.1; no baseline checkpoint found. Exact 1.4.3 source/wheel available.

- Final independent review found 8 Important and 1 Minor issue, no Critical. One corrective pass addressed every finding; initial 20 regression tests failed then passed, with three further interruption/reconciliation fixtures. Full final CPU suite: 47 passed; final GPU contract: 1 passed; post-review patrols: 9/9 succeeded.
- Physical return ruling: optional analytic return remains tested; physical return is explicitly rejected before execution. Required demonstrations use false. No unsupported return is reported as successful.
- Runtime source snapshot and byte-verified remote evidence are retained. Final local commit and delivery verification are recorded in the ignored delivery receipt after explicit staging.
