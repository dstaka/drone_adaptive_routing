# Dual drone learned coordination implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn allocation, reassignment and yielding for two independently loaded drones in one MuJoCo world, then measure the exact prospective gates.

**Architecture:** A central 242-action attention actor/critic drives two retained C-local-lqr-v1 controllers. A deterministic 100 Hz supervisor operates on continuous references. Experiment gates prohibit training before qualification and held-out exposure before joint freeze.

**Tech Stack:** Python, NumPy, native MuJoCo 3.15.0 float64 RK4/16, PyTorch; optional separately qualified MuJoCo Warp.

**Spec:** `specs/2026-10-27-dual-drone-learned-coordination/spec.md` (user approved 2026-10-27).

## Global Constraints

- Existing merged specs/results are historical, never dual-drone evidence. Work in the user's existing feature branch; preserve unrelated untracked files.
- Run: `outputs/dual_payload_coordination_20261027T144754Z/`. Start timestamp 2026-10-27T14:47:54Z is the first captured clock; prior duration is `not recorded`.
- Same physical world, two complete assemblies, independent loads/addresses, eight motor channels, 100 Hz control, 5 Hz decisions.
- 0.20 m full-assembly separation, floor >0.10 m, downwash exclusion radius 0.60 m / length 1.00 m, constraint residual ≤2e-5, tilt ≤60 degrees. No threshold relaxation.
- 30 s per-drone persistent leg deadline, 300 s team, 600 wall s per formal mission including raw flush; separate compilation.
- 3 seeds 42/43/44; 1,048,576 transitions each; stages 262,144/720,896/65,536. Exactly 4096 transitions/update, 5 PPO epochs, minibatch 512. Terminal checkpoints only.
- Validation300 selects once. Joint freeze precedes split30 routing300 (five methods) and separate split40 physical127. Failed cases score 600; no successful-intersection replacement.
- All §9 gates remain literal; no-eligible selection stops final campaign. Human review of exact final three full-length 1× videos is mandatory.
- Remote connection must be provided for this campaign. No previous host assumption; no push, PR creation, merge or publication.

## Review Focus

- An assignment churn or forced unavailability must not reset either physical state or leg clocks (Task 3/4).
- Callback/model address aliases must never put eight motor forces on a single body (Task 2).
- A stale observer must not report a dead job as running; monotonic clocks from different processes must not be subtracted (Task 1).
- Tampered/missing receipt or a partial raw stream must fail closed, including held-out access (Task 1/7).
- A moving box/link can cross between external samples even if center distances look safe (Task 2/3/7).

### Task 1: Observable contracts, declarations, deterministic cases

**Files:** Create `dual_coordination_progress.py`, `dual_coordination_schema.py`, `dual_coordination_cases.py` under `src/adaptive_drone_patrol/`; `scripts/run_dual_coordination.py`; `configs/dual_coordination.yaml`; corresponding `tests/test_dual_coordination_*.py`.

**Interfaces:** `RunJournal(root)` appends events, derives atomic snapshots and records timed attempts; `run_command(argv, item_id)` updates independently of chat. `generate_case(split, index, scenario, *, freeze=None)` returns strict dual-case-v1; held-out requires verified binding. `encode_action(t1,t2,priority)`/`decode_action(index)` round trip 242 entries. IDs 0..9 denote goals, 10 WAIT; public drone IDs 1/2.

- [ ] Write/run failing tests: overlapping execution union (10 + 10 overlapping 5 = 15 s), crash unknown end, retry preservation, stale after 600 s, malformed schema, field independence, deterministic rejection, held-out refusal.
- [ ] Implement append-only events with file lock, UTC+monotonic provenance and atomic progress; predeclare streams, fixed challenges and video IDs before learning. Hash declarations and original inputs.
- [ ] Run targeted tests and CLI declaration; retain red/green logs, current machine/version manifest; commit only relevant files.

### Task 2: A real shared two-payload MuJoCo model

**Files:** Create `dual_payload_mujoco.py`; tests `test_dual_payload_mujoco.py`. Reuse `payload_ball_asset.py` without changing original model semantics.

**Interfaces:** `DualMuJoCoFixture(original, output, payloads)` builds namespaced bodies/joints/geoms/sites/actuators; `reset(initials)`, `sample()`, `step_rpm((2,4), observe_stages=True)` share one model/data/time. Metadata resolves every address. `runtime()` records masses, geom coverage, exclusion pairs and source hashes.

- [ ] Write/run tests for disjoint joint addresses, 8 isolated motor channels, independent load/reset, one .01 s clock, per-RK-stage force, explicit self exclusions, intentional cross contacts.
- [ ] Preserve inertials, add massless conservative rotor/link/body collision coverage, retain all RK stages/substeps and their audit state.
- [ ] Run actual native physics tests with pinned source asset; no qualification claim until Task 5 passes.

### Task 3: Continuous references and conservative safety

**Files:** Create `dual_coordination_reference.py`, `dual_coordination_safety.py`; corresponding tests.

**Interfaces:** `ContinuousReference.sample(time)` returns p/v/a; `retarget(goal,time)` preserves p/v/a. `SafetySupervisor.decide(state,references,priority)` returns executed reference/brake state and reason. `audit_geometry(model,data)` returns full geometry distance/contact/floor/wash diagnostics.

- [ ] RED: retarget C2 continuity, bounded v/a/jerk, stopping reachability, hovering obstacle, rotated box/rotor/link proximity, between-sample crossing and persistent violations.
- [ ] Implement finite continuous transfers/brakes; horizon covers qualified braking+0.20 s. Conservative substep sweep bounds must be proven or fail closed.
- [ ] Qualify stopping envelopes on development load extremes; preserve failed attempts; freeze measured reference limits before production.

### Task 4: Mission semantics and learned actor/critic

**Files:** Create `dual_coordination_env.py`, `dual_coordination_policy.py`, `dual_coordination_train.py`; corresponding tests.

**Interfaces:** `MissionState.apply(action,tick)` atomically changes ownership; `advance` orders audit/dwell/events/decision; `legal_mask()` yields 242 bools. Central shared-goal encoder, two drone tokens, width128, two attention blocks, 242 logits and scalar value. Trainer persists all RNG/optimizer/environment/counter state.

- [ ] RED: exactly-once service, atomic swap, lock and WAIT semantics, unavailability event boundary, no deadline reset, observed-only features, identity/goal action permutation, reward and terminal bootstrap, resume parity.
- [ ] Implement policy at 5 Hz with current-state and predicted conflict features; no sampled mass/size/future event inputs. Record every proposal/execution and reward component.
- [ ] Run software tests and disposable development rollouts; no production seed updates before Task 5.

### Task 5: Baselines, native qualification and measured acceleration

**Files:** Create `dual_coordination_baselines.py`, `dual_coordination_benchmark.py`, qualification receipts; optional `dual_payload_warp.py` only after capability inspection.

**Interfaces:** Static/dynamic selectors consume own evolving causal state and legal mask; native qualification binds code/assets/config; benchmark receipt supplies projected campaign time and resource needs.

- [ ] RED: static eligibility, dynamic work conservation/ETA/conflict formula, canonical legal priority, no future event, unsafe recovery retains earlier failure.
- [ ] Implement exact §7 baselines. Run isolated actuation/reset, passive/powered, intentional collision, stopping, 330 s +6 endpoints and mixed cases.
- [ ] Obtain current GPU host details. Test CPU workers1/2/4/8 and GPU batches32/128/512/1024 subject to resources, three 4096-step+5-epoch repetitions after warmup, all component timing. GPU physics requires 36 paired cases and §6.2 tolerances.
- [ ] Measure raw size/full mission; project all native validation/audits/storage. Select qualified minimum median whole-campaign time, native favored within10%. Record resource blocks without silently replacing benchmark.

### Task 6: Exact finite training and validation selection

**Files:** Create `dual_coordination_campaign.py`, selection tests, campaign CLI commands.

**Interfaces:** `train` enforces 3 stage counts and complete resume; `validate` runs all three terminal models plus two baselines on common300; `select` returns eligible seed or explicit no-eligible failure.

- [ ] RED: budget overflow, missing seeds, intermediate checkpoint selection, threshold boundaries/ties, incomplete denominators, no-eligible stop.
- [ ] Freeze optimizer/normalization recipe and baseline formula; train seeds42/43/44 with retained failed attempts and exact counters.
- [ ] Native validation300, literal §9 selection, diagnostic ablations only after selection. If none qualifies, report learning unmet and do not open held-out.

### Task 7: Joint freeze, final physics evaluation and independent audit

**Files:** Create `dual_coordination_audit.py`, `dual_coordination_evaluate.py`; corresponding tests.

**Interfaces:** `freeze` requires immutable code/assets/weights/config/selection hashes; `evaluate-routing` opens split30 only after verified freeze; `evaluate-physical` opens separate127; audit reconstructs native raw independently of runtime success summaries.

- [ ] RED: stale/mixed hash, tampered/truncated raw, opened held-out without freeze, denominator omissions, bootstrap boundary and zero ablation held-time gate.
- [ ] Evaluate all5 methods×300 full-physics conditions; N-full physical9+18+100; stratified paired bootstrap10,000 seed10272799.
- [ ] Independently reconstruct state, substep safety, actions/locks/dwell/events/timers, numerics and attribution; report every failed gate unchanged.

### Task 8: Media, verified recovery and delivery

**Files:** Create `dual_coordination_media.py`, `dual_coordination_artifacts.py`; run reports; `docs/dual-coordination-handoff-ja.md` and English PR-description draft.

**Interfaces:** `render` consumes frozen audited native raw for the three predeclared IDs; `verify` checks every frame/timestamp and 5% panel border; `recover` builds independent source/destination size+SHA inventory.

- [ ] RED: shortened/nonmonotonic video, missing load panel/corner, stale review hash, missing/tampered recovery files.
- [ ] Render continuous full1× overview+two details with Japanese HUD. Immediately publish exact paths/hashes and physical metrics.
- [ ] Request human full-length1× review of all3 exact files; retain waiting interval. No completion claim before receipt.
- [ ] Recover all necessary evidence, timings/history; write Japanese outcomes/retrospective and English PR description linked to spec. Run full regression; fresh final review; local commits only.

## Initial rulings and execution conditions

- The spec's “proposed” numerical defaults are approved by the user's current instruction, not experimental measurements. Do not use the old wording to reopen approval or weaken a gate.
- Work in the existing dedicated feature branch and use explicit staging to avoid unrelated work. This follows the user's direction to proceed without generic confirmations.
- Current local Python3.11 environment supports Torch2.2.2 CPU; MuJoCo is absent. Install an isolated pinned native test environment. GPU data collection waits for the requested current connection details.
- No credible end-to-end ETA before full-world throughput/raw measurements. 1,500 routing-held-out missions alone are full physics, additional to validation/physical127.
- All unmeasured setup/host-adjustment intervals remain `not recorded`; UTC timestamp span is not CPU utilization.

Ruling 2026-10-28: spec §7 requires600 post-selection validation ablation diagnostic missions in addition to1500 selection missions. Implement them against the selected immutable terminal, without feeding diagnostics back into selection. Update time projection to include2100 total validation executions. The narrow env no-transfer admission guard must allow validation; preserve its previous source and prove the native stepping/physics code unchanged against the isolated81-case run. Both exact source versions and this compatibility decision remain explicit frozen inputs; no threshold, numeric configuration, or physical case changes. Current executable replay must be repeated after the guard change.

Ruling 2026-10-28T12:05:53.791215+00:00: qualification-batch001 failed7 cases after33000ticks each because the import snapshot omitted2 lazy audit modules;74 cases notlaunched. Do not salvage them as passing or use the previously considered source compatibility bridge. Fix prospective preloading and actual one-tick preflight, then rerun81 with currentenvguard. Original canonical training manifest has5indices/27initial-quaternion scalars differing from server regeneration by at most3.47e-18; use the originally declared local bytes verbatim on bothhosts, preserve both versions and the difference receipt. No rounding or case redefinition.

Ruling 2026-10-28: actual stopping maxima3.35s exposed a spec mismatch: v2 supervisor could predict only2s or local-brake-duration+.20s, without the worst-qualified-stop floor required by§4.3. Preserve its81-case/supplement outputs as diagnostics, stop pending benchmark/admission, enforce a4.0s global floor and independently qualify max measured stop+.20s against it. Current longer brakes still extend the horizon. Requalify the new executable; do not bridge old v2 admission. A stationary-state regression and a crossing outside2s both fail against preserved old source. This corrects implementation to the existing spec; it does not change physical/safety/deadline/acceptance criteria. Cost: repeated qualification and current-profile development replay before production.
