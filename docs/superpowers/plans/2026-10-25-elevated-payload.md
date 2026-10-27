# Elevated Payload Implementation Plan

> Execute inline with superpowers:executing-plans. User authorized the spec and requested continuous execution without generic confirmations.

**Goal:** Implement and measure elevated destinations, closer detail rendering, and the finite routing comparison without weakening acceptance.
**Architecture:** Add explicit versioned elevated modules and readers alongside immutable legacy contracts. Select routing before physical development, and jointly freeze before either held-out cohort is materialized.
**Tech Stack:** Python 3.11, NumPy PCG64, SB3 MaskablePPO/Torch CPU, qualified MuJoCo 3.15.0, FFmpeg.
**Spec:** [Revision 2](../../../specs/2026-10-25-elevated-payload-destinations-detail-zoom/spec.md)

## Global constraints

- Preserve old outputs/specs and unrelated workspace files. New run namespace only; local commit only.
- Fixed altitude transform z=2+1.5*(z-1); unchanged x/y/S, physics, objective, events, dwell and floor predicates.
- F42/F43/F44 weights from B; fresh Adam, each 50176 static + 50176 mixture transitions. Terminal-only selection.
- 300 validation, separate unopened 300 routing held-out, separate unopened physical 127. Root/key and selection rules in spec are binding.
- No flights counted from routing/camera evidence. No completion before final three exact-hash human reviews.

## Review focus

- Lazily sampled train collisions and independent keyed event fields must not leak model seed or cache state.
- Incomplete baseline strata must not suppress the 5% guard on other complete strata.
- Reset/counter behavior must preserve exactly two complete PPO rollout budgets and fresh optimizer.
- Higher coordinates must reach encoding unchanged; no hidden clipping or legacy hash bypass.
- Projected corners/rotors and raw substeps, rather than centers or runner flags, establish framing/clearance.

## Tasks

1. [x] Inventory: record namespace, baseline source/status, recovered baseline SHA/size and environment; rehash predecessor evidence and map qualification dependencies. Baseline CPU regression log.
2. [x] Layout: tests for affine transform, unchanged nongeometry fields, finite/domain rejection, physical separation and old profile replay; implement `elevated_layout.py`, new scenario YAMLs/layout manifest and old/new plot.
3. [x] Routing data: tests for keyed streams/splits, geometry-only rejection/exhaustion, collision rejection, saved exogenous events and pre-freeze held-out denial; implement `elevated_routing_data.py` and persisted declaration.
4. [x] Routing experiment: tests for fresh weight-only initialization, rollout accounting, unchanged normalization, 1%/5%/tie/incomplete boundaries; implement `elevated_routing_experiment.py` and script. Save effective optimizer/PPO/source/dependency settings before full runs; run exactly three terminal candidates and evaluate 300 validation against B/greedy/fixed/exact, including dynamic boundary exact suffixes.
5. [x] Media development: profile-aware camera, projections including all geometry, fixed pose-follow, overview goals; matched old/new frames. Retain mismatch/failure evidence. No old raw called elevated flight.
6. [x] Physical development: explicit selected checkpoint/scenario binding; clearance envelope; deadline and rotated-corner regressions; selected-model 15 missions and 12-case/120-transfer suite with independent audit. Any tuning bounded by spec and qualification dependency reruns.
7. [x] Joint freeze and formal: hash all code/model/config/prerequisites, open/check collisions/evaluate routing 300 once; physical 127 once, independent native/substep audit, Wilson and per-stratum results. Fail gates block acceptance without replacing trials.
8. [x] Media/delivery: 3 full native100Hz 1x recordings, lossless+HUD+MP4/recomposition/PTS/source/projection proofs; exact paths and measurements immediately; local recovery size/SHA inventory; human-review receipt; Japanese result/handoff and English PR description with spec link; branch review, tests and related local commit.

## Execution ledger

- Initial branch `feature/elevated-payload-destinations-detail-zoom`, baseline `2ea5cac`; spec/index preexist as uncommitted user inputs. Existing virtual environments and recovered outputs inspected without assuming remote availability.
- Ruling: Work in the current task-specific branch and checkout to preserve the supplied uncommitted spec and large recovered assets; no extra worktree or unrelated cleanup. User requested autonomous execution within this workspace.
- Ruling: The supplied detailed spec and explicit execution request satisfy design approval. No redundant design/plan approval round.
- Baseline CPU verification: 1189 passed,20 skipped,1 deselected; 504.25s. Log retained in new run at delivery.
- Predecessor actual rehash: 3608files/8641550668bytes and7519files/5137376485bytes, all exact. Physics/controller/reference source dependencies identical; actual Linux binary check remains mandatory before reuse.
- Routing preflight review: cache provenance, unavailable-metric JSON, absolute script snapshot path, immutable B hash, and partial-run progress preservation fixed. Cache and unavailable-stratum regressions observed RED→GREEN;16 elevated tests passed before full training; then callback plumbing8/8.
- Full routing declaration persisted before300 validation draws; F42/F43/F44 sequential, each two complete50176 stages. No held-out opened.
- Ruling: Partial-run receipts are auditable counters/sampling logs, explicitly not resumable state. No restart is attempted without full optimizer/RNG/environment state.
- New opt-in scenario verification changes only the acceptance reader; legacy default still binds dispersed bytes. Physical sources and duration source closure unchanged; elevated flight retains existing qualified ObservedFixture/StageWriter path, records every16x4 RK stage.
- SSH server authorized by user; source archive transfer initially rejected by automatic approval review and explicitly approved afterward. The user subsequently explicitly approved all necessary source/config/tests/spec/models/assets/receipts/raw/results/videos/logs in both directions for this task namespace. No workaround transfer attempted.

- Routing comparison completed: exactly301056 new transitions, all three terminal candidates retained. Validation selected F43 by the prospective rule (mean344.406874 vs B357.041243); no flight-driven selection.
- Restored Linux MuJoCo engine/library binding exactly equals predecessor, and all relevant physical/controller/reference source bytes match. No flight-NN training or controller/reference tuning. Inference portability:1200/1200 validation model/case orders and costs identical across execution environments.
- Selected F43 development15/15 and vertical12/12 (120 distinct transfers) completed; complete raw/RK/source recovery started. Independent prerequisite re-audit and full selected-development camera projections precede joint freeze.
- Final campaign review fixed campaign-scoped single-pass claims, preservation of pre-existing trials, missing/dangling/nested recovery subtrees, and all127 complete input/audit hash binding before and after final capture.37 focused tests pass. Two preflight preparation failures retained; neither exposed held-out inputs or changed physical conditions.
- Storage is predeclared from seed labels:45 cases in existing RAM (27 basic/endpoints plus j0/1 in each of nine held-out cells),82 on disk. This changes storage only. All links must be dereferenced and independently verified during recovery.

- Joint freeze completed after27 independent raw re-audits (9896512 actual RK stages; replay errors0) and all15 full-pose camera qualifications. SHA f2fbe39ebf90d55507627d4a4d2d9178b1ef8800784cd08d5f6ecf7b4bc3e6f6. Fresh routing held-out300 passed selected F43 absolute gates, mean345.061333 vs B355.530542, paired improvement2.9447%;170 wins/71 losses/59 ties and19 harmful applicable responses reported. No reselection.
- Physical127 single-pass launched under campaign-scoped exclusive claim; closed cases are recovered concurrently with symlink dereference and independent byte-size/SHA verification. Development15 and vertical12 recovery verified:8420files/6202772988bytes. Runtime engine209 source/library files recovered and matched against exact frozen hashes.

- Physical127 independent audit completed:9/9 basic,18/18 endpoints,100/100 fresh held-out;40148288 RK stages, replay errors0. All39104 case files/25899565860bytes recovered and independently verified. Strict-shadow0/1270 and transient box spin338.476deg/s retained as limitations.
- Final3 accepted seed1 recordings completed14526 frames; native/HUD lossless, MP4, recomposition/PTS/caption/geometry/source proofs pass. Exact local paths, actual altitude changes and all3 hashes presented; fresh human review requested and remains pending.
- Frozen original goal-marker bound underestimated cylinders/boxes. Without changing any frozen source/camera/flight, supplemental v2 encloses all styles with scene-based sqrt(3)r plus float conversion error, binds complete frame counts and accepted input hashes, and independently passes all18 recordings/probes with minimum84.530px against original40px criterion. Focused read-only review found no remaining concrete supplement blocker. Original proof and v1 retained.

- User confirmed all3 exact-hash final videos in full continuous1x, including altitude/framing/zoom/suspension/box/readability. Separate human-review-v1 receipt saved; original capture-time PENDING HUD/proof and reviewed video bytes preserved.

- Delivery complete: all59788 remote files/36851003632bytes independently size/SHA-256 verified locally. Final local inventory60127files/36921663846bytes also includes derived data and exact-hash user review. All remote entries match; no local directory symlinks. Related local commit follows the user-selected identity; no public action.
