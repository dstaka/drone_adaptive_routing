# Visible Multigoal Flight Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans in this session. Keep one final local commit, as explicitly requested.

**Goal:** Execute and record the approved ten-destination patrol in Genesis with persistent landmarks, synchronized priority annotations and physical evidence.

**Architecture:** Keep frozen CPU routing in its subprocess and the existing learned motor controller in a separate Linux environment. Generalize the physical mission runner, add immutable scene landmarks and world-projected annotations, and derive explanations from logged physical decision boundaries offline.

**Tech Stack:** Python, NumPy, pinned Genesis/PyTorch/rsl-rl, MaskablePPO CPU subprocess, FFmpeg/Pillow.

**Spec:** specs/2026-10-14-visible-multigoal-flight/spec.md (approved by operator).

## Global constraints

- Root implementation only; preserve all historical artifacts and other untracked work.
- Retain 0.20 m / 0.30 m/s / 0.20 s arrival and 0.50 s continuous inspection dwell, floor 0.1 m and tilt 60 degrees.
- Exactly one GPU job at a time; cumulative budget 28800 seconds, starting at 1233.7003595418064 seconds; supervised bounded jobs include initialization and retries.
- Freeze model pair, geometry, scenarios, camera, timing and acceptance settings before normal/emergency/cost_shift x seeds 1/2/3. Preserve development failures separately.
- No push, hosted PR, final video retiming/editing, server lifecycle changes or unrelated files in commit. Public paths are repository-relative.

## Review focus

- Terminal states never query the planner; malformed/masked/noninteger actions fail.
- A reset or interrupted dwell never completes an inspection.
- First/final frames and event changes preserve exact simulation-step references.
- Priority updates cannot alter unrelated marker colors or destroy visited landmarks.
- Physical explanations must match actual same-run boundary/action/hash references.

## Tasks

### 1. Baseline and isolated runtime
- [x] Confirm merged source, protected files, model/config/source hashes and cumulative ledger.
- [x] Transfer allowlisted root sources, locks and frozen artifacts into a new private workspace; compare sizes/hashes.
- [x] Install separate flight and CPU environments from exact locks; retain failures without silently substituting versions.
- [x] Supervise a short existing-model camera replay before modifying motor behavior.

### 2. Mission state machine
Files: `mission.py`, `evidence.py`, `run_demo_scenarios.py`, `tests/test_visible_flight.py`.
- [x] RED: N=10/12 unique completion, no terminal query, malformed actions, dwell interruption, failure/event ordering, frame boundary tests.
- [x] GREEN: derive IDs/schema from scenario; log snapshots, measured boundary pose, all attempts, dwell/leg statistics and state changes. Generalize evidence checks.
- [x] Run CPU suite and lint; expected: pass, GPU excluded.

### 3. Persistent scene and annotation
Files: new `scene.py`, `flight_agent.py`, `target_adapter.py`, annotation script and tests.
- [x] RED: palette anchors/fractions, independent state styles, extents camera/projection and event-frame synchronization.
- [x] GREEN: install fixed visual-only markers before scene build, fixed overview camera, manifest and initial capture. Inspect pinned runtime material/projection APIs; use documented neutral-marker/anchored-priority fallback when appropriate.
- [x] Verify same-seed marker on/off dynamics, actual frame projection/label readability and startup/final coverage.

### 4. Physical feasibility and fixed suite
Files: new validation orchestration and frozen acceptance config.
- [x] Test existing model on long transitions, reversals, repeated hover/retarget; inspect normalization/reset failures before considering training.
- [x] Freeze configuration and hashes; if geometry/model changes, repeat affected routing tests and full suite.
- [x] Run nine predeclared acceptance missions, log every outcome and record authentic raw media for at least one paired seed.
- [x] Validate 90 continuous physical dwells, events/decisions, frame associations and full video decode. Never label CPU mocks physical.

### 5. Physical explanations and delivery
Files: physical replay/verification script, compact evidence, Japanese handoff and PR description.
- [x] Regenerate and independently verify exact remaining-cost comparisons using actual physical run/decision boundaries; annotate L/H with pose-derived distances.
- [x] Inspect real frames at mission start, each retarget/event and mission end.
- [x] Transfer all new large artifacts locally; compare every size and SHA-256 in a delivery manifest.
- [x] Document results, retained failures, hashes, rerun commands, cumulative GPU time and unmet gates; update spec status/index.
- [x] Review complete staged content and metadata for private identifiers, paths and credentials. Run CPU suite/lint/diff check. Create exactly one local commit with neutral public identity, without push.

## Execution ledger

- Baseline: merged main `ff346a0`; tracked worktree clean; protected untracked files recorded privately. Root working directory explicitly required, so use a dedicated branch in place rather than relocating artifacts to a worktree.
- User explicitly requested plan then immediate execution; no additional approval gate. User requirement of one commit overrides per-task commits.

- Task 1: merged base/artifact hashes verified; 69 transferred inputs verified remotely. Exact locks installed in isolated Python 3.12.14 environments. Initial uv resolution failed due index precedence; retried identical versions with both declared trusted indexes. Baseline camera replay: 200 actual steps, 0 resets, full decode PASS (152.690200 s GPU job).
- Task 2: N=10/12, invalid actions, raw-pose/speed integrity and real intro samples implemented with observed RED→GREEN cycles. CPU suite reached 104 passing tests before later additions.
- Task 3: pinned Genesis camera uses OpenCV extrinsics; initial diagnostic incorrectly combined OpenGL projection and OpenCV extrinsics and failed before any flight step. Diagnostic corrected to intrinsics @ extrinsics; failed output retained.
- Task 4 Ruling: existing motor model fails a long B→J retarget at step 183, tilt 66.710922 degrees. Identical markers-on/off samples and failure; relative observations <=0.416, no reset or target-history mutation. Fine-tune from preserved parent into a new output directory, keeping 17 observations, four motors, unchanged world bounds/physics and 60-degree acceptance safety. Train full-envelope retargeting every 300 steps plus continuous hover, retain every checkpoint and failure. Freeze the selected result before acceptance. No routing model or geometry change.

- Task 4: fine-tuning v1 completed 1001 additional iterations, parent iteration 300 to selected model_1300, GPU wall 722.873721 s. Both markers-on/off eight-transition probes succeeded (1508 steps each), exact equality of pose/velocity/quaternion/actions. Development normal seed 11 completed all ten in 17.82 simulation seconds. An earlier development startup failed before mission stepping because upstream base_euler is initialized only on step; derive tilt directly from measured quaternion instead, with RED→GREEN regression.
- Review: independent reviewer identified physical L/H distance ordering must use actual pose as well as planning distance, and suite input/model provenance needed stronger binding. Added failing regressions and fixed both; reviewer reports no unresolved blocking CPU finding.
- Frozen acceptance v1: seeds 1/2/3, all three original routing scenarios, selected routing checkpoint unchanged, flight model_1300, fixed overview and 30/120/600 s leg/mission/wall limits, 2 s actual hover intro. Inherited plus development GPU usage before suite: 2318.808445 s; remaining 26481.191555 s. Supervised suite timeout 5400 s. Source archive and frozen manifest retained before launch. CPU suite before launch: 116 passed, one GPU test deselected; Ruff passed.
- Execution note: one source transfer was attempted before the local test/archive command completed, so it failed to find the archive. The following development launch used the already transferred source and hit the initial-pose bug above. All logs/status files remain; no accepted trial has been replaced by this development attempt.
- Final verification hardening: added decoded MP4 frame-count/PTS equality checks with RED→GREEN unit regression. This is offline evidence validation only; the remotely executing frozen flight source remains untouched throughout all nine trials. The exact executed source archive remains the runtime record.
- Frozen acceptance v1 completed: all nine worker statuses SUCCEEDED, 90 unique inspections, supervised GPU wall 484.764295 s. GPU contract test: one passed. No GPU work remains active.
- Offline verification v1 stopped on cost-event snapshot hash normalization: logged integer multiplier 5 was restored as float 5.0; actions and numeric snapshots agreed but canonical bytes differed. Added RED→GREEN regression and preserve the original logged snapshot encoding after numerical/action verification. Raw flight logs and fixed acceptance denominator remain unchanged; failed verification log retained.
- Annotation QA: add an always-visible crop of genuine same-frame raw vehicle pixels, before badges are drawn, so the overview labels cannot obscure the only readable vehicle view. No image synthesis, timing edit or physics rerun is involved.
- Offline verification v2 PASS: 9/9 missions, 90 inspections, physical explanations regenerated for every actual decision, all decoded raw frame counts and PTS match capture records. Annotated seed1 A/B/C clips retain the original frames/PTS. Independent standard-library replay verifies all recorded full suffix and forced-first component costs.
- Follow-up review of offline additions found no blocking issue; finite recorded PTS rejection was strengthened with an observed RED→GREEN regression. Real-frame inspection remains the final visual gate.
- Final raw visibility audit: all 7995 frames from all nine runs audited, zero landmark pixel-gate failures and zero out-of-frame drone telemetry samples. Human inspection of critical raw/annotated sheets confirmed all ten fixed landmarks and state transitions. Annotation v3 fixes observed caption/legend overlap and dark-priority ink contrast, and includes the genuine same-frame vehicle detail view.
- Remote final CPU suite: 123 passed, one GPU test deselected. Initial final Ruff scan found AppleDouble transfer metadata masquerading as .py files. Archive/remove only these generated remote metadata files, retain the failed lint log, then rerun Ruff; local personal files are untouched. These metadata files were already excluded from the frozen execution source hash.

- Delivery complete: 465 files / 91,402,466 bytes, every size and SHA-256 matched locally. Archive d6cba7b79cfeb60cdb9c29b7f94889fb69d6c650f974508dba4317a332d9a88f. Final visual QA confirms corrected priority text/event banner, all target transitions, final coverage and grayscale numeric readability. Local and Linux CPU suites: 123 passed, one GPU test deselected; GPU contract: one passed. Ruff and diff checks pass. Final user-authorized action is one local commit, no push or hosted PR.
