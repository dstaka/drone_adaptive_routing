# Scalable Adaptive Routing Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver the approved ten-destination CPU routing contract, learned models, exact comparisons, analytical replay, and reproducible handoff in one local commit.

**Architecture:** Parameterize scenarios, state, observations and IPC by ordered IDs. Use subset dynamic programming for the weighted completion objective and frozen boundary comparisons. Keep analytical replay independent of Genesis and all physical/video follow-up work.

**Tech Stack:** Python, NumPy, Gymnasium, CPU PyTorch, sb3-contrib MaskablePPO, pytest.

**Spec:** `specs/2026-10-14-scalable-adaptive-routing/spec.md` (approved by operator).

## Global Constraints

- Exactly ten destinations A–J plus S for delivery; regression sizes N=3 and N=12.
- Preserve departure-snapshot accounting, alpha=beta=1, speed=1, service=0.50 and no return by default.
- Retain seeds 42/43/44; validation-only selection; 100 static and 50 dynamic held-out cases per seed.
- Four-hour CPU execution allowance, durable bounded remote jobs; no flight/GPU work.
- English comments and new documentation; protect existing changes and other draft specifications.
- One final local commit includes this spec, implementation, tests and evidence. No push; prepare PR description locally, as clarified by operator.

## Review Focus

- Legacy unversioned three-point inputs must normalize explicitly; malformed versioned references fail.
- A model with the same N but different ID order or objective constants must fail before inference.
- Nonzero elapsed time and directed costs must remain in forced-first exact comparisons.
- Event-skipped and unchanged cases must remain in reported denominators.
- Explanation provenance mismatches or incomplete solver results must never assert optimality.

### Task 1: Scalable contracts and exact objective

**Files:** `scenario.py`, `objective.py`, `routing_env.py`, `baselines.py`, `tests/test_scalable.py`.
**Interfaces:** `Scenario.num_checkpoints`, `Scenario.checkpoint_ids`, `schema_for(scenario)`, `exact_remaining(state, first=None)` returning proved order/cost/components/status.

- [x] Add failing tests for N=3/10/12 episodes, legacy/versioned validation, hidden future events and once-only boundary updates.
- [x] Run new tests and observe failures; parameterize schemas, mappings, masks and accounting.
- [x] Add failing independent enumeration checks N=3..7, prefixes, forced actions, ties, asymmetric edges, return and the 18.0/20.5 trade-off fixture.
- [x] Implement subset DP: transition time increment is weighted by all still-unvisited priorities, making subset/current sufficient; add elapsed-time constant and reconstruct objective components.
- [x] Run targeted tests and the existing CPU suite. Expected: all pass.

### Task 2: Model, IPC, training and benchmarks

**Files:** `routing_agent.py`, `planner_ipc.py`, `benchmark.py`, routing CLI scripts/config, agent/IPC tests.
**Interfaces:** Frozen schema and ordered IDs travel with every checkpoint/request; `RoutingAgent.predict(state)` checks compatibility; benchmark emits every paired case and per-seed denominators.

- [x] Add and observe failing model/schema/IPC and CLI mismatch tests, plus real short train/save/load/IPC smoke.
- [x] Implement N-aware MaskablePPO training, manifest persistence, validation selection, strict inference and IPC dimensions.
- [x] Extend full no-event/post-event rollouts and paired post-event costs, exact status/time/gaps and quality gate summaries.
- [x] Run targeted and full CPU tests remotely. Expected: no Genesis imports, full completion and zero invalid actions.

### Task 3: Analytical replay and explanations

**Files:** new `analytical.py`, demo selection/replay scripts, explanation tests.
**Interfaces:** Versioned analytical boundary snapshots, stable hashes/run/decision IDs, all forced-first exact results, independently replayable components.

- [x] Add failing tests for full suffix comparisons, explanation integrity, suboptimal policy gap, and incomplete-solver labels.
- [x] Search declared demo split using frozen policy, balanced priorities, common compliant geometry, C-only emergency, directed cost-only update.
- [x] Retain attempt records and explicitly report unmet gates; never fabricate route changes or substitute oracle actions.
- [x] Run analytical replay/integrity tests. Expected: valid explained costs and provenance; incomplete examples labeled honestly.

### Task 4: Remote experiments, verification and local delivery

**Files:** execution driver, compact evidence, `docs/scalable-routing-handoff.md`, spec status/index, prepared PR body.
**Interfaces:** Reviewed source allowlist and checksums; unique remote/local run directories; source revision/hash, models, schemas, scenarios and results in handoff.

- [x] Transfer allowlisted source and run remote baseline/new tests in isolated routing environment.
- [x] Run smoke, then bounded three-seed static/event training within cumulative four-hour allowance; save actual rounded steps and wall time.
- [x] Freeze validation winner before held-out evaluation; run 100 static and 50 dynamic cases per seed, demo search and exact explanations.
- [x] Transfer artifacts using explicit allowlist and verify SHA-256/size; record all actual outcomes and pending physical acceptance.
- [x] Perform independent whole-change review, address important findings, run final tests/lint/integrity checks.
- [x] Stage only owned files (preserving pre-existing index edits), commit once, and prepare PR description without push.

## Execution ledger

- Initial branch: `feature/scalable-adaptive-routing`; existing changes saved under ignored `outputs/scalable_session/`.
- Ruling: work in the operator-designated root checkout on its existing feature branch; avoid moving or copying unrelated untracked drafts.
- Ruling: operator explicitly requests immediate execution after planning; no extra plan approval cycle.
- Ruling: operator clarified no push, so delivery ends at a local commit plus prepared PR description.
- Pre-flight: Task 2 consumes Task 1 schema/state; Task 3 consumes Task 1 solver and Task 2 agent/rollouts; Task 4 freezes all preceding interfaces before final evaluation.
- Task 1: complete. New RED: 11 failures for missing variable-count/schema/solver contracts. GREEN: 18 routing/scalable tests passed, including independent enumeration through N=7 and the 18.0/20.5 fixture.
- Task 2: model/IPC smoke complete. RED: three agent/CLI tests failed before implementation. GREEN: seven agent/IPC tests passed. Full intermediate CPU suite: 64 passed, one GPU test deselected.
- Task 3: explanation/counterfactual core complete. RED: analytical module missing. GREEN: four explanation/counterfactual/geometry tests passed. Demo outcomes remain pending frozen-model evaluation.
- Ruling: explicit `testpaths=["tests"]` prevents pytest from collecting the protected untracked historical project copy; baseline collection errors were caused by duplicate module names in that copy.
- Ruling: use a new completely isolated CPU PyTorch venv after inherited packages exposed dependency conflicts; preserve both the original interpreter and the unused first routing venv.
- Ruling: keep the historical default generator at N=3 for callers, but default the new training/environment CLI to N=10. All new benchmark/manifests pass N explicitly.
- CPU experiment: three seeds, 500,000 requested static plus 500,000 event-inclusive steps each; 1,000,448 rounded steps per seed; 512 rollout length, five epochs, [256,256] MLP. Four-hour cumulative CPU ledger, 10,000-second training-stage deadline. Final acceptance is based on actual measurements, not budget assumptions.
- Final review: independent fresh-context review completed; no critical findings. Four important findings reproduced: legacy objective bypass, nonfinite explanation costs, return travel omission and raw mask coercion. All corrected with regression coverage; targeted fix pass passed five tests. Additional teacher metadata validation test failed before the fix and then passed.
- Ruling: compare an additional shared-candidate neural policy with exact-label supervised initialization using training seeds only, then static/event PPO. This improves sample efficiency without an inference oracle; validation decides adoption. Both methods and every teacher input/label remain reproducible. If the comparison is ineffective, retain its measured failures rather than changing quality thresholds.
- Additional experiment: three spawned CPU seed processes, 12,000 training-only teacher states, 30 supervised epochs, 100,000 requested static plus 100,000 event-inclusive PPO steps per seed. Separate frozen source snapshot; maximum additional ledger allocation 4,380 s, deadline 4,200 s. Initial stage deadline 10,000 s plus cleanup and this allocation remain below four hours; final evaluation uses measured remaining allowance.
- Local verification before the second experiment: 83 passed, one GPU test deselected; lint passed. Exact count will be confirmed in the saved test output before final reporting.

- Ruling: candidate PPO's unscaled critic losses remained around 20,000 and validation is pending; compare a fixed positive optimizer-only reward scale of 0.01. Raw environment/objective values and optimal orders remain unchanged. Regression tests cover raw accounting/masks and save/reload. No held-out data informed this choice.
- Added an independent standard-library explanation cost replay CLI. Its test failed while the CLI was missing, then passed and rejected a corrupted travel component.
- Local full suite after contract hardening, candidate/teacher checks and independent replay: 86 passed, one GPU test deselected; lint passed.

- Candidate experiment: all three seeds completed; validation mean costs 176.163617 (42), 231.680909 (43), 173.625616 (44); selected seed 44 by validation only. Durable stage succeeded in 454.008206 s.
- MLP seed 42 completed 1,000,448 steps; validation cost 213.315640. Ruling: stop the remaining optional MLP comparison after the completed candidate method improves validation, preserving its complete checkpoint and incomplete seed logs. The mandatory three training seeds are already retained in the complete candidate experiment. Cost if wrong: a later MLP seed could have been better; this does not affect the independently frozen held-out test protocol.
- Optimizer-scale comparison starts in a third immutable source directory, with maximum additional allocation 2,400 s / deadline 2,200 s. Combined prior deadline plus completed candidate wall time plus this allocation remains below four hours; evaluation will use measured remaining allowance.

- Final adopted method: optimizer-scaled shared-candidate MaskablePPO. Validation means 166.281061639353 / 166.844177530977 / 166.644266932279 for seeds 42/43/44. Frozen winner seed 42, SHA-256 c4c2a937d26650c09a7810073382263e79ba58fcf238a752387fd41d45a2bc2a, before held-out opening.
- Held-out: 100 static + 50 dynamic per seed, 100% completion, zero invalid actions. Selected mean static exact ratio 1.029997146535; paired mean greedy/fixed deltas -11.595605712675 / -119.485166889784. All three quality gates passed.
- Final: fixed restored-prefix tolerance comparison — reproduced RED on validation seed 10000 at N=10; derive declared tolerance from recorded J* and independently check recomputed costs. Regression GREEN; full remote suite 93 passed / one GPU deselected. Original model and held-out measurements preserved.
- A replay launch raced with script upload and exited 127; after checking completed transfer, new job succeeded. Both statuses are retained.
- Demo 40004 selected after five attempts. C-only emergency benefit 25.219822682394; directed B→A cost-event benefit 7.035441364304. Both next actions change A→C. Nearer/lower-priority example retained at normal decision-2: E versus maximum-priority H.
- Independent verification: 30 decisions / 225 full orders, max component error 5.684341886080802e-14. All source, models, cases and results delivered locally: 104 files, every size/SHA-256 verified. Three models and all 30 actions/full suffixes reproduced locally.
- Measured supervised CPU time 2,004.196180806961 s, including failed/stopped attempts; no GPU work. No pending CPU acceptance gate; physical and video checks remain explicitly deferred.
- Final review outcomes: four important findings fixed with regression tests; both minor metadata/provenance observations addressed. No deferred findings. Declined-to-judge experiment/delivery items were resolved by actual runs and checksum verification; protected drafts and physical/video follow-ups remain outside this PR.
- Commit readiness: spec, implementation, tests, frozen fixtures, compact evidence, handoff and PR body prepared for one local commit. User explicitly prohibits push; preserve the current branch and unrelated working-tree changes.
