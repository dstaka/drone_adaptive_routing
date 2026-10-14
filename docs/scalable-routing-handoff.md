# Scalable routing session handoff

All CPU analytical acceptance gates in the approved [specification](../specs/2026-10-14-scalable-adaptive-routing/spec.md) passed. The operator authorized four hours of CPU execution and confirmed one local commit plus a prepared PR description, without push. No remote PR has been created. Physical flight, display and video remain follow-up work.

## Source, artifacts and environments

- Base revision: `006b7f96eb15fc7968e7a0d9924cc0bffbae75cb`.
- Delivery revision: the commit containing this record; obtain it with `git log -1 --format=%H`. A post-commit receipt is also saved under `outputs/scalable_session/final-commit.json`.
- All public artifact references below are relative to the repository root.
- Local artifact root: `outputs/scalable_20261015` (ignored by Git, fully delivered).
- The absolute remote workspace is recorded only in private execution configuration outside Git.
- Remote interpreter: `.venv-cpu/bin/python` relative to the private remote workspace (isolated Python 3.12.14, Torch 2.6.0+cpu). The tested closure is [routing-scalable-linux.lock.txt](../requirements/routing-scalable-linux.lock.txt).
- The original server environment, historical local models/evidence, protected reference copy and other draft specs were preserved. No Genesis, flight retraining, rendering or GPU experiment ran. No prior project/checkpoint was present in the inspected standard remote workspace; the historical local flight artifacts remain separate.
- Connection details and credentials stay out of Git. Use the operator-supplied SSH configuration.

The delivered archive contains 104 allowlisted files, 72,308,083 bytes. Every file size and SHA-256 matched locally. Archive SHA-256: `2f3a69197defa0fd28ff2ba009fe0f0f9a378aed8c9919b17cfab5498f35938f`. See [transfer manifest](evidence/scalable/delivery-manifest.json) and [receipt](evidence/scalable/delivery-receipt.json).

## Frozen model and contracts

The adopted model is seed 42 from the optimizer-scaled shared-candidate MaskablePPO experiment. Its validation mean objective was 166.281061639353, versus 166.844177530977 for seed 43 and 166.644266932279 for seed 44. All three beat the corresponding completed alternatives. Selection was frozen before held-out evaluation; all seven complete trial checkpoints and the intentionally stopped MLP seed-43 logs are retained.

- Local selected model: `outputs/scalable_20261015/normalized-src/outputs/training/seed_42.zip`.
- Remote selected model: `normalized-src/outputs/training/seed_42.zip` relative to the private remote workspace.
- Model SHA-256: `c4c2a937d26650c09a7810073382263e79ba58fcf238a752387fd41d45a2bc2a`.
- Schema sidecar SHA-256: `8d5aa5a75c11edebe78a53b903690886adfb50eac58fe6da0f77e9e3a334fb8a`.
- Schema: `routing-v2-n10-324`, 324 float32 features, ten actions, ordered IDs A–J. Alpha=beta=1, nominal speed=1 m/s, no return. Offsets, scales and objective constants are in the sidecar and [frozen manifest](evidence/scalable/models.json).
- Configuration: [routing_candidate_scaled.yaml](../configs/routing_candidate_scaled.yaml), SHA-256 `6434ef20406fac4705968cfb766255900bc4485943eb01c8592c512cadef3611`.
- Three-seed local manifest: `outputs/scalable_20261015/final-src/outputs/frozen/models.local.json`; this private manifest was remapped for the execution checkout. The original remote manifest remains unchanged alongside it.
- Model-loading verification: [all three transferred models loaded; all 30 actions and full suffixes reproduced locally](evidence/scalable/local-inference-verification.json).

Each adopted seed has 50,176 actual static plus 50,176 actual event-inclusive PPO steps (100,352 total). Initialization used 12,000 exact training-only states and 30 supervised epochs per seed. The teacher dataset SHA-256 is `f36263a9f2d8e80904422f9a348c6e5a86953a510721626860ff3f15abfe60e7`; its full inputs/labels are retained under `normalized-src/outputs/training/teacher/`. Positive optimizer reward scale 0.01 preserves the objective optimum; raw environment reward, accounting and reported costs stay unchanged. Inference contains no oracle, teacher or fallback. Actual architecture, steps and per-seed times are in [training records](evidence/scalable/training-records.json).

Source archives in the artifact root record exact executed code, rather than pretending the base revision includes uncommitted changes:

| Purpose | Archive | SHA-256 |
| --- | --- | --- |
| Adopted model training | `normalized-source.tar.gz` | `3096a4fa591a201780874d638a922ac1031362e8448de9fc3d506d304e2ee138` |
| Held-out evaluation | `final-source.tar.gz` | `824f1d9f2c42d12c90ca91953e505aa925e8cf47e4ffbc429bfd3e35adb0d72c` |
| Corrected analytical replay | `scalable-replay-source.tar.gz` | `3c35b2d51025cf19363e36ab862c882613c529f757b445a4b2b482146e0e8b02` |

## Measured acceptance

Remote CPU suite: **93 passed, one GPU test deselected**. Local CPU suite: **93 passed, one GPU test deselected**. Ruff and `git diff --check` passed. Coverage includes N=3/10/12, full enumeration for N=3 through N=7 with and without prefixes and every forced first action, asymmetric costs, return, ties, the 18.0/20.5 fixture, model/IPC smoke, invalid contracts, nonfinite evidence, event isolation, and restored-prefix roundoff. Remaining warnings are upstream Gymnasium unbounded-space, NumPy/PyTorch mask-conversion and local Matplotlib deprecations.

The same 100 static and 50 dynamic cases were evaluated per training seed. Every case is retained, including unchanged and skipped events. Static reference solver status is OPTIMAL for every case.

| Seed | Static cost mean | Mean learned/exact | Worst ratio | Delta vs greedy | Delta vs fixed | Quality gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 164.995237074 | 1.029997147 | 1.135945427 | -11.595605713 | -119.485166890 | PASS |
| 43 | 165.912782218 | 1.035796268 | 1.135945427 | -10.678060569 | -118.567621746 | PASS |
| 44 | 165.618922764 | 1.033669968 | 1.126528560 | -10.971920022 | -118.861481199 | PASS |

All 300 static and 150 dynamic learned episodes completed, with zero invalid actions. The selected model's worst static absolute gap is 21.704490269638; the quality criterion concerns mean ratio, not universal optimality. Its mean travel component is 16.347071368511 and mean priority completion penalty is 148.648165705161.

| Seed | Dynamic cost mean | Applicable / scheduled | Skipped | Changed / applicable | Beneficial / applicable |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 162.107492538 | 48 / 50 | 2 | 5 / 48 | 2 / 48 |
| 43 | 163.013025952 | 49 / 50 | 1 | 4 / 49 | 2 / 49 |
| 44 | 162.103662953 | 48 / 50 | 2 | 4 / 48 | 4 / 48 |

General dynamic responses are not universally beneficial: the selected model changes 5/48 applicable cases and improves the paired post-event objective in 2/48. Complete paired orders and components remain in `final-src/outputs/dynamic/scenario_results.json`. [Static](evidence/scalable/static-summary.json), [dynamic](evidence/scalable/dynamic-summary.json) and [acceptance](evidence/scalable/acceptance.json) records preserve the denominators and unchanged quality thresholds.

## Frozen A/B/C and explanation

Demo split search retained seed 40004 after five attempts. All scenarios use identical geometry, initial priorities in [2,3], unit initial multipliers, service, base, model bytes and deterministic inference. The specified grid has minimum destination separation 0.60 m and excludes the base from arrival spheres; its physical flight envelope remains unverified.

| Scenario | YAML SHA-256 |
| --- | --- |
| [normal](../configs/demo_scenarios_scalable/normal.yaml) | `b99a344a4016e7e2b36c83242fba0b22f6ce95e5530f07b201fc355154442dee` |
| [emergency](../configs/demo_scenarios_scalable/emergency.yaml) | `b1e8f35fc51365245301162ed5ddc7bd9cf9d49de41afe1270a196d4a9b310a5` |
| [cost_shift](../configs/demo_scenarios_scalable/cost_shift.yaml) | `8c97f363634d2e03ae18031638cd8f22ab6f5136464df77968537be72967c64c` |

The first completed analytical inspection is B. With no event, the remaining policy order is `A E G H I J F D C`. The emergency raises **C only** from 2.660865759308 to 5; the policy changes its next action from A to C and predicts `C D F J I H G E A`. C moves from ninth to first in the remaining order. Paired post-event remaining cost improves from **179.628171734002 to 154.408349051608** (25.219822682394 improvement).

The cost event changes **B→A only**, multiplier 1→5; A→B stays unchanged. Priorities stay unchanged. The predicted suffix changes to `C D F J I H G E A`; paired post-event total improves from **156.132036679599 to 149.096595315295** (7.035441364304 improvement), and travel component falls from 7.8 to 5.4. These are hypothetical suffixes, not committed physical routes. See [all selection attempts](evidence/scalable/demo-selection.json).

Featured explanation: `analytical-40004-normal / decision-2`, current A, visited B/A, eight destinations still unvisited. The actual policy selects **E**, priority 2.120156088347, distance 0.60 m. **H** is the maximum-priority remaining destination, priority 2.693373135601, distance 1.341640786500 m. Under frozen observed conditions:

- J* = Q*(E) = 147.981512600810.
- Q*(H) = 165.512743820382, strictly worse by 17.531231219572.
- J_policy = 147.981512600810; its full suffix happens to be optimal at this boundary, with zero gap and proven solver status. Other decisions retain their own measured gaps.
- Equality tolerance = 0.0000014798151260081.

The scope is `現在の条件を固定した残り巡回の最適解`; it excludes unknown future disturbances and physical/motor/energy optimality. [All 30 explanations](evidence/scalable/decision_explanations.json) include all eligible Q* alternatives and precise components. Independent standard-library replay verified **225 complete orders**, maximum component error **5.684341886080802e-14**. [Replay](evidence/scalable/analytical-replay.json) and [independent check](evidence/scalable/independent-cost-check.json) use `evidence_kind=analytical`.

## Runtime and preserved failed attempts

Measured CPU experiment window: **2026-10-15T00:40:42+08:00 to 2026-10-15T01:12:51+08:00**. Sum of supervised stage wall times: **2004.196181 s** (33 min 24.196 s), including cleanup, failed attempts and overlapping work; allowance 14,400 s. GPU work: zero. Per-seed training times remain in their records. [CPU ledger](evidence/scalable/cpu-execution.json) contains every status and deadline.

- Conventional MLP comparison stopped intentionally after seed 42 completed, because completed shared-candidate validation was better. Seed-42 checkpoint and seed-43 partial progress remain preserved. Its raw supervisor status is FAILED (SIGTERM cleanup), not a successful three-seed experiment.
- First final stage completed held-out evaluation, then failed on exact equality of a recomputed floating-point tolerance. A regression reproduced it; the verifier now derives the declared tolerance from the recorded J* and checks recomputed costs within that tolerance. The model and held-out results never changed.
- First corrected replay launch preceded completion of the script transfer and exited 127 immediately. Transfer completion was verified, and a new replay job succeeded. Both status files remain in the delivery.
- Independent review found four important validation/accounting issues; all were reproduced and fixed with tests. Candidate architecture metadata and teacher provenance checks were also added. No review findings remain deferred.

## Re-execution

From the local checkout, use the existing routing interpreter or an isolated compatible environment. The Linux experiment closure is separate from the historical optional plotting environment; reproduce it with the supplied lock and `PYTHONPATH=src` without installing unrelated extras.

```bash
export PYTHONPATH="$PWD/src"
.venv-routing311/bin/python -m pytest tests -m 'not gpu' -q
.venv-routing311/bin/python -m ruff check src scripts tests
.venv-routing311/bin/python scripts/eval_routing.py --models outputs/scalable_20261015/final-src/outputs/frozen/models.local.json --num-checkpoints 10 --split static --episodes 100 --out outputs/recheck_static_new
.venv-routing311/bin/python scripts/eval_routing.py --models outputs/scalable_20261015/final-src/outputs/frozen/models.local.json --num-checkpoints 10 --split dynamic --episodes 50 --out outputs/recheck_dynamic_new
.venv-routing311/bin/python scripts/select_demo_scenarios.py --models outputs/scalable_20261015/final-src/outputs/frozen/models.local.json --out outputs/recheck_demo_new
.venv-routing311/bin/python scripts/replay_decisions.py --analytical --run outputs/scalable_20261015/replay-src/outputs/demo --out outputs/recheck_replay_new
.venv-routing311/bin/python scripts/verify_analytical_costs.py --explanations outputs/scalable_20261015/replay-src/outputs/demo/decision_explanations.json --policy-hash c4c2a937d26650c09a7810073382263e79ba58fcf238a752387fd41d45a2bc2a --out outputs/recheck_costs_new.json
```

To retrain on the remote CPU interpreter, in a new source/run directory:

```bash
export PYTHONPATH="$PWD/src"
python scripts/save_scenario_manifests.py --num-checkpoints 10 --out outputs/manifests_new
python scripts/train_routing.py --config configs/routing_candidate_scaled.yaml --num-checkpoints 10 --seeds 42 43 44 --out outputs/training_new
python scripts/freeze_routing_models.py --models outputs/training_new/models.json --out outputs/frozen_new
```

The delivered `train-stage.sh`, `candidate-stage.sh`, `normalized-stage.sh`, `final-stage.sh` and `scalable-replay-stage.sh` record the actual commands. All output directories must be new. Runtime paths, schema/hash mismatches and incompatible destination counts are fatal; do not pad or reuse the old three-action model.

## Pending physical checks

1. Restore/verify the separately preserved Genesis environment, learned motor checkpoint, configuration and source hashes on the next authorized flight server.
2. Validate this frozen ten-point geometry against the controller's measured envelope, then prove arrivals, continuous dwell and complete seeded A/B/C missions under PR 2.
3. Bind explanations to real physical run/decision IDs and regenerate with `evidence_kind=physical`.
4. Implement the persistent destination display and annotations in PR 2, and recording/video presentation in PR 3.

Start the next session from the merged base branch. This session supplies CPU analytical evidence only; no flight or video acceptance is claimed.

## Public metadata privacy

Public evidence copies use repository-relative artifact references and omit machine-specific roots and local account names. Original execution manifests and archives remain unchanged in ignored local artifacts so their recorded hashes stay verifiable. These public copies may have different bytes from their original source artifacts.
