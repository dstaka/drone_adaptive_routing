# Scalable analytical routing

The approved [specification](../specs/2026-10-14-scalable-adaptive-routing/spec.md) defines this CPU-only change. Ten destinations A–J plus S are required for delivery. Three- and twelve-destination cases have separate contracts. Genesis flight, destination markers, recording and video delivery remain follow-up work.

Use an independent routing environment with the repository's routing dependencies. Do not install routing dependencies into a working Genesis environment.

```bash
export PYTHONPATH="$PWD/src"
python -m pytest tests -m 'not gpu' -q
python -m ruff check src scripts tests
python scripts/save_scenario_manifests.py --num-checkpoints 10 --out outputs/new_manifests
python scripts/train_routing.py --config configs/routing_scalable.yaml --num-checkpoints 10 --seeds 42 43 44 --out outputs/new_training
python scripts/eval_routing.py --models outputs/new_training/models.json --num-checkpoints 10 --split static --episodes 100 --out outputs/new_static
python scripts/eval_routing.py --models outputs/new_training/models.json --num-checkpoints 10 --split dynamic --episodes 50 --out outputs/new_dynamic
python scripts/select_demo_scenarios.py --models outputs/new_training/models.json --out outputs/new_demo
python scripts/replay_decisions.py --analytical --run outputs/new_demo --out outputs/new_replay_check
python scripts/verify_analytical_costs.py --explanations outputs/new_demo/decision_explanations.json --policy-hash FROZEN_SHA256 --out outputs/new_independent_cost_check.json
```

All output directories must be new. Training defaults to ten actions; the CLI rejects disagreement with `num_checkpoints` in configuration. Evaluation infers N from the frozen model and rejects a conflicting explicit argument. The low-level generator retains a three-point default for existing callers; new workflows pass N explicitly.

Scenario schema version 2 stores ordered `checkpoint_ids`, with S at coordinate row zero and action k at row k+1. Priority events name a checkpoint ID, and cost events name directed pairs of node IDs. Unversioned input is accepted only as an explicit three-point legacy format without IDs; integer references are normalized once. Newly serialized scenarios always use version 2. Existing historical artifacts are preserved.

The model sidecar fixes N, ID order, observation dimension, every field offset, normalization and objective constants. New observation sizes are 65 for N=3, 324 for N=10 and 434 for N=12. The schema ID is `routing-v2-nN-DIMENSION`; exact historical `routing-v1-65` sidecars remain explicitly supported. A three-action model cannot serve a ten-destination scenario. Policy requests carry this contract and the frozen policy hash. Action masks prohibit repeats and terminal policy queries. Flight workers run CPU model/scenario preflight before importing Genesis; ten-point physical execution remains explicitly unavailable until PR 2.

The exact reference uses subset dynamic programming for travel plus weighted absolute completion time. Each move delays every still-unvisited priority, so its time increment is charged against their sum. This makes remaining subset and current node sufficient, with current elapsed time contributing a shared additive term. Directed multipliers change logical travel cost, never travel time. The solver supports prefixes, elapsed time and forced first actions, reports proven status and timing, and refuses more than sixteen remaining points to avoid accidental exponential workloads. Normal N=10 benchmarks never enumerate permutations.

At an event boundary, two independent state copies produce full deterministic policy suffixes with no further events. Both suffixes are evaluated using the same post-event objective. Reports retain scheduled, skipped, unchanged, changed and beneficial cases with separate denominators. Static reports include paired fixed/greedy differences, exact mean/worst gaps, components and the unchanged quality threshold: mean learned/exact <=1.10, mean learned <=greedy and strictly below fixed.

The demo search uses only seeds 40000–40099 and the frozen validation-selected model. A/B/C share geometry, initial priorities, unit multipliers, service and model bytes. The emergency always raises C to 5 after the first completion. The cost event raises only the directed edge from the first completed node to the policy's previously predicted next node; its reverse remains unchanged. This is disclosed fixture selection, not a hand-authored route. All attempts and failed gates are preserved.

`configs/routing_candidate.yaml` defines an alternative MaskablePPO training experiment: shared neural candidate scoring, 12,000 exact training-only labels for supervised initialization, then static and event-inclusive PPO. The relation network is 4→32→32, the node network 44→64→64, the candidate scorer 128→64→1, and the critic 64→64→1. Weights are shared across destinations; learned logits and action masking determine every inference action. The exact teacher is imported only during training. The saved teacher manifest includes every frozen input and exact label. Reuse checks hashes, training split/seed ranges, N, dimensions and label eligibility. `net_arch` applies to the conventional MLP; the candidate architecture is recorded separately in each training record. `parallel_seeds: 3` trains independent seeds in separate spawned CPU processes.

`decision_explanations.json` contains every analytical replay boundary: complete frozen input snapshot, hash, run/decision identity, actual policy action and rollout, all eligible forced-first optima, J*, travel/priority components, tolerance and policy gap. Its verifier replays all order costs, re-solves claimed optima and checks provenance and candidate annotations. An optimal first action and an optimal full policy suffix are distinct claims. An incomplete solver is labeled `UNCONFIRMED` (display text: `最適性未確認`). Every record has `evidence_kind=analytical`; these files do not prove flight behavior.

For bounded remote jobs, use a unique tmux session and the existing supervisor with a separate CPU ledger:

```bash
python -m adaptive_drone_patrol.jobs --out outputs/new_cpu_job --ledger outputs/cpu-ledger.json --budget-s 14400 --timeout 10000 -- python scripts/train_routing.py --config configs/routing_scalable.yaml --out outputs/new_training
```

The ledger's total elapsed time includes process cleanup and rejects exhausted allowances. Keep the existing GPU ledger separate. A detached session does not establish success: inspect the saved status, exit code and artifacts. The session-specific handoff records actual paths, hashes, commands, measurements and pending physical checks.

`configs/routing_candidate_scaled.yaml` tests a fixed optimizer reward scale of 0.01. This positive constant preserves the objective optimum and leaves the base environment reward (`-leg_cost`), observations, action masks, raw cost ledger and all evaluation values unchanged. It reduces critic loss magnitude during PPO; its value is recorded in each training record. The experiment is selected using validation only.
