# Add scalable CPU adaptive routing and analytical evidence

The router was fixed to three checkpoints and six enumerated tours. This change supports versioned N-dependent scenarios, observations, masks and model/IPC contracts, with ten destinations for delivery and three/twelve-point CPU regressions. A subset dynamic program provides the exact weighted-completion reference, including elapsed prefixes and forced-first alternatives.

The same frozen learned policy serves the normal, C-priority emergency and directed-cost scenarios. Analytical replay records complete event counterfactuals and independently replayable decision explanations. Training compares a conventional MLP with shared-candidate MaskablePPO, using disclosed training-only exact-label initialization and optional positive optimizer reward scaling. Validation determines the frozen model before held-out evaluation.

Specification: [Scalable adaptive routing](../specs/2026-10-14-scalable-adaptive-routing/spec.md). Workflow and rationale: [CPU routing](../docs/scalable-routing.md). Actual models, hashes, metrics, commands and physical follow-up gates: [session handoff](../docs/scalable-routing-handoff.md).

Validation: 93 CPU tests passed locally and on the server (one GPU test deselected); all 450 held-out learned episodes completed with zero invalid actions. The validation-selected model has mean static learned/exact ratio 1.029997, beats greedy and fixed order, and passes both A/B/C event gates plus the nearer/lower-priority explanation gate. Independent replay verified 225 orders. Measurements and retained failures are recorded in the handoff and `docs/evidence/scalable/`. The implementation includes tests for N=3/10/12, independent enumeration through N=7, the 18.0/20.5 trade-off fixture, save/load/IPC, invalid contracts, full event suffixes, provenance and independent component replay.

Ten-point Genesis flight, display integration and video remain separate follow-up PRs. All new evidence is marked analytical. The operator requested no push; this is the prepared PR description for the local commit, and no remote PR has been created.
