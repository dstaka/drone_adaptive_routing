# Dual-drone learned coordination (implementation in progress)

This change implements a shared MuJoCo world containing two independently loaded drones and a centralized 242-action coordinator for assignment, reassignment, waiting, and crossing priority. Existing flight controllers remain fixed; flight-policy learning is outside scope.

Specification: [Learned coordination of two drones with suspended payloads](../specs/2026-10-27-dual-drone-learned-coordination/spec.md).

The implementation includes namespaced physical assemblies and motor metadata, atomic ownership and persistent deadlines, continuous references, conservative native auditing, fixed baselines, finite three-seed PPO and exact recovery, validation selection, post-selection ablation diagnostics, authenticated freeze/held-out runners, compact replay, video verification, and progress/timing/recovery tooling. The user approved replacing permanent full-stage raw retention with online auditing and reproducible inputs; physical and statistical gates are unchanged.

The current immutable source snapshot passes 1,735 tests, with 48 skips and one Genesis GPU deselection in 523.09 seconds; related Ruff checks pass. Corrected-source server-native preflight passes 47 tests in 28.64 seconds. A prior development B-static mission completes in 25.66 simulated seconds with five inspections per drone, no ownership transfers, minimum full-assembly clearance 3.38093252 m, and no detected violations. Its 2,567-frame video was machine checked and recovered with matching size/SHA-256. Prior development deterministic native replay matches all 129 decisions and 259 retained states exactly. These are development results, not production qualification or learned-performance acceptance. Genesis-only regression is excluded at the user's request.

Outstanding: full native qualification, complete CPU/GPU benchmarks and storage projection, three finite training seeds, selection on 1,500 validation executions followed by 600 ablation diagnostics, joint freeze, fresh routing300 comparisons and separate physical127, all numerical acceptance gates, and three final NN videos with fresh human review. Failed attempts remain distinct from successful reruns; held-out data remain unopened. No learned-performance or full-safety acceptance claim is made.

The former v2 source completed all 81 physical probes, but a subsequent check found that its supervisor horizon could be shorter than the measured worst stop plus policy latency required by §4.3. Those results remain diagnostic only, and pending benchmark/admission jobs were stopped before execution. The correction imposes a four-second minimum while extending longer current brakes, and qualification independently checks the measured seven-case worst stop against it. Corrected-source regression passes; a new 81-case qualification is running. No thresholds or deadlines were relaxed.

Three earlier development B-dynamic probes retained two successes and one leg-timeout failure. The user confirmed no server deadline or spending cap; all finite experiment budgets remain unchanged. Initial implementation commit: `55ef0bf` (local only); the horizon correction is being verified separately.

This is a draft description for local delivery. No pull request, push, merge, or publication is authorized.
