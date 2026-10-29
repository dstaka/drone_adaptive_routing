# Dual-drone learned coordination — paused, acceptance incomplete

This change implements a joint PPO dispatcher for two drones carrying separate suspended loads in one native MuJoCo world. The policy assigns destinations, transfers responsibility, selects WAIT and chooses crossing priority; existing flight controllers and the shared safety supervisor execute the requests. The feature aims to shorten team patrol time, but learned improvement has **not** been demonstrated.

Specification: [approved design](../specs/2026-10-27-dual-drone-learned-coordination/spec.md). [Japanese pause report and recovery instructions](dual-coordination-pause-ja.md).

The user paused development on2026-10-29 because elapsed time was excessive and learnability under the current task definition remained unclear. All campaign learning, follow-up evaluation, observer and maintenance processes were stopped. There is no automatic restart. No push, PR, merge or publication was performed; this is a local description draft.

Implementation includes shared-world two-load physics, bounded references and safety monitoring, mission/ownership semantics, joint NN/PPO, baselines and ablations, finite campaign/selection/freeze guards, evaluation, media verification and independent artifact recovery. Native qualification passed81/81 cases and10 supplemental checks. These are not the selected-policy physical127 acceptance.

The last full software regression passed1741 tests, skipped48 and deselected the nonessential Genesis test. Subsequent scoped CPU orchestration, progress/artifact and parallel scheduling checks passed. Full regression was not repeated for this documentation/shutdown-only handoff; source018 and its original LearningLock remain unchanged.

CPU measurements selected seven workers. Seed42 stopped at107/256 complete updates (438,272 transitions), with2,814 partial rollout rows retained (441,086 collected). Seeds43/44 never started. The exact-version recovery loader accepts the nonselectable checkpoint. Interruption left seven missing final-flush acknowledgements; original failure records are retained. No post-stop physical replay, learning or inference was performed.

Training observations do not establish convergence: critic loss decreased while patrol reward plateaued and saved recent training windows had only2–3 successes per140 terminal episodes. Most failures were persistent leg timeouts or uncertifiable requested references. These stochastic training windows are not validation results. One-seed, reduced-update and longer-deadline alternatives were discussed but never applied.

Validation selection, joint freeze, fresh routing/physical held-out evaluation, the final three NN videos and fresh full-length1x human review remain unperformed. The learned-coordination acceptance is incomplete. Physics, safety,30-second leg/300-second team deadlines and all original numeric gates were preserved.

The stopped-state packet contains42 files/43,755,252 bytes with independent source/destination size and SHA-256 agreement. It retains model/Adam/RNG/worker/rollout state, compressed training records, stopped process receipts, frozen runtime metadata and final remote timing. The earlier frozen-input packet was also recovered and verified. Large artifacts remain in the local run rather than Git; the Japanese handoff lists exact locations/hashes and restoration steps. The server instance itself was not stopped or destroyed.

Per-attempt timing preserves failures, retries, waits, overlapping work and unrecorded intervals. Development processes stopped46h46m01s after campaign start; handoff elapsed time is separately timestamped. UTC spans and monotonic measurements are reported separately where clocks disagree, rather than inventing a complete execution/wait partition.
