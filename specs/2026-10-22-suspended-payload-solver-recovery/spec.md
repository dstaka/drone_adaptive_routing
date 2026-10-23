# Suspended-Payload Solver Recovery and Evidence-Driven Learning

- Date: 2026-10-22.
- Status: Closed unsuccessful on 2026-10-23 at the user's direction after review of the hybrid/Genesis follow-up. Repaired N1 numerical qualification passed. Five pilots and a matched fresh development pair consumed7020000 valid transitions; shared settling remained zero. Both methods failed all six development conditions at300K/1M/3M, with all36 native replays passed. No final policy, validation, acceptance or accepted video was produced. Preserve this specification as the historical scope; closure does not satisfy its acceptance criteria.
- Expected starting revision: `e39cb098a9edc47007c5d427ae0d220939075dda`. Inspect the actual branch, HEAD, history, index and untracked files; preserve newer changes and these possibly uncommitted preparation files. Never reset to an older revision.
- PR: Not created. Include this spec, its index entry, preparation documents, implementation, tests and compact evidence in the same eventual implementation PR. Follow the execution instruction's one-new-local-commit delivery; no push, PR creation, merge or publication.
- Implementation root: repository root; `adaptive_drone_patrol_20261013/` and personal files are excluded.

## Goal and precedence

Deliver physically qualified ten-goal suspended flight, matched randomized/fixed learning, real-transition validation, all 254 paired acceptance missions and three complete 1x loaded videos with fresh human review. A numerical diagnostic, CPU test suite or faster simulator is an intermediate result.

This specification supersedes conflicting requirements in the [previous recovery spec](../2026-10-22-suspended-payload-flight-recovery/spec.md), [retry spec](../2026-10-21-suspended-payload-flight-retry/spec.md), [original payload spec](../2026-10-17-suspended-payload-flight/spec.md), their plans/instructions and inherited execution helpers. Unchanged physical, routing, acceptance and media requirements remain normative. Historical files and results stay intact.

The human explicitly removed all future GPU time limits and GPU usage tracking. The human then requested a new-session package based on the recommended engine flexibility and evidence-driven learning approach. The scope is:

| Topic | New controlling rule |
| --- | --- |
| GPU allowance | No cumulative allowance, remaining balance, phase allocation or GPU usage ledger; no stop because projected work exceeds an old allowance. |
| Physics implementation | Supported settings, targeted engine patches, necessary dependency upgrades, and an alternative rigid-body backend are authorized within the unchanged physical model and the procedure below. These bounded choices do not need repeated approval. |
| Training samples | 300000 transitions is an initial development checkpoint, not a ceiling or proof of sufficient learning. Determine and freeze an equal final sample/PPO plan from development evidence. |
| Numerical quality | Preserve positive-energy limits and explicitly check absolute unexplained passive energy balance, so a dissipative candidate is not adopted from its positive-growth score alone. |
| Delivery process | First present an implementation plan, then execute; do not stop at the plan or request another general approval. |
| Still excluded | Hybrid/residual control adoption, reduced angular domain, changed physical model, relaxed arrival/safety/mission limits and reduced acceptance denominators. These require a concrete separate amendment and explicit human approval. |

No success probability is asserted. Engine flexibility removes an implementation restriction; it does not establish physical correctness or learnability.

## Required context and current behavior

Read `AGENTS.md`, this complete spec, the [new handoff](../../docs/suspended-payload-solver-recovery-handoff.md), [implementation plan](../../docs/superpowers/plans/2026-10-22-suspended-payload-solver-recovery.md), previous recovery/retry/original specs and handoffs, and the five predecessor specifications/handoffs referenced by the original payload spec. Inspect dispersed adoption/failure/compatibility/acceptance evidence, both Linux locks, actual parent configuration and actual engine source.

Current evidence is under `docs/evidence/suspended-recovery/`; follow its relative paths to actual ignored artifacts. Rehash the relevant executed source, full receipts and raw data. Do not repeat all expensive historical simulations merely to rebuild context. Missing artifacts must be reported and recovered when possible, never fabricated.

- The old V10 representative 30-second passive trace was reproduced with identical sample SHA-256. Its maximum positive-energy increase was 530.703%.
- A separate planar Lagrangian local replay explains that trace's accumulated energy increments within about `8.92e-8 J`. This supports a case-specific discretization diagnosis, not a general 3D proof.
- The 128-substep candidate and 256-substep refinement produced only three complete candidate passive traces. Positive growth was below 5%, but nominal 170 degrees lost approximately 13% of initial energy; the 90-degree dt pair had large same-time trajectory differences. Long nonlinear trajectory divergence alone is not a definitive correctness test.
- Only 3/32 passive and 3/62 domain traces were screened; new motor 0/15, effective partial reset 0/30, both training runs 0, validation 0, acceptance 0/254, videos 0/3. No loaded policy is adopted.
- Local and exact-lock Linux CPU suites each passed 901 tests with one GPU test excluded. This does not qualify physics. Native admission is intentionally blocked, training/validation/acceptance/capture are not fully integrated, and the current batch implementation cannot yet run its base128/half256 comparison.
- The old engine's standalone midpoint path excludes the articulated/CONNECT-constrained model. Merely changing to CONNECT is not evidence of a midpoint fix.

The latest source differs from the actual GPU-executed candidate. Historical receipts cannot be relabeled as evidence for new source, settings or batch.

## GPU execution without usage accounting

Remove the old cumulative-budget gate from the new execution path. Do not reconcile or append GPU seconds to `outputs/gpu_ledger.json` or `docs/evidence/gpu_ledger.json`; do not create replacement usage ledgers, cumulative timing summaries or a new eight-hour balance. Preserve old ledger files as historical artifacts. Old `BudgetForecast` or launch helpers must not block this workflow on remaining seconds. Test this behavior using absent and exhausted historical ledgers.

Maintain one GPU-owning job at a time on the human-supplied server, with an independent process lock, heartbeat, termination and abnormal-output preservation. Per-job deadlock timeouts may be chosen from measured progress and adjusted before rerunning a diagnostic; they are not a total GPU allowance. A job timeout must retain its failed/incomplete attempt. Do not spin indefinitely on an unchanged failure.

Simulation timestamps, per-mission wall deadlines, profiler windows and training sample counts remain necessary scientific/operational data. They are not cumulative GPU usage tracking. Performance profiling is for choosing correct efficient execution, not deciding whether work fits a time allowance. The unchanged acceptance wall deadline is 600 seconds per mission; GPU allowance removal does not waive that gate.

Use the current connection command supplied privately for the new session, only when remote work is needed. If absent or invalid, ask for the current command and continue independent CPU work. Use a new remote workspace and isolated Python 3.12 flight/reference/routing environments, with unique `outputs/suspended_solver_<date>_<unique>/` local outputs. No machine-specific absolute paths or connection information in Git/public documents. Transfer external artifacts locally and verify sizes/SHA-256.

This authorizes use of the supplied GPU environment without a time cap, not new rentals, contract changes, additional paid servers or server destruction. Preserve one active GPU job; independent CPU work may proceed concurrently.

## Physical model and numerical candidate selection

Keep the actual original drone, its original rotor locations/spins/KF/KM/RPM mapping/root meaning and motor authority. Reverify the original free-drone mass `0.02700047753751278 kg` and actual mass/inertia/collision/frames. Keep the compression-capable rigid 0.30 m link, underside attachment `(0,0,-0.020)`, cube top-center attachment, unrestricted passive joints and reciprocal force/moment on the drone. Cube sides remain 0.030/0.045/0.060 m, equiprobable per episode, with an independent uniform mass ratio 0.10-0.30. Fixed comparison uses 0.045 m and 0.20.

No direct payload actuation, massless/point-mass substitution, slack-rope substitution, state projection, hidden swing cancellation, strong unexplained damping, contact-assisted stabilization or extra motor authority. Preserve the 0.01-second control interval and immediate current-action semantics.

Before each candidate run, save a declaration containing hypothesis, actual source/patch/build/lock hashes, settings, precision, geometry/physical parameters, exact initial states, cases, input schedule, numerical metrics/thresholds, sample collection, progress timeout and rejection conditions. Freeze the declaration before producing its data. Keep every failure and distinguish short diagnostics from full qualification.

Use the following escalation order; a disproved hypothesis need not be exhaustively swept:

1. **Supported settings:** inspect installed Genesis 1.4.3 source. Compare supported `implicitfast`/`approximate_implicitfast`, precision and timestep choices on the representative 90-degree and nominal 170-degree cases. Declare a small informative candidate set, not a blind full Cartesian sweep. Preserve actual effective properties when precision/import changes small masses or frames.
2. **Independent reference:** add a sufficiently refined higher-order reference to the independent planar model and, when useful, a full rigid-body reference such as MuJoCo. Compare identical model/initial states/forces. Check the reference's own step refinement; do not treat a second simulator's output as ground truth. CPU reference execution is acceptable.
3. **Targeted repair or backend change:** if settings do not resolve the observed mechanism, implement a source-supported coupled integration/constraint repair, evaluate a suitable engine release, or migrate the same model to a different backend. A targeted patch or backend change is permitted under this spec after its candidate declaration, without another permission request. Choose using measured numerical quality and feasibility, not brand or utilization.

For patches/upgrades, preserve the old locks and installed environments, add new versioned locks, record upstream revision plus exact patch/build hashes, licensing and dependency differences, and pin the selected artifact. Avoid unrelated upgrades. Adapt reset, raw observation, motor work and rendering explicitly. Source code changes must be reviewable; a private unrecorded engine edit is not admissible.

A backend change does not permit changing physical constants or reusing old qualification. Keep the original motor/property tolerances and remeasure them. Do not require byte-identical long trajectories across different integrators: compare identical motor mechanics and a refined numerical reference, retain old trajectories as controls, and document numerical compatibility separately from physical equivalence. Preserve same-backend unloaded/loaded motor-equivalence checks and all 15 motor cases.

## Numerical verification and operating-domain gates

Select a candidate using small diagnostics before full suites. For representative 90-degree and nominal 170-degree motion, compare at least three declared timestep resolutions, short-time state/acceleration/constraint errors, energy drift, dissipation and actual work. Predeclare comparison windows rather than choosing convenient windows after divergence. Analyze long-time differences without applying the powered 2 cm/2 degree threshold to a chaotic passive comparison.

Keep inherited positive passive-energy growth <=5%, constraint residual <=`2e-5 m`, physical-property relative error <=`1e-5`, collision extent error <=`1e-5 m`, small-angle period error <=10%, and finite/quaternion/speed guards. Add an explicit conservative passive energy-balance screen: the maximum absolute unexplained energy change, after independently reconstructed applied and declared physical dissipative work, must be <=5% of the same positive initial excitation-energy reference used by the inherited passive checker. Apply it over every complete passive trace. Predeclare the work/sign/reference conventions; an unobserved loss cannot be booked as physical damping. This makes the prior approximately 13% unexplained loss ineligible for adoption. The prior 20% emergency-loss stop was not a certificate of accuracy.

Do not repair energy by clipping state, resetting, filtering the recorded trajectory or widening the checker. Preserve every raw sample before transforms/guards/reset and a preceding raw window on abnormality, including finite but implausible state and partial backend exceptions. Independently reconstruct work, geometry, COM motion, quaternion norms, reaction and reset properties from actual getters.

The required operating domain remains all 62 traces:

- Passive32: nine small-angle anchors of at least 120 simulated seconds; nine 90-degree load combinations and nominal 5/45/80/89/91/120/170-degree cases of at least 30 seconds; seven additional 30-second half-dt runs (six size/mass endpoints at 90 degrees and nominal 170 degrees).
- Powered18: inherited load/initial-swing conditions, each at least 120 continuous seconds without reset.
- Powered refinement12: six matched base/half-dt pairs of the same 30-second segment; <=0.02 m/2 degrees with identical prescribed rotor commands. Distinguish open-loop numerical error from feedback-induced command differences.
- Motor15 and at least 30 actual selective resets are separate requirements, not included in 62.

Only independently reconstructed passive32 permits powered diagnostics. After that barrier, prioritize actual-backend hover, short out-and-back and retarget continuity with the diagnostic controller before expanding the remaining powered suite. Privileged information in that controller is declared; it is never the adopted policy.

Complete all62, motor15, reset30 and the actual chosen configuration's source/settings/property/work bindings before any optimizer. A new integrator/precision/backend/batch invalidates inherited work-bound assumptions; rederive and observe the actual integration path, including intermediate stages or constraint work as applicable. Preserve the legacy endpoint work guard where its quantities retain their meaning; an alternative rigorous work reconstruction must document why an old numerical bound no longer applies, not simply choose a looser allowance.

Complete final admission in code. Recompute raw evidence and actual training bindings at consumption; a status string, stale PASS or manually changed hard block cannot authorize training. A passed N=1 configuration may support an N=1 development optimizer pilot; an unqualified batch may not.

## Early flight, runtime integration and batch choice

Use one physical/runtime/observation contract across training, development, validation, acceptance and recording. Extend the recovery runtime/mission core or add a backend adapter; do not send new checkpoints through legacy Euler entry points. Test actual reset/retarget/current-action ordering and unloaded parent migration on the selected backend.

After N=1 qualification, establish a small learned hover and out-and-back development pilot on that actual backend. Record actual success/failure, saturation, floor/tilt, swing/settling and raw trajectories. Do not postpone this experiment until a full high-throughput infrastructure project is complete. A failed pilot is a development result, not evidence that the final fixed sample budget should be consumed unchanged.

Measure N=1/8/32/128 when supported, using evidence-enabled simulation/inference/reset/logging and actual PPO only on a qualified optimizer configuration. Keep per-environment geometry/mass/inertia, episode RNG, state/history/target/dwell/timers and masks independent; verify at least 30 partial resets with ongoing neighbors for N>1, actual RPM/held forces and geometry switches. N=1 still requires 30 real resets and actual property/input checks, but cannot claim neighboring-environment isolation. Reject invalid slots before PPO and quarantine affected rollouts. All selected-size/half-dt settings must be supported; remove the current base128/half256 implementation mismatch explicitly.

Require same-input N=1/selected-N numerical equivalence and full selected-batch qualification before batch learning. Keep per-case simulated duration unchanged. If batch correctness or implementation cost delays flight, retain qualified N=1 and continue; parallel speedup is optional, successful flight is required. Record profiler results without generating a GPU usage ledger or allowance forecast.

## Learning contract and equal sample selection

Verify model4296, its actual cfg and seed44 routing before use:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Parent flight model4296 | 475573 | `26658a31560f71c36fd9e9605b15d8e4f7b9cae8174c1465fd12dcd0d10b3431` |
| Actual parent cfg | 1628 | `35c6b0be8db90e4e799d453d447a484275317f85098a22cbdf04c19c84056e55` |
| Adopted routing seed44 | 297837 | `a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef` |

Keep `payload29-ideal-world-current-action-v2`, 29 measured fields, their frame/scale/clip semantics and four immediate motor actions. No mass, size, inertia, bank/geometry ID, seed or future event in actor inputs. Explicitly transfer compatible parent columns0-12; zero changed-history13-16 and new17-28 columns; copy other compatible tensors and create a fresh optimizer/iteration. Retarget preserves state/history. Engine choice is not permission to alter actor/control semantics. Any needed observation/controller redesign requires a concrete separate amendment.

Separate physics/controller development, learning pilots, training, model-selection validation and held-out streams. Predeclare development seeds/cases/checkpoints and selection rule before pilots. A development pilot may continue across increasing development sample checkpoints; none of its checkpoints initializes either final method. Failed model2000 is never an initializer.

Use a development sample ladder starting at 300000, 1000000 and 3000000 valid environment transitions per method; these are diagnostic checkpoints, not automatic final budgets or upper bounds. Test both load-training methods under a common schedule. If useful improvement continues, predeclare further rungs before running them; no GPU-time approval is needed. If progress stalls, investigate observations/actions, reward/curriculum, optimization and actual trajectories rather than repeating identical runs indefinitely. Common reward/curriculum/PPO tuning is permitted on development evidence, with unchanged physical task, actor inputs and final acceptance criteria. Report all development attempts; do not tune on held-out results.

Count `S=N*T*K` for complete rollouts, and independently count valid/invalid/generated/consumed transitions, vector steps, optimizer updates and per-stage episode experience. For nondivisible rung values, predeclare one common divisible S or implement a verified masked partial rollout; never round each method differently. Rollout boundaries do not reset bodies, loads or histories. Require declared long/continuous target sequences and all curriculum stages, not just high aggregate throughput.

PPO timeout bootstrapping applies only to time-limit truncation. A true terminal failure or the tenth actual shared completion takes precedence when it coincides with the episode limit; neither may be relabeled as a bootstrappable timeout. Reconstruct termination reasons from the complete native/transition stream and actual completed-target count independently of whether the limit was also reached. The existing terminal raw `timeout` field continues to mean that the clock limit was reached; it is not the PPO bootstrap flag and need not identify the goal count in isolation. This correction does not change the saved raw format. The current v19 development pair remains immutable through its declared comparisons; any later implementation correction receives a new source binding and does not rewrite the pair's historical evidence.

Before either final run, freeze the same S, seed7, N/T, architecture, optimizer/PPO parameters, reward, curriculum, commands, initialization tensors, development-derived layout and checkpoint/model-selection rule. Record this as `training-plan.json` with source/configuration hashes. The intended difference is randomized versus nominal training loads. Both final runs start fresh from the verified parent transfer. More samples are authorized without a GPU allowance calculation. If final development/validation necessitates changing this plan, invalidate that pair and restart both under a new frozen plan; do not extend only the weaker method. Preserve prior attempts.

### Next common development recipe declared before execution

The five diagnostic pilots used stage0 only and are never final-plan candidates. P5 added dense position cost to v4 and completed300000 transitions, but its matched20k/100k/300k mean policies all timed out at the first hover with zero shared completions. The300k error was0.277550m and package angular speed38.323479degrees/s. Full native action/state replay passed; these are learning failures, not physical admission failures.

The next nonpilot development pair starts fresh4296 for both randomized/fixed methods, seed71, N1/T100, S3000000 with predeclared checkpoints300000/1000000/3000000. Keep rewardv5 and all physical/control/acceptance inputs. Change the common PPO credit recipe from gamma0.99/lambda0.95 to gamma0.999/lambda0.99. This tests longer credit for observed position recovery and pendulum settling; it is a combined recipe experiment, not an isolated causal comparison. At100Hz the geometric discount scale becomes about10s and the gamma-lambda trace scale about0.91s, distinct from actual finite rollout/return support. Keep the100-step rollout and all other optimizer fields.

Set curriculum_steps1800000, providing about300000 transitions per initial stage and the remaining approximately1500000 at stage5. Current-command/episode boundaries may defer a transition between stages. Earlier300k/1M checkpoints remain reported even if full-stage exposure makes them ineligible. Select only the earliest common declared rung satisfying the unchanged randomized6/6 actual ten-target sequences and independent exposure for both methods. If3M fails, preserve both and diagnose before declaring a common next rung or revised recipe. No held-out tuning or pilot initialization.

Before the first run, freeze the source bundle, complete Torch/RSL Python/build dependencies and both declarations. Independent consumers regenerate method/seed reset draws, require complete native getters/contact consistency, and replay each final run's raw and fresh4296 initialization before final validation/freeze. Temporary RAM-backed output storage is permitted only inside owned namespaces with sufficient measured capacity; closed artifacts are offloaded with exact size/SHA before any persistent duplicate is removed. Qualification raw remains regular and immutable.

## Layout, validation, acceptance and media

Default `delta_z=0`. After physical repair, a development-only clearance study may choose the smallest justified common translation from 0/0.25/0.50/0.75/1.00 m, applied to S and all ten destinations. Preserve x/y, relative z, full-precision coordinates, priorities and original random draws. Freeze before both final trainings. Use a new versioned wrapper with base/transform hashes; check initialization, curriculum, observation clipping, routing normalization/quality, validation, all254 cases and camera/HUD. No post-acceptance height/seed rescue, no floor translation, no per-target change, no >1 m without approval.

Implement validation12 conditions/120 real target transitions across both policies: seeds11/12 x three prescribed loads x two policies, six cases/60 transitions each. Retarget real moving bodies through all ten prescribed targets; relabeled stationary samples do not count. Validate raw continuity/shared settling independently. Freeze selected models, actual source/backend/locks/settings, case draws, layout, routing, camera and checker bindings before held-out evaluation. Preserve untouched held-out cases and report if any were exposed; after exposure, a failed frozen run is not repaired by selective reruns or further tuning against that suite.

Run both policies on the same basic9, endpoint18 and held-out100 cases: 127 each, total254. Randomized gates remain 9/9,18/18,>=95/100. Fixed comparison must run all127 and be reported even if it performs better. Retain failures, denominators, distinct conditions, size/mass/scenario subgroups and Wilson95% intervals. Unexecuted cases are unexecuted, not successes or estimated rates.

Every destination requires all predicates for >=1.0 continuous second (100 complete .01-second intervals): drone error<=0.20 m/speed<=0.30 m/s; nominal package target error<=0.20 m/speed<=0.10 m/s; suspension angle<=5 degrees; swing rate<=10 degrees/s; package angular speed<=10 degrees/s. Derive attachment velocity with rotational offsets; nominal package target never follows actual drone drift. Require oriented drone/package lowest points>0.10 m, absolute roll/pitch<=60 degrees, leg30 s, mission120 s and wall600 s. These limits are not GPU usage accounting.

Keep full-precision dispersed coordinates and adopted routing contract `routing-v3-integer100-n10-324`; priority seed40029 and A81/B42/C43/D64/E23/F84/G6/H98/I22/J56; alpha1/beta0.05, importance divisor100, coordinate divisor2/distance divisor4, directed multipliers1-5. Keep nominal planner service0.50 s distinct from physical settling1 s. After the first new real shared completion, emergency changes unfinished C43 to100 and cost shift changes only F-to-A from1to5. Recompute actual-position/remaining-task Q*, J*, policy suffixes, counterfactuals and component costs; do not inherit old event times or routing improvement claims.

After randomized acceptance, record the adopted policy at basic seed1, nominal mass0.20/side0.045 for normal/emergency/cost-shift. Produce complete raw and Japanese-HUD videos through all ten shared completions at simulation-time1x. Show drone, package, link, all IDs/priorities/states/target, both altitudes, load conditions, swing and progress. Verify every decoded frame, PTS/time mapping, original pixels, subtitles and telemetry, not just metadata. Deliver final hashes and concrete event-before-to-next-shared-settling intervals; request fresh continuous1x human review. Pending human review remains pending. Keep CPU checks/documentation moving while waiting. No submission re-edit or `demo_full` work.

All public capture entry points must freshly recheck the full paired acceptance and bind the selected randomized trial, frozen case/model, native bytes and rendered proof. A saved success label alone cannot authorize capture or identify its policy. Independently reconstruct the complete successful mission event stream, including event order, step/time/pose, target IDs, completed order and actual-position post-event costs; the HUD must consume that verified stream. Rebinding outer hashes cannot legitimize changed display events or invented cost improvements.

## Evidence, compatibility and delivery

Use versioned candidate/run/qualification/training/freeze/acceptance/media records with backend ID, upstream revision, patch/build/lock/source hashes, units/frames, precision/control/substep settings, physical assets/properties, observations/actions, case/episode IDs, actual raw references and failures. Store public paths relative to the repository. Require strict schema and binding checks; unknown/stale/missing evidence fails closed. Do not copy an engine-specific work field into a different backend without defining and verifying its meaning.

Keep compact evidence under a new `docs/evidence/suspended-solver/` directory and large outputs ignored. Required compact records cover source/inputs, every candidate/attempt, numerical convergence, passive/powered/motor/reset/batch qualifications, runtime/initialization, training plan and actual samples, validation/freeze, all acceptance outcomes, videos/human review, transfers and CPU/PDF verification. No new GPU-time-summary requirement. Preserve old evidence/locks/checkpoints as history and old unqualified profiles as blocked.

Update Japanese README/report/handoff and affected PDFs; render and visually inspect every PDF page. Code/comments/docstrings remain English; explanations, subtitles and rerun guidance are Japanese. Run appropriate full CPU and actual-backend checks and obtain a fresh code review; CPU doubles never replace real backend qualification. Inspect exact staging and secret/path exclusions. Include preparation files and implementation in one new local commit using Author/Committer `dstaka <22057509+dstaka@users.noreply.github.com>` without global changes or amend. Prepare an English PR description linked to this spec. Do not push/create PR/merge/publish/submit.

## Acceptance and stopping decisions

Completion requires all of: qualified physical/backend/batch configuration; actual common runtime; matched fresh final training; validation12/120 and freeze; randomized9/9,18/18,>=95/100 plus all127 fixed cases; three independently verified complete videos and new human approval; accurate source/artifact/docs/PDF delivery. Report each gate separately.

There is no time-budget stopping state. On a failed candidate, preserve it and investigate the next justified in-scope numerical alternative. Do not end the work solely because it is slow, N=1 or requires more samples. Repeated unchanged experiments without a new hypothesis are not progress.

Stop dependent optimizer/acceptance/capture on unresolved physical correctness or missing qualification. If settings, targeted repairs and viable alternative backends leave a concrete unresolved blocker, document attempted alternatives, evidence and the smallest proposed scope change. If a change to hybrid control, angle domain, mass/geometry, observation meaning, motor authority, floor/settling/mission limits, >1 m/nonuniform layout or acceptance denominator is needed, present that specific amendment for human approval. Do not apply it silently. Server unavailability does not stop independent CPU integration/tests/documents. Preserve a truthful blocked result when no authorized path remains; never manufacture success.

## References

- [Genesis forward dynamics](https://genesis-world.readthedocs.io/en/v1.4.0/user_guide/theory/rigid_solver/forward_dynamics.html): numerical settings background; actual installed source is authoritative for each candidate.
- [MuJoCo numerical integration](https://mujoco.readthedocs.io/en/stable/computation/#numerical-integration): higher-order reference motivation, not proof that this model or any GPU backend passes.
- [Previous measured recovery](../../docs/suspended-payload-flight-recovery-handoff.md): actual failures, source differences, artifacts and incomplete runtime paths.
