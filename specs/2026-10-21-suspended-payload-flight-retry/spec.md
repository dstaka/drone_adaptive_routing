# Suspended-Payload Flight Retry with Operating-Domain Physical Qualification

- Date: 2026-10-21.
- Status: Retry specification prepared for a separate implementation session. Writing this document does not run probes, train, spend GPU budget, or authorize external access by itself. A human-submitted execution instruction starts implementation; do not interpret this preparation status as a prohibition after that instruction is received.
- Expected starting revision: `2d0e00324c63d96a9376aa4f23128ec4d9e2bde4`, the failed payload prototype. Inspect actual HEAD, history, staged changes and untracked files; preserve intervening work without resetting.
- PR: Not created. Include this spec with the retry code/tests and link it from the eventual PR description. Do not rewrite earlier specifications as if the retry had already succeeded.
- Implementation root: the repository root. `adaptive_drone_patrol_20261013/` remains historical material.

## Goal and scope

Complete physically coupled suspended-load flight through all ten inherited dispersed 3D destinations, train randomized and nominal-load policies fairly, evaluate both on 127 paired cases each, and produce three independently verified complete 1x scenario videos with fresh operator review. First repair and qualify the physical representation; do not resume learning on the failed representation or merely increase the number of solver substeps.

This is a follow-up to the [original suspended-payload specification](../2026-10-17-suspended-payload-flight/spec.md). Its payload distribution, task semantics, settling/safety limits, evaluation denominators, media scope and compatibility requirements remain normative unless this document explicitly changes them. This document takes precedence for the retry baseline, physical representation selection, pretraining qualification, evidence admission and GPU allocation. Earlier implementation-status statements describe history, not authorization for this session. No acceptance gate is relaxed here.

Required context:

- Repository `AGENTS.md`; the original payload spec; [failed-attempt handoff](../../docs/suspended-payload-flight-handoff.md).
- [Failure outcome](../../docs/evidence/suspended/feasibility.json), [independent raw-state replay](../../docs/evidence/suspended/failure-check.json), [acceptance status](../../docs/evidence/suspended/acceptance.json), [source/test binding](../../docs/evidence/suspended/verification.json), [transfer receipt](../../docs/evidence/suspended/transfer.json), [static clearance](../../docs/evidence/suspended/geometric-clearance.json).
- The dispersed-layout, integer-priority, scalable-routing, visible-flight and realtime-video predecessor specs and handoffs linked by the original spec. Read their relevant adoption, failure, compatibility and acceptance records under `docs/evidence/dispersed/`.
- `requirements/flight-linux.lock.txt`, `requirements/routing-scalable-linux.lock.txt`; actual pinned Genesis source; current `payload*.py`, probe/training/evaluation/checker scripts and payload configuration files.

## Established failure and remaining uncertainty

The previous implementation used a 0.30 m rigid rod, serial upper two-axis and lower three-axis Euler joints, tiny auxiliary links, and a real cube. Thirteen short probes and thirty property/reset checks passed within their limited coverage. The powered coupled probe lasted only 0.60 seconds; it did not establish operating-domain stability.

Randomized training was stopped at 215500 environment steps after catastrophic reward/state excursions. Re-execution of the same intermediate checkpoint recorded the following failures with bounded four-motor actions:

| Physics substeps | Physics dt at control dt 0.01 s | Recorded samples | Maximum package speed |
| --- | --- | ---: | ---: |
| 2 | 0.005 s | 2789 | 1.6377276136982605e8 m/s |
| 8 | 0.00125 s | 914 | 5.443570084566984e11 m/s |
| 16 | 0.000625 s | 3714 | 8.158574990269932e12 m/s |

The intermediate `model_2000.pt` SHA-256 is `dc4cc37f30f4fea70eb756b94aa3bf4c3cafcf06da9e614d843f6bff23c6b16d`. It is failure evidence, not an adopted model or default training initialization. The three raw hashes and actual paths are in the failure receipt. A diagnostic process exiting successfully means evidence was written; it is not a physical PASS.

Small auxiliary inertia, joint-coordinate singularities, solver conditioning, reset/state conversion, integration and force application are hypotheses to distinguish, not an established root cause. Failure after timestep refinement does not prove that every fixed-link model is infeasible. The local terminal-state/envelope/abort guards added after the failed GPU run have CPU tests but have not repaired or physically validated the dynamics.

No loaded policy was adopted. Baseline training, validation, all 254 acceptance missions and final videos were not run. No loaded human approval exists. The historical `preflight-v4.json` and any receipt for the failed physical signature remain invalid for operational admission.

## Immutable task and initialization contracts

### Inherited artifacts

Recheck actual sizes and SHA-256 against the original spec, handoffs and manifests before use:

| Artifact | Repository-relative path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Adopted unloaded initialization | `outputs/dispersed_20261020_v1/remote-final/outputs/dispersed/finetune_v2/model_4296.pt` | 475573 | `26658a31560f71c36fd9e9605b15d8e4f7b9cae8174c1465fd12dcd0d10b3431` |
| Actual parent configuration | `outputs/dispersed_20261020_v1/remote-final/outputs/dispersed/finetune_v2/cfgs.pkl` | 1628 | `35c6b0be8db90e4e799d453d447a484275317f85098a22cbdf04c19c84056e55` |
| Adopted integer routing | `outputs/integer_20261020_v1/training/seed_44.zip` | 297837 | `a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef` |

Do not revert to `model_1300.pt`, a failed dispersed model or fractional routing. The previous failure-source snapshot is `outputs/suspended_20261021_v1/source-before-substeps.tar.gz`, 28074 bytes, SHA-256 `8de9045eb9fe4301ff87ba9d38f9fabc3a6e766ae5376687211dec81fc2c28b8`. The external archive is 14234257 bytes, SHA-256 `f859f5c87761e34cdc29750d5da32664c9e67042fc3339e2197b837cffc5a330`; its 278-file manifest is under `outputs/suspended_20261021_v1/remote-final/`. Verify evidence from actual files; if missing, report what cannot be reconstructed instead of inventing a replacement.

### Geometry, routing and payload

- Preserve exact coordinates in `configs/demo_scenarios_dispersed/`, layout seed 51000/attempt 3353, coordinate hash `21dd7cf41bf12c592ccdb4bf9c8817352feeafb9597103a3ba626e51a6d910e0`, S=(0,0,1), and the absolute 1–3 m target envelope. Never redraw or raise the mission layout for a load, policy, seed, scenario or frame.
- Preserve priority seed 40029 and A81/B42/C43/D64/E23/F84/G6/H98/I22/J56; real/display values are integers 1–100, normalization 100, alpha 1, beta 0.05, directed multipliers 1–5. Preserve the routing schema, coordinate/distance divisors and nominal planner service time 0.50 s.
- After the first new shared completion, emergency changes unfinished C 43→100; cost shift changes only F→A 1→5. Do not change A→F or reuse unloaded event times.
- Fixed attachment-to-attachment length 0.30 m; centered underside drone offset `(0,0,-0.020)` m and top-center cube attachment. Same load for every target in an episode/mission.
- Randomized training: each episode independently selects side 0.030/0.045/0.060 m equiprobably and mass ratio uniformly in [0.10,0.30]. Mass and size are independent. Fixed training: ratio 0.20, side 0.045 m. Scene creation alone is not resampling.
- Use the actual original vehicle link-mass sum, excluding suspension/load. The measured previous value was 0.02700047753751278 kg because nominally massless rotors acquire solver epsilon. Audit again under the new importer without silently changing original physical properties. Explain any discrepancy before training; retain the equivalent reference across both methods and all scenes.
- Uniform-cube inertia is `m*s*s/6` per axis. Verify effective runtime mass, inertia, collision geometry, rendering and attachment frames after every reset/geometry switch. Inactive bank entries exert no force, contact or inertia on an active episode; selective resets leave other active environments unchanged.

## Physical representation selection

Retain the physical model of a passive, fixed-length, compression-capable suspension with real body mass/inertia/collision and two-way reaction on the drone. Explicitly authorize investigation and implementation of a different joint/constraint representation **within this model**, after the execution instruction is submitted. This removes the need to ask again merely to replace the failed Euler implementation.

Evaluate candidates using the pinned Genesis 1.4.3 implementation:

1. Preferred investigation: native passive spherical/ball articulation, using a supported asset format such as MJCF only if actual pinned construction and stepping work. A conceptual API or support in another release is insufficient.
2. Alternative: a supported solver constraint between rigid bodies that realizes the same rigid suspension and unrestricted package orientation, with measured reciprocal reaction and constraint stability.
3. Retain or revise serial hinges only if source/measured conditioning analysis explains the failure and the entire new qualification passes. Renaming the old profile or adding substeps is insufficient evidence.

Do not assume any candidate is available or stable. Record why the selected representation was chosen and why rejected candidates failed. A native ball joint may remove coordinate issues but does not by itself prove correctness. Avoid fictitious tiny auxiliary bodies when the pinned representation does not require them. All actual suspension masses/inertias, passive damping, friction, armature, base damping, solver tolerances, precision and substeps are declared and identical across methods. Numerical regularization must not conceal a different effective load or artificial swing damping.

If a new importer no longer provides the Drone-specific motor API, an explicit four-rotor force/torque adapter is permitted only after verifying the same retained RPM mapping, thrust/drag coefficients, rotor frames, spins, application points and original body properties. Apply physical forces/torques to the correct vehicle links; do not apply a total COM/root shortcut that omits moments or drives the payload directly. Validate unloaded motor equivalence and rerun every affected predecessor gate. A new importer must not silently reinterpret the root as the total center of mass.

No decorative payload, prescribed body-pose trajectory, pose tracking of the load, state projection after each step, direct payload torque, teleport/reset during a target change, hidden stabilizer, larger motor authority, tension-only force law, flexible chain, changed length/distribution or dependency upgrade is an implicit fallback. A change outside the fixed-link model requires a specific spec revision and human approval before execution.

## Qualification before PPO optimization

Declare cases, sources, actual settings, finite timeouts and numerical limits in a new versioned probe manifest **before** executing the qualification. Existing numeric probe tolerances remain starting requirements: constraint residual<=2e-5 m, mass/inertia relative error<=1e-5, collision extent error<=1e-5 m, small-angle period relative error<=0.10, passive positive-energy-growth allowance<=0.05, and matched-time half-dt position/angle differences <=0.02 m/2 degrees. Define reference energies, expected physical dissipation, timestamp alignment and all additional excursion guards before their data are inspected. Do not reduce a failed duration, condition set or limit to obtain PASS.

Use these stages in order:

1. **Forensics and admission tests (CPU first).** Recheck old raw-state hashes and failure classification. Inspect executed archived sources separately from the later CPU guards. Add negative tests demonstrating rejection of old PASS receipts, changed settings/source/physics signatures, incomplete long probes, omitted terminal samples, nonfinite values and process-success/physical-failure confusion.
2. **Construction and motor audit.** Measure rotor identity, thrust/torque direction/application, root semantics, original/suspension/load masses and inertias, all joint freedoms and collision exclusions. Include no-load and paired loaded/unloaded rotor excitation. Measure reciprocal coupled motion; record which internal reaction forces are observable and which are unavailable. Never claim measured rope tension from an unavailable API.
3. **Passive diagnostics.** Across all 3 sizes×ratios 0.10/0.20/0.30, run the anchored small-angle fixture for at least 120 simulated seconds per condition. Include a separately declared large-angle joint-orientation sweep spanning potential coordinate singularities; label it a diagnostic, not safe flight. Check gravity, period, energy, quaternions, physical state and constraints. Passive energy loss can be expected from declared physical damping; unexplained energy gain fails. A fixed-anchor result cannot certify free flight.
4. **Long powered coupled probes.** Across the same 9 loads, run at least 120 continuous simulated seconds with no body/joint/history reset per case, for both zero-swing and 10-degree perturbed starts: 18 declared runs. Include hover, short/long transfers, reversals, ascent, descent, simultaneous XYZ changes and repeated commands covering the inherited envelope. Retain the original floor and attitude limits. An untrained controller need not meet the final one-second learning gate, but contact-assisted stabilization, detachment, unexplained energy growth or numerical excursion cannot pass physical qualification.
5. **Refinement and reset isolation.** Repeat all 6 size×endpoint-mass conditions over an identical predeclared 30-second command/initial-state segment at the candidate physics dt and half dt; compare same-time measured states. Freeze substeps from these results, not from acceptance success. Perform at least 30 episode/partial resets covering all sizes, runtime properties and nonselected-scene isolation. Check S and all ten nominal targets, every declared initial perturbation, and measured oriented-geometry clearance during powered transitions. Do not equate static clearance with dynamic clearance.
6. **Independent admission.** Reconstruct every declared case and threshold from raw samples and actual property records, not the runner's PASS flag. All required stages must finish and pass before an optimizer step is permitted. Save per-condition outcomes, complete sample counts, simulation duration, force/energy observability, source/settings/asset/model hashes and checker hash.

Probe-only prescribed motor excitation, weight-compensated hover or a deterministic diagnostic flight controller is allowed to exercise physical dynamics. It acts exclusively through the same four real rotors, is labeled in the manifest, and is never passed off as the learned controller or used inside acceptance. Explicitly record any privileged parameters available to that diagnostic controller. If powered tracking fails, distinguish insufficient diagnostic control, physical-limit failure and solver/numerical failure; an unresolved cause leaves admission blocked rather than proving universal infeasibility.

Evaluate energy in the appropriate system: powered flight can gain energy through motors, so record input/work or a justified bound; do not apply a passive conservation rule to powered trajectories or claim a complete energy balance without the needed measurements. Collect short-window samples before an abnormality and the offending raw state **before** reset, normalization or clipping. A later valid reset cannot erase an invalid episode.

### Admission lifecycle

The existing failed profile remains blocked. Develop a distinct experimental physical signature and run feasibility probes under that experimental status; this is permitted without falsely setting training admission to PASS. The initial experimental status permits diagnostics only.

A new qualification receipt admits only its actual measured physics signature, source/asset/config hashes, complete declared probes and numerical settings. Every training, validation, acceptance and freeze entry point recomputes these bindings before GPU initialization. The known failed signature and old receipts remain rejected. Do not unlock training by hand-editing `operating_domain_admission`, changing a status string or weakening a checker. The new profile may become training-admissible only through independently verified new evidence. Physical/source changes invalidate that admission and require the relevant complete qualification again.

If the new representation cannot pass within its phase budget, stop training/acceptance/filming, preserve candidates and failures, continue independent CPU/document work, and report the precise blocker plus a concrete next proposal. Do not claim full delivery for completing infrastructure alone.

## Observation, learning and validation

Reuse `payload29-ideal-world-v1` only after checking actual frames, scaling, clipping, attachment kinematics and action-history timing for the new physical backend. Its 12 added values are measured package-relative position/velocity, suspension direction and direction rate. No exact mass, dimensions, inertia, bank/geometry ID, seed or future event is an actor input. Hidden parameters remain verification/HUD data. Any semantic/history/normalization change gets a new contract and explicit compatibility mapping, even if the dimension stays 29.

Inspect actual legacy action timing: the previous environment's latency flag did not implement delayed action, and observation construction/history-update order differed between learning and retarget observation. Define and test the retry's actual training/inference timing consistently; changing it requires a versioned contract, new loaded training and affected predecessor checks. Do not infer behavior from the flag alone.

Initialize both policies from the verified unloaded model_4296 with the documented 17→29 first-layer transfer (copy legacy columns, zero additional columns, copy other compatible tensors, new optimizer/iteration 0), or a newly documented mapping if the observation contract changes. Do not resume the failed intermediate optimizer or use its checkpoint as a successful loaded prior.

Match architecture, initialization seed 7 and initial tensors, reward and components, optimizer, command distribution/curriculum, perturbation bounds, parallelization semantics and actual environment-step budget. Default to 300000 environment steps per method, as previously planned. Fix any revised matched budget before either new training run, explain throughput/remaining-budget evidence, and record it in the implementation plan and manifests; it must be identical across methods and cannot be selected on held-out outcomes. Full mass/size distribution is present at every randomized curriculum stage. Throughput is not an excuse for sampling once per environment. A single training seed permits conclusions about these two trained policies only, not seed-independent algorithmic superiority.

Curriculum covers stationary hover, short/long flight, reversal, ascent/descent, combined XYZ changes and multiple continuous target switches. Keep load/body/joints/history across targets. Log clipped/unclipped observations, saturation duration, actual lowest collision point, contact/constraint flags, rewards, motor commands and failures. Restore and test training safety/envelope/abort guards before GPU use. A finite but nonsensical state fails; do not optimize on catastrophic reward samples. Bound anomaly guards in advance and distinguish them from final acceptance thresholds.

Training, diagnostic development, validation and held-out streams are separate. Validation initially retains `configs/payload_validation.yaml`: seeds 11/12, loads (.030,.13)/(.045,.20)/(.060,.27), ten ordered transitions each, both policies: 12 continuous cases / 120 transitions. Compare actual checkpoints/actor tensors, initial conditions, targets, source/configuration and all raw samples. Validation must establish all required real transitions and full shared settling before held-out opens. Use the declared final matched-budget checkpoint rule; any validation-only redesign is a new recorded development attempt, preserving failures. Never select on held-out acceptance data.

## Completion, safety and paired acceptance

The following remain simultaneous requirements for at least 1.0 uninterrupted seconds, with 100 real valid intervals at dt 0.01 s:

| Quantity | Maximum |
| --- | --- |
| Drone center error / speed | 0.20 m / 0.30 m/s |
| Package center error from nominal hanging target / speed | 0.20 m / 0.10 m/s |
| Suspension angle / angular rate | 5 degrees / 10 degrees/s |
| Actual package angular speed, including axial spin | 10 degrees/s |

Compute attachment velocity with angular-velocity cross-offset terms. Compute swing rate from `norm(cross(r,v_b-v_a))/norm(r)^2`; a turning point is not settling. The nominal cube center is determined from the fixed drone target, declared attachments, length and half-side; do not recenter it on the drifting drone or expose size to the actor. Any invalid predicate resets dwell. A target change preserves both bodies, joints, velocities and policy history. Only shared completion marks a visit and triggers events/replanning.

Require oriented cube lowest point>0.10 m, retained drone floor>0.10 m, |roll|/|pitch|<=60 degrees, no contact-assisted damping, no drone/load collision, valid attachment and finite state. Preserve leg 30 s, mission 120 s and wall 600 s. If probe evidence shows a limit unsuitable, request an explicit spec revision before acceptance; no automatic relaxation is authorized.

For **each** policy run exactly these 127 new cases:

| Suite | Conditions | Randomized-policy gate |
| --- | --- | --- |
| Basic | Three scenarios×execution seeds 1/2/3, ratio 0.20/side 0.045 | 9/9 |
| Endpoints | Three scenarios×three sizes×ratios 0.10/0.30 | 18/18 |
| Held-out | 100 predeclared random cases | >=95/100 |

The fixed baseline runs the same 127, total 254, with no mandatory superiority/gate for the baseline. Report both honestly. Failures and timeouts stay in their original denominator. Predeclare Wilson-score 95% binomial intervals and report size/scenario/mass subgroup counts; no interval is reported for zero attempted trials.

Preserve the prior declared case draws: `outputs/suspended_20261021_v1/cases-v1.json`, SHA-256 `f98e0632d96a4d7b40feda667bff6e9ca831fc7f67e1734f2438e3c527ecf01c`, or independently reconstruct the same physical cases from `payload_cases.py`. Held-out allocation is 11 per scenario×size plus 1 smallest-size normal case. Initial swing is uniform 0–10 degrees/direction 0–2pi, x/y independently±0.02 m, roll/pitch independently±2 degrees, zero yaw, zero velocities, z 1 m, with constraint-consistent package state. Basic scenarios share their initial state per execution seed; both methods share every case; endpoints use 10-degree swing and the recorded deterministic directions.

The old manifest's provisional attachment/admission labels do not constitute a freeze. Produce a new versioned wrapper bound to the qualified physical backend; preserve exact case IDs/loads/initial draws, recording choices and scenario bytes, and document metadata-only migration. Do not reroll to match a preferred outcome. A changed accepted model/settings requires a fresh complete suite with failed previous suites retained; never combine model revisions into 254 allegedly paired trials.

Before acceptance freeze actual coordinates/priorities/events/distribution/initial cases, both model/config hashes, routing model/sidecar, observation/action timing, backend/assets/numerical settings, all thresholds/timeouts, camera, actual execution source and complete prior validation telemetry. The independent checker reconstructs every target, reset continuity, shared completion, event time and actor/batch/source binding from actual samples. Corrupted metadata, missing samples and stale receipts fail closed. Recompute routing counterfactuals, Q*, J*, learned suffixes and raw/weighted costs from new real decision boundaries, vehicle poses and remaining tasks; distinguish an optimal next action from full suffix optimality.

## Video, documentation and delivery

Freeze basic-suite seed 1 for normal/emergency/cost-shift footage, nominal load 0.20/.045 and the adopted randomized controller. Preserve complete `raw.mp4`, `frames.csv`, telemetry/events/decisions, settings and source manifests. Generate `full_normal.mp4`, `full_emergency.mp4`, `full_cost_shift.mp4`, including all ten shared completions with no omitted flight intervals at simulation-time 1x. No substitution of a more attractive seed after failure.

The Japanese HUD shows all ten IDs/integer priorities/states/current target, actual drone/package altitudes, load conditions, link, swing and settling progress. Label the compression-capable fixed-link approximation accurately. Check genuine frames for package/link/drone visibility, overlap/occlusion, height differences and readable labels. If an inset is needed, crop the same actual source frame. Never animate a decorative payload separately.

Independently decode every raw/final frame, check timestamps/source-step coverage/unit time slope, actual HUD text/pixels and telemetry agreement, and final stabilization coverage. Preserve failed encodes and fix encoding into a new output rather than weakening pixel/text limits. Link final files and concrete continuous 1x intervals from before each event through the next shared completion; request fresh human review tied to final hashes. Continue CPU/document work while waiting, but operator-pending is not PASS. Prior video approvals are historical only. Combined `demo_full.mp4` and 60–180 s submission editing remain out of scope.

Update README, Japanese report and affected PDF; render and inspect all regenerated PDF pages. Create `docs/suspended-payload-flight-retry-handoff.md`, a new compact evidence directory and English PR description linked to this spec. Record base revision, rejected/selected physical candidates and exact approximation, root/rotor audit, actual mass/inertia/geometry, observation/action contract and transfer, all hashes, source execution differences, sampling/training/validation/test separation, all failures and 254 results, media/review, GPU totals and reproduction commands. Keep earlier handoffs/specs/evidence as history. Missing goals must remain explicit in every summary.

## GPU and execution constraints

The 2026-10-21 snapshot of both ledgers is cumulative 7484.093172315159 s, remaining 21315.90682768484 s from 28800 s, 39 jobs, no active job; both ledger SHA-256 values are `0290c9a1aec9ddc4094025542439aa3d4fcc7ee73b5ba710d59e5ce27950c200`. Reconcile again at startup; any newer authorized use supersedes this snapshot. Never reset totals, double-count historical jobs, or drop failures.

Initial allocation from that snapshot:

| Phase | GPU wall-time ceiling (s) |
| --- | ---: |
| Physical candidate/probes/qualification | 3000 |
| Randomized training | 3500 |
| Fixed training | 3500 |
| Validation | 1500 |
| All 254 acceptance trials | 6500 |
| Complete raw/HUD scenario recording | 2000 |
| GPU-dependent final verification | 500 |
| Reserve | 815.90682768484 |

This is a planning allocation, not a promise of feasible throughput. Re-estimate from measured throughput before training; retain both complete evaluation and recording reserves. Explicitly record reallocations within the remaining total. If the plan cannot fit, report that rather than silently reducing trials, target scope, validation, or video coverage.

Use one GPU job at a time with supervisor-enforced finite timeouts that fit remaining phase and total budgets. Charge initialization, inference, training, filming, failed attempts and retries. Do independent CPU checks/docs during GPU jobs. The previous GPU instance is being retired by the operator. Use the replacement instance created by the operator and its newly supplied SSH endpoint; never reuse the old endpoint or assume its filesystem survives. If that endpoint is not provided, ask once before GPU-dependent work and continue independent CPU work meanwhile. Check the human-authorized replacement only when GPU work is needed. Use a fresh workspace, experiment/output root and isolated Python 3.12 flight/CPU-routing environments with the existing locks; verify exact package versions. Do not copy stale absolute server paths. No rental/contract change, added rental, server destruction or dependency upgrade is authorized.

Store model/video/large logs under a unique ignored `outputs/suspended_retry_<date>_<unique>/`; never overwrite `outputs/suspended_20261021_v1/`. Transfer external sources, actual settings, models and all telemetry locally and verify file sizes/SHA-256 before consuming or reporting them. Public documentation uses repository-relative artifact paths; connection details, local account paths and secrets remain outside Git.

## Acceptance criteria and stopping states

1. A physically coupled fixed-link representation passes every new declared short/long/refinement/reset/motor probe; old receipts remain rejected. Probe control and learned control evidence are distinct.
2. All episode/property/isolation and actor-hidden-parameter requirements are verified from effective runtime data and negative CPU tests.
3. Both policies complete matched, auditable learning and independent real-transition validation; observation/checkpoint/physical mismatches fail before inference.
4. One frozen randomized controller passes 9/9, 18/18 and >=95/100 independently. The nominal comparison executes the same 127 and is reported without manufactured superiority.
5. All real shared completions, safe continuous transfers and scenario events are reconstructed independently, including exact routing explanations where their proof conditions hold.
6. Three complete raw/HUD 1x videos pass every frame/time/text/telemetry check and fresh human review for final hashes.
7. Source/artifact/validation/acceptance bindings, external integrity, cumulative GPU accounting, Japanese docs/PDF and the new spec/index are complete and truthful.
8. One new local commit contains only this retry's spec, code, tests and necessary compact docs/evidence, authored/committed as `dstaka <22057509+dstaka@users.noreply.github.com>`. Do not amend, change global identity, push, create/merge a remote PR, publish videos or submit the course.

Use explicit states such as `EXPERIMENTAL_DIAGNOSTICS_ONLY`, `QUALIFICATION_FAILED`, `QUALIFIED_FOR_TRAINING`, `VALIDATION_FAILED`, `ACCEPTANCE_FAILED`, `HUMAN_REVIEW_PENDING` and `ACCEPTED`. Incomplete qualification blocks optimizer/trials/filming; later failures block adoption and honest completion. Distinguish missing evidence, insufficient control, actual physical failure, numerical divergence, exhausted budget and unavailable server. Preserve useful CPU work and produce a failure handoff/local commit when GPU progress is blocked; never label it accepted.

## API, data and compatibility impacts

Introduce a new physical profile/signature and qualification-receipt version rather than erasing the failed profile's status. Define source units/frames and profile-bound scene/observation/checkpoint/reset/probe schemas. Preserve legacy unloaded entry points and loaders. Migrate case metadata explicitly without changing physical draws. Validate effective physical properties and actual actor tensor signatures; labels, shape alone, summary-only target sequences and file hashes without semantic reconstruction are insufficient.

Shared observations, motor adaptation, routing or unloaded execution changes require affected predecessor quality gates to run again, with changed sources and new evidence bound together. If those gates cannot run, record compatibility as unverified. Do not combine separately passing historical components into a new integration claim.

## Out of scope

Tension-only/slack/elastic/deformable rope, payload release/replacement, multiple loads, hidden swing stabilization, wind/drag modeling, sensor estimation, payload-energy routing retraining, new hardware, changed coordinates/load domain, real aircraft, dependency upgrades, combined/submission video editing, publication and GPU rental changes. Such changes require their own explicit review rather than becoming a fallback for a failed fixed-link experiment.


## Implementation record (2026-10-22)

Human execution instructions started implementation from the expected revision. A separate native two-ball profile, actual four-rotor adaptation, read-only substep work observations, independent raw checkers and retry-only observation/training safeguards were implemented. The original failed profile and all historical evidence remain blocked and unchanged. New short motor and reset checks passed within their scope; a long powered condition violated the unchanged 0.10 m package-floor gate at 39.35 s. Qualification is therefore not admitted, and no retry optimizer, validation, 254 acceptance missions or final videos have run. All 55 declared conditions were attempted once with the frozen V10 source: 8 stopped on the package-floor limit, and all 6 matched dt/half-dt comparisons exceeded the predeclared tolerances (2 also lacked the full 30 seconds). The independent checker also found passive positive-energy growth above 5% in 14 of 16 large-angle conditions, despite normal runner completion. The nine small-angle 120-second anchor conditions passed within scope. These diagnostics do not establish numerical repair or general physical infeasibility. No failed case was erased or reclassified as passing. Post-qualification CPU-only guard/binding corrections have no new physical admission. See the [retry handoff](../../docs/suspended-payload-flight-retry-handoff.md) for final counts, hashes, limitations, budget and reproduction. No acceptance requirement above is changed by this record.
