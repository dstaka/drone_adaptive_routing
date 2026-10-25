# Video-First Suspended-Payload Transport — MuJoCo Controller-Only Completion

- Date: 2026-10-23.
- Revision: 3 (2026-10-25), implementing the user-directed MuJoCo controller-only completion scope. Revision 2 requirements retained below are historical where superseded by §0.
- Status: Completed under Revision3: MuJoCo controller-only acceptance127/127, all required evidence, three final fixed-overview native1× videos, full local size/SHA verification and explicit fresh human review passed. Related implementation, tests, evidence and documentation are retained in the final local commit; no push/PR/merge/publication.
- PR: Not created or requested; retain this specification with the final local commit. Do not open a PR.
- Goal: Complete the qualified MuJoCo controller-only runtime, robustness, formal acceptance and three enlarged-fixed-overview full-speed videos. Preserve all existing evidence and report remaining failures honestly.

## 0. Revision 3: User-directed scope change (2026-10-25)

**This section takes precedence over every conflicting statement in the retained Revision 2 sections below.** The user selected MuJoCo as the final primary backend based on three controller-only ten-destination development successes. Genesis has partial numerical, parallel and short-control evidence, but no demonstrated full operational ten-destination flight, deadline margin or learning advantage. No further proof of Genesis impossibility is required.

1. Stop new Genesis experiments, retries, optimization and learning infrastructure work. Preserve partial raw, logs, settings and sources before safely stopping only this session's jobs. Record user-directed interruption separately from numerical pass/fail; protect previous data, checkpoints, source bindings and RAM keepers. See [Genesis interruption record](../../docs/suspended-payload-genesis-interruption.md).
2. Adopt the existing qualified MuJoCo 3.15.0 / float64 / RK4 / substeps16 / N=1 and controller `C-local-lqr-v1` as first choice. The Genesis prerequisites of §5.4, parallel execution or N=1 exception of §6, learning gate, C/H development comparison, NN pilots and fixed/random NN training are **not required for this completion**. NN work needs a demonstrated controller limitation and an explicitly presented additional scope; do not resume automatically.
3. Rebind actual source, physical assets, motors, configuration and preserved raw. Reuse valid MuJoCo numerical/motor/reset and C0/C1/C2 evidence; repeat only checks invalidated by a change. Never change physics, damping, motor authority, goal coordinates or routing objective without authorization.
4. Keep §8.1 exactly: `payload-settling-video-v1`, package speed ≤0.15 m/s, swing speed ≤15°/s, complete box angular speed ≤30°/s, inherited position/inclination thresholds and simultaneous continuous one-second dwell. Keep leg30s and wall600s; no aggregate120s/12000-step mission termination. Audit ten real destinations independently. Retain old strict predicates on the same raw as diagnostics without relabeling historical failures.
5. Before final freeze, declare and execute15 development robustness missions: three nominal scenarios, six independent interior loads (two per scenario, covering three sizes and independently sampled ratios), and six endpoints (two per scenario), with distinct development seeds. Diagnose failures and allow only finite, justified controller changes; changed physical control source requires affected gates again. Complete nominal plus six endpoint continuous330s powered traces without hidden resets. These duration tests are not missions.
6. Retain C validation12 conditions/120 actual commanded-and-dwelled legs (seeds11–14 × inherited three loads). The inherited ten-target sequence includes an initial S hold and a repeated final S hold: report96 coordinate transfers and24 holds separately, rather than claiming120 distinct-coordinate moves. Each hold still requires actual closed-loop physical intervals and full dwell; preserve the exact inherited coordinates. Freeze runtime before exposing formal cases. Run the inherited127 cases once: basic9/9, endpoints18/18, held-out≥95/100, retaining every failure and confidence intervals. Never use held-out for development, tuning or teacher data.
7. **First deliverable:** re-render one existing successful development raw by enlarging the original **fixed overview containing all ten destinations**, adjusting zoom/look-at to reduce empty sky/floor. Preserve the original individual drone/package detail view without enlarging it. This follows the user's explicit camera clarification; an enlarged detail-follow layout was rejected and remains an archived display experiment. Verify travel and settling visibility. This is camera-only improvement, not another physical success. Preserve full100Hz native frame coverage and actual1× timing with no cuts or pose modification. Show actual controller/backend/profile and explicit development/final status.
8. After formal acceptance, capture normal, emergency and cost_shift final videos with the same improved camera. Report exact routing-objective gaps; any positive gap prohibits a claim of optimal routing. Collect required raw/model/video/log/config/source artifacts locally with sizes and SHA-256. Produce Japanese results/handoff and English PR description. Request specific fresh human review of all final continuous1× videos; pending review is not completion. Commit only related work locally using the requested author/committer; do not push, open PR, merge or publish.

Completion requires valid unchanged MuJoCo qualification, development robustness, seven330s durations, C validation12/120, formal127 with its gates, independent raw and route evidence, three final enlarged-fixed-overview videos, verified local recovery, fresh human review and documentation. Genesis completion, NN training and parallel/GPU demonstrations are outside the remaining mandatory scope. Preserve Revision2's historical prospective rules and results below; they must not silently become active gates again.


## 1. Scope and Relationship to Existing Specifications

This specification is a follow-up to [Solver Recovery and Evidence-Driven Learning](../2026-10-22-suspended-payload-solver-recovery/spec.md). The control, learning, backend-selection, settling, deadline, capture-order, and branch-specific completion rules below take precedence over conflicting predecessor specifications, instructions, plans, and automated guards. Other physical-property, numerical-verification, routing, evidence, and recording requirements are inherited. Preserve previous specifications and experimental results as historical records.

| Topic | Rule under this specification |
| --- | --- |
| Control architecture | Establish a controller-only baseline first. After the learning gate, compare it with bounded NN correction in finite development trials. Either may be the final flight controller; NN flight training is not a prerequisite for the first video. |
| Observations | Remove the fixed 29-dimensional restriction. Measurable or estimable payload orientation, angular velocity, history, and related signals may be added. Ground-truth mass, dimensions, and similar parameters must not be supplied at runtime. |
| Initialization | Transferring specific columns from model4296 is no longer mandatory. Use development comparisons to select a common controller, pretraining with common teacher data, compatible existing weights, or fresh initialization. |
| Learning tasks | Permit a curriculum that advances based on demonstrated skills instead of only a fixed transition count. |
| Primary backend | Prioritize GPU-parallel execution in Genesis. Do not retain the existing MuJoCo CPU N=1 path as the main training path by inertia. |
| Role of MuJoCo | Permit short control-feasibility diagnostics and an early development video on the already qualified configuration before Genesis qualification. This is separate from the evidence-based primary-backend fallback in §5.4. |
| Settling | Use the versioned video-first profile in §8.1: retain position, suspension angle, and one-second simultaneous dwell; moderately relax three speed limits. Also report the former strict profile as a diagnostic. |
| Deadlines | Remove the 120-second aggregate simulated mission deadline. Retain the 30-second per-leg and 600-second per-mission wall-clock limits, with precise clock semantics in §8.1. |
| Deliverables | Deliver a verified development video as soon as one full mission succeeds. Full completion still requires three final videos, branch-specific evaluation, route evidence, and fresh human review. |
| Optimization objective | Reduce wall-clock time to reliable settling and transport. GPU utilization, episode count, or transitions per second alone are not success metrics. |
| GPU time | No time cap or cumulative GPU-usage tracking. Retain short performance measurements, elapsed-time measurements, and mission deadlines. |

Creating this document does not stop or modify existing jobs, initiate SSH access, or start training. When implementing this specification, inspect any ongoing previous comparisons, source bindings, and artifact preservation, then move to a new run namespace without modifying old runs. Do not confuse automatic execution helpers for previous comparisons with instructions to execute this specification.

The user authorized moderate settling relaxation and removal of the aggregate 120-second limit. The specific thresholds, finite comparison procedure, and controller-only completion branch below are the proposed implementation choices made under that direction. This revision does not authorize additional physics changes or retrospective relabeling of old failures.

## 2. Background and Current Problems

The [handoff](../../docs/suspended-payload-solver-recovery-handoff.md) consulted when writing this document reports that the MuJoCo CPU, float64, RK4 physical configuration passed qualification, while multiple learning attempts failed to achieve shared settling. Update progress from saved evidence at execution time; do not treat the numbers in this document as the live state of running jobs.

- In early tasks, some trajectories satisfy position and translational-velocity limits while retaining transverse payload rocking at approximately 4.2 Hz. Position holding and active vibration suppression are distinct skills.
- In one 100-second interval, mean payload angular speed was approximately 96 degrees/s, and the spin cost reached its cap for approximately 99% of the interval. This concerns the discriminating power of that direct cost; it does not mean the entire PPO gradient is zero.
- In 107 selected failure episodes from later tasks, insufficient supporting thrust led rapidly to the floor condition. Failure returns exceeded a constructed distant-hover benchmark. The benchmark is not an executed policy and does not reproduce actual target changes, so this does not establish intent to crash, optimality, or causality.
- Existing N=1/8/32/128 throughput measurements were serial diagnostics of independent scenes. They do not establish the limits of actual GPU or CPU parallelism.
- Incorporate the timeout-bootstrap fix for simultaneous tenth-target success and deadline expiry into the next source version, but do not present it as the cause of zero shared completions.

Evidence: [reward diagnosis](../../docs/evidence/suspended-solver/discounted-reward-diagnosis.json), [trajectory diagnosis](../../docs/evidence/suspended-solver/long-credit-feedback-diagnosis.json), and [PPO integration verification](../../docs/evidence/suspended-solver/ppo-integration-review.json).

FLARE supports Genesis-based suspended-payload RL and a collective-thrust/body-rate action interface; its passage tasks do not establish full rigid-box angular settling. ASTER supports informed exploration, but its taut/slack seeding equations do not directly apply to this massive rigid link. The single-pendulum energy expression alone omits the box's rotational mode. Use these studies as design evidence, not as proof that this model or its acceptance suite will succeed; see References.

## 3. Selected Approach and Alternatives

The first candidate, C, is a controller responsible for vehicle stabilization, target tracking, and active payload vibration suppression. Use a cascaded position/attitude controller with payload-state feedback as the initial approach; use geometric control or a local linear design where appropriate. Consider MPC only when a diagnosed limitation justifies its model and batching cost. Establish actual C success before learning a correction.

The second candidate, H, adds a bounded NN correction to the same C. It may learn compensation for model error, load differences, and vibration suppression. Compare C and H under §7.4. Both architectures are tried in development once the learning gate passes, but do not delay a successful C video until H finishes. If C remains preferable, adopt C and mark flight-NN final training as not applicable, with the development comparison retained.

An optional third candidate pretrains or distills an NN from successful controller trajectories. Use it only to address a recorded deployment or performance limitation of C/H; it is not another mandatory search branch. Teacher success must not be reported as NN success. Perform actual closed-loop simulated flight verification after distillation as well.

Continuing purely neural direct motor control while only adding large numbers of samples is not the main path. Any such comparison must be a small development experiment with its own hypothesis and stopping decision. If the controller alone meets the requirements, report that result; do not claim the NN is necessary for performance without demonstrating an improvement.

The control architecture, experiment sequence, and decision procedures below are design choices that make the user's goal and backend priorities concrete. They are not methods already proven successful.

## 4. Hybrid Control and Observation Contract

### 4.1 Control Path

1. Generate a reference trajectory with limits on acceleration, jerk, and related quantities toward the current target selected by the existing routing system. Preserve the routing contract, coordinates, and the revised deadlines in §8.1. Intermediate reference points do not count as completed destinations.
2. Build a baseline for attitude, altitude, and position stabilization and payload vibration suppression. Compare options such as geometric control, vibration suppression with state estimation, and MPC based on development feasibility and GPU-batching cost.
3. Apply the NN correction at one explicitly defined layer, such as acceleration references or thrust/moment references. Record correction bounds, units, frames, action scaling, saturation, and anti-windup before execution. Verify that zero correction at initialization reproduces the baseline.
4. All actuation must continue to pass through the four motors corresponding to the physical model. Neither the controller nor its correction may exceed the original RPM limits or alter KF/KM, rotor locations, or spin directions. Do not apply direct forces to the payload or overwrite poses or velocities.
5. Preserve the external 0.01-second control interval and current-action semantics. Do not update discrete controller state extra times during physics substeps or intermediate RK stages. Evaluate state-dependent coordinate transformations of motor forces correctly along each backend's actual integration path.

If safety limits or correction suppression are used, record their activation and the actions before and after correction, and use the same processing during training, evaluation, and recording. These mechanisms do not constitute a flight guarantee. Resets, teleportation, and unrecorded switches to reference control that conceal failures are prohibited.

### 4.2 Observations and Information Boundaries

- The runtime contract applies to the entire control system, including the baseline, state estimator, and trajectory generator, not just the NN. Do not bypass the restriction by supplying ground-truth mass or similar information outside the NN.
- In addition to vehicle state, target error, relative payload position/velocity, and link direction/angular velocity, consider payload orientation, full angular velocity, and necessary history. Fix each quantity's units, frame, normalization, clipping, timestamp, and reset initialization. Reconsider scales that continuously clip important variations.
- The baseline must resolve the fast box-rotation mode through measured payload attitude/angular velocity or a validated estimate from permitted history. Link deviation alone is not assumed sufficient. Record estimation bandwidth, delay, and error against checker-only truth during development.
- Do not supply true mass, dimensions, inertia, bank ID, random seed, or future events as runtime inputs. Online estimation from measurement history is permitted, but do not embed ground truth in initial values or internal state. Verify estimation error and reset behavior.
- The diagnostic teacher and training-time critic may use privileged state. Distinguish this from the runtime measurement/estimation path, and test that teacher information does not leak into the actor or baseline.
- Explicitly identify simulator getters used as ideal sensors. Do not claim demonstrated transfer to real hardware. Separately document sensor availability and development evaluations of delay and noise.

### 4.3 Explicit Gate Before Learning

First use the already qualified MuJoCo float64 RK4 N=1 path for short controller-only diagnostics after checking the actual source, assets, precision, timestep/substeps, motor mapping, and qualification bindings. These diagnostics may precede Genesis work and do not require a primary-backend fallback decision. They permit controller design and development recording, not early PPO or substitution of a new unqualified physics configuration. A physics/backend change requires fresh qualification.

Predeclare a development manifest with keyed seeds, initial states, target lists, load conditions, the §8.1 profile, controller source/configuration, and all attempt identifiers. Use no formal validation or held-out cases. Run the following gates in order with frozen parameters within each complete gate:

| Gate | Executed cases | Pass criterion |
| --- | --- | --- |
| C0: nominal hover | 10 independent nominal-load starts: five at 5-degree and five at 10-degree initial link inclination, varied azimuths and inherited vehicle perturbations | 10/10 attain simultaneous §8.1 settling within 30 simulated seconds, with no safety failure |
| C1: nominal move/return | 10 runs: each of the offsets (+0.50,0,0), (-0.50,0,0), (0,+0.50,0), (0,-0.50,0), (0,0,+0.50) m from S, with two independent starts; command S, S+offset, S in that order | At least 9/10 complete all three targets, each within its own 30-second leg, and all 10 remain safe |
| C2: endpoint transfer | Each of six size/mass endpoint combinations, five independent starts with 10-degree initial link inclination; use each C1 offset once per combination, including return | At least 4/5 complete all three targets for every combination, and all 30 remain safe |

The 0.50 m offsets exceed twice the 0.20 m position tolerance, so consecutive target neighborhoods do not overlap. Audit actual body displacement and target changes; relabeled stationary samples cannot pass. Nominal means side 0.045 m and mass ratio 0.20; endpoints are three sizes times mass ratios 0.10/0.30. Initialize constraint-consistent bodies, the upright box, and zero velocities as in the inherited distribution. Do not add arbitrary box axial spin: for the ideal isotropic cube and torque-free top-center ball attachment, that component cannot generally be damped by support translation alone. These starts test the actual intended domain, not arbitrary-state controllability.

Any nonfinite state, invalid constraint, contact, clearance/tilt violation, or undeclared reset fails the gate. A leg timeout is an unsuccessful case even if it has no safety violation. Preserve every attempt; after changing a controller, rerun its whole failed gate and all earlier gates with the new frozen configuration. Do not average failed versions into a passing result. Initially declare at most three controller hypotheses; further attempts need a new diagnosis and recorded design change, not unchanged repetition.

Record full time histories and per-case first settling time, all predicate margins, longest joint-valid interval, low-frequency link motion, fast transverse box rotation, body-axis angular-speed components, and amplitude envelopes. Estimate dominant frequencies from each run rather than fixing them at 4.22 Hz. Compare identical initial states with the vibration-feedback term enabled/disabled while retaining the same vehicle stabilization. Report envelope decay (or absence of decay), command-to-motion phase/delay near observed modes, motor saturation duration, and position/safety excursions. A decay fit or phase estimate without sufficient excitation must be marked unidentifiable. Reward increase or lower mean angular speed never substitutes for the C0–C2 counts.

**Learning admission requires C0, C1, and C2 plus base numerical/motor/reset/batch qualification (all62/motor15/reset30 and batch isolation) of the actual learning runtime.** Repeat C0–C2 after porting to the selected Genesis configuration (or justified fallback). On a batch, scatter predeclared cases across environment indices and partial resets; do not accept a passing N=1 result as batch evidence. In addition to §5's same-input refinement, compare feedback-on/off and refined-timestep results to establish that apparent settling is not unexplained numerical damping.

A controller failure is a finding about that design, not proof of physical impossibility. If a gate fails, diagnose authority, observability, phase, estimation, or the deadline before another design. Do not launch large-scale RL to bypass a failed gate. A successful full development mission can be recorded before all learning gates finish under §8.2, with its evidence and provisional scope explicit.

## 5. Genesis-First Numerical Consistency and Fallback

### 5.1 Unchanged Physics

Preserve the original vehicle and four motors, masses/inertias/frames, the 0.30 m rigid link with nonzero mass, passive ball joints at both ends, attachment locations, payload sizes of 0.030/0.045/0.060 m, and the independent mass ratio of 0.10–0.30. The fixed-load comparison uses 0.045 m and 0.20. Do not change the formal evaluation distribution.

This specification does not authorize adding damping, replacing the link with a rope, replacing bodies with point masses, or adding motor authority. Treat damping or attachment changes based on physical measurements as separate physical-model changes. Do not disguise numerical dissipation as physical damping to obtain settling.

### 5.2 Selecting Genesis Candidates

1. Inspect the actual installed source and dependency versions. Declare a small set of candidates, each with a hypothesis, covering supported GPU integrators, float32/64 precision, substeps, and constraint settings.
2. Use the current MuJoCo float64 RK4 configuration and an independent model as references. Measure timestep convergence with identical physical properties, initial states, and inputs. Refine MuJoCo's timestep as well to assess reference error. The short CPU control-feasibility path in §4.3 and early capture in §8.2 are permitted independently of the conditions for switching primary training to MuJoCo.
3. If the cause of inconsistency under standard settings can be identified, consider necessary Genesis version updates or targeted integration/constraint-processing fixes. A new RK4 implementation is not mandatory; it is a candidate only when measurements establish a need.
4. Do not require the same integration order or byte-identical trajectories. Judge physical properties, forces/moments, reactions, energy balance, constraint error, oscillation period/phase, and short-time state convergence. Long-duration divergence of large-angle trajectories alone does not establish inconsistency.

### 5.3 Qualification and Thresholds

Inherit all 62 traces from the existing specification: passive32, powered18, and powered refinement12, plus motor15 and at least reset30. Do not start powered diagnostics on a new configuration before passive32 passes. Do not run a learning optimizer until the selected backend, precision, integrator, and batch configuration have full qualification and §4.3 passes. Existing qualification may support only the unchanged, re-bound MuJoCo reference configuration; do not reuse N=1 qualification for Genesis or a new batch.

Preserve positive passive-energy growth of at most 5%; absolute unexplained energy change, after accounting for declared dissipative and external work, of at most 5%; constraint residual of at most 2e-5 m; physical-property relative error of at most 1e-5; collision-extent error of at most 1e-5 m; and small-angle period error of at most 10%. Preserve the powered same-input base/half-dt limits of 2 cm/2 degrees. Follow the predecessor specification for each case's initial conditions, duration, work reconstruction, and other guards.

Verify work definitions and intermediate integration stages for the new backend. Do not transplant fields from another method by merely matching their names. Do not loosen tolerances, angle ranges, test durations, or load conditions for performance.

Removing the mission deadline does not shorten the existing 120-second numerical test traces. Before final freeze, add continuous powered controller runs covering 330 simulated seconds at nominal load and all six size/mass endpoints, without resets, to check the longer possible operating horizon. Keeping the last reached target is permitted in these explicitly labeled duration tests; it is not ten-destination mission evidence. Apply the same safety, work/constraint, and source-binding checks. These additional duration checks are final-freeze gates, separate from the base qualification required before development learning; they do not delay short diagnostics or early previews. Record a shorter preview as provisional pending these extended checks.

### 5.4 Conditions for Making MuJoCo the Primary Backend

An inability to achieve consistency does not require an infinite candidate search. Evaluate a small set of predeclared settings, precision/timestep convergence, independent references, and feasible fixes or version updates that address the observed cause. Identify promising untested candidates explicitly before making the decision.

Fallback requires a decision record containing candidates, executed source, raw comparisons, failed metrics, cause analysis, repair results, and why remaining remedies are not feasible. Not having started, inconvenient GPU implementation, greater familiarity with MuJoCo, or Genesis being slow are not sufficient reasons for fallback. If a consistent Genesis configuration is found, minimize time among consistent Genesis configurations.

Only after the fallback conditions are met, compare GPU-batched MuJoCo options such as MuJoCo Warp/MJX first. GPU RK4 does not inherit CPU float64 qualification; reverify precision, inputs, resets, and other required properties. If GPU candidates are also unqualified or unavailable in the actual environment, CPU parallelism may be used with a recorded reason. Use CPU N=1 mainly for reference calculations, local diagnostics, and fault isolation, and explicitly justify any need to use it for primary training.

Resolve the N=1 completion exception explicitly: if no GPU/CPU parallel configuration qualifies or runs reliably in the available environment, a qualified N=1 runtime on an otherwise permitted backend may complete the task. This includes a qualified serial Genesis configuration; choosing MuJoCo still requires the primary-backend fallback decision above. Record the tested candidates, failed metrics or availability evidence, feasible repair attempts, and why a measured parallel alternative cannot be used. Execute at least one actual parallel candidate when the environment supports it; a feature/API availability probe may instead establish an unsupported path. Passing throughput alone is not qualification. Report `parallel_execution=NOT_ACHIEVED_WITH_JUSTIFIED_EXCEPTION`; never claim GPU speedup or waive the retained flight, wall-clock, evidence, and video requirements. Finishing a controller-only flight task does not require artificial NN training to demonstrate GPU usage.

## 6. GPU Parallelism and Time Reduction

- Batch physics, the baseline, NN inference, rewards, termination checks, resets, rollout buffers, and the optimizer on device wherever possible. Avoid Python loops that process environments serially and CPU round trips on every step. Identify CPU requirements through measurement.
- Run multiple environments within one managed GPU job. Do not launch unrelated competing jobs; protect existing jobs and artifacts.
- Use N=1 as a reference, then investigate 8/32/128 and larger sizes supported by memory and measurements. Do not impose a fixed batch-size or substep ceiling; select from verified valid configurations.
- Keep physical properties, RNGs, controller/estimator/NN history, targets, dwell counters, termination masks, and resets independent per environment. Test partial resets with continuing neighboring environments and compare N=1 against batched execution under identical inputs.
- Measure the complete path, including inference, control, actual physics, resets, required evidence recording, and the optimizer. Report compilation/setup time, valid transitions per wall-clock time, wall-clock time and samples to successful settling, and evaluation duration separately. Do not select a configuration based only on GPU utilization.
- Reevaluate rollout length, update frequency, and experience diversity when increasing batch size. More short failure episodes do not count as improved learning efficiency.
- Recording may use batched transfers and asynchronous writes, but must retain the raw data required for existing independent audits. Verify transfer/write backlog handling, missing records, ordering, and final flushing.
- Focus improvements on major latency sources rather than developing unlimited candidates. Do not create a cumulative GPU-time ledger or remaining-time calculation.

## 7. Learning, Teacher Data, and Fair Comparisons

### 7.1 Rewards and Tasks

Compare returns for safe progress, no-progress hover, early failure, and success using executed trajectories with the same target-change, deadline, and termination rules. Do not choose coefficients from constructed-state arithmetic alone. Inspect accumulated distance costs, success/failure rewards, discounting, and timeout bootstrapping together.

Design a vibration-suppression signal that distinguishes improvement even at high payload angular speeds; removing the cap alone is not a complete solution. Record progress, vibration suppression, action saturation, and correction magnitude terms independently. Distinguish true success/failure from time truncation, and use a new source version containing the known bootstrap fix for simultaneous success and deadline expiry.

Stage the curriculum through hover settling, small movements, continuous retargeting, and ten destinations. Progression must depend on success and safety over multiple episodes under criteria fixed before testing; do not advance solely because time has elapsed before the criteria are met. Mix in previously mastered tasks and monitor forgetting. When progression stalls, revise the hypothesis rather than adding samples indefinitely under unchanged settings.

### 7.2 Teacher and Initialization

Record the teacher's source, information access, load conditions, successes/failures, states/actions, collection seeds, imitation loss, and closed-loop performance. Do not use formal evaluation or held-out trajectories as teacher data. Prefer initializing the NN correction near zero, and compare development results with and without teacher pretraining.

The final randomized-load and fixed-load methods must start with the same baseline, observations, NN architecture, initial tensors, teacher data, and optimizer initialization. Generate common teacher data using only the fixed nominal load. If a comparison instead uses randomized-load teacher data for only one method, explicitly define it as a new comparison that includes different load exposure; do not confuse it with the existing fair comparison.

Apply the previous model4296 transfer rules only when compatible with the new architecture. Do not initialize just one final method from a trained checkpoint of a previous pilot. Compare the controller alone, the controller with correction, and, if needed, a distilled NN under the same development conditions, and select the simplest stable configuration.

### 7.3 Freezing Final Training When an NN Is Selected

After development demonstrates sufficient settling/movement success and experience across all stages, freeze the common S, N/T, seed7, controller, observations, initialization, reward, curriculum rules, optimizer, and checkpoint selection for both methods. With adaptive progression, use common rules and also report differences in actual stage exposure.

Use equal numbers of valid training transitions for both methods, and separately disclose the cost and experience associated with teacher collection and pretraining. Distinguish generated, invalid, and consumed transitions, vector steps, and update counts. Do not extend training or change conditions for only one method. When revising the plan, restart both under a new common plan.

These paired final-training requirements apply only to the NN-selected branch in §8.3. For the controller-only branch, record NN weights, S, training seed7, optimizer, and randomized/fixed final training as `NOT_APPLICABLE_CONTROLLER_ONLY`. Freeze the complete controller/estimator/trajectory-generator runtime instead. Do not label two identical controller copies as independently learned methods. The inherited routing NN remains in use and is distinct from the optional flight NN.

### 7.4 Finite Controller-Only / NN Comparison and Selection

After §4.3 passes, freeze C and try H. Use the same qualified runtime, permitted observations, target protocol, settling profile, and development cases. Predeclare at most two NN-correction recipes, each with a hypothesis and a common pilot allowance of 300,000 valid transitions per load method, with checkpoints at 100,000 and 300,000. Within a recipe, run randomized-load and nominal fixed-load pilots from identical initial tensors, common nominal-only teacher data if used, and fresh identical optimizer state. These are development pilots, not final seed7 training. Do not extend only a favorable arm or repeat an unchanged failed recipe. Further learning requires a new evidence-backed development plan; no cumulative GPU-time ledger is required.

Before comparing, declare a paired development suite of 12 full ten-destination missions: two independent nominal starts for each of the three scenarios (six), plus one for each size/mass endpoint (six, allocated two per scenario). Freeze its distinct seeds and allocation independently of validation/held-out/video fixtures. Run C once for each unchanged baseline and each pilot checkpoint on the same cases; retain all failures. Initial settling at S, if used, is separate from the ten destinations. Do not use the successful video as training or selection data.

Select by the following ordered rules, fixed before results:

1. An eligible candidate must complete all 12 missions with no safety failure, using §8.1. If none does, return to diagnosed controller/development work; do not freeze an ineligible final runtime.
2. If only C is eligible, select C. If C is ineligible and an NN candidate is eligible, select that NN architecture/recipe for fresh paired final training.
3. If both are eligible, prefer C unless an NN candidate reduces median full-mission simulated completion time by at least 10%, does not increase the maximum completion time, and does not worsen the suite's minimum floor clearance, peak absolute roll/pitch, or maximum settling-window payload angular speed beyond independent checker resolution. Report all per-case changes, not only the median. This rule rewards demonstrated benefit rather than obligatory NN inclusion.
4. For ties between eligible NN recipes/load methods, use lower maximum mission time, then lower median mission time, then lower measured end-to-end runtime cost; an otherwise exact tie favors the randomized method. Freeze the selected recipe and intended adoption method before final training. Either fixed-load or randomized-load may be selected on development evidence; the NN final branch still trains and evaluates both fresh methods fairly. Do not switch the adoption method after inspecting held-out outcomes.

Pilot evidence may select the architecture and common recipe; it may not replace fresh final training or guarantee its outcome. If fresh final NN evaluation fails, do not silently adopt its favorable pilot or choose another controller using held-out outcomes. Preserve that failed freeze and make an explicit new plan with an uncontaminated evaluation protocol.

## 8. Final Acceptance and Deliverables

### 8.1 Versioned Video-First Settling and Deadline Contract

Use `payload-settling-video-v1` consistently for development gates, rewards, curriculum success, validation, acceptance, route-event boundaries, and recording. The numerical choices are moderate prospective relaxations to obtain visibly stable transport; they are not justified by retroactively making an old run pass.

| Simultaneous predicate | Former strict profile | Video-first profile |
| --- | --- | --- |
| Vehicle target error | <=0.20 m | <=0.20 m |
| Vehicle speed | <=0.30 m/s | <=0.30 m/s |
| Payload error from nominal hanging target | <=0.20 m | <=0.20 m |
| Payload speed | <=0.10 m/s | **<=0.15 m/s** |
| Suspension angle from downward vertical | <=5 degrees | <=5 degrees |
| Suspension swing-rate magnitude | <=10 degrees/s | **<=15 degrees/s** |
| Full payload angular-speed magnitude | <=10 degrees/s | **<=30 degrees/s** |
| Uninterrupted shared-valid duration | >=1.0 second | >=1.0 second |

The full angular-speed limit includes all axes; do not discard axial spin or replace the magnitude with a time average. For a single small sinusoidal mode at 4.22 Hz, 30 degrees/s corresponds to approximately 1.13 degrees of angular amplitude, compared with approximately 0.38 degrees under the former limit. This is a scale estimate, not a general bound on arbitrary trajectories or proof of visual quality. Preserve exact raw metrics and inspect the actual video.

Keep the original definitions of the nominal payload target and attachment-point velocity, including rotational offsets. The checker may use true geometry; runtime control may not receive it. Any failed predicate or target change resets dwell. At 100 Hz, require 100 complete consecutive valid intervals, not 100 isolated samples or a turning point. Only this profile's shared completion marks a destination visited or triggers the inherited routing/scenario event. Compute the former strict profile independently on the same raw trajectory and report its per-target pass/fail/dwell; label it a shadow diagnostic, not a counterfactual strict-profile mission success. Later targets followed the video-first event stream.

Preserve oriented drone/package lowest-point clearance >0.10 m, absolute vehicle roll/pitch <=60 degrees, collision/constraint/contact checks, and the original motors. The deadline contract is:

- **No aggregate simulated mission timeout.** Remove the inherited 120-second/12,000-control-step termination from every new training, validation, evaluation, capture, replay, and scheduling path. Do not replace it with another aggregate success cutoff.
- Each commanded destination has **30 simulated seconds**, including travel and shared settling, starting at command issuance. Intermediate reference points and arrival do not restart this clock. If initial settling at S is used, it has its own 30-second window and never counts among the ten destinations. Ten destination windows plus initialization therefore bound a successful uninterrupted run by at most 330 simulated seconds as a consequence, not a new mission-time termination rule.
- Retain the **600-second wall-clock mission limit**, measured from initial physical reset through terminal-state evidence enqueue, including control, stepping, online inference, and required recording backpressure. Compilation/setup and post-run video encoding are separately reported. A wall timeout is a failed mission; CPU fallback does not waive it. Final evidence flushing must still complete before success can be certified.
- If a final valid interval completes exactly at a leg deadline, count that target as complete; a safety failure at that state still takes precedence. Tenth-target completion is a true terminal success, never timeout-bootstrap. Explicit training rollout boundaries may truncate collection, but must preserve unfinished physical state and clocks or be recorded as truncations, never fabricated mission failures or successful resets.

Version and bind the complete profile and clock contract. Do not relax values further after seeing validation/acceptance failures. Legacy readers retain the old profile and deadlines for historical evidence. All former failures retain their original verdicts.

### 8.2 Deliver a Successful Development Video First

The first milestone is one complete normal-scenario, nominal-load, ten-destination flight with C, followed by emergency and cost-shift development videos as soon as those succeed. Predeclare separate development video seeds; do not use formal basic/validation/held-out cases early. Record every failed attempt and choose the first successful attempt under the frozen development configuration. A successful C video must not wait for NN pilots, paired final training, held-out evaluation, or all three scenarios.

This early recording may use the re-bound qualified MuJoCo reference configuration under §4.3 without a primary-backend fallback decision. It must satisfy the video-first profile and safety/deadline rules for the whole actual mission, with independent raw/event/routing verification, source/profile hashes, complete simulation-time 1x playback, and Japanese HUD. Label it `開発実証・動画優先静定基準 / 頑健性評価前`, identify the actual controller/backend, and deliver it with the measured result and pending gates. No resets, teleportation, omitted failed legs, simulated-state edits, or accelerated playback may create apparent completion.

Introduce a separate development-capture admission path. It checks qualification bindings, this full successful mission, and media integrity; it does not pretend that the predecessor's full paired acceptance has passed. Keep the final-capture path strict and branch-aware under §8.3. An early video is a useful delivered milestone, not full specification completion. If the chosen runtime or profile changes later, rerun and record final videos under the final freeze.

### 8.3 Final Adoption Branches and Evaluation

Select the branch from §7.4 before formal evaluation; never select from held-out outcomes.

| Requirement | C selected: controller-only flight | NN selected: hybrid or justified distillation |
| --- | --- | --- |
| Fresh final flight-NN training | Not applicable; freeze C and all estimator/reference settings | Both randomized/fixed methods under the common §7.3 plan |
| Validation | 12 distinct conditions / 120 real target transitions: four seeds (11–14) x the inherited three validation loads, all executed by C | Inherited 12 conditions / 120 transitions: seeds 11/12 x three validation loads x two methods |
| Formal missions | The original 127 physical cases executed once by C | The same original 127 cases per learned method, 254 total |
| Adoption gates | C passes basic9/9, endpoint18/18, held-out >=95/100 | The preselected adoption method passes those same gates; report all 127 outcomes for the comparison method |
| Final videos | C, three complete ten-destination scenarios | Preselected adopted method, three complete ten-destination scenarios |

Controller-only acceptance explicitly supersedes the inherited requirement to train two flight NNs and run 254 missions: do not duplicate identical C executions solely to meet an irrelevant two-policy denominator. It preserves the 127-condition robustness domain and thresholds for the adopted runtime, and 12-condition/120-transition validation. For an NN, fixed-load adoption is permitted only when preselected in development; this replaces the prior requirement that the adopted method always be randomized, while retaining the full paired experiment.

Freeze the complete runtime, profiles, layout, case manifests, route policy, dependencies, and checker source before formal evaluation. Preserve original acceptance draws and the untouched held-out suite. Report failures, exact denominators, distinct initial conditions, size/mass/scenario breakdowns, and Wilson 95% intervals. Never pool basic/endpoints/random cases to compensate for an unmet gate. A 95/100 empirical pass does not prove population success probability >=95%. Successful videos alone do not establish robustness.

Final capture uses the nominal basic seed1 fixtures for normal/emergency/cost-shift after the selected branch's acceptance. Verify fresh branch-specific receipts, actual source/runtime/profile bindings, full raw data, decoded frames, PTS/simulation-time mapping, event order, and HUD values. Preserve the Japanese HUD and display the adopted settling profile; show `基準内で安定` rather than implying exact zero motion. Retain three full raw and HUD videos and request fresh continuous 1x human review. Report pending review honestly. Rendering must replay the adopted physical runtime/data; switching the simulation backend requires a new freeze and evaluation, not reuse of an old pass.

### 8.4 Meaning of Optimal Routing

Preserve all ten full-precision destinations, priorities, cost/event definitions, and the adopted routing policy. Interpret optimality through the inherited objective and exact remaining-route comparator: recompute Q*, J*, policy suffix cost, and the optimality gap from actual decision boundaries. State the modeled-cost scope and equality tolerance `max(1e-6, 1e-8 * abs(J*))`. Distinguish an optimal next choice from an optimal full remaining order. If the learned route has a gap, show it instead of claiming global optimality or substituting an oracle route without declaration.

Removing the mission-time cutoff does not change nominal planner service time, routing rewards/objective, or the safety-first physical controller. Report actual physical travel/settling times separately; do not claim optimization of unknown future events, measured flight energy, or minimum physical flight duration. The existing route-quality and explanation gates still apply to full completion.

## 9. API, Data, and Compatibility

- Assign new schemas/versions to the observation/action contract and policy artifact. Do not load old 29-dimensional, four-direct-action checkpoints merely because tensor shapes match.
- Bind artifacts to more than NN weights: include controller/estimator/trajectory-generator configuration and source, correction scaling/bounds, normalization, history initialization, backend/precision/dt, physical properties, and dependency locks/builds.
- Raw data must support reconstruction of controller state, baseline commands, NN corrections, combined/saturated commands, actual RPM, environment IDs, timestamps, targets, and termination reasons. Preserve dedicated readers for historical formats.
- Include controller-only artifacts with explicit absent NN/optimizer fields, branch-specific freezes, settling-profile hashes, deadline schemas, early/final capture receipts, and original-strict shadow metrics. Reject implicit defaults that silently restore the old 120-second timeout or old speed limits.
- Close the known predecessor audit gap: independently regenerate command RNG streams, target permutations and stage/mastery decisions from the declaration; check commanded target ID/coordinates across RESET, RETARGET, and every STEP. Native state/action replay alone is insufficient. Negative checks must reject a rebound pilot manifest labeled as non-pilot, changed target order, and mismatched event/STEP targets.
- Save new candidate declarations, numerical comparisons, fallback decisions, performance measurements, control comparisons, teacher data, training plans, and acceptance records under new output namespaces with source hashes. Do not overwrite old runs' raw data, passes, or failures.

## 10. Execution Sequence and Verification Completion Criteria

1. Read existing artifacts/status and establish a new run namespace. Implement and CPU-check the versioned profiles, clock semantics, artifact branches, and independent command-stream checks before using their results.
2. Re-bind the qualified MuJoCo N=1 reference and test C0/C1/C2 controller feasibility. Attempt and deliver the first complete development video as soon as a full mission succeeds, without waiting for NN development or Genesis qualification. Report failed attempts and pending gates.
3. Narrow Genesis GPU candidates using representative comparisons and passive32; then complete powered/refinement/motor/reset checks, batch isolation, and repeat C0–C2 on the selected runtime. Use §5.4's recorded fallback procedure when necessary. Early MuJoCo progress is not itself a fallback justification.
4. After both control and qualification gates pass, compare C and finite H pilots under §7.4. Verify correction/teacher/history/partial-reset/terminal semantics. Generate the remaining development videos whenever ready.
5. Select the simplest eligible runtime under the declared rule. If an NN is selected, freeze a common plan and train both load methods freshly; for C, mark that training not applicable. Complete the extended continuous-duration checks and branch-specific validation, then freeze formal evaluation.
6. Execute the selected branch's complete acceptance, final three-scenario capture, fresh human review, and documentation. Preserve early videos as development history rather than relabeling them final.

Full completion requires (a) qualified Genesis or a qualified justified fallback, (b) actual parallel execution/performance measurements or the explicit justified N=1 exception, (c) the finite C/H development comparison and declared selection, (d) a frozen runtime without information leakage, (e) branch-specific training/validation/acceptance, and (f) route evidence, three final videos, human review, and documentation. The first successful development video is an earlier milestone. Report numerical consistency, parallelism, flight success, NN contribution, and media status separately.

CPU tests should cover frames/units, zero-correction equivalence, correction bounds and motor authority, history/reset isolation, teacher information boundaries, schema rejection, terminal precedence, and independent curriculum reconstruction. Add profile-boundary/dwell-reset checks, strict-shadow isolation from live routing, successful missions continuing beyond 120 simulated seconds, leg-clock reset only on actual target change, retained wall timeout, controller-only versus NN acceptance denominators, and early/final capture admission. They do not replace physical qualification or flight success. Verify numerical results and required-record completeness before and after optimization.

Out of scope are real-hardware trials; changes to damping, geometry, mass, or the rope/link model; increased motor authority; further threshold/deadline relaxation beyond §8.1; reduction of a branch's acceptance domain/counts beyond §8.3; changes to the routing objective or an undeclared oracle substitution; and unauthorized contracts for additional servers. If any becomes necessary, present concrete evidence and the smallest additional change required.

## References

- [FLARE, arXiv:2508.09797v2](https://arxiv.org/html/2508.09797v2): Genesis suspended-payload RL with collective thrust/body-rate actions and passage tasks; not proof of rigid-box terminal settling.
- [ASTER, arXiv:2603.10715v1](https://arxiv.org/html/2603.10715v1): Physics-informed state seeding for taut/slack cable maneuvers; adapt exploration principles rather than transplanting its physical model.
- [Genesis forward dynamics](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_solver/forward_dynamics.html): Published integration methods and substeps. The pinned source used for execution is authoritative.
- [MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html): GPU-batching use cases and precision differences from CPU execution. Measure speed and qualification on this model.
- [MuJoCo Warp RK4 API](https://mujoco.readthedocs.io/en/stable/mjwarp/api.html#rungekutta4): The existence of an RK4 implementation does not qualify this physical system.
- [MuJoCo XLA](https://mujoco.readthedocs.io/en/stable/mjx.html): Candidate GPU backends and supported features.
