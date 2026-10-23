# Hybrid Control and Genesis GPU-Parallel Learning for Stable Suspended-Payload Transport

- Date: 2026-10-23.
- Status: New specification based on the user's stated direction. Implementation and new experiments under this specification had not started when it was written.
- PR: Not created. Include this specification in the same PR as its implementation and tests, and link it from the PR description. Add the PR URL when available.
- Goal: Prioritize practical success in transporting the payload stably to every destination over demonstrating purely neural direct motor control. Preserve physical consistency and reduce wall-clock time to success, with GPU-parallel simulation in Genesis as the first choice.

## 1. Scope and Relationship to Existing Specifications

This specification is a follow-up to [Solver Recovery and Evidence-Driven Learning](../2026-10-22-suspended-payload-solver-recovery/spec.md). The control, learning, and backend-selection rules changed below take precedence. All other physical-property, numerical-verification, routing, final-acceptance, evidence, and recording requirements are inherited. Preserve previous specifications and experimental results as historical records.

| Topic | Rule under this specification |
| --- | --- |
| Control architecture | Permit and prioritize a controller for stabilization, payload vibration suppression, and target tracking combined with a neural network (NN). Pretraining from successful trajectories is also permitted. |
| Observations | Remove the fixed 29-dimensional restriction. Measurable or estimable payload orientation, angular velocity, history, and related signals may be added. Ground-truth mass, dimensions, and similar parameters must not be supplied at runtime. |
| Initialization | Transferring specific columns from model4296 is no longer mandatory. Use development comparisons to select a common controller, pretraining with common teacher data, compatible existing weights, or fresh initialization. |
| Learning tasks | Permit a curriculum that advances based on demonstrated skills instead of only a fixed transition count. |
| Primary backend | Prioritize GPU-parallel execution in Genesis. Do not retain the existing MuJoCo CPU N=1 path as the main training path by inertia. |
| Role of MuJoCo | Initially an independent numerical reference. Fallback to MuJoCo as the primary training and evaluation backend is permitted only when evidence establishes that Genesis cannot meet the required physical consistency. Prioritize GPU parallelism in that fallback as well. |
| Optimization objective | Reduce wall-clock time to reliable settling and transport. GPU utilization, episode count, or transitions per second alone are not success metrics. |
| GPU time | No time cap or cumulative GPU-usage tracking. Retain short performance measurements, elapsed-time measurements, and mission deadlines. |

Creating this document does not stop or modify existing jobs, initiate SSH access, or start training. When implementing this specification, inspect any ongoing previous comparisons, source bindings, and artifact preservation, then move to a new run namespace without modifying old runs. Do not confuse automatic execution helpers for previous comparisons with instructions to execute this specification.

## 2. Background and Current Problems

The [handoff](../../docs/suspended-payload-solver-recovery-handoff.md) consulted when writing this document reports that the MuJoCo CPU, float64, RK4 physical configuration passed qualification, while multiple learning attempts failed to achieve shared settling. Update progress from saved evidence at execution time; do not treat the numbers in this document as the live state of running jobs.

- In early tasks, some trajectories satisfy position and translational-velocity limits while retaining transverse payload rocking at approximately 4.2 Hz. Position holding and active vibration suppression are distinct skills.
- In one 100-second interval, mean payload angular speed was approximately 96 degrees/s, and the spin cost reached its cap for approximately 99% of the interval. This concerns the discriminating power of that direct cost; it does not mean the entire PPO gradient is zero.
- In 107 selected failure episodes from later tasks, insufficient supporting thrust led rapidly to the floor condition. Failure returns exceeded a constructed distant-hover benchmark. The benchmark is not an executed policy and does not reproduce actual target changes, so this does not establish intent to crash, optimality, or causality.
- Existing N=1/8/32/128 throughput measurements were serial diagnostics of independent scenes. They do not establish the limits of actual GPU or CPU parallelism.
- Incorporate the timeout-bootstrap fix for simultaneous tenth-target success and deadline expiry into the next source version, but do not present it as the cause of zero shared completions.

Evidence: [reward diagnosis](../../docs/evidence/suspended-solver/discounted-reward-diagnosis.json), [trajectory diagnosis](../../docs/evidence/suspended-solver/long-credit-feedback-diagnosis.json), and [PPO integration verification](../../docs/evidence/suspended-solver/ppo-integration-review.json).

## 3. Selected Approach and Alternatives

The primary candidate combines a controller responsible for vehicle stabilization, target tracking, and active payload vibration suppression with a bounded NN correction. The NN learns compensation for model error, differences in load conditions, and assistance with vibration suppression. Always measure the controller alone as a baseline.

The second candidate pretrains an NN on successful trajectories from a diagnostic controller, followed by further learning in a qualified simulator. It may reuse successful trajectories from the primary candidate, but teacher success must not be reported as NN success. Perform actual closed-loop simulated flight verification after distillation as well.

Continuing purely neural direct motor control while only adding large numbers of samples is not the main path. Any such comparison must be a small development experiment with its own hypothesis and stopping decision. If the controller alone meets the requirements, report that result; do not claim the NN is necessary for performance without demonstrating an improvement.

The control architecture, experiment sequence, and decision procedures below are design choices that make the user's goal and backend priorities concrete. They are not methods already proven successful.

## 4. Hybrid Control and Observation Contract

### 4.1 Control Path

1. Generate a reference trajectory with limits on acceleration, jerk, and related quantities toward the current target selected by the existing routing system. Do not change target order, coordinates, or mission deadlines. Intermediate reference points do not count as completed destinations.
2. Build a baseline for attitude, altitude, and position stabilization and payload vibration suppression. Compare options such as geometric control, vibration suppression with state estimation, and MPC based on development feasibility and GPU-batching cost.
3. Apply the NN correction at one explicitly defined layer, such as acceleration references or thrust/moment references. Record correction bounds, units, frames, action scaling, saturation, and anti-windup before execution. Verify that zero correction at initialization reproduces the baseline.
4. All actuation must continue to pass through the four motors corresponding to the physical model. Neither the controller nor its correction may exceed the original RPM limits or alter KF/KM, rotor locations, or spin directions. Do not apply direct forces to the payload or overwrite poses or velocities.
5. Preserve the external 0.01-second control interval and current-action semantics. Do not update discrete controller state extra times during physics substeps or intermediate RK stages. Evaluate state-dependent coordinate transformations of motor forces correctly along each backend's actual integration path.

If safety limits or correction suppression are used, record their activation and the actions before and after correction, and use the same processing during training, evaluation, and recording. These mechanisms do not constitute a flight guarantee. Resets, teleportation, and unrecorded switches to reference control that conceal failures are prohibited.

### 4.2 Observations and Information Boundaries

- The runtime contract applies to the entire control system, including the baseline, state estimator, and trajectory generator, not just the NN. Do not bypass the restriction by supplying ground-truth mass or similar information outside the NN.
- In addition to vehicle state, target error, relative payload position/velocity, and link direction/angular velocity, consider payload orientation, full angular velocity, and necessary history. Fix each quantity's units, frame, normalization, clipping, timestamp, and reset initialization. Reconsider scales that continuously clip important variations.
- Do not supply true mass, dimensions, inertia, bank ID, random seed, or future events as runtime inputs. Online estimation from measurement history is permitted, but do not embed ground truth in initial values or internal state. Verify estimation error and reset behavior.
- The diagnostic teacher and training-time critic may use privileged state. Distinguish this from the runtime measurement/estimation path, and test that teacher information does not leak into the actor or baseline.
- Explicitly identify simulator getters used as ideal sensors. Do not claim demonstrated transfer to real hardware. Separately document sensor availability and development evaluations of delay and noise.

### 4.3 Control Feasibility Checks

During the powered-diagnostic phase permitted by the qualification procedure, test nominal-load hover, small movements, out-and-back flight, and retargeting. Separate the low-frequency pendulum component from the fast rotational component, and measure decay rates, phase, saturation, position error, and continuous shared-settling duration. Then extend to size/mass endpoints.

Freeze development conditions and pass/fail criteria before testing. First reproduce all shared-settling conditions at one destination from multiple independent initial states. One success does not establish skill acquisition. A diagnostic-controller failure does not establish physical impossibility; determine whether the obstacle concerns control authority, information, the model, or deadlines.

## 5. Genesis-First Numerical Consistency and Fallback

### 5.1 Unchanged Physics

Preserve the original vehicle and four motors, masses/inertias/frames, the 0.30 m rigid link with nonzero mass, passive ball joints at both ends, attachment locations, payload sizes of 0.030/0.045/0.060 m, and the independent mass ratio of 0.10–0.30. The fixed-load comparison uses 0.045 m and 0.20. Do not change the formal evaluation distribution.

This specification does not authorize adding damping, replacing the link with a rope, replacing bodies with point masses, or adding motor authority. Treat damping or attachment changes based on physical measurements as separate physical-model changes. Do not disguise numerical dissipation as physical damping to obtain settling.

### 5.2 Selecting Genesis Candidates

1. Inspect the actual installed source and dependency versions. Declare a small set of candidates, each with a hypothesis, covering supported GPU integrators, float32/64 precision, substeps, and constraint settings.
2. Use the current MuJoCo float64 RK4 configuration and an independent model as references. Measure timestep convergence with identical physical properties, initial states, and inputs. Refine MuJoCo's timestep as well to assess reference error. Short CPU reference calculations are permitted independently of the conditions for switching primary training to MuJoCo.
3. If the cause of inconsistency under standard settings can be identified, consider necessary Genesis version updates or targeted integration/constraint-processing fixes. A new RK4 implementation is not mandatory; it is a candidate only when measurements establish a need.
4. Do not require the same integration order or byte-identical trajectories. Judge physical properties, forces/moments, reactions, energy balance, constraint error, oscillation period/phase, and short-time state convergence. Long-duration divergence of large-angle trajectories alone does not establish inconsistency.

### 5.3 Qualification and Thresholds

Inherit all 62 traces from the existing specification: passive32, powered18, and powered refinement12, plus motor15 and at least reset30. Do not start powered diagnostics before passive32 passes. Do not run an optimizer until the selected backend, precision, integrator, and batch configuration have full qualification. Do not reuse previous N=1 qualification for Genesis or a new batch.

Preserve positive passive-energy growth of at most 5%; absolute unexplained energy change, after accounting for declared dissipative and external work, of at most 5%; constraint residual of at most 2e-5 m; physical-property relative error of at most 1e-5; collision-extent error of at most 1e-5 m; and small-angle period error of at most 10%. Preserve the powered same-input base/half-dt limits of 2 cm/2 degrees. Follow the predecessor specification for each case's initial conditions, duration, work reconstruction, and other guards.

Verify work definitions and intermediate integration stages for the new backend. Do not transplant fields from another method by merely matching their names. Do not loosen tolerances, angle ranges, test durations, or load conditions for performance.

### 5.4 Conditions for Making MuJoCo the Primary Backend

An inability to achieve consistency does not require an infinite candidate search. Evaluate a small set of predeclared settings, precision/timestep convergence, independent references, and feasible fixes or version updates that address the observed cause. Identify promising untested candidates explicitly before making the decision.

Fallback requires a decision record containing candidates, executed source, raw comparisons, failed metrics, cause analysis, repair results, and why remaining remedies are not feasible. Not having started, inconvenient GPU implementation, greater familiarity with MuJoCo, or Genesis being slow are not sufficient reasons for fallback. If a consistent Genesis configuration is found, minimize time among consistent Genesis configurations.

Only after the fallback conditions are met, compare GPU-batched MuJoCo options such as MuJoCo Warp/MJX first. GPU RK4 does not inherit CPU float64 qualification; reverify precision, inputs, resets, and other required properties. If GPU candidates are also unqualified or unavailable in the actual environment, CPU parallelism may be used with a recorded reason. Use CPU N=1 mainly for reference calculations, local diagnostics, and fault isolation, and explicitly justify any need to use it for primary training.

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

### 7.3 Freezing Final Training

After development demonstrates sufficient settling/movement success and experience across all stages, freeze the common S, N/T, seed7, controller, observations, initialization, reward, curriculum rules, optimizer, and checkpoint selection for both methods. With adaptive progression, use common rules and also report differences in actual stage exposure.

Use equal numbers of valid training transitions for both methods, and separately disclose the cost and experience associated with teacher collection and pretraining. Distinguish generated, invalid, and consumed transitions, vector steps, and update counts. Do not extend training or change conditions for only one method. When revising the plan, restart both under a new common plan.

## 8. Final Acceptance and Deliverables

Preserve the predecessor's validation across 12 conditions/120 actual target transitions, the same 127 conditions per method for 254 missions in total, randomized acceptance of basic9/9, endpoint18/18, and at least held-out95/100, and full reporting of all 127 fixed-load outcomes. Controller-only development success or a speedup alone does not constitute completion.

Each destination requires all of the following simultaneously for one continuous second: vehicle error at most 0.20 m and speed at most 0.30 m/s; payload target error at most 0.20 m and speed at most 0.10 m/s; suspension angle at most 5 degrees; swing angular rate at most 10 degrees/s; and full payload angular speed at most 10 degrees/s. Preserve lowest-point clearance above 0.10 m, absolute roll/pitch at most 60 degrees, a 30-second leg deadline, a 120-second mission deadline, and a 600-second wall-clock deadline.

Inherit routing, coordinates, event conditions, acceptance denominators, three complete ten-destination videos, simulation-time 1x playback, the Japanese HUD, independent raw-data verification, and fresh human review. Freeze the complete final runtime, including the baseline, NN, estimator, and action limits, and keep it consistent across training, evaluation, and recording. Do not reuse an old freeze when switching backends for recording.

## 9. API, Data, and Compatibility

- Assign new schemas/versions to the observation/action contract and policy artifact. Do not load old 29-dimensional, four-direct-action checkpoints merely because tensor shapes match.
- Bind artifacts to more than NN weights: include controller/estimator/trajectory-generator configuration and source, correction scaling/bounds, normalization, history initialization, backend/precision/dt, physical properties, and dependency locks/builds.
- Raw data must support reconstruction of controller state, baseline commands, NN corrections, combined/saturated commands, actual RPM, environment IDs, timestamps, targets, and termination reasons. Preserve dedicated readers for historical formats.
- Save new candidate declarations, numerical comparisons, fallback decisions, performance measurements, control comparisons, teacher data, training plans, and acceptance records under new output namespaces with source hashes. Do not overwrite old runs' raw data, passes, or failures.

## 10. Execution Sequence and Verification Completion Criteria

1. Read existing artifacts, running-job status, and the latest diagnoses. Identify new runs and intended changes. Do not operate existing jobs merely because this document was added.
2. Narrow Genesis GPU candidates using representative numerical comparisons and pass passive32. Address failures by cause, and make a documented fallback decision only if necessary.
3. Verify baseline hover, small movements, and vibration suppression during powered diagnostics, then complete all62/motor15/reset30 and qualification of the selected batch.
4. Verify the new observations/actions, baseline, NN correction, teacher, partial resets, and terminal handling, then run short GPU-parallel development training.
5. Compare success rates, safety, and time under the same development conditions, and narrow observation/reward/control hypotheses. Do not automatically advance to large-scale learning without improved success.
6. Freeze a common plan, train both methods, and complete validation, acceptance, recording, human review, and documentation.

Completion requires (a) Genesis GPU qualification or qualification of a justified fallback, (b) actual parallel execution and performance measurements, (c) comparison against the controller alone, (d) a common runtime without information leakage, (e) fair final training, and (f) all acceptance, video, and human-review requirements. Report GPU acceleration, numerical consistency, and flight success separately.

CPU tests should cover frames/units, zero-correction equivalence, correction bounds and motor authority, history/reset isolation, teacher information boundaries, schema rejection, terminal precedence, and related behavior. They do not replace physical qualification or flight success. Verify numerical results and the completeness of required records before and after optimization.

Out of scope are real-hardware trials; changes to damping, geometry, mass, or the rope/link model; increased motor authority; relaxed acceptance thresholds or mission deadlines; reduced acceptance counts; and unauthorized contracts for additional servers. If any becomes necessary, present concrete evidence and the smallest additional change required.

## References

- [Genesis forward dynamics](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_solver/forward_dynamics.html): Published integration methods and substeps. The pinned source used for execution is authoritative.
- [MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html): GPU-batching use cases and precision differences from CPU execution. Measure speed and qualification on this model.
- [MuJoCo Warp RK4 API](https://mujoco.readthedocs.io/en/stable/mjwarp/api.html#rungekutta4): The existence of an RK4 implementation does not qualify this physical system.
- [MuJoCo XLA](https://mujoco.readthedocs.io/en/stable/mjx.html): Candidate GPU backends and supported features.
