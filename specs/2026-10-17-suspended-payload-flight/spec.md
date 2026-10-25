# Robust Ten-Goal Flight with a Suspended Payload

> 2026-10-21実行状態：同日のユーザーの明示指示で実装を開始した。以下の未承認・未開始の文言は作成時点の歴史記録である。今回の関節表現は長時間モーター駆動で数値破綻し、物理実現性ゲート不合格。訓練は中断、受入・比較・動画は未達。[実行記録](../../docs/suspended-payload-flight-handoff.md)を参照。分布・閾値・timeoutを緩める改訂は行っていない。

- Date: 2026-10-17
- Status: Written specification for operator review. Scope, payload distribution, observation access, settling gates, trial counts and media scope were confirmed on 2026-10-17. The explicit design defaults below are included for final review; implementation is not yet approved.
- Baseline review: 2026-10-21. Updated after the integer-priority and dispersed-3D implementations; this document update does not start payload implementation or training.
- PR: Not created. Link this specification from the eventual implementation PR.
- Implementation root: the repository root. Historical copies are reference material only.
- Original inspected baseline: realtime-video commit `23c7f772f194dab00ade2bd77e64cbd16452874c` (historical).
- Current inspected baseline: merge commit `f3e33747ebd62d5e5e0466cb671a1b39069d182f`, including dispersed-layout implementation `1e4e53e5fd6c59dd913a2e2d57094deb4b8a97bd`. Check HEAD and local changes again at implementation time; do not reset or overwrite intervening work.
- Dependencies: [scalable routing](../2026-10-14-scalable-adaptive-routing/spec.md), [visible multigoal flight](../2026-10-14-visible-multigoal-flight/spec.md), [authentic video timing](../2026-10-14-realtime-demo-video/spec.md), [integer priorities](../2026-10-17-integer-priorities/spec.md), and [dispersed 3D destinations](../2026-10-17-dispersed-destination-layout/spec.md). The merged specifications remain historical records.

## Goal and agreed intent

Demonstrate a small drone carrying one suspended package around the existing ten destinations A-J. At every destination, both the drone and its load must settle before the mission advances. The package stays attached throughout the mission: arrival is a delivery-like demonstration, not an actual package release or completed delivery.

Train one motor-control policy across episode-varying package masses and sizes. The same frozen policy must handle the declared payload distribution without selecting a separate policy for each package. Evaluate robustness on held-out physical episodes, rather than treating domain-randomization configuration or training reward as evidence of robustness.

Operator decisions confirmed on 2026-10-17:

1. Use all ten existing destinations and wait for payload stabilization at each one. Circular orbit flight is not the requested meaning of circulation.
2. Retain the existing small drone; choose feasible payload dimensions and masses relative to that vehicle.
3. Observe payload motion using ideal simulator measurements. Do not give the actor the exact randomized mass or dimensions. Camera perception and physical sensor estimation are outside this change.
4. Package dropping is outside scope. Start with a simple suspended-body approximation before considering slack-rope dynamics.
5. Independently sample one of three cube sizes per episode, plus a continuous package mass. Use the proposed 3/4.5/6 cm sizes and 10-30% vehicle-mass range as the initial domain, subject to the documented pre-training feasibility gate.
6. Require simultaneous drone arrival and payload stabilization for 1.0 second: package position error <=0.20 m, suspension angle <=5 degrees, package speed <=0.10 m/s, swing rate <=10 degrees/s and package angular speed <=10 degrees/s.
7. Accept the randomized policy only if it completes all nine basic missions, all eighteen mass/size endpoint missions, and at least 95 of 100 held-out random missions. Evaluate the nominal-policy comparison on the same 127 cases.
8. Deliver complete 1x raw footage and three Japanese-HUD scenario videos. Re-editing `demo_full.mp4` or `submission.mp4` belongs to a separate PR.

## Current behavior

The flight environment uses the Genesis Crazyflie `cf2x` model, a 0.01-second control step, four motor actions, and a 17-component policy observation. There is no payload body or suspension constraint. The inference loader explicitly checks the existing observation/action contract.

The adopted unloaded controller is now the dispersed-domain `model_4296.pt`; `model_1300.pt` is its historical initialization, not the current adoption. The dispersed implementation completed nine unloaded missions and 90 inspections on the new 3D layout. Those results do not validate a payload. Adding a payload changes the physical system and requires new training or justified initialization, physical evaluation, telemetry, and recordings.

Use the following adopted artifacts as the reproducible unloaded starting point. Verify size and SHA-256 against the [dispersed handoff](../../docs/dispersed-destination-layout-handoff.md) and artifact manifests before use; inspect the actual configuration and embedded/sidecar contracts as well as file dimensions.

| Artifact | Repository-relative path | Bytes | SHA-256 |
| --- | --- | --- | --- |
| Flight checkpoint | `outputs/dispersed_20261020_v1/remote-final/outputs/dispersed/finetune_v2/model_4296.pt` | 475573 | `26658a31560f71c36fd9e9605b15d8e4f7b9cae8174c1465fd12dcd0d10b3431` |
| Flight configuration | `outputs/dispersed_20261020_v1/remote-final/outputs/dispersed/finetune_v2/cfgs.pkl` | 1628 | `35c6b0be8db90e4e799d453d447a484275317f85098a22cbdf04c19c84056e55` |
| Routing checkpoint | `outputs/integer_20261020_v1/training/seed_44.zip` | 297837 | `a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef` |

The unloaded flight observation contract is `hover17-relative-world-v1`: relative world position is scaled by 1/3 and clipped to [-1,1]. Long dispersed transitions produced actual observation saturation; identical observation shape is not proof of compatible meaning or adequate control. The former altitude-1 m, narrow-horizontal configuration cannot be restored as the loaded operating envelope. Neither the failed dispersed `model_2800.pt` nor a fractional-priority routing model is an accepted fallback.

The existing physical arrival gate checks drone position, speed and continuous dwell. It cannot establish that a suspended package has settled. The route planner runs separately and selects the next destination at a physical completion boundary. Routing cost remains a nominal analytical objective, separate from measured flight duration and energy.

## Intended behavior

### First suspension model and alternatives

Use a fixed-length, lightweight rigid link with passive joints to represent a taut suspension. The load must swing in both horizontal directions. The package is a separate rigid body with mass, inertia, pose and velocity. Account for the link's own nonzero mass and inertia if required by the solver. Neither the link nor its joints have actuators, pose controllers or artificial swing cancellation.

Use an underside attachment centered on the drone and a top attachment centered on the package. Specify their local frames and the connection length explicitly. The load's reaction forces and moments must affect the drone through the physical solver. Moving a decorative package under the drone, increasing only drone mass, or prescribing a pendulum trajectory does not satisfy this specification.

Prefer an articulated model compatible with the pinned drone motor implementation. A short feasibility test must confirm that adding links does not change propeller indexing, rotor thrust/torque application, or drone root-pose semantics. Choose the exact joint representation only after inspecting and exercising the pinned implementation; a documented ball-joint API in a different release is not proof of compatibility.

The fixed link is an approximation: it can carry compression and cannot represent slack. Label it as such in documentation and Japanese video text. Restrict this first experiment to suspended, noncontact flight. Measure constraint residuals and inspect reaction-force behavior, including whether the tested trajectories enter compression. Report that audit's observability and any unavailable solver force measurements honestly. Do not describe such evidence as validation of a tension-only rope. Any passive joint damping must be declared, constant across methods, and physically justified; large hidden damping cannot substitute for learned stabilization.

Alternatives considered:

- A tension-only force model can represent slack, but also requires attachment-point forces/torques, damping, numerical integration, and slack-to-taut transition validation. Defer it to a separate specification if the fixed-link experiment establishes the need.
- A chain of many links or deformable cable adds bending/contact complexity that is unnecessary for this first transport experiment. It is not an implicit fallback.

If the pinned engine cannot support a physically coupled fixed-link drone, report the feasibility failure and propose a revised approach before changing the physics model or dependency versions.

### Payload distribution

The operator selected the following initial mass/size distribution. The design default for fixed suspension length is 0.30 m; verify its physical feasibility together with the distribution before training:

| Parameter | Contract / design default |
| --- | --- |
| Suspension length | 0.30 m between attachment points; fixed during training and evaluation |
| Package shape | Cube; centered mass distribution |
| Cube side length | Independently sample 0.030, 0.045 or 0.060 m with equal probability at each episode reset; confirmed |
| Package mass | Independently sample uniformly from 0.10 to 0.30 times the original unloaded drone mass; confirmed |
| Reference drone mass | Actual sum of the original vehicle link masses, excluding new payload and suspension |
| Within an episode | Mass, size, inertia and suspension parameters remain constant across all ten visits |

Record physical values in SI units as well as the sampled ratios. Mass and size must vary independently; do not keep density constant and call both independently randomized. Compute the inertia tensor from the selected shape and mass. For a uniform cube of side `s`, each principal inertia is `m*s*s/6`. Update actual collision geometry and rendering consistently; visual-only scaling is insufficient.

First verify the proposed maximum load against available motor thrust, numerical stability, initial ground clearance and attachment geometry. Record the measured or asset-derived thrust margin using the retained motor limits. An infeasible range is a design finding, not a reason to silently shrink the acceptance domain after training.

Every episode reset in randomized-policy training must resample that episode's payload. A value sampled once per parallel environment and then reused forever does not meet the request. The fixed-payload comparison deliberately retains its nominal mass and size. Evaluation resets apply the frozen case manifest without making replacement draws. For engines that cannot change geometry in place, a validated prebuilt geometry bank or an episode-level scene rebuild is acceptable. Randomized training must still sample the active size per episode, and inactive geometry must exert no forces, contacts or inertia. Selective resets must leave ongoing environments unchanged.

Do not assume mass setters also update inertia, or that a changed visual scale updates collision geometry. Verify the solver's effective mass, inertia and collision dimensions, not just a configuration file. Record RNG streams and draw indices so episodes can be reproduced without sharing the test stream with training.

### Physical checks before reinforcement learning

Run small, recorded probes before allocating a full training job:

1. Verify motor/rotor identity, total mass and inertia, attachment coordinates, and passive joint degrees of freedom.
2. Hold the suspension anchor fixed for a separate pendulum diagnostic. Check gravity direction, small-angle oscillation against the declared model, and absence of unexplained energy growth. This diagnostic is not a flight result.
3. Release the coupled drone and load under the retained motor interface. Confirm the load changes drone dynamics and that no external support or pose correction is active.
4. Exercise every size with minimum and maximum masses. Check collision geometry, ground clearance, constraint errors and finite state.
5. Repeat a deterministic short probe with a smaller physics timestep or additional solver substeps. Record changes in trajectory, constraint residuals and energy; justify numerical tolerances before freezing evaluation settings.
6. Reset selected environments repeatedly and verify that sampled mass, dimensions and inertia agree with runtime state, while other environments remain unchanged.

Also check the starting point and every inherited 3D hover target with all package sizes, attachment offsets and the declared initial perturbations. Check the lowest point of the oriented package geometry throughout transitions, not only nominal hanging clearance. An unloaded trajectory's ground clearance cannot establish loaded clearance.

These checks determine feasibility. They do not establish learned control success.

### Learned controller and training

Preserve four learned motor commands as the actuation interface. Add payload relative position and velocity, suspension direction and swing-rate information to a versioned observation contract. Define coordinate frames, units, quaternion conventions, normalization, clipping, history and reset behavior. Compute relative velocity consistently with the chosen rotating or inertial frame.

Use simulator state as an explicitly ideal observation source. The actor must not receive payload mass, dimensions, density, inertia, geometry-bank identity, randomization seed, or future changes. Payload state can naturally reveal physical behavior; the requirement is to avoid directly supplying hidden episode parameters. Both comparison policies use the same observation contract. Privileged critic inputs are not required for the first implementation.

The unloaded checkpoint is not directly compatible with additional observation fields. If used for initialization, document an explicit parameter-transfer mapping and initialization of new inputs; never load by silently truncating the observation. Preserve the original checkpoint and legacy loader path. Save the new observation schema and physical-model signature with every payload checkpoint, and reject mismatches before starting inference.

Use a staged curriculum: stationary loaded hover, short transfers, reversals and long transfers, ascent, descent and simultaneous horizontal/vertical transfers, then multiple target switches across the inherited dispersed 3D envelope with the final payload distribution. Vary initial swing and drone pose within the declared bounds; subsequent payload motion must arise from the coupled dynamics. Final evaluation includes these perturbed initial states. Do not train only on the final demonstrated ten-point order.

Before acceptance, evaluate declared long-horizontal, ascent, descent, diagonal, reversal and repeated-settling transitions with actual drone/load telemetry. Record unclipped and clipped observations, saturation duration, floor clearance and constraint behavior. Changing validators or observation scaling alone is not evidence of compatibility. Any normalization/history change requires a new observation contract and an explicit checkpoint-transfer/evaluation procedure. Preserve failures: the dispersed implementation's first broad-domain fine-tune failed its floor gate, while its later staged curriculum succeeded unloaded; neither outcome proves loaded feasibility.

For the first training design, vary command difficulty through the curriculum while retaining the full mass/size distribution in every randomized-policy stage. The fixed-payload baseline follows the same command curriculum with its fixed load. Any later proposal to narrow or ramp the physical distribution must be recorded as a training-design revision.

Reward target progress and stable settling, penalize excessive payload motion and unsafe behavior, and record each reward component. Prevent the agent from maximizing a low-swing reward by refusing to travel. Compute terminal failure and reward before a training reset erases the failing state. No automatic reset is allowed inside an evaluation mission.

Train a nominal fixed-payload baseline with the same observations, architecture and optimizer budget, plus the randomized-payload policy. Freeze selection using validation data before held-out evaluation. Record training seeds, actual steps, curriculum, checkpoint-selection rule and compute for both. If only one training seed per method is affordable, limit conclusions to those trained policies rather than claiming seed-independent algorithmic superiority.

Use a 0.20 reference-mass ratio and 0.045 m cube for the fixed-payload baseline. Match the training initialization seed, network initialization/transfer procedure, command distribution, initial-state randomization, reward, and environment-step budget across methods. The intended difference is payload mass/size randomization. Record unavoidable throughput differences. An unloaded checkpoint is a historical reference, not a substitute for this matched loaded baseline.

### Destination completion and routing integration

Retain the full-precision A-J **3D** coordinates from `configs/demo_scenarios_dispersed/{normal,emergency,cost_shift}.yaml`, with nominal start S=(0,0,1). The coordinate hash is `21dd7cf41bf12c592ccdb4bf9c8817352feeafb9597103a3ba626e51a6d910e0`; generation version, seed 51000 and accepted attempt 3353 are recorded in `docs/evidence/dispersed/layout.json`. Do not regenerate coordinates per policy, load, scenario, execution seed or frame. Preserve the inherited geometry checker, including sorted consecutive triples for the 0.10 m horizontal-band rule and the specified height-band boundaries. Small declared initial drone perturbations change initial state, not S or destination definitions.

The target envelope is x in [-2.7,2.7], y in [-1.8,1.8], and absolute z in [1,3] m. Actual destination altitudes range from 1.0844227584485355 to 2.911967352347588 m. These are drone hover targets, not package centers or hard bounds on transient drone altitude. Define each nominal suspended-package position from its hover target, attachment offsets and package dimensions. Do not flatten targets to z=1, raise all targets, or shrink the geometry to obtain loaded success without an explicit specification revision. Nominal horizontal area enlargement, actual convex-hull area and physical flight distance remain distinct quantities.

Keep the integer-priority contract in routing, exact comparisons, telemetry and HUD: values are integers 1-100, independently sampled uniformly with duplicates allowed, with normalization 100, alpha=1 and beta=0.05. Reuse priority seed 40029 and A81, B42, C43, D64, E23, F84, G6, H98, I22, J56 for every case. Layout, payload and initial-state randomization use independent streams. Preserve `routing-v3-integer100-n10-324`, coordinate divisor 2, distance divisor 4 and the adopted seed-44 route policy; do not silently change nominal planner service time (0.50 s) to the physical settling dwell. The three scenarios share the same initial coordinates and priorities.

The operator approved these simultaneous settling predicates:

| Quantity | Limit |
| --- | --- |
| Drone center position error from its hover target | <=0.20 m, Euclidean world-space distance |
| Drone center speed | <=0.30 m/s, world-space magnitude |
| Package center position error from its nominal hanging target | <=0.20 m, Euclidean world-space distance |
| Package center speed | <=0.10 m/s, world-space magnitude |
| Suspension angle from downward vertical | <=5 degrees |
| Suspension swing-rate magnitude | <=10 degrees/s |
| Package angular-velocity magnitude | <=10 degrees/s, including rotation around the suspension |
| Shared uninterrupted interval satisfying every predicate | >=1.0 second |

Let `a` be the drone attachment point, `b` the package attachment point, and `r=b-a`, all in world coordinates. Compute the angle from `r` and downward gravity. Compute swing rate as `norm(cross(r, v_b-v_a))/norm(r)^2`, converted to degrees/s. Include the angular-velocity cross-offset term when deriving each attachment velocity from a rigid body's center velocity. The package spin predicate uses its own actual angular velocity. A turning point with momentarily low package speed is not proof of stabilization.

For a level drone at hover target `g`, a centered underside attachment offset `d`, and an upright cube with top-center attachment, the nominal package center is `g + d - (0,0,L+s/2)`, where `L` is attachment-to-attachment length and `s` is cube side length. Define `d` in the level world orientation, including its downward sign. Compute this verification target from the episode manifest; do not feed the hidden size to the policy. Never continually recenter the goal on the actual drifting drone.

Any failed predicate resets the shared settling dwell. At a 0.01-second control step, count at least 100 consecutive valid step intervals, without counting a single instantaneous sample as a duration. A valid arrival confirmation can retain the existing 0.20-second threshold, but a visited flag requires the full 1.0-second shared interval. The old 0.50-second unloaded completion threshold cannot bypass it. A target switch changes the command only; it must preserve drone/load state, joint state, velocities and policy history. The package remains attached with the same mass and geometry for the entire mission.

Only the new shared completion boundary triggers a visited flag, routing decision or scenario event. Preserve the inherited trigger after the first completed destination: emergency raises unvisited C from 43 to 100; cost shift changes only the directed F-to-A multiplier from 1 to 5, not A-to-F. Keep all multipliers within 1-5. Freeze these event definitions before acceptance and recheck their physical-boundary ordering. Do not replay the unloaded event timestamp (4.72 s) or its subsequent completion timestamps.

Recompute counterfactual choices, Q*, J*, learned-policy suffixes and raw/weighted cost components from the loaded run's actual decision time, vehicle position and remaining tasks. When presenting L/H optimality, retain the inherited priority-gap >=10 and other proof conditions, distinguish nominal planner distances from current-pose distances, and distinguish an optimal next action from a potentially suboptimal full policy suffix. Do not reuse unloaded improvements or optimality examples as loaded results; report any unmet explanation condition. Keep nominal planner service time and measured physical settling time separately labeled. Do not imply that the existing routing objective optimizes measured payload energy or delivery duration.

Use Japanese viewer labels such as `吊り荷搬送`, `荷物安定待ち` and `到着・安定化完了`. Do not show `投下済み` or imply that the package was released. Flight safety also covers package ground contact, drone/package collision, detached or invalid constraints, nonfinite state, timeouts and resets. A package touching the floor to suppress swing is a failure.

Retain the drone's 60-degree maximum roll/pitch and 0.10 m floor gate. Also require the lowest point of the oriented package collision geometry to stay above 0.10 m, not merely its center. Initially retain the existing 30-second leg, 120-second mission and 600-second wall-time limits. If physical preflight shows these are unsuitable, seek an explicit specification revision before the frozen acceptance suite rather than relaxing limits after seeing failures.

### Robustness evaluation and evidence

Freeze inherited coordinates/priorities, event definitions, payload distribution, initial-state perturbations, timeouts, settle thresholds, both payload-policy hashes, the shared route-policy hash, camera/scene settings and numerical settings before acceptance runs. Bind the freeze record to the actual execution source, model/configuration contents, and the manifest, summary and full telemetry hashes of the pre-acceptance transition evaluation. Bind subsequent acceptance results to that frozen record and their own measured telemetry hashes. Independently reconstruct commanded transitions and shared settling from the predeclared cases and measured samples; a self-reported target sequence, stationary hover repeated under different labels, or a stale PASS receipt cannot establish completion. Recompute these bindings when consuming evidence. Separate training, validation and held-out episode streams. Use exactly the same physical evaluation cases for the nominal and randomized policies, including failures; do not compare different load samples or hide retries.

The evaluation must include nominal loads, combinations of extreme masses and sizes, unseen in-range masses, and perturbed initial swing. Report complete-mission success, per-stop success, stabilization time, payload/drone position errors, swing angle/rate, peak tilt, contact/constraint failures, motor saturation and actual elapsed simulation time. Break results down by size and mass range instead of reporting only a pooled average.

The prior nine unloaded missions are historical evidence. New loaded missions require new measurements. A repeated deterministic trajectory under several seed labels is not evidence of robustness to payload variation. State the actual number of distinct physical conditions.

The operator approved this acceptance suite:

| Suite | Cases per policy | Randomized-policy pass gate |
| --- | --- | --- |
| Basic demonstration | 3 scenarios x execution seeds 1/2/3, nominal 0.20 mass ratio and 0.045 m cube | 9/9 complete missions |
| Parameter endpoints | 3 scenarios x 3 sizes x mass ratios 0.10/0.30 | 18/18 complete missions |
| Held-out random | 100 predeclared cases with independent in-range masses and initial perturbations; balanced across the 3 sizes and 3 scenarios | >=95/100 complete missions |

All gates are separate. Passing the pooled total cannot compensate for a failed endpoint or basic-case gate. A mission succeeds only after all ten unique destinations satisfy the shared settling gate without a safety failure, reset or timeout. The matched nominal baseline must run and be reported, but is not required to meet the adoption gates. Do not require manufactured superiority: if both policies pass or the baseline is better, report that result without claiming an unsupported benefit from randomization.

For the 100 random cases, allocate 11 cases to each size/scenario combination, plus one additional case to the smallest-size normal combination. Draw mass independently within the declared interval. Record a held-out manifest before evaluation, including every seed, parameter and initial condition. Report pooled and subgroup outcomes, denominators and a 95% binomial confidence interval for the random-suite success fraction. A measured 95/100 is an empirical gate, not proof that population reliability is at least 95%.

Initial-state design defaults, included in the complete written review, are a suspension angle uniform from 0 to 10 degrees with uniform horizontal direction, drone x/y offset independently uniform within +/-0.02 m of S, and roll/pitch independently uniform within +/-2 degrees. Start with zero body/joint velocities, zero yaw, and nominal drone altitude; initialize all bodies in a consistent constrained configuration. Use the same perturbed state for paired policies and paired basic scenarios. Endpoint cases include 10-degree initial swing, with deterministic directions recorded in the frozen manifest. Do not add arbitrary independent poses that violate the suspension constraint at reset.

Training and validation can use the same perturbation bounds but independent RNG streams. Freeze the exact draws and endpoint directions before test execution. Do not reroll failed loads, redefine seeds to repeat a favorable deterministic trajectory, choose checkpoints on this suite, or adjust thresholds after reading test results. Retain all attempts; any model/configuration revision requires a new complete suite and preserves the failed suite.

No loaded training or acceptance result is asserted by this document. This specification does not claim generalization to other rope lengths, untrained box dimensions, non-cubic shapes or parameter ranges beyond those tested.

### Required local deliverables and visual verification

Use a new unique output root and retain the following:

- A complete evaluation manifest and telemetry/results for all 127 declared cases for each of the two policies, including failures. Save training/validation logs, model/configuration hashes, curriculum and checkpoint-selection evidence.
- Complete recorded runs for normal, emergency and cost shift with the adopted randomized policy, the same nominal payload and the same execution seed. Prefer the predeclared basic-suite seed 1. Freeze this choice before recording; failures cannot be concealed by selecting a more attractive successful run.
- Each selected run's complete `raw.mp4`, `frames.csv`, physical telemetry, scenario, events, decisions, payload/scene manifests and mission summary, from initial state through final stabilization.
- `full_normal.mp4`, `full_emergency.mp4` and `full_cost_shift.mp4`, with Japanese HUD, all ten stabilized stops, natural durations and no omitted flight intervals.
- Per-video timeline, selection, verification and manifest records; a global size/SHA-256 manifest; compact evidence and Japanese handoff/reproduction documentation. Preserve raw originals separately if timestamp normalization is necessary.

The HUD shows A-J, integer priorities on the fixed 1-100 color/numeric scale, target/visited state, actual drone and package altitude and position, package mass and size, suspension length, current swing angle and settling progress. Mark the physical approximation with a readable Japanese description such as `固定長リンクによる吊り下げ近似（弛みなし）`. Mass and size in the viewer HUD are verification/display information, not actor inputs.

The vehicle, package and suspension must remain identifiable at starts, transfers, scenario events and stabilization. Adjust camera framing before the frozen recording suite if needed and verify all ten IDs, priorities, states and the current target remain readable across the 3D envelope. Check actual projected overlaps, occlusion and height differences, including the package below the lowest target; the unloaded camera approval is not a loaded-scene approval. When a detail inset is needed, use the same genuine source frame and identify it as an enlargement. Do not animate the suspension or load separately from physics to make the result look smoother.

Apply the existing realtime spec's physical-step/frame/encoded-PTS checks to these new recordings. Independently verify unit-slope source-to-output time, full decoding, terminal coverage, all ten stabilization events, and the actual HUD text against same-frame payload telemetry. Do not rely only on `playback_rate=1` metadata. Inspect extracted real frames and continuous 1x playback around a transfer, an event, and the next stabilization. Record whether review was performed by the agent or the operator. Request fresh operator review of the final three loaded videos, with file links and concrete intervals from before each event through the next shared stabilization at continuous 1x speed. Bind approval to the final file hashes; previous integer/dispersed-video approvals do not cover loaded videos.

Validate actual encoded frames as well as unencoded overlays. The dispersed delivery required an encoding-quality correction because compressed height text failed the unchanged pixel checker. Use the current badge-scene encoding as a starting point, test representative loaded HUD frames before full rendering, and run the independent checker on every final frame. Do not loosen text/pixel thresholds to accept compression damage. Preserve failed encodes and use a new output directory; an encoding-only correction can reuse unchanged verified physics, but requires renewed final-media verification and review bound to the delivered hashes.

The 60-180-second submission constraint and combined `demo_full.mp4` are explicitly outside this PR. Keep their previous versions as unloaded historical deliveries. If existing README/report text would imply that old recordings validate loaded flight, update it and regenerate/inspect any affected PDF. Describe the new capability and tested limits without relabeling prior evidence.

## API, data and compatibility

- Introduce a distinct payload-enabled execution profile; retain unloaded execution and historical artifacts.
- Save a versioned payload manifest with model type, link/joint definitions, attachments, reference drone mass, package mass/dimensions/inertia, effective runtime properties, randomization stream, initial state and model hashes.
- Extend physical samples with drone and package world poses/velocities, suspension endpoints, swing metrics, constraint residuals, contact flags, motor commands and shared settling-dwell state. Define source frames and units for every field.
- Record mass and dimensions for verification and viewer HUD, while keeping them outside policy observations.
- Preserve per-run scenario, events, decisions, telemetry, mission summary, raw frames and timing records. Add physical payload completion evidence to each visited destination.
- Version new observations and telemetry; legacy data must not be interpreted as payload evidence by adding fabricated default measurements.
- Keep the CPU route-planner/flight-process boundary and input/model hash checks. Regenerate explanations from the new physical decision records when showing route comparisons.
- Integer priorities and dispersed 3D geometry are now merged prerequisites, explicitly inherited above. Validate the combined payload/scenario/schema/model contract and all loaded acceptance cases. If later changes land before implementation, inspect and document their compatibility before freezing the suite. Do not combine separately successful results into an untested integration claim. If a shared observation, routing model or unloaded execution path changes, rerun the affected predecessor quality gates as well as the loaded suite.

## Acceptance criteria

1. A real simulated load and passive fixed-length suspension affect drone dynamics; all solver-property and reset-isolation probes pass.
2. Every randomized-policy training episode samples the declared mass and size distribution, and each episode retains its sampled load across target changes. Inertia and collision geometry match the sampled package. The fixed-payload baseline and manifest-driven evaluation follow their explicitly different parameter-assignment contracts.
3. One frozen four-motor policy operates across the declared load distribution without direct actor access to mass or size. Observation/checkpoint mismatches are rejected.
4. All ten visited flags in an accepted mission follow simultaneous drone and payload settling, with no drops, resets, contact-assisted stabilization or hidden state correction.
5. New normal/emergency/cost-shift runs preserve physical-boundary scenario semantics and use the same frozen model pair and paired initial conditions.
6. The randomized policy passes 9/9 basic missions, 18/18 endpoint missions and >=95/100 held-out random missions. A matched nominal-training comparison runs all 127 cases. Preserve all failures and report the limits of any claimed improvement.
7. Three complete raw runs and `full_normal.mp4`, `full_emergency.mp4`, `full_cost_shift.mp4` visibly show the drone, suspension, package and all ten settled stops. Each flight uses independently verified simulation-time 1x playback, truthful Japanese completion labels and complete source timing. The new payload HUD and event/settling transitions pass independent actual-frame checks and fresh operator continuous-playback review tied to the final hashes.
8. New artifacts, model signatures, hashes, reproduction instructions and compact evidence are saved without overwriting prior deliveries. Mocks and analytical pendulum probes are not reported as mission success.

## Verification and execution constraints

CPU tests cover parameter validation, inertia calculation, randomization reproducibility and stream isolation, inherited coordinate/priority contracts, observation field selection, schema mismatch rejection, shared dwell interruption, turning-point false positives, event ordering and metadata integrity. Include negative tests for swapped targets, stale evidence hashes and incompatible actual checkpoint configurations, rather than trusting a matching shape or sidecar declaration. Test fixtures establish logic only.

Use the pinned flight environment first. Inspect the actual pinned source for articulation, per-environment property updates and geometry switching before choosing APIs. Public documentation for another version is architectural guidance, not a reason to silently update `requirements/flight-linux.lock.txt`. Keep the flight and CPU environments separate.

No GPU training or server access is authorized merely by drafting this specification. At execution time, inspect both current GPU ledgers and the then-authorized server and budget. At this 2026-10-21 review, `outputs/gpu_ledger.json` and `docs/evidence/gpu_ledger.json` agree: cumulative use is 5779.474017477245 seconds, leaving 23020.525982522755 seconds of the existing 28800-second allowance. This is a snapshot, not a reservation or fresh allowance; reconcile any newer recorded use before starting and never reset the ledger. Use unique jobs/output directories, bounded timeouts, one GPU job at a time, and preserve failed attempts. Stop and report unmet gates if the remaining budget cannot complete them.

Before training, allocate the actual remaining budget across feasibility probes, both matched training runs, validation, all 254 evaluation missions, recording and artifact verification. Reserve evaluation/recording time instead of consuming the allowance on training alone. A CPU test or partial pilot cannot replace an incomplete physical acceptance suite.

When remote execution is separately authorized, transfer required sources, checkpoints, recordings and logs back locally and verify file sizes/SHA-256. Publish only repository-relative artifact references and sanitized compact evidence. No publication, course submission, push or remote PR creation is part of this specification-writing task.

## Out of scope

Package release or replacement at a destination; multiple simultaneous packages or drones; winches; slack, elasticity, rope-ground contact or entanglement; deformable cable simulation; wind and aerodynamic package drag; route-policy retraining for payload-aware energy; sensor estimation; actual aircraft deployment; circular-orbit flight as a separate objective; blanket robustness outside the tested distribution; combined-demo and submission editing.

## Design references

- [Genesis rigid-body constraints](https://genesis-world.readthedocs.io/en/v1.4.0/user_guide/robot_control/constraints.html): conceptual constraint types; compatibility with pinned 1.4.3 still requires source inspection and a physical probe.
- [Genesis domain randomization](https://genesis-world.readthedocs.io/en/latest/user_guide/policy_training/best_practices/domain_randomization.html): distinction between batched physical properties and state randomization. This specification deliberately requires episode-level resampling rather than relying on a once-at-build distribution.
- Repository `third_party/genesis_drone/hover_env.py`, `src/adaptive_drone_patrol/flight_agent.py`, `src/adaptive_drone_patrol/patrol_runner.py`, and `src/adaptive_drone_patrol/transition_check.py`: motor, observation, completion and independent transition contracts.
- [Dispersed-layout handoff](../../docs/dispersed-destination-layout-handoff.md) and [integer-priority handoff](../../docs/integer-priorities-handoff.md): current adopted artifacts, compatibility, geometry, event semantics, failures and evidence. [Visible-flight handoff](../../docs/visible-flight-handoff.md) remains historical context.

## 2026-10-21 仕様レビュー記録

整数重要度と分散3D配置のマージを受け、起点commit、採用モデルと設定、3次元座標、重要度、イベント、GPU残予算を更新した。今回の実測で明らかになった観測飽和、荷物を含む地面余裕、実遷移と証拠hashの照合、圧縮後の字幕検証、最終動画への新しい人手確認を検証方針へ反映した。旧モデル・旧動画の成功を吊り荷の成功として扱わない。

荷物の3寸法・質量10〜30%、固定長0.30 m、同時安定1秒、基本9件・端点18件・未学習100件、比較方策を含む254試行、および全編3本という成果物範囲は変更していない。この変更はspecの更新のみであり、吊り荷の実装・学習・受入試行は未開始。

## 2026-10-21 実装・検証の結果記録

明示の実装指示を受け、実装計画、実質量・慣性・衝突cubeと受動0.30 m固定リンク、episode再抽選、29観測、同時1秒settling、paired case宣言、hash検証と独立checkerを追加した。採用無荷物model_4296/cfgと整数routing seed44の実サイズ・hashを照合し、全精度配置・重要度・イベント規則を維持した。元の機体実solver質量は0.02700047753751278 kgであり、その10〜30%を荷物質量の基準とした。

13短時間probeと30 resetの限定条件では成立したが、ランダム化学習中に数値破綻した。同じ途中checkpointのraw診断は2/8/16 substepで全て破綻したため、この物理表現を不採用としてGPU作業を停止した。短時間の旧PASSは運用域の許可として無効化し、現行設定で再開を拒否する。固定リンク一般の不可能性、張力のみのロープ、長時間反力の正確性を検証したとは主張しない。

固定比較学習、validation、両方策の254試行、routing実判断説明、raw/HUD全編3本、動画照合・今回の人手確認は未達。予定条件を減らした合格判定はしない。今後の候補はpinned版での受動球関節表現と、学習前の長時間モーター・全遷移probeだが、代替方式は未実装。受入基準の変更には別途明示確認が必要であり、今回その変更は行っていない。詳細な失敗、全成果物hash、CPU修正版と実行版の違い、GPU台帳と再実行入口は引き継ぎに記録した。
