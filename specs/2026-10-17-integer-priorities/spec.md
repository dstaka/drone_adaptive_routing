# Intuitive Integer Priorities for the Physical Patrol Demo

- Date: 2026-10-17
- Status: Approved by the operator on 2026-10-17, including the numerical acceptance criteria below. Implemented and verified on 2026-10-20; all seven acceptance criteria pass. See the [measured handoff](../../docs/integer-priorities-handoff.md).
- PR: Not created. One implementation PR for this specification.
- Implementation root: repository root; the historical nested source directory is not an implementation target.
- Baseline: [Scalable routing](../2026-10-14-scalable-adaptive-routing/spec.md), [visible physical flight](../2026-10-14-visible-multigoal-flight/spec.md), and [accepted realtime video](../2026-10-14-realtime-demo-video/spec.md).
- Separate change: [Dispersed destinations](../2026-10-17-dispersed-destination-layout/spec.md). Neither specification implicitly approves the other.

## Goal and context

Make importance immediately understandable to viewers by replacing closely spaced fractional priorities such as 2.56 and 2.27 with randomly assigned integers from 1 to 100. A larger number means greater importance, not a mandatory visit rank. Keep numeric labels, color encoding, policy inputs and cost explanations honest and consistent.

The operator requested an independent specification and further clarification where necessary. At specification-authoring time this was not an instruction to implement, retrain, film, or use a server. The operator explicitly initiated those activities on 2026-10-20.

## Current behavior

`Scenario` permits real-valued priorities in [1,5]. The retained demo has initial values between approximately 2.12 and 2.95, with the emergency raising C to 5. The general training/evaluation generator already emits integers 1-5, while the demo fixture search uses fractional values in the narrower range.

Priorities are actual objective weights: the completion-time component includes priority multiplied by nominal completion time. Observations normalize priorities by 5. Model metadata, event validation and the scene palette also encode the 1-5 contract. Video labels print two or three decimals. Simply rounding labels could conceal genuine differences between model inputs; simply accepting 100 in a validator could silently violate the existing checkpoint contract.

## Intended behavior

On 2026-10-17 the operator first selected actual integer weights and matching labels, then explicitly revised the range to random integers on an inclusive 1-100 scale. The latest 1-100 requirement supersedes the earlier 1-10 proposal. The policy and objective must use those actual weights. A display-only index or rounding of the historical fractional values does not satisfy this specification.

Each displayed integer must be the importance weight used for that snapshot's cost calculation. The operator approved the coefficient and numerical verification requirements below together with the other specification decisions.

### Numeric and objective contract

Require integer-valued priorities in [1,100] initially and after every event. Generate the initial ten values using independent discrete-uniform draws from that inclusive domain, with replacement: equal values at different destinations are explicitly allowed. Do not deduplicate, require ten distinct priorities, use a fixed 1-10 permutation, round old demo values, or constrain the draw to a small handpicked subset. Save the generator/version, fixed seed and resulting per-ID values. Repeating the same generation command/configuration must reproduce the same values; all three compared scenarios and the fixed acceptance execution seeds share that initial vector. Keep the priority random stream separate from geometry generation. Once a mission starts, initial values stay fixed except for explicit logged priority events; never redraw them per frame or decision.

Demo candidate search may select a qualifying random fixture using disclosed, frozen analytical criteria, with every attempted seed and rejection reason retained. Do not hand-edit a draw to manufacture an explanatory example, or describe a conditionally selected demo as an unbiased random benchmark. Training, validation and test inputs remain separate from demo selection.

In the emergency scenario, raise only C's priority to 100 after the first verified physical inspection and before the next decision, preserving the predecessor's event boundary. C must be unvisited and its initial value must be below 100. A candidate with C already at 100, C already visited, or no required policy response fails the disclosed demo-selection gate; keep the attempted seed and reason rather than manually altering its draw. Other priorities do not change. Ties at 100 are valid: identify all maximum-priority destinations and apply a deterministic ID-based tie rule when choosing an H example.

New-profile scenario/event inputs use integer numeric values in YAML/JSON. Reject booleans, strings, nonfinite values, fractions and out-of-range values rather than coercing or rounding them; integral floating tokens such as 7.0 also fail the external integer-input contract. Float32 policy tensors may represent these validated integers exactly. Keep canonical integer values in scenario snapshots, event records and displayed labels; objective components remain real-valued.

Update priority-specific bounds independently from directed edge multipliers; multipliers remain in [1,5]. For the new profile, normalize priority observations by the fixed constant 100, not the current mission's maximum. Version that normalization and objective metadata explicitly. Never silently feed a new interpretation through an old schema/checkpoint merely because array shapes still match.

The operator chose coefficient adjustment rather than leaving the priority term at its old coefficient of 1. The approved constants are alpha=1 and beta=0.05: total objective = sum(distance * directed multiplier) + 0.05 * sum(priority * nominal completion time). Preserve nominal speed 1 m/s, service 0.50 s and no return to base. The factor is 5/100, compensating for the increase in the priority-scale maximum from 5 to 100. Record raw priority-time components and their weighted contributions separately. All learning, exact evaluation, event comparisons and viewer explanations must use the same frozen constants. A displayed remaining cost excludes the already incurred prefix while retaining absolute modeled completion times, as in the predecessor objective.

This scale compensation does not reproduce the historical priority distribution or guarantee the same route; in particular, random values from 1-100 can have much larger relative contrasts than the former balanced demo values. Observation normalization by 100 serves a different purpose and is not a second objective rescaling. The predecessor's narrow balanced-normal-demo restriction is replaced for this new profile. Do not tune beta after examining held-out results or physical acceptance outcomes; any change to the frozen constants requires a documented specification revision and re-evaluation.

### Viewer-facing contract

Display priorities as integers without .00 in landmark labels, tactical maps, destination lists, event captions and L/H comparison cards. Distances, times and objective costs remain appropriately precise real-valued quantities; the request does not integerize unrelated metrics.

Use one fixed low-to-high palette spanning 1-100, with exact value/color mapping in the scene manifest and the same mapping across scenarios and time. Keep numeric labels authoritative and readable, including the three-digit value 100. Do not normalize colors against each frame's or randomly selected mission's min/max. Target rings, visited checks and emergency symbols remain distinct from importance color.

Introduce the Japanese scale and explain that importance is not the visitation order. Cost cards show the configured 0.05 coefficient so viewers can reproduce the weighted components. Emergency captions show the actual old integer followed by 100 and change at the correct physical event. For the featured L/H example, require an importance difference of at least 10 points, H remaining a maximum-priority destination, in addition to all inherited distance, actual-policy-action and exact-cost proof gates. This replaces explanations based on nearly indistinguishable priorities; it does not replace proof of the route trade-off.

### Evaluation and evidence

Preserve three scenarios with shared geometry, initial priorities and model identity. Regenerate demo selection and exact Q*, J*, policy rollout and component costs for any changed real weights. Re-establish changed-next-action and positive post-event benefit gates under identical boundary conditions, including C remaining unvisited when its priority rises. Clearly handle any ties at maximum priority.

Apply the integer profile to newly generated training, validation, held-out evaluation and demo scenarios for the new policy contract; the runtime must support every integer from 1 through 100. Keep the predecessor's disjoint splits and evaluation denominators. Existing datasets and checkpoint interpretations remain unchanged. Version scenario/observation compatibility metadata to identify the priority domain, normalization by 100, alpha and beta; the old shape-compatible contract is insufficient.

Assess the adopted routing checkpoint on this new contract; explicitly establish compatibility or train/select a compatible model using the retained train/validation/test separation. Silent loading under changed metadata is forbidden. Never use an exact solver or heuristic fallback as the deployed learned policy. Record failed attempts and actual training/evaluation work. If training or selection changes the model, rerun the affected predecessor learning/held-out quality gates rather than reporting an old benchmark as the new model's result.

Produce new physical trial evidence rather than applying new labels to the old fractional-priority mission. Preserve all arrival/dwell, stability and 1x video gates. Recompute explanations from new physical snapshots.

## Compatibility and independent PR scope

This PR does not disperse or enlarge destination coordinates. It must be implementable on the accepted rectangular geometry. If the geometry PR has landed first, record that baseline and revalidate the combined result; do not mix its coordinate changes into this PR without an explicit scope decision.

Preserve historical fractional-priority scenarios and media as historical artifacts. Introduce an explicit version/profile contract for new inputs and compatibility errors or documented legacy loading; do not retroactively mutate old logs or silently round them. Update current Japanese README/report/PDF, palettes, schema documentation and rerun guidance.

Use unique output paths and repository-relative public references with size/SHA-256 manifests. At implementation time read both GPU ledgers and continue the cumulative budget. This specification grants no fresh allocation or server access. CPU compatibility and analytical checks precede GPU work, and filming requires current execution details.

## Acceptance criteria

1. Actual weights and matching labels use integers [1,100], independent uniform initial draws with replacement, fixed-seed reproduction, C's emergency update to 100, normalization by 100, alpha=1 and beta=0.05. Generated values, seeds and profile/model metadata reproduce this contract; no manual editing or hidden rounding occurs.
2. All priorities in the new profile satisfy the approved integer contract initially and after events. Fractional/out-of-range/invalid values fail clearly. Legacy artifacts retain their original values and interpretation.
3. Every displayed priority is a correct integer; palette, labels, events and logged values agree. No hidden display rounding is used to claim different cost weights. Real-valued distances/times/costs retain their own units and precision.
4. Observation normalization and checkpoint metadata match the chosen semantics. Incompatible checkpoints fail before inference; any replacement model is selected with disclosed evaluation and no held-out selection leakage.
5. Required normal/event/L-H demonstrations and independent exact-cost/component replay pass using the actual new contract. The featured L/H priorities differ by at least 10, with all inherited exact-proof gates also satisfied. Captions continue to distinguish an optimal next action, optimal full suffix and unproved optimality. If no qualifying candidate/model is found, report the gate as unmet instead of weakening it silently.
6. Actual integer weights are demonstrated by a new fixed physical 3-seed by 3-scenario suite with 9/9 missions and 90/90 inspections, preserving failures, and by new complete 1x media passing inherited independent and human checks.
7. Current Japanese docs, regenerated/visually checked PDF, logs, manifests and handoff describe the same contract and actual measurements. Prior accepted artifacts are preserved.

## Verification approach

Use CPU tests for boundary values 1 and 100, rejection of 0/101/fractions, random generation reproducibility and bounds, acceptance of duplicate priorities, event kinds, legacy loading, checkpoint/schema mismatch, normalization, tie handling, fixed palette and formatting. Verify objective components and every featured complete route against independent calculations. Do not treat synthetic cases as physical measurements.

For actual-weight changes, evaluate/freeze the compatible policy and fixture, run the fixed physical suite, and generate explanations from the recorded boundary states. Independently decode and validate final videos and event synchronization; inspect all ten labels, min/max colors and target/visited distinction, and obtain human playback review. Regenerate and visually inspect affected report pages.

## Approval record

On 2026-10-17 the operator explicitly approved both independent specifications, including the additional numerical acceptance rules. This approves beta=0.05, the minimum 10-point featured L/H difference, strict integer input representation and new-profile training/evaluation coverage together with random 1-100 actual weights/labels, repeats, fixed-seed reproduction and C raised to 100. These are requirements, not new physical measurements. Approval here does not start implementation, training or filming.

## Out of scope

Destination relocation, integerization of times/distances/costs, priority as mandatory visiting rank, changing logical multipliers into physical effects, unrelated flight-control changes, publication and course submission.


## Implementation and measured acceptance (2026-10-20)

Implemented from `7da4d23721770613d3ca910a20718ed6fd0e9371` in the repository root, preserving the rectangular geometry and historical artifacts. Scenario v3 / `integer-100-v1` and embedded checkpoint contracts enforce the approved semantics; v2 remains unchanged. Three new routing models were trained; validation selected seed44 before held-out evaluation. All three static quality gates pass on 100 static and 50 dynamic cases per seed, with 450 learned completions and no invalid actions.

The new fixed physical suite completed 9/9 missions and 90/90 inspections. Actual boundary explanations contain 33 qualifying L/H decisions and 675 independent full-order/component checks. New 1x videos pass independent decode/PTS/pixel/HUD checks: 7545 edited frames and 2741 normalized raw frames. The operator confirmed all three full clips and both event-through-next-inspection intervals at 1x on 2026-10-20. The four-page Japanese PDF was regenerated, rendered and visually checked. All seven criteria pass; no required item remains unverified.

See [compact acceptance evidence](../../docs/evidence/integer/acceptance.json), [implementation plan](../../docs/superpowers/plans/2026-10-20-integer-priorities.md), [Japanese handoff](../../docs/integer-priorities-handoff.md) and [prepared English PR description](../../docs/integer-priorities-pr-description.md). Delivery is one new local commit; no push, remote PR, publication or submission. GPU use continues the existing ledger at 3198.9943080281373 seconds, leaving 25601.005691971863 seconds. Dispersed-layout and suspended-payload specifications remain outside this commit.
