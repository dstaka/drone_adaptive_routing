# Scalable Adaptive Routing and Three Scenario Contracts

- Date: 2026-10-14
- Status: Approved and implemented in the 2026-10-15 session; all CPU analytical acceptance gates passed. See the [measured handoff](../../docs/scalable-routing-handoff.md).
- PR: Local commit and prepared PR description only. The operator explicitly confirmed no push; no remote PR can be created for the unpublished commit.
- Implementation root: repository root (`src/`, `scripts/`, `configs/`, `tests/`). The untracked `adaptive_drone_patrol_20261013/` copy is historical reference, not the implementation destination.
- Dependencies: None. Followed by [physical integration](../2026-10-14-visible-multigoal-flight/spec.md) and [video delivery](../2026-10-14-realtime-demo-video/spec.md).

## Goal and context

Demonstrate that one learned routing policy changes its inspection order when observed priorities or travel costs change. The required patrol has at least ten distinct inspection destinations, with learned flight control executing the choices in the subsequent PR. Optimize a declared distance/priority objective and measure the gap to its optimum; do not equate a changed route with an improved route.

This specification supersedes the three-checkpoint restriction, fixed 65-feature routing contract, six-tour oracle, and demonstration event definitions in the [original specification](../2026-10-13-add-drone-routing-rl/spec.md). Its flight/evidence safeguards remain applicable unless the follow-up specifications explicitly replace them. This is a specification-authoring deliverable, not an instruction to execute the original specification's historical SSH, training, or commit authorization now.

## Single-session execution and PR boundary

When the operator supplies this specification with an explicit implementation request, complete its planning, implementation, training/evaluation, verification, local artifact delivery, and one PR in that session. The current operator request supplies execution authorization; the historical specification supplies baseline context. Do not rerun its entire three-point project or inherit its old prohibition on push when the operator explicitly requests a new PR.

The intended execution environment is the operator's existing GPU server reached through their supplied SSH method. Routing training/evaluation remains CPU-based in an isolated routing environment on that server; GPU hardware presence does not require converting MaskablePPO training to CUDA. Preserve the working Genesis environment and flight checkpoints. This PR does not require flight retraining, ten-point Genesis missions, camera rendering, or video composition.

Obtain or confirm the SSH alias, approved remote working directory, local checkout, known interpreter/checkpoint locations, and the applicable execution time budget from current operator input or verified existing configuration. Keep connection details and credentials out of Git. Do not assume the previous eight-hour allowance starts again. Record CPU job deadlines and artifact paths; retain the GPU ledger if any separately authorized GPU work occurs. Use durable remote jobs, separate run directories, reviewed transfer allowlists, and checksum verification as in the baseline workflow.

PR 1 acceptance ends at CPU analytical evidence: the frozen ten-action model, scenario manifests, benchmarks, and reproducible decision explanations. PR 2 binds these explanations to actual physical run/decision IDs and proves the selected behavior in Genesis. PR 3 presents the verified recordings. Mark planning-only outputs with `evidence_kind=analytical`; they must never masquerade as flight evidence.

Include only this PR's specification, implementation, tests, and relevant index/documentation changes in the PR. Preserve the other draft specifications locally for their later sessions. After validation, deliver a handoff record with source revision, model/schema/scenario hashes, selected model rationale, CPU evidence paths, remote and local artifact locations, actual elapsed time, and pending physical checks. Add the created PR URL here and link this specification from its description. The next session starts after this PR is merged, from the updated base branch.

## Current behavior

- `scenario.py` defines `IDS=('A','B','C')` and fixed `(4,3)`, `(3,)`, `(4,4)` shapes.
- `objective.py` and `routing_env.py` use three visited flags, `Discrete(3)`, and `routing-v1-65`.
- `baselines.py` enumerates six orders. Expanding that implementation directly to ten destinations would enumerate 3,628,800 orders per instance.
- The current emergency YAML updates action index 1, which is checkpoint B. `select_demo_scenarios.py` can choose an alternate checkpoint instead of preserving C as the emergency target.
- The existing departure-snapshot objective, action masks, event-at-inspection-boundary semantics, and separate learned motor controller are suitable foundations.

## Intended behavior

### Destination and model contract

The default and mandatory demonstration uses exactly ten destinations, IDs A through J, plus a separate start/base S that does not count toward ten. Derive all array sizes, loops, labels, masks, and terminal checks from an ordered `checkpoint_ids` list. Support at least N=3 for legacy regression, N=10 for delivery, and N=12 for a scalability check. Arbitrarily large N and one model that generalizes to every N are not required.

Use a new versioned scenario schema with `schema_version`, `checkpoint_ids`, and the existing fields. Coordinate row 0 is S; row `k+1` belongs to action `k` and `checkpoint_ids[k]`. Shapes are `coords=(N+1,3)`, `priorities=(N,)`, `service_duration_s=(N,)`, and directed `multipliers=(N+1,N+1)`. IDs must be unique, nonempty, and different from S. Validate finite values, distinct coordinates, positive nominal speed, nonnegative service/objective coefficients, priorities and multipliers in [1,5], and event references before training or inference.

New event configurations reference checkpoint IDs and directed pairs of node IDs. Resolve them through one mapping, rather than duplicating index arithmetic. Legacy unversioned three-point files must be explicitly recognized and normalized with IDs A/B/C; reject malformed or ambiguous input. Newly saved scenarios use the new version.

Keep a fixed observation/action shape for each trained N. `Discrete(N)`, an N-element mask, feature ordering, normalizations, N, ID ordering, and objective constants form the checkpoint contract. Generate and save field offsets and dimension in the model schema; use a new schema ID. Future event target/time/value remain absent from policy observations. Counts normalize by N; time/cost scales cover the declared training distribution without silently clipping larger patrols.

A three-action checkpoint cannot run a ten-action mission. Validate model, scenario, schema, and IPC dimensions before starting physics. Reject mismatches with an actionable retraining message. Do not pad the old model, truncate destinations, or fabricate actions. Legacy evidence and models remain immutable; new models and results use separate output directories.

### Objective and optimization evidence

Preserve the shared objective for every planner. For each leg from i to j, snapshot its departure state:

```text
d = Euclidean distance(coords[i], coords[j])
travel_cost = d * multiplier[i,j]
completion_time = previous_model_time + d / nominal_speed + service[j]
leg_cost = alpha * travel_cost + beta * priority[j] * completion_time
reward = -leg_cost
```

Default alpha=beta=1, nominal speed=1 m/s, service=0.50 s, and `return_to_base=false`. Multipliers affect logical cost only, not simulated speed or modeled travel time. Keep measured flight time/distance separate. Do not retrospectively reprice completed legs after an event.

Compare fixed ID order, the existing distance-times-multiplier/priority greedy rule, the learned policy, and an exact static reference. Use an exact method practical for N=10, such as subset dynamic programming for the weighted completion-time objective; a distance-only TSP solver is not an oracle for this objective. Verify the method against full enumeration for N=3 through N=7. Record solver status, time, and proven cost; a timeout or bound alone cannot be labeled an exact optimum. No factorial enumeration is allowed in the normal ten-point benchmark path.

At an event boundary, create two deep copies of the same visited/current/time state: one without the event and one with it. Run the same deterministic frozen policy through the remaining destinations on each copy, with no further events. Log both full hypothetical orders and first actions. Evaluate both orders using the post-event objective from the identical boundary to measure the benefit of replanning. These rollouts are diagnostic predictions, not precommitted physical routes or access to future events.

### Explaining distance versus priority with exact remaining-route comparisons

A destination's priority is an importance weight, not a mandatory visiting rank. A nearer, lower-priority destination can be the optimal first choice because the objective includes travel and priority-weighted completion times for every remaining destination. Provide numerical evidence for this trade-off to the video compositor; a color legend or the immediate-leg cost alone cannot establish it.

For each decision featured in an explanation, freeze the actual analytical boundary state b: current node, visited set, elapsed modeled time, current priorities, directed costs, service durations, and objective constants. Let U be its unvisited destinations. Compute:

```text
J*(b)    = minimum remaining objective cost over every complete order of U
Q*(b,a)  = minimum remaining objective cost over orders whose first visit is a
J_policy = remaining objective cost of the frozen policy's deterministic rollout
```

Use the same exact objective solver as the static reference, extended to accept a visited prefix, current node, elapsed modeled time, and optional forced first action. Include absolute modeled completion times in remaining priority penalties. Exclude the already incurred prefix cost from all three comparisons; it is common to them. Compare full remaining orders, not just two destinations while ignoring the others. Compute Q* for all eligible first actions at the featured boundary, and store the minimizing order and travel/priority-penalty components for each. An alternative first action must receive its own optimal suffix, not an arbitrarily poor continuation.

Freeze currently observed conditions for these comparisons, with no hidden future-event information. Prefer Scenario A or a boundary after the sole event in B/C for the explanatory example. The display claim is `現在の条件を固定した残り巡回の最適解`; it is not an optimum over unknown future disturbances, actual motor trajectories, energy use, or unmodeled obstacles. Compute this offline from logs so exact analysis does not pause physics or replace the learned controller.

Require at least one retained decision in the ten-destination A/B/C analytical replay, with at least three destinations still unvisited, illustrating all of the following. PR 2 must subsequently demonstrate it in an actual recorded mission:

- The actual policy chooses L, while H is a maximum-priority remaining destination, with priority[L] < priority[H] and Euclidean distance(current,L) < distance(current,H).
- The explanation artifact identifies L/H and their numeric priorities and planning distances, including exactly which H is used if several share maximum priority. Their visible annotations are delivered by PR 2/3.
- Q*(b,L) equals J*(b) within tolerance, while Q*(b,H) exceeds J*(b) by more than tolerance. Thus the chosen first action is part of a globally optimal remaining order under the frozen conditions, and forcing H first is strictly worse.
- Report J_policy separately. A first action compatible with an optimum does not prove that the policy's full remaining order is optimal.

Use equality tolerance `max(1e-6, 1e-8 * abs(J*(b)))`; show rounded values with enough precision to preserve any claimed strict difference. Label the full policy rollout optimal only if it equals J* within this tolerance and the solver has proved optimality. If the solver is incomplete, display `最適性未確認`; if the policy is worse, show its measured gap rather than presenting the oracle's route as the policy's route.

Choose this explanatory fixture using the declared validation/demo split and frozen policy, preserving the shared initial conditions and existing A/B/C acceptance requirements. Record selection attempts and criteria. Do not force an action, change the emergency target, or tune on held-out tests to obtain the example. If no valid example is found, mark the demonstration explanation gate incomplete; do not manufacture an optimality narrative. Existing held-out quality gates remain separate from this illustrative selection.

As an analytical unit fixture only, use S=(0,0,1), L=(0.5,0,1), H=(2,0,1), priorities 1 and 5, service=0.50 s, unit multipliers/speed, alpha=beta=1, and no return. L→H costs 18.0 (travel 2.0, priority penalty 16.0); H→L costs 20.5 (travel 3.5, priority penalty 17.0). These are the only two orders, so L first is optimal for this small example. This fixture is not ten-point demonstration evidence or a validated physical layout.

### Shared demonstration geometry and initial state

All three scenarios share one frozen geometry, base pose, initial priorities/multipliers, service durations, policy checkpoint, and inference mode. Use a nondegenerate layout with minimum destination separation 0.50 m for the retained 0.20 m arrival radius. The base must lie outside every arrival sphere. Geometry must subsequently pass the flight envelope checks in PR 2.

A starting candidate is the ten points from x in {-0.9,-0.3,0.3,0.9} m and y in {-0.6,0,0.6} m, excluding (-0.3,0) and (0.3,0), at z=1 m; S=(0,0,1). Assign A–J in ascending y then x. This candidate is not a claim of verified flight capability. Validation may choose another compliant geometry before freezing the final manifest; record the reason and re-evaluate all scenarios together. Do not relax spacing or arrival thresholds to fit ten points.

### Scenario A — Normal patrol (`normal`)

Use priorities in [2,3] with max/min <=1.5, unit edge multipliers, and no events. This provides relatively balanced importance while retaining distance/priority trade-offs. Show the learned traversal of every destination and its total objective value against baselines.

### Scenario B — Emergency inspection (`emergency`)

After the first physically completed inspection and before the next decision, raise checkpoint C's priority to 5. Other priorities, geometry, and multipliers remain unchanged. C must still be unvisited at that boundary. The accepted demonstration must show C moving earlier in the remaining policy rollout than in the no-event rollout, a changed next action, and a strictly lower post-event remaining objective cost (difference >1e-6). C need not be the immediate next action if the objective justifies an intervening stop.

Select a suitable demonstration fixture from a declared validation/demo split with the frozen policy; never change the emergency target away from C to obtain a successful example. If no fixture satisfies the conditions, record the unmet gate and improve training using validation data. Do not hand-author the actual route. For general benchmark episodes where C was already visited, skip the update and report the case separately rather than changing its target.

### Scenario C — Travel-cost change (`cost_shift`)

After the same first-completion boundary, increase one or more declared directed edge multipliers from 1 to a value in (1,5]. The affected edges must involve a possible remaining route and be fixed in the demonstration manifest before final physical trials. Priorities remain unchanged. State explicitly whether one direction or both directions are updated.

The accepted example must change the policy's next action and remaining order, and the new order must have strictly lower post-event remaining objective cost (>1e-6 improvement) than retaining the no-event order. Report both the travel component and total objective; changed order alone is insufficient. This represents a logical travel penalty, not simulated wind, obstacles, or energy consumption.

### Event delivery and training

Use the same model bytes across A/B/C, with no scenario-specific weights, policy branching, retraining, or heuristic fallback during demonstration. Deliver each event exactly once, after inspection completion and before the following action. Preserve the departure snapshot and all visited destinations. Continue selecting one next destination online at each boundary.

Train a new ten-action MaskablePPO using the existing CPU/flight separation. Retain three training seeds (42/43/44), static and event-inclusive training, saved manifests, and disjoint training/validation/test/demo splits. Reassess training effort for N=10 and persist actual steps and wall time; the old three-point training budget is not evidence that ten-point learning will converge. Select the final model by validation only, freeze its hash, and then evaluate 100 held-out static and 50 dynamic cases per training seed. Save all cases, including no-change and skipped-event cases.

## API, data, and compatibility impacts

- Update scenario loading/generation, routing state/environment, agent loading, baselines, benchmark, scenario selection, and CPU-facing replay together.
- Version the observation/IPC contract and carry N, ordered IDs, schema ID, mask, and policy hash. Wrong request IDs, dimensions, eligible actions, or process failures remain fatal.
- Extend benchmark/decision records with boundary ID, event applicability, no-event and post-event rollout references, component costs, exact-reference status, and schema version.
- Add a versioned `decision_explanations.json` linked to analytical replay run/decision IDs and boundary-state hashes, explicitly marked `evidence_kind=analytical`. Include snapshot inputs, frozen-condition scope, policy hash/action/rollout, all Q* values and minimizing orders, J*, solver proof/status/tolerance, objective components, candidate priorities/distances, selected L/H, and policy gap. PR 2 regenerates the explanation from actual flight boundary logs with physical run/decision IDs and `evidence_kind=physical`. Preserve precision in machine-readable data; presentation rounding must not change claims.
- Keep existing CLI entry points and add a documented `--num-checkpoints` option to routing training and evaluation; default new training to 10. Evaluation must reject a value incompatible with its frozen model. The configuration and explicit CLI value must not disagree silently.
- Historical three-point artifacts remain readable through explicit legacy handling. They never satisfy new ten-point acceptance.

## Acceptance criteria

1. Valid N=10 and N=12 scenarios complete CPU episodes with exactly N unique visits; invalid/masked actions never create a visit. No policy query occurs after all destinations are visited.
2. Scenario/model/IPC mismatches and invalid event references are rejected before mission execution. Legacy three-point regression fixtures remain valid.
3. The objective/reward invariant and departure-snapshot accounting hold for all supported test sizes. The exact reference matches brute force on small fixtures.
4. A/B/C use the same frozen N=10 routing checkpoint and shared initial fixture. B updates C only; C updates costs only. Both event examples meet their action/order/improvement conditions.
5. Report completion ratio, invalid-action count, objective components, paired fixed/greedy differences, exact gap, and dynamic response rate with denominators per seed. CPU completion must be 100%, with zero invalid model actions.
6. Proposed routing-quality gate for the selected model: mean static learned/exact cost ratio <=1.10, and mean static learned cost no worse than greedy and strictly below fixed ID order on the held-out set. Report worst-case gap and failures too. This is a bounded quality target, not a claim of global optimality; if unmet, mark routing quality incomplete rather than change the threshold after seeing results.
7. Source, tests, schema, frozen scenario manifests, benchmark summaries, and reproduction commands accompany this specification in its implementation PR.
8. At least one ten-point analytical replay decision meets the nearer/lower-priority example above. Independent cost replay confirms every full-order/component cost and the strict disadvantage of forcing the farther maximum-priority destination first. Distinguish an optimal first action from an optimal full policy rollout. Actual recorded-flight confirmation is a PR 2 gate, not a dependency for merging this PR.

## Verification approach

Run CPU unit/property checks for N=3/10/12, invalid shapes/IDs, deterministic regeneration, masks, full completion, boundary ordering, event isolation, and no future-event features. Use independent small exact fixtures for objective/oracle correctness. Run a real short train/save/load/IPC smoke before full training, then the frozen held-out evaluation and deterministic A/B/C planning replay. CPU tests must not import Genesis. Record commands, exit codes, actual measurements, and failures in the implementation's verification document.

Verify the two-point trade-off fixture and compare every forced-first optimum against enumeration on small problems, including nonzero elapsed time, visited prefixes, asymmetric costs, ties, suboptimal policy rollouts, and incomplete solver status. Independently recompute the selected ten-point explanation's order costs from its snapshot. Reject mismatched decision IDs, snapshots, objective constants, or models. An explanation is a mathematical comparison of observable inputs and outcomes, not an introspective account of the neural policy's internal reasoning.

## Out of scope

Genesis marker rendering, ten-point physical acceptance, final video editing, immediate mid-flight emergency diversion, dynamic destination insertion/removal, hardware operation, perception, collision avoidance, a joint routing/motor network, and an unconditional proof that the learned policy is globally optimal.

## Implementation experiment record

The operator authorized four hours of CPU execution on the supplied server, local artifact delivery, one final commit, and preparation of a PR description without push. Implementation and measurements use separate directories from all historical models and evidence.

Two disclosed CPU training methods are compared using validation only: a conventional MaskablePPO MLP and a shared-candidate neural MaskablePPO policy. The candidate policy can receive supervised initialization from exact frozen-condition labels generated exclusively from the training split, followed by static and event-inclusive PPO. Every teacher input/label is saved with hashes. This is a training method; inference never invokes an exact solver, teacher, baseline, or heuristic fallback. Three independent training seeds remain 42/43/44. The final handoff records the actual model choice, configuration, elapsed time, held-out measurements and any incomplete gates.
