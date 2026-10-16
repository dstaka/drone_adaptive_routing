# Stable Ten-Destination Flight and Persistent Goal Visibility

- Date: 2026-10-14
- Status: Approved by the operator for implementation and physical verification in this session. See the implementation handoff for measured results.
- PR: Not created. Add the PR URL when available.
- Implementation root: repository root (`src/`, `scripts/`, `configs/`, `tests/`). The untracked `adaptive_drone_patrol_20261013/` copy is historical reference, not the implementation destination.
- Dependency: [Scalable adaptive routing](../2026-10-14-scalable-adaptive-routing/spec.md).

## Goal and context

Execute all ten destinations in actual Genesis physics while making the complete mission visible from its first frame. A viewer must understand that every destination already exists and that the AI selects among them. Demonstrate stable arrival/hovering and actual flight after priority/cost-driven decisions, using one frozen routing model and one frozen motor-control model for all scenarios.

This specification replaces the original three-destination physical acceptance and single-active-marker presentation. Preserve the [original specification's](../2026-10-13-add-drone-routing-rl/spec.md) learned motor control, source/checkpoint provenance, no-reset target changes, pre-reset failure detection, and authentic same-run evidence requirements. GPU execution requires valid runtime access and a recorded remaining budget; previous execution allowances are not reset by this PR.

## Current behavior

`mission.py` embeds three-action bounds and A/B/C IDs. `flight_agent.py` enables the upstream target visualization, and `TargetAdapter.set_target()` moves one `env.target` to the current goal. `video.py` shows all three points on an offline map, but that does not make all destinations present in the Genesis scene. The current 0.6 m triangular layout and its existing flight results do not establish safe execution of a wider ten-point layout.

## Intended behavior

### Integrated mission execution

Use the versioned scenarios and models from PR 1. Derive IDs, completion counters, action validation, telemetry, and summaries from the configured N. The mandatory demo has A–J and starts at S. Run normal, emergency, and cost-shift using identical initial geometry, initial physical conditions for each paired seed, frozen policy hashes, and arrival settings.

Before physics starts, check schema/model compatibility, world-frame target bounds, minimum spacing, base clearance, renderer setup, and finite positive per-leg/mission timeouts. Timeouts must accommodate the measured ten-point mission and be frozen before acceptance trials; retain failure bounds instead of disabling timeouts. Change targets without resetting position, velocity, motor history, or the flight policy's required state. Refresh target-relative observations before inference.

Complete each inspection using the following retained contract:

| Measurement | Requirement |
| --- | --- |
| Distance to target center | <=0.20 m |
| Linear speed | <=0.30 m/s |
| Arrival confirmation | Both conditions continuously hold for >=0.20 simulation seconds |
| Inspection completion | Both conditions continuously hold for >=0.50 simulation seconds total; the first 0.20 s is included |
| Interrupted hold | Reset accumulated dwell when either condition fails |
| Failure | Reset, termination, fall, nonfinite state, target mismatch, or timeout cannot count as arrival/completion |

Use actual pose/velocity samples, not elapsed time alone or analytical planner transitions. Log hover residuals and speed across each dwell, per-leg maximum/percentile tracking error, measured path length/time, and failure counts. Preserve declared flight safety limits and record tilt/altitude; do not relax limits to pass a longer route. Changing the active goal must not move the vehicle instantly.

At a verified inspection completion, account for the departed leg, mark the destination visited, apply the event exactly once if due, build the new observation, and query the frozen routing policy. B updates C and C updates the declared directed cost edges as specified in PR 1. The aircraft remains at its physical state while the next goal is selected. No in-flight interruption is required in this version.

### All destinations visible from the start

Create N persistent, noncolliding visual markers at the scenario coordinates before capturing the first mission frame. Markers must not add forces, collisions, or alter the vehicle dynamics. Keep every marker in place and visible through mission completion, including visited destinations. The current target is a highlighted existing marker; never implement it by hiding or relocating the others.

Use a fixed overview camera or a validated framing rule that keeps the entire patrol area and drone visible throughout every flight segment. Camera framing must derive from the scenario extents plus a margin, not the current target alone. At 1280x720 presentation size, destination positions and their IDs must be distinguishable without relying only on color. If one camera cannot provide both overview and readable drone detail, keep an always-visible Genesis overview beside a detail view; the map alone does not substitute for persistent scene markers.

Use stable IDs and a consistent state legend:

| State | Required visual distinction |
| --- | --- |
| Unvisited | Persistent marker and readable ID |
| Current target | Additional ring/outline and target label |
| Visited | Persistent marker retaining its priority color, plus a check mark and distinct outline |
| Emergency C | Additional alert symbol and changed priority indication |
| Increased travel cost | Directed affected-edge indication in the tactical overlay, with its old/new multiplier |

IDs may be native scene labels or screen-space labels anchored to the actual projected world position. Projection/label visibility must be verified on real camera frames. Do not invent a Genesis text API. Log a versioned marker/scene manifest containing IDs, world coordinates, visual-only status, camera parameters, and style mapping. Expose target/visited/priority state changes in the event stream so PR 3 can reconstruct the display precisely.

The first frame already shows all ten markers. At least one introductory view explains their IDs and initial priorities before the first departure. That view cannot be used to hide missing markers later. Avoid identical overlapping labels, tiny markers, and HUD panels that cover destinations. Retain visual symbols when marker colors change.

### Priority encoding independent of visit state

Every destination must communicate its currently observed numeric priority throughout the video. Use one fixed light-to-dark red sequential scale for priorities 1–5: suggested anchors `#fee5d9`, `#fcae91`, `#fb6a4a`, `#de2d26`, `#a50f15`. Interpolate deterministically for fractional priorities. Save the exact palette in the scene manifest and use it unchanged across scenarios and time; never normalize colors to the current minimum/maximum. A priority of 3 has the same encoding before and after C rises to 5.

Show a readable anchored label such as `C｜優先度 5` for each marker and a persistent `薄い色＝低優先度 / 濃い赤＝高優先度` legend with numeric scale. Light markers must remain opaque with a contrasting border; lighting, distance, or transparency must not be mistaken for priority. The numeric label is authoritative even if scene lighting changes its apparent color. Marker size remains constant in world coordinates so size does not introduce an undocumented importance cue.

Reserve marker fill for priority. Use a separate ring and `選択中` for the current target, a check mark and `訪問済み` for completed destinations, and an alert symbol for an emergency. Do not recolor the selected target red simply because it was selected, or dim a visited marker into a different apparent priority. Cost-increase edges use a distinct dashed-arrow style and their own legend rather than the priority scale.

At a priority event, update the affected marker's displayed color/value and map entry at the first frame representing the event, while all unrelated priority encodings remain unchanged. Event text includes the old and new values, such as `C：優先度 2 → 5`. At the decision chosen for the distance/priority explanation, annotate the nearer L and farther H with their current distance from the actual drone pose; both remain visible during the subsequent actual departure. Keep analytical node-to-node planning distances separate if they differ from the drone's measured position.

The inspected baseline already defines a colored target through `gs.surfaces.Rough` and `gs.textures.ColorTexture`; existing offline composition can draw projected labels. At implementation time verify runtime material-update support in the pinned Genesis source. If dynamic scene recoloring is unavailable, use neutral real noncolliding landmarks and add a clearly identified, world-anchored colored priority badge/ring in the composed footage, plus the same encoding on the tactical map. A supported screen-space annotation satisfies the priority display requirement; do not rebuild/reset physics or claim the annotation was a native raw-scene material update. Record the chosen rendering method. With native recoloring, marker fills track current priorities; with the overlay fallback, neutral fills avoid competing stale priority colors and the dynamic badges carry the priority scale.

### Flight feasibility and frozen comparisons

Validate the proposed layout from PR 1 against the actual controller's command envelope and observation normalization. Endpoint coordinates within bounds do not prove long inter-goal transitions work: exercise distant transitions, direction reversals, observation saturation, and repeated hover/retarget sequences. A failed validation requires a justified geometry revision or additional flight training, followed by re-evaluation of all three scenarios. Do not compress markers below the separation constraint.

Reuse the proven motor checkpoint first. If it fails, diagnose the target adapter, resets, or coverage before deciding on fine-tuning. Record any new flight checkpoint and validate it before final trials. Freeze the routing model, flight model, scenario manifest, camera, timeouts, and arrival settings before the acceptance suite. Model changes invalidate prior comparative trials and require the complete suite again.

## API, data, and compatibility impacts

- Update mission/flight/target-rendering integration to consume arbitrary configured IDs and N. Preserve the CPU-planner subprocess boundary and runtime signature checks.
- Record `checkpoint_ids`, N, schema versions, both model hashes, scenario hash, execution seed, marker manifest, and full attempted/visited orders in each run summary.
- Preserve same-run `scenario.json`, `events.jsonl`, `decisions.jsonl`, `telemetry.csv`, `frames.csv`, and `raw.mp4`. Add marker visibility/projection evidence without replacing physical telemetry.
- Extend the scene manifest with priority range/palette, fractional-value mapping, labels, orthogonal state symbols, rendering method, and event-to-display mapping. Preserve original raw footage if labels/badges are added offline, and identify the annotated output explicitly.
- Frame records must associate each camera capture with its actual simulation step/time and encoded PTS. Include initial and final mission state coverage; capture cadence must allow PR 3 to preserve 1x time.
- Existing three-point recordings remain historical evidence only. Do not overwrite them or treat them as ten-point verification.

## Acceptance criteria

1. Every complete recording starts with all A–J destinations present in the Genesis scene and keeps all markers visible until the final inspection. Each marker's world position agrees with its scenario entry, independent of the active target.
2. After one target is completed, it remains visible while another existing target is highlighted. Resetting/repositioning a marker or the aircraft cannot produce a completion.
3. For predeclared execution seeds 1, 2, and 3, each scenario completes all ten unique destinations: 3/3 successes per scenario, nine successful missions total. Preserve and report all attempts, including earlier development failures; repeated retries do not replace the fixed acceptance-suite denominator. This is a demonstration stability gate, not a general reliability guarantee.
4. Every one of the 90 inspections in that suite satisfies the recorded position/speed/continuous-dwell criteria. No mission contains resets, falls, target mismatch, nonfinite state, or timeout.
5. The emergency and cost events occur at the specified physical completion boundary, and their actual first post-event action matches the logged frozen-policy decision. A/B/C share the same frozen model pair. Counterfactual predictions are explicitly distinguished from actual flight.
6. At least one paired seed has full authentic raw footage for all three scenarios with complete timing/telemetry. Record every acceptance trial's telemetry and outcomes, even when video is not selected for delivery.
7. Real-frame visual inspection confirms readable IDs, target/visited/emergency distinction, drone visibility, and no hidden destinations at mission start, every target change, each event, and mission end.
8. All ten destinations have readable current priority values and consistent priority colors in the annotated video. Selection/visited state never changes the encoded priority. The emergency update changes only C's priority encoding at the correct frame. The nearer/lower-priority versus farther/higher-priority pair from PR 1 is visibly identifiable in the actual recorded mission.
9. Regenerate `decision_explanations.json` from the actual recorded boundary state with physical run/decision IDs and `evidence_kind=physical`. Verify that at least one actual ten-point decision satisfies PR 1's nearer/lower-priority versus farther/maximum-priority exact-cost conditions. An analytical replay identifier alone does not establish this gate.

## Verification approach

First run CPU state-machine tests with ten and twelve destinations, including skipped/duplicated visits, injected resets, dwell interruption, malformed actions, event ordering, and marker-state transitions. These mocks establish logic only.

Check the priority-to-color mapping at 1–5 and fractional values, stable colors under unrelated events, independent selected/visited symbols, and synchronization of event-driven labels/badges. Inspect real rendered light and dark markers against the actual background at the final resolution, including grayscale readability through numeric labels. Verify pose-derived distance annotations against same-frame telemetry and label any separate analytical planning distance.

On the authorized GPU runtime, run a short actual camera/marker smoke and endpoint/long-transition checks using the pinned environment. Compare matched seeded dynamics with markers enabled/disabled to confirm visual markers do not affect physics. Check world-to-camera projections against rendered frames and inspect actual labels/occlusion. Then execute the frozen nine-mission suite and validate all retained event/telemetry/frame references, full-video decode, and physical metrics. Log actual runtime and remaining GPU allowance; if unavailable or exhausted, mark physical acceptance blocked while preserving CPU progress.

Deliver compact machine-readable summaries, representative frame evidence, Japanese reproduction notes, and all large raw artifacts through the existing manifest workflow. Final edited timing and HUD verification belong to PR 3.

## Out of scope

Video retiming/composition, obstacle or wind simulation, physically altered flight dynamics for logical cost shifts, sensing actual defects, emergency interruption during a leg, return-to-base requirements, real aircraft, or additional inspection points implemented only as decorative markers without physical visits.

## Implementation and measured acceptance

Implemented from merged scalable routing base ff346a0. See [Japanese handoff](../../docs/visible-flight-handoff.md) and [compact evidence](../../docs/evidence/visible/acceptance.json).

The predeclared normal/emergency/cost_shift by execution seeds 1/2/3 suite completed 9/9 missions and 90/90 continuous physical inspections. All nine raw recordings retain initial/final coverage and full capture-step/encoded-PTS correspondence. Source, routing checkpoint, selected motor checkpoint, geometry, camera and acceptance settings were frozen before the suite. All development failures and the initial offline-verification failure remain in the ignored delivered artifacts.

The existing motor checkpoint passed the baseline but failed a long-retarget safety probe. Diagnosing unchanged state/history, unsaturated observations and identical marker-on/off failure justified fine-tuning. The selected model_1300 preserves the motor architecture, target envelope and inference physics, then passes long transitions, reversals, repeated hover and the full suite. Routing model bytes and geometry are unchanged.

Persistent neutral Genesis landmarks use world-anchored priority badges with fixed palette/numeric values, orthogonal state symbols and an always-visible genuine same-frame vehicle detail crop. Offline annotation fixes do not alter raw footage or physical trials. All 7995 raw frames pass supplementary visibility checks; critical real frames receive human inspection. The seed1 annotated clips preserve 2665 source frame indices/PTS, without timing edits.

All 90 actual decision boundaries have physical explanations, including 42 qualifying nearer/lower-priority cases. Independent standard-library replay checks 675 complete orders. Cumulative GPU use, including inherited history, is 2803.57274048205 seconds of 28800. This is bounded deterministic demonstration evidence; disturbance/general reliability tests and final edited/retimed submission video remain outside this PR. No remote PR or push was performed.
