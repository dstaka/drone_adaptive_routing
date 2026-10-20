# Dispersed Destination Layout for the Physical Patrol Demo

- Date: 2026-10-17
- Status: Approved by the operator on 2026-10-17, including the numerical acceptance criteria below. Specification approval was followed by explicit implementation authorization on 2026-10-20. Implementation ran on 2026-10-20 and was finalized on 2026-10-21. All seven acceptance criteria passed; measured evidence is recorded in docs/dispersed-destination-layout-handoff.md.
- PR: Not created. One implementation PR for this specification.
- Implementation root: repository root; the historical nested source directory is not an implementation target.
- Baseline: [Scalable routing](../2026-10-14-scalable-adaptive-routing/spec.md), [visible physical flight](../2026-10-14-visible-multigoal-flight/spec.md), and [accepted realtime video](../2026-10-14-realtime-demo-video/spec.md).
- Separate change: [Integer priorities](../2026-10-17-integer-priorities/spec.md). Neither specification implicitly approves the other.

## Goal and context

Make the destinations in the video look spatially dispersed rather than arranged in regular rectangular rows, and expand the physical flight area. On 2026-10-17 the operator selected both a larger area and irregular placement, then explicitly requested different heights for different destinations. Change the actual three-dimensional world coordinates used by routing and Genesis, so the drone physically visits the displayed locations. Retain ten destinations A-J and a clear explanation of adaptive routing unless the operator explicitly changes that scope.

Historical authoring scope (2026-10-17): the operator requested a separate specification for this improvement and further questions where requirements were ambiguous. At that time, specification authoring did not authorize training, recording, server use or implementation. The later explicit implementation authorization and completed acceptance are recorded above and below.

## Current behavior

The selected demo configuration uses ten points at z=1 m, chosen from x={-0.9,-0.3,0.3,0.9} m and y={-0.6,0,0.6} m. The two interior positions (-0.3,0) and (0.3,0) are absent. S=(0,0,1). All three scenarios share this geometry. `analytical.demo_fixture()` and `scripts/select_demo_scenarios.py` are part of the fixture selection path.

The general scenario generator already samples random coordinates, but it does not by itself provide a validated, fixed physical demo layout with readable labels and the required explanatory examples. Replacing only a tactical map or annotations would leave the underlying physical mission unchanged and would not meet this goal.

## Intended behavior

### Geometry and reproducibility

Use an irregular set of ten world positions generated with a fixed seed, shared by normal, emergency and cost_shift. The operator selected fixed-seed reproduction for both the completed demo's layout and its initial priorities. Keep destination IDs stable throughout a mission. Freeze the selected coordinates, generator/version, seed, candidate rejection reasons, and scenario hashes before acceptance runs. Repeating the same generation command/configuration must reproduce the same destination coordinates. Do not independently resample positions per scenario, frame, or execution seed.

Vary actual destination z coordinates within the operator-confirmed absolute height range [1.0,3.0] m above ground, targeting a 2.0 m vertical span. Keeping every destination at z=1 m does not satisfy this specification. Report actual min/max destination heights and freeze the height-distribution tolerances before selecting the fixture; merely enlarging empty vertical camera space does not count. Represent heights consistently in routing, target commands, telemetry, markers and captions. A screen-space offset or height label without the corresponding physical flight is insufficient.

Use an expanded area rather than perturbing positions only within the old footprint. On 2026-10-17 the operator selected a target horizontal footprint of 5.4 m by 3.6 m: three times the previous width and depth, and nine times its area. Retain S=(0,0,1) and use world bounds x in [-2.7,2.7] m, y in [-1.8,1.8] m and z in [1,3] m. Enlarge the actual destination distribution, not merely empty camera margins or map bounds. Report the measured destination bounding-box width/depth/height alongside the nominal volume; ninefold area describes the nominal horizontal envelope, not the convex hull or a measured flight-distance increase.

Use deterministic constrained random sampling in that volume, followed by operator review and freezing one layout. Keep the layout random stream separate from any priority-generation stream so changing one generator does not silently change the other PR's inputs. Small perturbations that preserve an obvious row/column pattern do not achieve the intended visual change. The approved numerical gates are:

- Destination spans cover at least 90% of each nominal axis: x span >=4.86 m, y span >=3.24 m and z span >=1.80 m, while every destination remains inside the bounds.
- No three destination centers occupy any horizontal-coordinate band of width 0.10 m in x or y. Check all sorted consecutive triples rather than only arbitrary bin boundaries. This avoids recreating the old rows/columns; it is not a claim to detect every possible geometric pattern.
- At least one destination appears in each of the three equal-height thirds of [1,3] m. Assign internal boundary values to the higher band and z=3 to the top band.
- Every candidate also passes the inherited physical separation and projected visibility checks. Numerical gates supplement human inspection; they do not establish flight success or guarantee a pleasing layout.

Freeze these generation/selection checks before the acceptance fixture search; record all seeds/attempts and reasons for rejection. Any later tolerance revision requires a documented spec change and a fresh selection/evaluation, not post-hoc relaxation to rescue a failed flight. One selected layout is the fixed demo acceptance fixture; success does not establish robustness across arbitrary new layouts.

Preserve at least 0.50 m of three-dimensional Euclidean separation between destination centers for the existing 0.20 m arrival radius; keep S outside every destination arrival sphere. Do not reduce physical separation, arrival requirements, or label readability to make a candidate fit. All x/y/z coordinates must satisfy the adopted flight controller's world-target bounds. Validate actual long horizontal, ascending, descending and combined transitions, reversals and ground clearance, not just endpoint bounds. If the expanded volume exceeds the existing checkpoint's supported target range, explicitly revise the flight configuration and establish compatible trained behavior rather than only widening a validator.

### Routing and physical evidence

Recompute distances, logical costs, policy decisions, event comparisons and exact remaining-route explanations for the new geometry. Preserve the distinction between planning distances and same-frame pose distances. Old tours, Q*, J*, cost improvements and physical completion measurements cannot be relabeled as results of the new layout.

Retain the three scenario meanings and their shared initial state. Re-establish the priority-event and directed-cost-event response gates and a physically grounded nearer/lower-priority example from the predecessor specifications. Select fixtures using a disclosed demo search, never held-out test outcomes. Record unsuccessful candidates and flights.

Prefer the adopted routing and flight checkpoints initially. Assess routing quality and flight capability on the changed geometry. Any necessary model change requires separate recorded justification and the affected predecessor acceptance checks; reusing a model does not imply that new geometry is already validated.

### Visibility and delivered media

Derive scene framing and tactical map extents from the actual three-dimensional coordinates. Keep every physical marker and its ID visible throughout complete flights, with readable priority, target and visited indicators. Review projected label collisions, depth occlusion, apparent height differences and same-frame drone detail on real rendered frames. An overhead x/y map alone does not establish three-dimensional visibility: show destination height values and a clearly labeled projection or complementary view that makes vertical separation understandable. Physical landmarks, telemetry, scenario coordinates, labels and map positions must agree.

Produce new raw recordings, physical logs and complete 1x per-scenario videos, plus the full demo and submission deliverables required by the realtime video specification. Full flights have natural duration with every inspection retained; the 60-180 second submission may use explicitly disclosed cuts if the expanded missions require them. Preserve independent timing/pixel checks, terminal coverage, event-to-next-inspection context and authentic-flight-majority requirements. If the required submission context cannot fit, report the requirement unmet; do not change speed. New geometry cannot be demonstrated by moving labels over old flight footage.

## Compatibility and independent PR scope

This PR changes demo geometry and the evidence derived from it. It does not change priority values, ranges, color semantics, objective weights or the policy observation schema merely to disperse destinations. It must be implementable against the currently accepted priority contract. Record the expanded coordinate/target domain in model and flight metadata; if coordinate normalization genuinely must change, version that contract and re-evaluate compatible models explicitly. Endpoint bounds and unchanged array dimensions alone do not establish compatibility. If the separate integer-priority PR has landed first, explicitly record that baseline and rerun the affected combined checks.

Keep arrival radius 0.20 m, arrival speed <=0.30 m/s, arrival hold 0.20 s and inspection dwell 0.50 s, with no reset/teleport or marker relocation to create success. Retain the predecessor safety gates. Larger-area leg/mission timeouts may only be increased with documented transition evidence, a bounded job budget and values frozen before the fixed acceptance suite; never disable them or retroactively reclassify a timeout.

Do not rewrite historical merged specifications or overwrite accepted models, recordings and results. Use unique output paths, repository-relative public references and manifests with sizes/SHA-256. Update current README/report/PDF and document the new layout without presenting prior results as new measurements.

At implementation time, consult both GPU ledgers and continue the cumulative budget; do not reset it. This specification creates no new GPU allocation or server authorization. CPU layout design and analytical checks precede new GPU filming. Confirm current execution details when filming becomes necessary.

## Acceptance criteria

1. A committed, reproducible layout manifest identifies all ten actual coordinates, generator/version, fixed seed and selection rule; all three scenarios use the same geometry. The same generation inputs reproduce the same layout independently of the physical execution seed.
2. All destinations lie within the stated x/y/z bounds; actual spans are at least 4.86/3.24/1.80 m. The 0.10 m row/column-band and height-third gates pass. CPU checks reject invalid bounds, insufficient 3D spacing, base overlap, nonfinite coordinates and unreadable/occluded projected markers.
3. Before freezing acceptance runs, operator review of a spatial preview confirms an irregular 3D distribution. Final real-frame review separately confirms distinguishable positions/heights and all ten readable labels. Numeric geometry checks alone do not substitute for either review.
4. Three fixed execution seeds across normal/emergency/cost_shift complete all ten real inspections: 9/9 missions and 90/90 inspections, using unchanged arrival/dwell criteria and one frozen model pair for that suite. Preserve every failure and do not replace seeds after observing outcomes.
5. Updated routing/event/exact-comparison evidence is tied to actual new-run snapshots. Geometry-specific analytical gates and physical stability gates pass independently.
6. New full and submission media pass the inherited independent timing, source correspondence, visibility and human playback reviews. All full flights retain every actual inspection at 1x.
7. Handoff records include models/hashes, coordinates, generation and reproduction commands, failures, verification results, GPU cumulative usage and output manifests. Accepted historical artifacts remain intact.

## Verification approach

First test deterministic generation, invalid candidate rejection, three-dimensional physical separation and projection calculations on CPU. Compare candidate spatial plots, including height, with the current layout and obtain operator agreement on what counts as adequately dispersed. Include configurations with adequate 3D separation but overlapping image projections as rejection/reframing cases. Test routing/event behavior and exact-cost reconstruction before freezing the fixture.

Then validate controller transitions, run the fixed physical suite, and regenerate explanations from actual boundary snapshots. Decode and check final videos independently; inspect start, every goal switch/inspection, event boundaries and terminal frames, and watch event-to-response intervals at 1x. Regenerate and visually inspect affected Japanese report pages.

## Approval record

On 2026-10-17 the operator explicitly approved both independent specifications, including the additional numerical acceptance rules. This approves the centered bounds, 90% span threshold, 0.10 m row/column-band rule and height thirds as requirements. No actual layout, checkpoint or physical trial has been validated by this specification-authoring work. The generated layout is reviewed during implementation before acceptance trials; approval here does not start implementation, training or filming.

## Out of scope

Integer priority conversion, new destinations beyond the agreed ten, obstacles/wind, altered physical dynamics for logical edge costs, in-flight emergency interruption, real aircraft, publication and course submission.


## Implementation acceptance record (2026-10-21)

The operator explicitly authorized implementation after the specification-authoring phase above. The implementation preserves the approved numerical requirements and integer-priority contract. Layout seed51000/candidate3353 was independently reproduced and approved before freezing. Expanded routing quality gates,45 declared flight transitions, fixed9 new physical missions/90 inspections, all5 final videos/11,361 decoded frames, final operator playback review and PDF review passed. Development failures remain preserved; no threshold or acceptance seed was relaxed or replaced.

See [handoff](../../docs/dispersed-destination-layout-handoff.md) and [seven-criterion acceptance evidence](../../docs/evidence/dispersed/acceptance.json). Historical predecessor specifications remain unchanged. No remote PR or publication was created.
