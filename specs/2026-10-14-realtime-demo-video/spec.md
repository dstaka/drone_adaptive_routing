# Real-Time Playback and Evidence-Backed Adaptive Patrol Videos

- Date: 2026-10-14
- Status: Approved for implementation in the 2026-10-16 session; accepted after operator visual approval on 2026-10-17. Delivery measurements and review provenance are recorded in `docs/realtime-video-handoff.md`.
- PR: Not created. Add the PR URL when available.
- Implementation root: repository root (`src/`, `scripts/`, `configs/`, `tests/`). The untracked `adaptive_drone_patrol_20261013/` copy is historical reference, not the implementation destination.
- Dependencies: [Routing/scenarios](../2026-10-14-scalable-adaptive-routing/spec.md), [physical integration](../2026-10-14-visible-multigoal-flight/spec.md).

## Goal and context

Show actual Genesis flight at exactly 1x simulation speed, with all destinations visible and clear evidence that the same learned policy adapts to priorities and travel costs. A viewer must be able to distinguish actual motion, the current target, future predictions, and changes in environmental conditions.

This replaces the [original specification's](../2026-10-13-add-drone-routing-rl/spec.md) allowance for accelerated/slow playback and its three-point storyboard. Preserve authentic footage, Japanese audience-facing explanations, traceable telemetry, and the 60–180 second course submission requirement by separating full mission videos from an optional-cut submission edit.

## Current behavior

`video.compose()` divides a requested duration (default 120 seconds) into three equal scenario slots, computes `rate = source_span / scene_duration`, and resamples each patrol to fit. It hardcodes an additional 0.02-second source-frame interval and labels clips as slow playback. Its HUD includes A/B/C-only strings, `/3` counters, and fixed map scaling. The media checker validates the existing mapping but does not require 1x playback.

Changing the displayed speed label or output FPS alone cannot fix this: the source-to-presentation time mapping must change.

## Intended behavior

### Definition of 1x playback

For every retained continuous flight segment:

```text
presentation_time - segment_presentation_start
    = simulation_time - segment_simulation_start
playback_rate = 1.0
```

Simulation time comes from captured steps and the actual physics/control timestep. It is independent of how quickly the GPU computed or rendered those steps. Slow offline computation is acceptable; slow-motion or accelerated flight in the exported video is not.

For raw footage, encoded PTS differences must match source simulation-time differences within the declared frame quantization tolerance. For edited footage, map output PTS directly to that source time at unit slope. Select/resample source frames by their recorded timestamps, preserving duration. Normal frame repetition or dropping for 50-to-30 FPS conversion is allowed only to quantize sampling, never to stretch or compress the mission. No optical-flow generated motion is required.

Derive the final frame's coverage from the capture schedule and logged mission-end time, rather than hardcoding 0.02 seconds. Handle a forced off-cadence final frame explicitly using a documented timestamp-aware encoding/sampling rule. Reject missing, duplicate, nonmonotonic, or unexplained discontinuous timing records. The final inspection/completion state must be represented without adding a fictitious extra flight interval.

### Deliverables and duration

Use one selected paired execution seed from PR 2; record selection and preserve all trial summaries.

| Artifact | Content and duration |
| --- | --- |
| `raw.mp4` per selected run | Complete authentic Genesis recording with `frames.csv`, at 1x simulation time |
| `full_{normal,emergency,cost_shift}.mp4` | Complete per-scenario flight, Japanese HUD, no omitted mission intervals, 1x; natural duration |
| `demo_full.mp4` | Three full scenario videos plus clearly separate intro/transitions/credits; no fixed duration cap |
| `submission.mp4` | 60–180 seconds, target 120 only when feasible; every retained flight interval remains 1x |
| Per-output timeline and selection JSON | Source run/frame/time mapping, cards, cuts, policy/scenario hashes, selection reason |

The full videos are mandatory proof of all ten visits. A short submission is an edit for explanation, not independent proof that every omitted visit occurred.

If the full patrols and explanations exceed 180 seconds, make explicitly labeled cuts in the submission and keep the complete videos. Show continuous 1x segments around each event and the resulting physical departure/arrival, including at least two seconds before and after an event when available, extending through the next verified inspection. Show the initial all-destinations view, representative normal patrol, emergency response, cost response, and completion evidence for each scenario. Indicate omitted simulation intervals and visit counts at cuts; never splice runs into a fictitious continuous patrol.

If the material is shorter than 60 seconds, use separate explanatory cards, measured comparison graphics, or clearly labeled static diagrams. Do not stretch flight, loop it as new progress, or freeze a drone while a simulation-time counter continues. Keep authentic flight as the main visual for more than half the submission duration, as required by the original specification. If these constraints cannot be met with the recorded material, report the missing footage and plan additional genuine recording; do not silently change playback speed.

### Viewer-facing narrative and HUD

Display Japanese scenario titles with these meanings:

- Scenario A: 通常巡回 — 距離と優先度のバランス.
- Scenario B: 緊急点検 — C地点の優先度上昇で残りの巡回を再計画.
- Scenario C: 移動コスト変化 — 一部の有向移動コストが増加し、低コストな順序に変更.

Keep actual Genesis footage prominent and preserve the all-goal visibility established in PR 2. The persistent tactical map also shows every destination from the beginning, the current vehicle position from telemetry, visited states, and current target. Compute map bounds from the whole scenario; never clip outer destinations with fixed three-point scaling. Use scenario IDs and N in every label/counter rather than `ABC`, `SABC`, or `/3` constants.

Show `シミュレーション時間`, `1.0倍速`, current target, visited count `/10`, current priorities, and the relevant changed directed edges/multipliers. Distinguish physical seconds/meters from abstract objective cost. All values and status changes come from the selected run's logs. Emergency C must be legible without the viewer confusing it with Scenario C.

For B/C, show the pre-event and post-event next decision and the remaining-order comparison from PR 1. Label the no-event rollout `更新なしの予測` and the post-event rollout `更新後の予測`; label the measured flown path separately. Predicted route graphics cannot imply that every future stop has already been committed or physically flown. Display the measured analytical improvement and its objective definition; do not claim physical energy/time savings from a logical multiplier change.

The same policy across scenarios is explained in Japanese; full model hashes and schema IDs belong in metadata/credits/documentation. Report actual benchmark/flight outcomes and limitations. Update existing Japanese README/report source and its generated PDF if it contains outdated three-point or playback claims; regenerate and visually verify the affected final report pages as part of delivery.

### Showing why a nearer, lower-priority destination can be optimal

Use the priority colors, numeric labels, and independent visit/target symbols defined in PR 2 on every destination in the camera annotations and tactical map. Keep the fixed 1–5 legend visible. State `優先度は重要度であり、訪問順そのものではありません` early in the video. For the selected example, show L and H simultaneously, with their priority values and distances, while the actual policy-selected destination receives a ring and `AIの選択` label.

Build a compact comparison panel from PR 1's `decision_explanations.json`. Compare the complete remaining orders using identical boundary state and current conditions:

| Row | Value and meaning |
| --- | --- |
| `近いLを先に訪問した場合の最小コスト` | Q*(b,L), with its optimal suffix and component costs |
| `遠い高優先度Hを先に訪問した場合の最小コスト` | Q*(b,H), with its own optimal suffix and component costs |
| `AIの残り巡回予測` | J_policy and the actual frozen policy's rollout, which may differ from either optimal suffix |

Show `総コスト＝移動コスト＋優先度を重みとした完了時刻のペナルティ（小さいほど良い）`, with configured alpha/beta weights reflected in component values. Use stacked bars or a concise numeric table to expose the travel/priority-time trade-off. Do not rename the completion-time term as pure hover/wait time, or describe the total as physical energy. Show the difference between the two forced-first optima and explain `高優先度でも、直行するより近い地点を先に回る方が、残り全地点の総コストを小さくできる場合があります` only when the recorded comparison supports it.

Label exact suffix routes `条件付き最適解（比較用）`, the policy rollout `AIの予測`, and the flown trajectory `実飛行`. A solid short arrow can identify the actual next target, while dashed routes depict alternatives on a separate map/panel; neither alternative is synthetic flight footage. For readability, show the two alternatives side by side or sequentially on separate explanation cards rather than layering every possible route over the drone view.

Display `この選択は現在条件で最適な次の一手` only when Q*(b,L)=J*(b) with a completed proof. Display `AIの残り巡回全体が最適` only when J_policy also equals J*. Otherwise show its absolute/relative gap to the exact reference. Always include the scope `現在の条件を固定した残り巡回` and use `最適性未確認` for incomplete solver results. Comparing two routes alone does not prove global optimality; the J* reference must cover all remaining orders.

This is an explanation supported by input/output cost analysis, not a claim to read the policy's internal thoughts. Explain what the chosen action achieves under the objective. Bind every displayed number, order, distance, and optimality caption to the actual decision ID and snapshot. At a later boundary, clear the old panel or label it with its source decision/time so it cannot appear to describe the current state. Planning distances are node-based; pose-to-goal distances in the live HUD are measured and labeled separately.

Preserve 1x playback during flight. If the comparison needs more reading time, insert a separate, explicitly labeled `判断時点の比較（静止図）` card with the decision timestamp and no advancing simulation clock. Do not slow down the flight or disguise a freeze as ongoing hovering. Include the verified trade-off example in both `demo_full.mp4` and the submission edit; full per-scenario flight remains uncut at 1x. Introduce the palette/legend before the example and retain the actual resulting departure/arrival so viewers can connect comparison and behavior.

### Timeline and compatibility

Create a versioned timeline containing `segment_type` (`flight` or `card`), run/clip ID, source frame range, source simulation start/end, presentation start/end, output FPS, `playback_rate=1.0` for flight, and explicit cut/disclosure records. Record boundary conventions and mission-end coverage so another tool can independently recompute duration and mapping. Cards have no fictional source simulation interval.

Remove fixed-duration flight retiming from the default demo build path. Keep the existing `build_video.py` entry point, add explicit `--mode full|submission`, and reject configurations requesting non-1x flight for these deliverables. A duration option may constrain the submission edit/card plan; it must not derive playback rate. Document new configuration fields and migrate `configs/video.yaml` accordingly. Keep legacy timeline parsing separate from new 1x acceptance; old files remain immutable and are not re-labeled as compliant.

Use H.264 MP4, at least a 1280x720 presentation canvas, and 25 or 30 output FPS. Preserve raw source resolution/FPS metadata, letterbox rather than distort the camera view, and document licensed Japanese fonts. Save new artifacts under unique paths and update the artifact manifest only after hashes, decoding, and timing checks pass.

## Acceptance criteria

1. Every raw and edited flight segment has actual 1x timing, not just a `1.0` metadata field. Source/presentation span error is <=one output frame for each edited segment; raw PTS error is <=one raw frame. No cumulative mission drift beyond one relevant frame is permitted.
2. An independent checker derives expected times from physics steps and encoded frame PTS, and compares them with the timeline and HUD event presentation. It must not pass merely by copying the composer's declared playback rate or emitting a constant zero alignment error.
3. Three full scenario videos show all ten actual completed inspections without time compression or omitted intervals. `demo_full.mp4` preserves their natural durations plus declared cards/transitions.
4. The submission decodes, lasts 60–180 seconds, and contains only 1x retained flight. Every omission is recorded in the timeline and visibly disclosed. Required event/response context remains present.
5. All A–J markers and IDs remain readable in every flight segment. The HUD shows correct active/visited states, C's priority update, and Scenario C's changed directed costs in sync with actual frames, within one output-frame interval of the represented source event.
6. A/B/C use matching frozen model hashes and geometry, and the delivery links physical success, routing-quality measurements, and video timing to the same retained evidence. Failures and prediction/actual distinctions are disclosed.
7. Japanese documentation, report, videos, timelines, and hashes are consistent. No three-point completion text, unlabeled slow playback, fictitious extra visits, or unsupported claim of global optimality remains in the new deliverables.
8. Viewers can identify each destination's priority from color plus numeric label and distinguish priority from target/visited state. The verified L/H example shows priorities, distances, both forced-first optimal remaining costs, policy cost, and the scope of the exact comparison. Captions distinguish optimal first action, optimal full policy route, and unverified optimality accurately.
9. Explanatory cards/panels preserve 1x flight timing and link to the actual logged decision. Displayed values and chart components reproduce the explanation artifact within declared rounding; no illustrative unit-test numbers are presented as measured ten-point results.

## Verification approach

Use short synthetic timestamp/frame fixtures solely for compositor tests: varied capture FPS, nonzero simulation starts, 50-to-30 FPS conversion, final partial intervals, forced final captures, event boundaries, and multiple cuts/cards. Include negative cases that reproduce the current fixed-120-second retiming, alter an encoded PTS, mismatch a run ID, or omit frame records. Synthetic fixtures are not flight evidence.

Add explanatory-display checks for a proven optimal first action with a suboptimal policy suffix, tied actions, incomplete exact solving, stale/mismatched snapshots, policy/oracle identity confusion, and priority updates. Confirm that each case yields the appropriate caption, not an unconditional global-optimum claim. Recompute displayed costs from complete orders independently of the compositor. Visually review the chosen L/H scene at final playback size for label/legend readability, color/state separation, cost-panel readability, and clear association with the actual departure.

Then validate the actual selected recordings with ffprobe frame timestamps and complete ffmpeg decode. Independently recompute segment durations, source correspondence, cut mappings, final mission coverage, and event alignment from original telemetry. Inspect frames at the first frame, every goal switch, before/after both events, every cut/transition, and the final inspection; also watch continuous event/response clips at normal playback speed. Verify Japanese glyphs, all ten labels, marker visibility, drone detail, and report pages. Record exact commands, statuses, output paths, measured duration/FPS, and failures in verification notes.

## Out of scope

Re-training policies for presentation alone, synthetic replacement of Genesis motion, video publication/course submission, interactive web dashboards, music/voiceover requirements, interpolated fake flight, and maintaining arbitrary playback speeds in the new demo delivery mode.
