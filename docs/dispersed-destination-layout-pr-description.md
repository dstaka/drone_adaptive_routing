## Summary

Replace the rectangular, single-height ten-goal fixture with a reproducible, operator-reviewed 3D layout. Routing, physical targets, telemetry, markers and Japanese height labels share the same actual coordinates. Preserve integer priority seed 40029, alpha=1, beta=0.05 and priority normalization 100.

Specification: [Dispersed destination layout](../specs/2026-10-17-dispersed-destination-layout/spec.md)

The layout spans 5.061 × 3.411 × 1.828 m, passes sliding-triple row/column checks and has 0.865 m minimum separation. Retain every rejected candidate and historical artifact. Add independent geometry, declared-transition/source-integrity, frozen-input and encoded-height checks. Keep the adopted seed44 routing model after expanded-domain evaluation; train the accepted flight policy through four progressively wider 3D stages and freeze the resulting model/configuration before acceptance.

## Validation

- Deterministic coordinate/scenario/rejection replay and independent geometry checks pass.
- Expanded routing: 450/450 learned held-out episodes complete, invalid actions 0; all three models pass static quality gates. Adopted seed44 mean cost/exact ratio 1.078192.
- Transition validation11/12 and held-out31/32/33: all 45 legs pass independent sample/command/observation/safety/dwell checks, including observation saturation.
- New fixed physical seeds 1/2/3 × three scenarios: 9/9 missions, 90/90 inspections, unchanged arrival and safety thresholds. All 14,673 original frames decode; frozen inputs match. Independently recompute 90 decisions/675 complete orders, maximum component error1.819e-12.
- Local CPU suite: 196 passed. Pinned Linux physics/checker source: 196 passed. GPU contract: 1 passed. Final encoding is independently checked on every output frame. Lint, compile and whitespace checks pass.
- All five final outputs/11,361 frames pass independent decode, timing, source-pixel, visible-height and cost checks. New full 1x flights, integrated demo and 140.9-second submission preserve all flight intervals. Operator separately approved the spatial preview and reviewed all three new full flights plus both continuous event intervals.

## Failures and limits

The first CRF18 video failed the unchanged height-glyph pixel gate (4.194 > 4). Preserved it and regenerated CRF1/YUV420 outputs; the final operator review is bound to the new hashes.

Preserved the first expanded model's ground-clearance failure at validation step57, its checkpoint/configuration/samples/frame and the unsuccessful fixed-badge layout search. Retrained from the previously accepted model, without relaxing acceptance thresholds or replacing seeds. The adopted routing policy is not always optimal: the featured first action is optimal, but its full suffix has a 1.651246 cost gap. Deterministic simulation results do not establish disturbance robustness or real-aircraft safety.

See [Japanese handoff](dispersed-destination-layout-handoff.md) for artifact sizes/SHA-256, commands, model compatibility, failures, video verification, operator review and cumulative GPU usage (5779.474017 seconds of 28800). Models, videos and large logs stay outside Git. No push, remote PR, publication or course submission is performed by this change.
