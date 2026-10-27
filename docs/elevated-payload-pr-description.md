## Summary

The previous destination heights and detail camera did not make vertical transport sufficiently visible. Add a separate elevated profile using `z_new = 2 + 1.5 * (z_old - 1)` and move the native right-hand camera from 1.25 m to 0.90 m. Preserve x/y coordinates, physical assets, controller, reference dynamics, safety limits and simultaneous settling criteria. Legacy readers, merged specifications and artifacts remain available.

Implement the finite Masked PPO comparison in [the specification](../specs/2026-10-25-elevated-payload-destinations-detail-zoom/spec.md): compare the baseline with three independent weight-initialized candidates, retain terminal checkpoints, and select on 300 common validation cases using the prospective rule. Bind the selected F43 checkpoint, controller, geometry, camera, source and actual engine before opening either held-out cohort. No flight neural network is trained or used.

Independent audits reconstruct native/RK dynamics, work, oriented clearance, dwell, continuity and flush clocks. Final capture verifies all 127 physical input/audit inventories before and after rendering. Recovery rejects missing roots, nested declared cases and dangling RAM links.

## Validation

- Routing training: three runs of exactly 100,352 transitions, totaling 301,056. All four models complete all 300 validation cases without invalid actions. The fixed selection rule chooses F43: mean cost 344.406874 versus baseline 357.041243. Cross-environment inference matches all 1,200 orders and costs.
- Selected-model development: 15/15 missions and 12/12 vertical-validation cases with 120 distinct transfers. Complete raw re-audit precedes joint freeze.
- Fresh routing held-out: all 300 cases complete without invalid actions; F43 passes absolute quality gates. Mean cost 345.061333 versus baseline 355.530542, a 2.9447% reduction; 170 wins, 71 losses and 59 ties. No reselection.
- Separate physical acceptance: basic 9/9, endpoints 18/18 and fresh held-out 100/100. All 127 are single-pass and independently audited over 40,148,288 RK stages, with zero replay error. Minimum package/drone ground clearance is 0.608760/0.977117 m; maximum reset-through-flush wall time is 167.019 s.
- Three predeclared seed-1 final recordings: 14,526 native frames, full continuous 1×, with complete recomposition, decode, PTS, caption, source and asset checks. Detail enlargement is 41.50%; minimum assembly border is 91.43 px. Fresh full-length human review passes for all three exact video hashes.
- Frozen-source CPU regression: 1,226 passed, 20 skipped and one GPU-only test deselected (282.28 s); 37 focused tests passed.

## Evidence and limitations

The original decorative-marker check underestimates cylinder/box extents. Preserve that frozen source and its reports, and add a separate scene-based enclosure covering every style with `sqrt(3) * r` plus float-conversion error. It binds complete frame counts and accepted input hashes for all 15 development and three final recordings, and verifies at least 84.53 px of marker margin against the unchanged 40 px criterion. No camera or flight configuration changes after held-out exposure. Label readability and viewing comfort pass the separate user review.

The policy is not optimal: positive exact gaps remain at 385 of 1,270 physical decision boundaries. Random routing held-out includes 71 cost regressions and 19 harmful applicable event responses. Physical acceptance uses the inherited video-first simultaneous one-second settling rule. Transient box spin reaches 338.48°/s; stricter shadow settling passes 0/1,270 stops on the same trace. Preserve these diagnostics without claiming a strict-mode mission result or changing thresholds.

Large raw/stage/model/media assets remain outside Git in the dedicated run namespace. All 59,788 remote namespace files (36,851,003,632 bytes) were recovered and independently verified by size and SHA-256. A separate delivery inventory also covers local derived evidence and the human receipt. The [recovery receipt](evidence/elevated/recovery.json) and Japanese [results](elevated-payload-result-ja.md) and [handoff](elevated-payload-handoff-ja.md) provide delivery details. This change is prepared for a local commit only; no push, PR creation, merge or publication is performed.
