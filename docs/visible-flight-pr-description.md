Execute ten-destination Genesis patrols with persistent noncolliding landmarks, numeric priority badges, synchronized target/visited/emergency/cost displays, and a genuine same-frame vehicle detail view. The mission runner supports configured destination IDs and confirms inspections from continuous physical pose/velocity telemetry.

Specification: [Visible multigoal flight](specs/2026-10-14-visible-multigoal-flight/spec.md). Results, artifact hashes and reproduction: [Handoff](docs/visible-flight-handoff.md).

The original motor checkpoint failed the diagnosed long-retarget safety probe. Fine-tuning preserved the motor architecture, target envelope, physics and safety limits; the selected checkpoint passed long transitions, reversals and repeated hover. Routing checkpoint bytes and all three scenario geometries remain unchanged.

Validation:
- 123 CPU tests and one real GPU contract test passed.
- The predeclared normal/emergency/cost_shift × seeds 1/2/3 suite completed 9/9 missions and 90/90 physical inspections, with no resets, falls or timeouts.
- All 7995 raw frames and encoded PTS were checked. Three paired annotated recordings preserve source timing and priority-event synchronization.
- All 90 physical decision explanations were regenerated; 675 complete-order cost replays matched independently.
- Cumulative GPU use, including inherited history and failed attempts: 2803.573/28800 seconds.

Development failures, full logs, checkpoints and raw media are retained outside Git and covered by the verified local transfer manifest. This is deterministic demonstration evidence. Final editing, retiming and submission composition remain for the subsequent realtime-demo-video PR.
