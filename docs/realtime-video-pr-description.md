Implement simulation-time 1x playback for the ten-goal physical patrol demo. The previous compositor fitted missions into fixed-duration slots; this change preserves natural flight duration, normalizes forced boundary captures without altering originals, and separates Japanese explanation cards from flight.

Specification: [Realtime demo video](specs/2026-10-14-realtime-demo-video/spec.md).

- Generate three complete scenario videos, a full demo, and a 96.30-second submission containing all selected seed1 flights without cuts. Authentic flight occupies 52.30 seconds (54.31%).
- Show A–J priorities, target/visited states, measured current-target distance, synchronized updates, and snapshot-bound L/H comparisons with Q*, J*, policy rollouts and component costs.
- Independently decode original and normalized raw footage and all edited frames; verify encoded PTS, physical timing, visible Japanese HUD/cards, source pixels, event alignment and terminal coverage. Preserve all nine physical trial summaries and failed development artifacts.
- Update Japanese documentation and regenerate the three-page report. Keep large media and models outside Git; retain local manifests and compact review evidence.

Validation: 142 CPU tests passed, one GPU test deselected; Ruff and diff checks passed. The inherited physical suite remains 9/9 missions and 90/90 inspections. No GPU use was added; cumulative usage remains 2803.57274048205 seconds. No model, geometry, camera or physical acceptance criteria changed.

Acceptance: the operator reported visual approval of the videos on 2026-10-17, closing the remaining human review item. Automated media checks and static frame/report reviews are recorded separately. Specification acceptance is PASS with no remaining items. Suggested improvements are deferred to a separate specification and PR.

Artifacts: `outputs/realtime_20261016_v2/`. Reproduction, hashes, measured results and limitations: [handoff](docs/realtime-video-handoff.md). No push, remote PR, publication or course submission was performed.
