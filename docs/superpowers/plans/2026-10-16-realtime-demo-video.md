# Realtime Demo Video Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans. The operator approved the specification and explicitly requested immediate execution in this session and one final local commit.

**Goal:** Deliver complete, independently verified 1x ten-goal videos and Japanese explanations from the retained physical recordings.

**Architecture:** Preserve original captures. Normalize their documented forced same-step captures into a timestamp-indexed VFR raw delivery, then compose CFR flight segments at unit simulation-time slope. Keep rendering, timing validation, and analytical cost verification separate.

**Tech Stack:** Existing local CPU Python environment, Pillow, FFmpeg/ffprobe, ReportLab, bundled Poppler.

**Spec:** [Approved specification](../../../specs/2026-10-14-realtime-demo-video/spec.md)

## Global Constraints

- Implement in the repository root; preserve personal untracked files and all previous artifacts.
- H.264 MP4, at least 1280x720, edited output at 25 or 30 FPS; flight playback_rate=1.0.
- Submission 60–180 seconds and authentic flight more than half its duration.
- Japanese viewer copy; English code, comments and docstrings.
- Use the frozen finetuned flight model and routing seed42; no training or GPU needed for editing.
- No push, PR creation, publication or submission. One local commit with the requested identity.

## Review Focus

- Repeated same-step captures are explained by explicit forced boundary captures, not silently accepted as new time.
- Off-cadence terminal frames represent completion with no invented flight interval.
- Checker reads encoded PTS and physical steps independently of the compositor's declared rate.
- Explanation snapshot, policy identity and exact solver proof must agree before optimality captions appear.
- Source metadata can contain private deployment paths; only sanitized relative references enter Git.

### Task 1: Preserve and validate inputs; timestamp contract

**Files:** Create `src/adaptive_drone_patrol/video_timing.py`, `tests/test_video_timing.py`.

**Interfaces:** `capture_schedule(frames, telemetry, events)` returns validated dt, unique capture mapping and mission coverage; `flight_samples(start, end, fps)` returns a timestamp-aware CFR schedule.

- [x] Record initial Git status, verify merged ancestors and all 465 delivery hashes, models and GPU ledgers.
- [x] Write and run failing fixtures for 50-to-30 FPS, nonzero starts, forced duplicates, partial ends, missing and malformed timing.
- [x] Implement strict timing validation and documented terminal-point sampling; run passing tests.

### Task 2: Normalize raw and render Japanese full/submission outputs

**Files:** Modify `video.py`, `scripts/build_video.py`, `configs/video.yaml`; create `video_render.py`, `video_delivery.py`, `tests/test_video_delivery.py`.

**Interfaces:** `compose(..., mode='submission', explanations=...)` builds a versioned timeline; full mode creates the three complete flights plus demo; submission preserves all footage when it fits.

- [x] Write failing tests for modes, duration/rate rejection, cuts/cards and snapshot-scoped captions.
- [x] Implement VFR normalized raw with original capture index provenance, isolated cards, all-goal annotations and dynamic tactical map.
- [x] Verify L/H components and physical provenance, event rollout comparisons, fixed palette and independent target/visited symbols.
- [x] Generate to a new ignored output directory; retain complete original inputs and all nine trial summaries.

### Task 3: Independent encoded-media acceptance

**Files:** Create `video_check.py`, `scripts/check_realtime_media.py`, `tests/test_video_check.py`; leave legacy acceptance separate.

**Interfaces:** `check_delivery(root)` probes/decodes actual media, reconstructs expected source time from original steps and compares per-frame encoded HUD/camera evidence and timeline.

- [x] Add failing corruption fixtures for accelerated timing, PTS alteration, cross-run references, omitted records and stale snapshots.
- [x] Independently recompute costs from complete orders and verify final coverage, every event and retained frame mapping.
- [x] Human playback review: operator reported videos visually OK on 2026-10-17, closing the previously pending item. Agent continuous playback remained blocked by browser policy; full decode and encoded-frame inspection are separate evidence.
- [x] Write per-output verification and manifest only after checks; preserve failures as development records.

### Task 4: Documentation, report, review and commit

**Files:** Update `README.md`, `docs/report_ja.md`, `scripts/build_report.py`, `specs/README.md`; create `docs/realtime-video-handoff.md`, `docs/realtime-video-pr-description.md`, compact `docs/evidence/realtime/` evidence.

- [x] Replace outdated current three-goal and retimed-video claims; regenerate PDF and inspect every rendered page.
- [x] Record paths, hashes, commands, measured results, limits and unchanged GPU cumulative usage.
- [x] Run CPU suite, lint, diff and privacy checks; request one independent whole-change review and resolve material findings.
- [x] Stage only this task's explicit paths and create one local commit as dstaka; verify final identity, tree and remaining personal changes.

## Execution ledger

- Initial base: `974274d`; merged ancestors `ff346a0`, `91cf985` confirmed.
- Input delivery: all 465 sizes and SHA-256 match; archive matches the supplied digest.
- Ruling: Work in the existing root checkout as explicitly requested. Do not create a second checkout or copy historical sources.
- Ruling: Original recorder emits cadence frames plus forced boundary captures, including same-step duplicates. Keep originals immutable; normalize by choosing the last capture at each step (latest event state), recording every removed duplicate. Reject duplicates that lack an event-sequence advance or documented boundary.
- Ruling: Use all three complete seed1 flights in submission; their combined physical duration is about 52 seconds, permitting readable cards while keeping flight above 50%. No omission is necessary.
- GPU ledgers: both record 2803.57274048205 seconds. This CPU-only delivery adds zero GPU time.

- Task 1: complete. Baseline123 CPU tests passed; all465 delivery hashes and explicit model hashes matched.
- Task 2: implemented. Development v1 retained; final v2 corrects pose-to-current-target distance at same-step target changes. Full flights plus cards fit96.30s without cuts.
- Task 3: automated checker implemented with original raw decoding, encoded PTS, raw/full-frame pixels, Japanese text and independent costs. The operator initially left human review pending, then reported visual approval on 2026-10-17; acceptance is now PASS.
- Task 4: Japanese report regenerated and rendered; final CPU142 passed,1 GPU deselected; final local commit is authorized, without push.
- Review: original raw trust, unverified captions/candidates, normalized raw pixels, manifest self-hash and weighted accounting findings fixed; focused reviewer found no further issues.
- Ruling: preserve all v1 output as development evidence and use a new `outputs/realtime_20261016_v2/` delivery rather than overwrite generated clips.
