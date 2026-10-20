# Implement integer priorities with compatible learned routing and new physical evidence

The patrol used fractional importance weights and a checkpoint contract tied to a 1–5 scale. This change adds an explicit scenario v3 / `integer-100-v1` profile with strict integer weights and labels from 1 to 100, reproducible independent uniform draws with replacement, priority normalization by 100, and the frozen objective `travel + 0.05 * priority_completion_time`. Directed multipliers remain 1–5 and destination geometry is unchanged. Legacy artifacts retain their original interpretation.

New routing checkpoints and teacher/evaluation data carry the profile contract. Three models were trained and selected using validation only; seed44 was frozen before held-out evaluation. The selected demo uses C43→100 after the first physical inspection and a separate H→F multiplier1→5 event. Exact replay, policy rollout, weighted components, fixed palettes and Japanese captions use the same contract, including the >=10-point L/H requirement.

Specification: [Intuitive Integer Priorities](../specs/2026-10-17-integer-priorities/spec.md).
Implementation and rerun details: [Japanese handoff](integer-priorities-handoff.md).

Validation:
- Local and locked Linux CPU suites: 177 passed each; GPU contract test: 1 passed. Lint and whitespace checks pass.
- New held-out evaluation:100 static and 50 dynamic cases per training seed; 450 learned episodes complete, zero invalid actions. All three static quality gates pass; selected mean learned/exact ratio 1.040070.
- New fixed physical suite: 9/9 missions and 90/90 inspections; 90 physical explanations, 33 qualifying L/H decisions, 675 independently replayed complete orders. Prior footage is not used as new evidence.
- The operator confirmed all three complete clips and both event-through-next-inspection intervals at 1x with no issues. Full encoded-media and PDF checks are recorded in the handoff.

Large artifacts remain outside Git. Only this specification, its index entry, implementation, tests, documentation and compact evidence are included. Dispersed layout, suspended payload and unrelated local files are excluded. GPU usage continues the inherited 8-hour ledger: 3198.994308 seconds used, 25601.005692 seconds remaining. No push, PR creation/merge, public video upload or course submission was performed.
