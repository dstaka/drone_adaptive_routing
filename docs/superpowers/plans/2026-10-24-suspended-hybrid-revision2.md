# Suspended payload Revision 2 execution plan (historical)

> Superseded by the explicit2026-10-25 MuJoCo-only directive and [Revision3 completion plan](2026-10-25-suspended-mujoco-completion.md). Preserve this original plan as history; it does not authorize restarting Genesis, H, NN training or old execution helpers.

> **For agentic workers:** Use superpowers:executing-plans, task by task.

**Goal:** Deliver a verified ten-destination development video first, then the complete Revision 2 acceptance package.
**Architecture:** Preserve historical solver consumers. Add explicit versioned settling/clock contracts and a controller runtime using permitted ideal sensors. Rebind unchanged qualified physics before short controller diagnostics; independently reconstruct every new command stream.
**Tech Stack:** Python, NumPy, MuJoCo float64 RK4, Genesis GPU, existing routing actor and media pipeline.
**Spec:** [Revision 2](../../../specs/2026-10-23-suspended-payload-hybrid-genesis/spec.md).

## Global Constraints

- Working-tree Revision 2 is authoritative. Preserve old source, raw, checkpoints and failures; use `suspended_hybrid_20261024_r2a`.
- Unchanged geometry, damping, motors, coordinates and routing objective. No runtime true mass/size/inertia/seed inputs.
- `payload-settling-video-v1`: .20/.30/.20/.15 m or m/s, 5/15/30 degrees or degrees/s, 100 consecutive complete .01s intervals.
- No aggregate simulated mission deadline; leg30s, mission wall600s from reset through terminal enqueue; final flushing required.
- Qualification precedes new-backend powered runs; C0/C1/C2 and actual learning-runtime qualification precede any optimizer.
- Preserve held-out unopened until final freeze. Development videos use separate seeds and remain provisional.
- No push, PR, merge, publication or connection information in tracked files. One final local commit with requested identity.

## Review Focus

- Terminal success at exact leg boundary, safety precedence and nonmonotonic wall clocks.
- Measurement leakage through reference generator/controller state, not only NN inputs.
- Stationary relabeling and target-stream tampering; independent physical displacement checks.
- Legacy reader isolation and hidden 12,000-step cutoffs in new execution paths.
- Truncated raw, stale source receipts and video/backend mismatches.

### Task 1: Versioned contracts and independent audit primitives

**Files:** Create `src/adaptive_drone_patrol/payload_hybrid_contract.py`, `tests/test_payload_hybrid_contract.py`.
**Interfaces:** Produces `SettlingProfile`, `VIDEO_PROFILE`, `STRICT_PROFILE`, `MissionClock`, explicit acceptance denominators and capture admission checks. Consumes existing SI metrics and consecutive-interval dwell logic.

- [x] Write failing tests for boundaries, strict shadow isolation, >120s continuation, exact leg completion, wall timeout and branch denominators/admission.
- [x] Run `.venv-routing311/bin/python -m pytest tests/test_payload_hybrid_contract.py -q`; expected missing module failure.
- [x] Implement immutable canonical profiles and clock state with success/safety precedence; reject nonfinite/inconsistent input.
- [x] Repeat tests; expected all pass. Preserve old defaults and raw verdicts.

### Task 2: Rebind physical reference and controller development

**Files:** Create controller/runtime/diagnostic modules, command manifest, independent raw checker and related tests.
**Interfaces:** Consumes Task 1; produces full trace, native input/state records, gate summaries, runtime/source hashes. Controller receives explicitly filtered measurements only.

- [x] Reconcile current source/asset/dependency hashes with complete all62/motor15/reset30 evidence. New environment differences block inherited qualification until resolved.
- [x] Predeclare keyed development starts, offsets and maximum three controller hypotheses. Test information boundaries, reset, zero correction and motor mapping before use.
- [x] Run C0 first, then C1, C2 with parameters frozen; compare feedback on/off and refinement. Preserve failures and record new hypotheses before retry.
- [x] Run first complete normal development mission when justified. Audit all raw and actual route boundaries independently; record and deliver first successful full 1x Japanese-HUD video immediately.

### Task 3: Genesis qualification and actual parallelism

**Interfaces:** Consumes fixed physical model/reference; produces candidate/repair/fallback decision and actual batch qualification.

- [ ] Inspect installed source; predeclare finite supported precision/integration/substep candidates.
- [ ] Execute passive32, then powered18/refinement12/motor15/reset30/batch isolation for passing candidates. Measure actual N1/8/32/128 where feasible, rather than serial scenes presented as parallel.
- [ ] Repeat C0–C2 and feedback/refinement on selected runtime. Record justified fallback/exception only with required evidence.

### Task 4: Finite comparison and final acceptance

**Interfaces:** Consumes qualified runtime and passing gates; produces frozen selected branch and acceptance/capture evidence.

- [ ] Predeclare 12-mission paired development suite and at most two H recipes, equal 300k transitions per method, checkpoints100k/300k.
- [ ] Apply eligibility/time/safety selection rules from spec. For NN, fresh fair paired final learning; for C, mark flight learning not applicable.
- [ ] Execute seven continuous330s powered checks, branch validation12/120, frozen acceptance127 or254, exact routing gaps and three final captures. Request fresh human 1x review; never mark pending review complete.

### Task 5: Recovery, review and local commit

- [ ] Recover required artifacts; independently verify each size/SHA-256 and final flush.
- [ ] Run focused and broad applicable tests; inspect actual results. Separate software verification, physical qualification, flight and media verdicts.
- [ ] Prepare Japanese report/handoff, compact evidence and English PR description linking spec. Record every unachieved requirement.
- [ ] Review whole diff, fix material findings with regression tests. Stage only relevant files and commit as dstaka; do not push.

## Execution rulings

- User explicitly authorized continuous execution and supplied the design/spec; omit further brainstorming approval and intermediate commits. Work in the named current checkout to preserve its uncommitted Revision 2; do not create a HEAD-based copy that loses it.
- All later experiment tasks are conditional on measured prerequisites. A failed prerequisite remains a failure, not permission to bypass it.
