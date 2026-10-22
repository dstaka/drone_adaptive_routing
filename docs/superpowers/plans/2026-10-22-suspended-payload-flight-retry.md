# Suspended Payload Flight Retry Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution; independent CPU forensics and inherited-contract audits use parallel agents. User explicitly authorizes execution without another plan approval, implementation in this root, and one final local commit.

**Goal:** 運用域の物理qualificationを先行し、成立時だけ公平な学習・254評価・全編動画へ進む。

**Architecture:** failed Euler profileを保持し、native ball MJCF候補を別module/profileへ隔離する。probe宣言、raw測定、独立checker、source/settings/signature bindingを分離する。失敗時もCPU検証、失敗記録、文書、commitまで保存する。

**Tech Stack:** Python 3.12 on Linux, Genesis 1.4.3, existing flight/routing locks, pytest, Japanese ReportLab PDF.

**Spec:** [retry spec](../../../specs/2026-10-21-suspended-payload-flight-retry/spec.md)

## Global Constraints

- 0.30 m compression-capable passive fixed link, underside offset (0,0,-0.020), real cube with unrestricted orientation and reciprocal reaction. No direct load control or damping substitution.
- Preserve exact dispersed coordinates, model4296/cfg, seed44 routing, priority integers and event definitions. Never resume failed model2000.
- Experimental diagnostics only until complete independently verified qualification. Old blocked profile remains blocked.
- Qualification: 9 loads × >=120 s anchored; large angles; 18 powered × >=120 s; 6 endpoints × matched 30 s dt/half-dt; >=30 resets; rotor/root/properties/geometry/reaction audits.
- Constraint <=2e-5 m; properties relative <=1e-5; extent <=1e-5 m; passive positive energy <=.05; period relative <=.10; half-dt <=.02 m/2 degrees. Floor >.10 m, tilt <=60 degrees in powered probes.
- Both policies seed7/model4296, matched 300000 env steps. Before either run, measured throughput must preserve full evaluation/recording reserves.
- 12 validation cases/120 transitions, 127 paired cases per policy, 254 total. No reduced denominator. Videos basic seed1, nominal .20/.045, fresh review.
- Budget starts used7484.093172315159/remaining21315.90682768484 s unless actual ledger newer. Phase ceiling3000 s physics, 3500+3500 training,1500 validation,6500 acceptance,2000 recording,500 final verification,815.90682768484 reserve.
- One bounded GPU job at a time; fresh isolated Python3.12 flight and routing environments. No dependency changes, rentals, push/PR/publication.
- All explanatory docs Japanese; code/comments English. Sanitize connection, local user paths and secrets. One new commit, no amend, identity dstaka.

## Review Focus

- Old PASS or hand-edited status never admits a failed signature.
- Missing final sample or duration and unexplained finite energy/speed excursions block qualification.
- MJCF importer may change original epsilon masses, base semantics, damping and rotor frames; measure before policy reuse.
- Ball-joint support in a parser does not prove stepping stability or fix the old failure.
- Server/environment failure must leave all downstream gates unattempted and budget truthful.

### Task 1: Preserve and reconstruct inputs

**Files:** new `docs/evidence/suspended-retry/{forensics,inputs,inherited-contracts}.json`; ignored unique output root.
**Interfaces:** consumes existing raw/archive/source/locks/ledgers; produces verified inputs, historical failures and compatibility obligations.
- [x] Preserve status, staged and unstaged diffs, spec/instruction originals; verify input/archive/raw hashes and executed/current source differences.
- [x] Read all required predecessors and actual cfgs; run existing raw failure checker and baseline CPU suite.

### Task 2: New candidate and fail-closed qualification

**Files:** create `src/adaptive_drone_patrol/payload_ball_asset.py`, `payload_qualification.py`, `configs/payload_retry_probe_manifest.json`, `tests/test_payload_retry.py`, and new probe/check scripts.
**Interfaces:** asset writer consumes pinned original vehicle properties; produces passive two-ball articulation with rotor frames retained. Qualification consumes declaration+raw records and produces signature-bound diagnostics/receipt; cannot admit partial evidence.
- [x] Write and run failing CPU tests for passive ball geometry and old/stale/incomplete/nonfinite/terminal sample rejection.
- [x] Implement minimal asset and declaration/checking APIs; preserve legacy paths and blocked config.
- [x] Inspect actual pinned importer/solver; declare sources, settings, cases, thresholds, timeout and energy observability before executing.
- [x] Build fresh remote environment only when construction/stepping needed, verify both locks and archive actual source/settings.

### Task 3: Physical operating-domain qualification

**Files:** new ignored output raw and compact `docs/evidence/suspended-retry/qualification.json`.
**Interfaces:** consumes Task2 declarations; produces independently checked receipt or precise blocked result.
- [x] Native ball construction/stepping and rotor/root/mass/inertia/shape/DOF/reaction audit; if unavailable, inspect supported equivalent constraint without changing model.
- [ ] Complete anchored, large-angle, powered, refinement, reset and clearance cases in manifest; save pre-abnormality and offending states before clipping/reset.
- [x] Reconstruct all actual raw/signatures and run independent checker. Do not change thresholds after observing data.
- [x] On unresolved failure or phase budget/server unavailability, stop dependent GPU work, retain candidates/failures/next proposal, continue Tasks5/6.

### Task 4: Conditional learning, validation, evaluation and media

**Files:** versioned retry observation/action contract and case wrapper; training/evaluation/freeze/media integration; compact results.
**Interfaces:** requires Task3 complete new admission bound to actual source; otherwise all items BLOCKED.
- [x] Verify 29-field semantics/action history; create new contract if changed; matched explicit17→29 transfer.
- [ ] Verify throughput and freeze matched budgets before optimizer; train both, save terminal state/reward before reset and abort abnormalities.
- [ ] Independent 12-case/120-transition validation then freeze; preserve old physical case draws in new metadata wrapper.
- [ ] Execute all254, reconstruct real routing decisions/costs and Wilson95% subgroup outcomes.
- [ ] Produce complete seed1 raw/HUD3 videos, decode every frame, verify timing/pixels/telemetry, request final-hash human continuous1x review.

### Task 5: Documentation and regression

**Files:** `README.md`, `docs/report_ja.md`, `scripts/build_report.py` if needed, Japanese PDF output, new `docs/suspended-payload-flight-retry-handoff.md`, PR description, spec index and compact evidence.
**Interfaces:** consumes actual results only; no historical PASS reused.
- [x] Run targeted CPU tests and affected full CPU regressions; affected GPU predecessor gates require fresh evidence or explicit unverified status.
- [x] Record each gate, all attempts, missing goals, cumulative GPU/transfer integrity, exact commands and concrete next proposal.
- [x] Regenerate affected PDF and render/inspect every page.

### Task 6: Review and one local commit

**Files:** explicit stage allowlist only.
**Interfaces:** consumes Tasks1–5 results; produces local commit and English PR description linked to retry spec.
- [ ] Fresh independent whole-change review; fix material findings and rerun relevant checks.
- [ ] Verify staged content excludes ignored outputs, historical copy/personal files, secrets, user paths and SSH details.
- [ ] Include existing new spec/public instructions/index; create one local commit using per-command author/committer dstaka <22057509+dstaka@users.noreply.github.com>. No push/amend/remote PR.

## Execution record

2026-10-22: HEAD2d0e003, branch feature/suspended-payload-flight-retry. User root preference overrides worktree creation and user one-commit requirement overrides per-task commits. Existing dirty files are protected. Plan self-review covers all spec sections; conditional downstream tasks remain blocked unless qualification succeeds.

2026-10-22 review correction: V8 initial swing azimuth was +Y although declaration0means+X. Stopped owned driver, retained five completed and one interrupted anchor raw; no V8 long receipt admits learning. V9 fixes axis with27 CPU analytical checks and reruns source-bound motor/long cases. Same threshold/case draws retained. Physics phase ceiling increased3000→3700s, reserve815.90682768484→115.90682768484s; both3500s training,1500s validation,6500s254acceptance,2000s recording,500s final unchanged. Reset proof remains reusable only because its exact six bound producer/backend/dependency files and manifest are unchanged. Review also requires retry-only raw finite guards and every-step training trace before any optimizer.

2026-10-22 energy evidence correction: endpoint wrench/twist is insufficient to bound hidden rotational work. Stopped V9 domain after six completed anchors and one interrupted anchor; preserved all raw/ledger. Added read-only per-actual-substep qpos/vel_next/world-wrench evidence, declared the conservative root-path work bound before long powered data, preserving the original .02J+1.1 endpoint guard as a separate mandatory screen. V10 reruns all source/manifest-bound motor, reset and55domain cases. Cumulative before V10 is8949.565231329412s (122jobs), session physics used1465.472059014253s; remaining physics2234.527940985747s within3700s ceiling. Planned actual motor~150s/reset~63s/domain<=2000s; unchanged complete learning/evaluation/recording reserves. No optimizer has run.


## Final stopping-state record (2026-10-22)

- All55 frozenV10 operating-domain conditions attempted once. Runner47completed/8floorfailures; powered12/18 reached120s, refinement10/12 reached30s. All6matched dt/halfdt pairs exceed.02m/2deg;2pairs also incomplete. Completion of attempted diagnostics is not qualification/admission.
- Independent aggregate:33/55 individualpass;14/16largeangle passiveenergyfail despite runnercompletion.9smallangle120s anchorspass. PassiveCOMcentraldifference crosscheck corroborates largeenergyincrease;rootcauseunresolved. Nextproposalprioritizes motor0anchor refinement before sameRPMpowered diagnosis.
- Motor15 and actualreset30 independently pass within their scope. Fullraw aggregate receipt is saved in compact qualification evidence; training stays blocked. Oldfailedprofile and all oldreceipts unchanged.
- Currentpostfailure vector-norm guard and training source-list fixes are CPU-only and not physical-repair evidence. SourceV10 and currentCPU source are separately hash-bound.
- Matchedlearning, validation,254acceptance, nativevalidation/media runtime integration, final3videos and freshhumanreview remain BLOCKED/unfinished. No optimizer or heldoutcase opened.
- Bothlocal+isolatedlockedLinux:547CPU tests pass,1GPUdeselected;188source hashes equal. Initial independent review found3P1 defects and fixes were verified; no second independent whole-change agent review after those corrections was available. This limitation remains explicit.
- Japanese report/PDF12pages all rendered and visually inspected; allraw transferred and1631memberhashes verified. CumulativeGPU10824.12788819254s/remaining17975.87211180746s;155sessionjobs/3340.034715877381s.53executablepath arguments normalized in publicledger only; timing/status/order unchanged and raw kept outsideGit.
- Failure handoff and concrete same-RPM refinement/control diagnosis proposal complete. Final explicit stage/privacy audit and singlelocalcommit close the authorized failure path; deployment/push/remotePR/publication remain prohibited.
