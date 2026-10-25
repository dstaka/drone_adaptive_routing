"""Seven continuous C-only MuJoCo reference duration traces; no mission/learner.

Each trace has one explicit constraint-consistent reset and exactly33000 future
100Hz intervals. Settling is descriptive, never a reason to stop or reset.
The native RK4 implementation, motor callback and controller stay unchanged.
"""

import copy
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np

from .payload_cases import initial_state
from .payload_contract import Payload
from .payload_hybrid_contract import VIDEO_PROFILE, STRICT_PROFILE
from .payload_hybrid_controller import Controller, SENSOR_FIELDS
from .payload_hybrid_flight import ReferenceAdmission
from .payload_hybrid_runtime import SmoothReference
from .payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG
from .payload_solver_runtime import SolverRuntime
from .payload_solver_qualification import physical_errors
from .payload_solver_work import pack_stages, WIDTH, total_mechanical_energy
from .payload_solver_work_check import reconstruct_batch_work
from .util import sha256

SEED = 82630071
SCOPE = "CONTINUOUS_C_DURATION_NOT_MISSION"
_VERIFIED = object()
HISTORICAL_RECEIPT_SHA256 = "ea39f291737210984488911e830460d0810cd629809912f2569cf58acd091cb7"
RAW_MANIFEST_SHA256 = "d40d67855b46d56658eab4c097b57270cbc13b7b4d0f504c099139ac3c1d2fd1"
RAW_FILES, RAW_BYTES = 2669, 7723572007
SOURCE_NAMES = (
    "payload_hybrid_duration",
    "payload_hybrid_duration_audit",
    "payload_hybrid_flight",
    "payload_hybrid_controller",
    "payload_hybrid_runtime",
    "payload_hybrid_contract",
    "payload_hybrid_audit",
    "payload_hybrid_native",
    "payload_solver_mujoco",
    "payload_solver_runtime",
    "payload_solver_qualification",
    "payload_solver_reference",
    "payload_solver_work",
    "payload_solver_work_check",
    "payload_solver_trace",
    "payload_solver_validation",
    "payload_solver_initial_reset",
    "payload_solver_engine",
    "payload_ball_asset",
    "payload_ball_bank",
    "payload_contract",
    "payload_passive_check",
    "payload_cases",
    "payload_probe_controller",
    "payload_retry_contract",
    "util",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def duration_cases():
    result = []
    loads = [(0.045, 0.2)] + [(s, r) for s in (0.03, 0.045, 0.06) for r in (0.1, 0.3)]
    for index, (side, ratio) in enumerate(loads):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 330, index]))
        before = copy.deepcopy(rng.bit_generator.state)
        result.append(
            dict(
                case_id=f"duration330-{index:02d}",
                scope=SCOPE,
                seed=SEED,
                seed_domain=[SEED, 330, index],
                side_m=side,
                mass_ratio=ratio,
                initial=initial_state(rng, side),
                rng_before=before,
                rng_after=copy.deepcopy(rng.bit_generator.state),
                target=[0.0, 0.0, 1.0],
                target_id="S",
                duration_s=330.0,
                intervals=33000,
            )
        )
    return result


def validate_duration_case(case):
    if case not in duration_cases():
        raise ValueError("Exact prospective development-only330s duration case required")


def duration_declaration(cases=None):
    cases = duration_cases() if cases is None else cases
    if cases != duration_cases():
        raise ValueError("All seven exact prospective development cases required")
    return dict(
        schema="hybrid-mujoco-continuous-duration-declaration-v1",
        scope=SCOPE,
        cases=cases,
        intervals_per_trace=33000,
        native_samples_per_trace=33001,
        duration_s=330.0,
        reset_count_per_trace=1,
        dt_s=0.01,
        substeps=16,
        native_execution="independent SceneN1",
        backend="mujoco",
        precision="64",
        integrator="RK4",
        controller="C-local-lqr-v1",
        vibration_feedback=True,
        profile_sha256=VIDEO_PROFILE.sha256,
        strict_shadow_sha256=STRICT_PROFILE.sha256,
        leg_s=None,
        mission_wall_s=None,
        mission_simulated_deadline=None,
        targets_visited_claim=False,
        learning_admitted=False,
        termination="33000 actual intervals or first actual safety/runtime failure; no retry",
        native_replay_atol=1e-10,
        stage_replay_atol=1e-10,
        signed_work_residual_limit_j=0.02,
    )


def sources(source_root):
    root = Path(source_root)
    names = ["src/adaptive_drone_patrol/" + n + ".py" for n in SOURCE_NAMES]
    names += [
        "scripts/qualify_payload_hybrid_duration.py",
        "scripts/check_payload_hybrid_duration.py",
        "scripts/verify_payload_hybrid_duration_reference.py",
    ]
    return {name: sha256(root / name) for name in names}


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _historical_chain(rebinding, historical_receipt):
    record = json.loads(Path(rebinding).read_text())
    old = json.loads(Path(historical_receipt).read_text())
    manifest = old["raw_manifest"]
    if (
        sha256(historical_receipt) != HISTORICAL_RECEIPT_SHA256
        or record["historical_receipt_sha256"] != HISTORICAL_RECEIPT_SHA256
        or canonical_digest(manifest) != RAW_MANIFEST_SHA256
        or record["raw_manifest_sha256"] != RAW_MANIFEST_SHA256
        or record["raw_files"] != RAW_FILES
        or len(manifest) != RAW_FILES
        or record["raw_bytes"] != RAW_BYTES
        or sum(r["bytes"] for r in manifest.values()) != RAW_BYTES
        or old["actual_binding"] != record["actual_binding"]
    ):
        raise ValueError("Exact pinned historical receipt/raw manifest/binding required")
    return record, manifest


def verify_preserved_raw(rebinding, historical_receipt, locations):
    """Hash actual retained raw on its storage host, without importing an engine."""
    started = time.time()
    record, manifest = _historical_chain(rebinding, historical_receipt)
    files = {}
    for name, item in manifest.items():
        stage, relative = name.split("/", 1)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Historical raw path escaped its declared stage")
        path = Path(locations[stage]).resolve() / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Actual regular historical raw file required")
        before = path.stat()
        actual = sha256(path)
        after = path.stat()
        if (
            before.st_size != item["bytes"]
            or actual != item["sha256"]
            or (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValueError("Actual preserved historical raw bytes changed/differ")
        files[name] = dict(
            path=str(path),
            bytes=after.st_size,
            sha256=actual,
            inode=after.st_ino,
            device=after.st_dev,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
    receipt = dict(
        schema="hybrid-preserved-reference-raw-verification-v1",
        historical_receipt_sha256=HISTORICAL_RECEIPT_SHA256,
        raw_manifest_sha256=RAW_MANIFEST_SHA256,
        actual_binding=record["actual_binding"],
        raw_files=RAW_FILES,
        raw_bytes=RAW_BYTES,
        files=files,
        verification_started_unix_s=started,
        verification_ended_unix_s=time.time(),
        storage_host=platform.node(),
        verifier_pid=os.getpid(),
        verifier_source_sha256=sha256(__file__),
        verifier_executable_sha256=sha256(sys.executable),
        scope="ACTUAL_PRESERVED_RAW_BYTES_ONLY_NO_NEW_PHYSICS_QUALIFICATION",
        learning_admitted=False,
    )
    receipt["canonical_receipt_sha256"] = canonical_digest(receipt)
    return receipt


def verify_raw_receipt(path, expected_sha256, rebinding, historical_receipt):
    """Consume an explicitly pinned storage-host receipt; never a pass boolean."""
    record, manifest = _historical_chain(rebinding, historical_receipt)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or sha256(path) != expected_sha256:
        raise ValueError("Explicit exact storage-host raw receipt SHA256 required")
    receipt = json.loads(Path(path).read_text())
    unsigned = {k: v for k, v in receipt.items() if k != "canonical_receipt_sha256"}
    times = [receipt["verification_started_unix_s"], receipt["verification_ended_unix_s"]]
    if (
        receipt["schema"] != "hybrid-preserved-reference-raw-verification-v1"
        or receipt["canonical_receipt_sha256"] != canonical_digest(unsigned)
        or receipt["historical_receipt_sha256"] != HISTORICAL_RECEIPT_SHA256
        or receipt["raw_manifest_sha256"] != RAW_MANIFEST_SHA256
        or receipt["actual_binding"] != record["actual_binding"]
        or receipt["raw_files"] != RAW_FILES
        or receipt["raw_bytes"] != RAW_BYTES
        or set(receipt["files"]) != set(manifest)
        or not np.isfinite(times).all()
        or not 0 < times[0] <= times[1]
        or not isinstance(receipt["storage_host"], str)
        or not receipt["storage_host"]
        or type(receipt["verifier_pid"]) is not int
        or receipt["verifier_pid"] <= 0
        or receipt["verifier_source_sha256"] != sha256(__file__)
        or receipt["learning_admitted"] is not False
    ):
        raise ValueError("Complete canonical pinned actual storage-host verification required")
    for name, item in manifest.items():
        actual = receipt["files"][name]
        if (
            actual["bytes"] != item["bytes"]
            or actual["sha256"] != item["sha256"]
            or not Path(actual["path"]).is_absolute()
            or any(
                type(actual[key]) is not int or actual[key] < 0
                for key in ("bytes", "inode", "device", "mtime_ns", "ctime_ns")
            )
        ):
            raise ValueError("Canonical per-file actual raw receipt entry differs")
    return receipt


class DurationReference:
    """Existing reference capability plus every actual preserved historical raw."""

    def __init__(self, admission, proof, paths, token):
        if token is not _VERIFIED:
            raise ValueError("Actual historical raw verification required")
        self.admission, self.proof, self.paths = admission, proof, paths

    @classmethod
    def from_proof(cls, proof):
        return cls.from_files(
            proof["rebinding_path"],
            proof["historical_receipt_path"],
            proof["locations"],
            raw_verification=proof["raw_verification_path"],
            expected_receipt_sha256=proof["raw_verification_sha256"],
        )

    @classmethod
    def from_files(
        cls,
        rebinding,
        historical_receipt,
        locations=None,
        *,
        raw_verification=None,
        expected_receipt_sha256=None,
    ):
        record, manifest = _historical_chain(rebinding, historical_receipt)
        admission = ReferenceAdmission.from_rebinding(record)
        if raw_verification is None:
            if locations is None or expected_receipt_sha256 is not None:
                raise ValueError("Actual local raw mapping or explicitly pinned split-host receipt required")
            receipt = verify_preserved_raw(rebinding, historical_receipt, locations)
        else:
            if locations is not None:
                raise ValueError("Choose one explicit raw evidence host mode")
            receipt = verify_raw_receipt(
                raw_verification, expected_receipt_sha256, rebinding, historical_receipt
            )
        paths = {row["path"]: row["sha256"] for row in receipt["files"].values()}
        proof = dict(
            rebinding_path=str(Path(rebinding).resolve()),
            rebinding_sha256=sha256(rebinding),
            historical_receipt_path=str(Path(historical_receipt).resolve()),
            historical_receipt_sha256=sha256(historical_receipt),
            locations={k: str(Path(v).resolve()) for k, v in locations.items()}
            if locations is not None
            else None,
            raw_verification_path=str(Path(raw_verification).resolve())
            if raw_verification is not None
            else None,
            raw_verification_sha256=expected_receipt_sha256,
            raw_manifest_sha256=RAW_MANIFEST_SHA256,
            raw_files=RAW_FILES,
            raw_bytes=RAW_BYTES,
            actual_binding=admission.binding,
        )
        return cls(admission, proof, paths, _VERIFIED)

    def require_current(self):
        self.admission.require_current()
        for key in ("rebinding", "historical_receipt"):
            if sha256(self.proof[key + "_path"]) != self.proof[key + "_sha256"]:
                raise ValueError("Bound historical reference receipt changed")
        if self.proof["raw_verification_path"] is not None:
            verify_raw_receipt(
                self.proof["raw_verification_path"],
                self.proof["raw_verification_sha256"],
                self.proof["rebinding_path"],
                self.proof["historical_receipt_path"],
            )


class ObservedFixture(MuJoCoFixture):
    """Request native recording through the existing frozen callback API only."""

    def step_rpm(self, rpm, *, observe_stages=True):
        return super().step_rpm(rpm, observe_stages=True)


class StageWriter:
    def __init__(self, root, *, chunk_intervals=32):
        if type(chunk_intervals) is not int or not 1 <= chunk_intervals <= 128:
            raise ValueError("Bounded integer stage chunk size required")
        self.root, self.chunk_size = Path(root), chunk_intervals
        self.buffer, self.index, self.closed = [], [], False

    def append(self, stages, qacc):
        if self.closed:
            raise RuntimeError("Final stage stream is closed")
        if (
            stages.shape != (64, WIDTH)
            or qacc.shape != (64, 12)
            or stages.dtype != np.float64
            or qacc.dtype != np.float64
            or not np.isfinite(stages).all()
            or not np.isfinite(qacc).all()
        ):
            raise ValueError("Complete64 actual RK stages and pre-solve accelerations required")
        self.buffer.append((stages.copy(), qacc.copy()))
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        path = self.root / f"stages-{len(self.index):05d}.npz"
        np.savez_compressed(
            path,
            stages=np.stack([r[0] for r in self.buffer]),
            qacc_before_solve=np.stack([r[1] for r in self.buffer]),
        )
        self.index.append(
            dict(
                path=path.name,
                first_step=sum(r["intervals"] for r in self.index),
                intervals=len(self.buffer),
                sha256=sha256(path),
            )
        )
        self.buffer.clear()
        write_json(self.root / "stages-index.json", self.index)

    def close(self):
        self.flush()
        self.closed = True
        write_json(self.root / "stages-index.json", self.index)


def run_case(*, original, output, case, design, reference, source_root, clock=time.monotonic):
    validate_duration_case(case)
    if not isinstance(reference, DurationReference):
        raise ValueError("Verified actual historical reference capability required")
    reference.require_current()
    if design.get("recipe") != "C-local-lqr-v1" or design.get("design_load") != dict(
        side_m=0.045, mass_ratio=0.2
    ):
        raise ValueError("Frozen nominal C-local-lqr-v1 design required")
    original, root, source_root = (
        Path(original).resolve(),
        Path(output).resolve(),
        Path(source_root).resolve(),
    )
    if (
        source_root != Path(__file__).resolve().parents[2]
        or sha256(original) != reference.admission.binding["original_asset_sha256"]
    ):
        raise ValueError("Actual executed project/original asset binding differs")
    root.mkdir(parents=True, exist_ok=False)
    binding = sources(source_root)
    for name, digest in binding.items():
        destination = root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / name).read_bytes())
        if sha256(destination) != digest:
            raise ValueError("Source changed while snapshotting")
    write_json(root / "case.json", case)
    write_json(root / "design.json", design)
    write_json(root / "declaration.json", duration_declaration())
    write_json(root / "reference-proof.json", reference.proof)
    setup_started = clock()
    f = ObservedFixture(
        original, root / "fixture", Payload(FREE_MASS_KG, case["mass_ratio"], case["side_m"]), substeps=16
    )
    properties = f.runtime()
    if physical_errors(properties, dict(case, substeps=16), fixed=False):
        raise ValueError("Actual compiled qualified physical properties differ")
    write_json(root / "runtime-properties.json", properties)
    metadata = dict(
        scope=SCOPE,
        learning_admitted=False,
        original=str(original),
        original_sha256=sha256(original),
        source_binding=binding,
        reference_binding=reference.admission.binding,
        profile_sha256=VIDEO_PROFILE.sha256,
        strict_shadow_sha256=STRICT_PROFILE.sha256,
        pid=os.getpid(),
        executable=sys.executable,
        executable_sha256=sha256(sys.executable),
        setup_excluded_wall_s=clock() - setup_started,
        substeps=16,
        sample_order="preaction sensor projection; C tick; original RPM held16x4 RK stages; native; work; predicates; flush",
        observation_fields=list(SENSOR_FIELDS),
        optimizer=None,
        nn=None,
    )
    write_json(root / "metadata.json", metadata)
    core, stages = SolverRuntime(f, root / "runtime"), StageWriter(root)
    controller = Controller(design["gain"], design["hover_action"], vibration_feedback=True)
    control = gzip.open(root / "control.jsonl.gz", "wt")
    result = dict(
        case_id=case["case_id"],
        scope=SCOPE,
        success=False,
        failure=None,
        reset_count=1,
        learning_admitted=False,
        steps=0,
        native_samples=0,
    )
    started = clock()  # No compilation, model setup or source copying in duration timing.
    result["reset_started_monotonic_s"] = started
    work_total = loss_total = bound_total = 0.0
    try:
        core.reset_case(initial=case["initial"], target=case["target"], target_id="S")
        ref = SmoothReference(core.current_sample["drone_xyz"], case["target"])
        initial_energy = total_mechanical_energy(core.current_sample, properties, case["side_m"])
        from .payload_hybrid_audit import independent_metrics, interval_update

        metrics = independent_metrics(core.current_sample, case["target"], case["side_m"])
        live = (VIDEO_PROFILE.valid(metrics, contact=core.current_sample["contact"]), 0)
        strict = (STRICT_PROFILE.valid(metrics, contact=core.current_sample["contact"]), 0)
        initial_unsafe = bool(
            core.current_sample["contact"]
            or min(metrics["drone_floor_m"], metrics["package_floor_m"]) <= 0.1
            or max(abs(metrics["drone_roll_deg"]), abs(metrics["drone_pitch_deg"])) > 60
            or metrics["constraint_residual_m"] > 2e-5
        )
        if initial_unsafe:
            result["failure"] = "SAFETY_FAILURE"
        else:
            for tick in range(33000):
                position, velocity = ref.sample(tick * 0.01)
                action = controller.action(
                    {k: copy.deepcopy(core.current_sample[k]) for k in SENSOR_FIELDS},
                    position,
                    tick=tick,
                    reference_velocity=velocity,
                )
                before = clock()
                row = core.step(action)
                after = clock()
                packed = pack_stages(f.stage_rows)
                acc = np.asarray([r["qacc_before_solve"] for r in f.stage_rows], dtype=np.float64)
                stages.append(packed, acc)
                work = reconstruct_batch_work(
                    packed[None],
                    substeps=16,
                    first_step=tick,
                    total_mass_kg=sum(r["mass_kg"] for r in properties["links"]),
                )
                work_total += float(work["motor_work_j"][0])
                loss_total += float(work["dissipative_work_j"][0])
                bound_total += float(work["endpoint_absolute_work_bound_j"][0])
                energy = total_mechanical_energy(core.current_sample, properties, case["side_m"])
                metrics = independent_metrics(core.current_sample, case["target"], case["side_m"])
                live = interval_update(live, VIDEO_PROFILE.valid(metrics, contact=row["contact"]))
                strict = interval_update(strict, STRICT_PROFILE.valid(metrics, contact=row["contact"]))
                residual = energy - initial_energy - work_total - loss_total
                verdict = (
                    "SAFETY_FAILURE"
                    if row["unsafe"]
                    else "WORK_FAILURE"
                    if abs(residual) > 0.02 or energy - initial_energy > 0.02 + 1.1 * bound_total
                    else "DURATION_COMPLETE"
                    if tick == 32999
                    else "CONTINUE"
                )
                record = dict(
                    kind="STEP",
                    step=tick + 1,
                    time_s=(tick + 1) * 0.01,
                    target=case["target"],
                    target_id="S",
                    reference_position=position.tolist(),
                    reference_velocity=velocity.tolist(),
                    control=copy.deepcopy(controller.last_audit),
                    metrics=metrics,
                    dwell_s=live[1] * 0.01,
                    strict_dwell_s=strict[1] * 0.01,
                    energy_j=energy,
                    motor_work_j=work_total,
                    dissipative_work_j=loss_total,
                    absolute_work_bound_j=bound_total,
                    signed_work_residual_j=residual,
                    native_step_started_monotonic_s=before,
                    native_step_ended_monotonic_s=after,
                    reset_elapsed_wall_s=clock() - started,
                    verdict=verdict,
                )
                control.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                control.flush()
                if verdict != "CONTINUE":
                    result.update(
                        success=verdict == "DURATION_COMPLETE",
                        failure=None if verdict == "DURATION_COMPLETE" else verdict,
                    )
                    break
    except BaseException as exc:
        result.update(success=False, failure="RUNTIME_FAILURE", error=repr(exc))
        raise
    finally:
        core.close()
        control.close()
        stages.close()
        result.update(
            steps=core.step_count,
            native_samples=core.step_count + 1,
            simulated_s=core.step_count * 0.01,
            recorded_stage_intervals=sum(r["intervals"] for r in stages.index),
            final_flush_elapsed_wall_s=clock() - started,
            final_flush_complete=True,
            raw_sha256=sha256(root / "runtime/raw.jsonl.gz"),
            control_sha256=sha256(root / "control.jsonl.gz"),
            stages_index_sha256=sha256(root / "stages-index.json"),
            source_after=sources(source_root),
        )
        result["final_flush_elapsed_wall_s"] = clock() - started
        write_json(root / "result.json", result)
        finished = clock()
        write_json(
            root / "terminal-evidence.json",
            dict(
                result_sha256=sha256(root / "result.json"),
                evidence_complete_monotonic_s=finished,
                evidence_complete_elapsed_wall_s=finished - started,
                scope=SCOPE,
                ipc_claim=False,
            ),
        )
    return result
