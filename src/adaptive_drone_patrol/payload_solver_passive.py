"""Full newly bound passive32 with pre-transform raw, no powered or optimizer."""

import gzip
import json
import time
import traceback
from pathlib import Path
from collections import deque
import importlib.metadata as metadata
import numpy as np
from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG
from adaptive_drone_patrol.payload_contract import Payload
from adaptive_drone_patrol.payload_solver_qualification import (
    make_manifest,
    passive_cases,
    PHYSICAL_SOURCES,
    verify_passive_probe,
    require_passive_barrier,
)
from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.util import sha256


def record_passive_domain(output, original):
    """Record and independently verify every original passive case before powered."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)

    original = Path(original)
    manifest = make_manifest(base_substeps=16)
    write_json(out / "manifest.json", manifest)
    bindings = {p: sha256(p) for p in PHYSICAL_SOURCES}
    dist = metadata.distribution("mujoco")
    from .payload_solver_engine import engine_hashes

    engine = engine_hashes(dist)
    plan = dict(
        schema="solver-passive32-plan-v1",
        manifest=manifest,
        source_binding=bindings,
        source_driver_sha256=sha256(__file__),
        engine_source_and_build=engine,
        original_asset_sha256=sha256(original),
        lock_sha256=sha256("requirements/payload-reference-linux.lock.txt"),
        initial_state="Fixed original base xyz0,0,1; identity wxyz; zero velocities; declared upper ball displacement; lower identity.",
        input_schedule="Zero RPM and actual applied/passive generalized forces; no damping or contact.",
        thresholds=manifest["limits"],
        raw="Every control interval before transforms, guards or reset; initial/terminal included.",
        work="No applied or physical dissipative work in fixed conservative model. Signed residual is E-E0; both signs <=5% of positive E0.",
        case_timeout_s=600,
        physical_training_admitted=False,
    )
    write_json(out / "plan.json", plan)
    results = {}
    locations = {}
    for case in passive_cases(manifest):
        identifier = case["id"]
        run = out / identifier
        run.mkdir()
        assert all(sha256(p) == h for p, h in bindings.items())
        declaration = dict(
            schema="solver-case-declaration-v1",
            case=case,
            settings=manifest["settings"],
            source_binding=bindings,
            diagnostic_duration_s=None,
            plan_sha256=sha256(out / "plan.json"),
        )
        write_json(run / "declaration.json", declaration)
        f = MuJoCoFixture(
            original,
            run / "fixture",
            Payload(FREE_MASS_KG, case["mass_ratio"], case["side_m"]),
            fixed=True,
            substeps=case["substeps"],
        )
        f.reset(swing_deg=case["initial_swing_deg"], direction_rad=case["initial_direction_rad"])
        write_json(run / "runtime.json", f.runtime())
        recent = deque(maxlen=128)
        started = time.monotonic()
        summary = dict(status="RUNNING", physical_training_admitted=False, case_id=identifier)
        try:
            with gzip.open(run / "raw.jsonl.gz", "wt") as stream:
                for step in range(round(case["duration_s"] / 0.01) + 1):
                    row = f.raw_state()
                    row["step"] = step
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
                    recent.append(row)
                    if not np.isfinite(f.data.qpos).all() or not np.isfinite(f.data.qvel).all():
                        raise ValueError("NONFINITE_RAW")
                    if any(row["warning_counts"]):
                        raise ValueError("ENGINE_WARNING")
                    for body in ["drone", "rod", "package"]:
                        if (
                            np.linalg.norm(row[body + "_velocity"]) > 20
                            or np.linalg.norm(row[body + "_angular_velocity"]) > 200
                        ):
                            raise ValueError("FINITE_SPEED_EXCURSION")
                    if not np.isclose(row["time_s"], step * 0.01, atol=1e-7, rtol=0):
                        raise ValueError("INTERNAL_RESET_OR_TIME_DISCONTINUITY")
                    if time.monotonic() - started > 600:
                        raise TimeoutError("Case progress deadline")
                    if step == round(case["duration_s"] / 0.01):
                        break
                    f.step_rpm(np.zeros(4))
            summary["status"] = "RAW_COMPLETE_UNCHECKED"
        except BaseException as exc:
            summary.update(status="FAILED", error=repr(exc))
            try:
                actual = f.raw_state()
            except Exception as raw_exc:
                actual = {
                    "capture_error": repr(raw_exc),
                    "qpos": f.data.qpos.tolist(),
                    "qvel": f.data.qvel.tolist(),
                }
            (run / "abnormal-window.json").write_text(
                json.dumps(dict(preceding=list(recent), actual=actual), allow_nan=True) + "\n"
            )
            (run / "traceback.txt").write_text(traceback.format_exc())
            raise
        finally:
            summary.update(
                raw_sha256=sha256(run / "raw.jsonl.gz"), runtime_sha256=sha256(run / "runtime.json")
            )
            write_json(run / "summary.json", summary)
        check = verify_passive_probe(run, manifest)
        write_json(run / "independent-check.json", check)
        results[identifier] = check
        locations[identifier] = str(run)
        write_json(out / "checks.json", results)
        print(
            identifier,
            check["passed"],
            check.get("energy_balance", {}).get("maximum_absolute_unexplained_fraction"),
            check["failure_codes"],
            flush=True,
        )
        if not check["passed"]:
            raise SystemExit(2)
    checks = require_passive_barrier(manifest, locations)
    write_json(
        out / "barrier.json",
        dict(
            status="PASSIVE32_INDEPENDENTLY_PASSED",
            cases=checks,
            physical_training_admitted=False,
            remaining="powered30,motor15,reset30,work/source bindings",
        ),
    )
    return checks
