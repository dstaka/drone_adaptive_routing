"""Bounded SDK candidate benchmark; never qualifies or adopts the candidate."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from adaptive_drone_patrol.dual_coordination_native_audit import build_native_audit, CompiledAuditShadow
from adaptive_drone_patrol.dual_coordination_progress import atomic_json
from adaptive_drone_patrol.util import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--headers", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--rpm", type=float, default=15000.0)
    parser.add_argument("--compare-fast-path", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or not 1 <= args.repetitions <= 3:
        parser.error("Bounded benchmark requires1..100 steps and1..3 repetitions")
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="dual-native-candidate-benchmark-v1",
        production_qualified=False,
        source_model=dict(path=str(args.model.resolve()), sha256=sha256(args.model)),
        runtime_reconstruction="NativePointAuditor authored angular damping; XML initial state",
        point_acceleration_checks="live integration finite plus independent scratch forward finite",
        command_rpm=args.rpm,
        copy_to_fixture_included=False,
        audited_boundary_ledger_included=True,
        repetitions=[],
        status="running",
    )
    atomic_json(args.output / "benchmark.json", record)
    try:
        tree = ET.parse(args.model)
        assets = []
        for asset in tree.getroot().findall("asset/*"):
            filename = asset.get("file")
            if filename is None:
                continue
            path = Path(filename)
            if not path.is_absolute():
                path = args.model.parent / path
            if not path.is_file():
                choices = list(args.asset_root.rglob(Path(filename).name))
                if len(choices) != 1:
                    raise ValueError(f"Missing/ambiguous asset {filename}")
                path = choices[0]
            path = path.resolve()
            assets.append(dict(path=str(path), sha256=sha256(path)))
            asset.set("file", str(path))
        model = args.output / "model.xml"
        tree.write(model, encoding="utf-8", xml_declaration=True)
        record.update(assets=assets, rebased_model_sha256=sha256(model))
        library = build_native_audit(args.output / "build", headers=args.headers, library=args.library)
        record["build_sha256"] = sha256(library)
        for fast in [False, True] if args.compare_fast_path else [False]:
            with CompiledAuditShadow.from_xml(model, library) as shadow:
                initial = shadow.inspect()
                for repetition in range(args.repetitions):
                    shadow.reset_state(initial["qpos"], initial["qvel"])
                    counters = dict(
                        native_seconds=0.0,
                        interval_seconds=0.0,
                        rk_points=0,
                        native_intervals=0,
                        body_intervals=0,
                        certified_intervals=0,
                        python_fallback_intervals=0,
                        safety_flags=0,
                    )
                    start = time.perf_counter()
                    for step in range(args.steps):
                        result = shadow.step_audited(np.full((2, 4), args.rpm), fast_path=fast)
                        audit = result["interval_audit"]
                        counters["native_seconds"] += result["native_elapsed_seconds"]
                        counters["interval_seconds"] += audit["elapsed_seconds"]
                        counters["rk_points"] += len(result["rk_points"])
                        for key in (
                            "native_intervals",
                            "body_intervals",
                            "certified_intervals",
                            "python_fallback_intervals",
                        ):
                            counters[key] += audit[key]
                        for point in result["rk_points"] + result["boundaries"]:
                            counters["safety_flags"] |= point["flags"]
                    elapsed = time.perf_counter() - start
                    snapshot = shadow.snapshot()
                    state_digest = hashlib.sha256(
                        json.dumps(
                            {key: snapshot[key] for key in ("state", "warning_state", "last_rpm")},
                            sort_keys=True,
                            allow_nan=False,
                        ).encode()
                    ).hexdigest()
                    if record["repetitions"]:
                        baseline = record["repetitions"][0]
                        if baseline["terminal_integration_state_sha256"] != state_digest or any(
                            baseline[key] != counters[key]
                            for key in (
                                "rk_points",
                                "native_intervals",
                                "body_intervals",
                                "certified_intervals",
                                "safety_flags",
                            )
                        ):
                            raise ValueError(
                                "Candidate terminal integration state or safety classification differs"
                            )
                    record["repetitions"].append(
                        dict(
                            fast_path=fast,
                            repetition=repetition + 1,
                            control_steps=args.steps,
                            simulated_seconds=args.steps * 0.01,
                            wall_seconds=elapsed,
                            terminal_qpos=shadow.inspect()["qpos"],
                            terminal_integration_state_sha256=state_digest,
                            cumulative_audit=snapshot["audit"],
                            **counters,
                        )
                    )
                    atomic_json(args.output / "benchmark.json", record)
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(args.output / "benchmark.json", record)
        raise
    atomic_json(args.output / "benchmark.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
