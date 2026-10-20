#!/usr/bin/env python
"""Exercise full-domain transitions; retain every physical sample and failure."""

import argparse
import json
import math
import numpy as np
from adaptive_drone_patrol.util import load_config, unique_dir, json_write, sha256
from adaptive_drone_patrol.patrol_runner import ArrivalGate, FlightSample
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.scene import scene_manifest, project
from adaptive_drone_patrol.transition_check import execution_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--transitions", default="configs/dispersed_transitions.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--camera", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    suite = load_config(a.transitions)
    if a.seed not in suite["validation_seeds"] + suite["heldout_seeds"]:
        p.error("Use a predeclared validation or held-out seed")
    out = unique_dir(a.out)
    source_hashes = execution_sources()
    scene = None
    if a.camera:
        s = Scenario.from_dict(load_config(cfg["scenario_files"][0]))
        scene = scene_manifest(s)
        cfg["scene_manifest"] = scene
        json_write(out / "scene_manifest.json", scene)
    from adaptive_drone_patrol.flight_agent import FlightSession

    flight = FlightSession(cfg, seed=a.seed, camera=a.camera)
    if a.camera:
        from PIL import Image

        Image.fromarray(flight.cam.render()[0]).save(out / "initial.png")
        # Confirm pinhole implementation against real renderer camera matrices.
        for marker in scene["markers"]:
            xyz = np.array([*marker["world_xyz"], 1.0])
            q = flight.cam.intrinsics @ (flight.cam.extrinsics @ xyz)[:3]
            if not np.allclose(q[:2] / q[2], project(marker["world_xyz"], scene["camera"]), atol=1e-4):
                raise RuntimeError("Runtime camera matrix differs from scene projection")
    samples = []
    legs = []
    status = "SUCCEEDED"
    error = None
    step = 0
    try:
        for i, entry in enumerate(suite["targets"]):
            target = entry["xyz"]
            before = flight.sample()["xyz"]
            flight.set_target(target)
            gate = ArrivalGate(
                flight.dt,
                radius=suite["arrival_radius_m"],
                max_speed=suite["arrival_speed_mps"],
                hold_s=suite["arrival_hold_s"],
                service_s=suite["inspection_dwell_s"],
                floor=suite["floor_m"],
                max_tilt_deg=suite["max_tilt_deg"],
            )
            start = step
            leg_samples = []
            for _ in range(math.ceil(suite["max_leg_s"] / flight.dt)):
                r = flight.advance()
                step += 1
                residual = float(np.linalg.norm(np.array(r["xyz"]) - target))
                speed = float(np.linalg.norm(r["velocity"]))
                row = {
                    "step": step,
                    "target_index": i,
                    "target": target,
                    "residual_m": residual,
                    "speed_mps": speed,
                    **r,
                }
                samples.append(row)
                leg_samples.append(row)
                flags = gate.update(
                    FlightSample(r["xyz"], r["velocity"], r["done"], r["reset"], r["tilt_deg"]), target
                )
                if "WAYPOINT_REACHED" in flags:
                    legs.append(
                        {
                            "index": i,
                            "kind": entry["kind"],
                            "target": target,
                            "start_xyz": before,
                            "start_step": start,
                            "end_step": step,
                            "duration_s": (step - start) * flight.dt,
                            "min_altitude_m": min(x["xyz"][2] for x in leg_samples),
                            "max_tilt_deg": max(x["tilt_deg"] for x in leg_samples),
                            "saturated_samples": sum(
                                x["relative_observation_absmax"] >= 1 for x in leg_samples
                            ),
                            "max_speed_mps": max(x["speed_mps"] for x in leg_samples),
                        }
                    )
                    if a.camera:
                        Image.fromarray(flight.cam.render()[0]).save(out / f"hover_{i}.png")
                    break
            else:
                raise RuntimeError(f"Transition timeout at {i}:{entry['kind']}")
    except Exception as exc:
        status = "FAILED"
        error = repr(exc)
    finally:
        with (out / "samples.jsonl").open("x") as f:
            for row in samples:
                f.write(json.dumps(row) + "\n")
        json_write(
            out / "summary.json",
            {
                "status": status,
                "error": error,
                "execution_seed": a.seed,
                "split": "validation" if a.seed in suite["validation_seeds"] else "heldout",
                "legs": legs,
                "steps": step,
                "dt": flight.dt,
                "evidence_kind": "physical",
                "checkpoint_sha256": cfg["flight_checkpoint_hash"],
                "config_sha256": cfg["flight_cfg_hash"],
                "suite_sha256": sha256(a.transitions),
                "execution_sources": source_hashes,
                "hover_source_sha256": cfg["hover_source_hash"],
                "camera": a.camera,
            },
        )
    raise SystemExit(0 if status == "SUCCEEDED" else 2)


if __name__ == "__main__":
    main()
