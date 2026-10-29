"""One development open-loop comparison, not the required36-case qualification."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from adaptive_drone_patrol.dual_payload_mujoco import DualMuJoCoFixture
from adaptive_drone_patrol.payload_contract import Payload
from adaptive_drone_patrol.payload_solver_mujoco import FREE_MASS_KG, HOVER_RPM, KF
from adaptive_drone_patrol.dual_coordination_progress import RunJournal, atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--original", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=False)
j = RunJournal(root)
a = j.start(
    "open-loop-gpu-parity", "1s fixedmotor native vs32 duplicate GPU worlds", category="qualification"
)
result = dict(scope="DEVELOPMENT_PILOT_ONLY", qualified=False, production_transitions=0)
try:
    import mujoco_warp as mjw
    import warp as wp

    f = DualMuJoCoFixture(args.original, root / "fixture", [Payload(FREE_MASS_KG, 0.2, 0.045)] * 2)
    result["versions"] = {p: importlib.metadata.version(p) for p in ["mujoco", "mujoco-warp", "warp-lang"]}
    initial = f.sample()
    wp.init()
    then = time.monotonic()
    gm = mjw.put_model(f.model)
    gd = mjw.put_data(f.model, f.data, nworld=32, nconmax=64, njmax=256)
    rpm = np.full((2, 4), HOVER_RPM * np.sqrt(1.2 + 0.0001 / FREE_MASS_KG))
    ctrl = np.tile((KF * rpm**2).reshape(1, 8), (32, 1)).astype(np.float32)
    wp.copy(gd.ctrl, wp.array(ctrl, dtype=wp.float32, device="cuda:0"))
    # Warm compilation on a disposable copy, never advance the paired initial.
    warm = mjw.put_data(f.model, f.data, nworld=32, nconmax=64, njmax=256)
    mjw.step(gm, warm)
    wp.synchronize()
    result["initialization_warmup_seconds"] = time.monotonic() - then
    then = time.monotonic()
    with wp.ScopedCapture() as capture:
        for _ in range(16):
            mjw.step(gm, gd)
    result["capture_seconds"] = time.monotonic() - then
    errors = []
    native_seconds = 0.0
    gpu_seconds = 0.0
    for tick in range(100):
        then = time.monotonic()
        f.step_rpm(rpm)
        native_seconds += time.monotonic() - then
        then = time.monotonic()
        wp.capture_launch(capture.graph)
        wp.synchronize()
        gpu_seconds += time.monotonic() - then
        gq = gd.qpos.numpy()[0].astype(float)
        max_position = max(
            np.linalg.norm(
                gq[a["free_qpos"] : a["free_qpos"] + 3] - f.data.qpos[a["free_qpos"] : a["free_qpos"] + 3]
            )
            for a in f.addresses
        )
        quaternion_angle = 0.0
        for addr in f.addresses:
            for qstart in [addr["free_qpos"] + 3, addr["link_qpos"], addr["package_qpos"]]:
                q1 = gq[qstart : qstart + 4]
                q1 = q1 / np.linalg.norm(q1)
                q2 = f.data.qpos[qstart : qstart + 4]
                quaternion_angle = max(
                    quaternion_angle, float(np.degrees(2 * np.arccos(np.clip(abs(q1 @ q2), 0, 1))))
                )
        errors.append(
            dict(time_s=(tick + 1) * 0.01, position_m=float(max_position), angle_deg=quaternion_angle)
        )
    result.update(
        native_physics_seconds=native_seconds,
        gpu32_physics_seconds=gpu_seconds,
        max_position_error_m=max(e["position_m"] for e in errors),
        max_angle_error_deg=max(e["angle_deg"] for e in errors),
        max_first_second_gates_met=max(e["position_m"] for e in errors) <= 0.01
        and max(e["angle_deg"] for e in errors) <= 1,
        missing=[
            "36 paired mixed-load trials",
            "30s closed loop",
            "all-stage safety/numerics",
            "device resident controller/reference/mask",
            "4096+5epoch end-to-end benchmark",
        ],
    )
    atomic_json(
        root / "samples.json",
        dict(
            initial=initial, errors=errors, native_final=f.sample(), gpu_final_qpos=gd.qpos.numpy().tolist()
        ),
    )
    j.finish(a, status="done", reason="One open-loop development comparison only")
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
    j.finish(a, status="failed", reason=result["error"])
atomic_json(root / "result.json", result)
print(json.dumps(result, indent=2), flush=True)
raise SystemExit(1 if "error" in result else 0)
