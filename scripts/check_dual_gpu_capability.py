"""Installed-engine capability probe; no qualification or production learning."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
result = {"scope": "CAPABILITY_ONLY", "qualification": False, "production_transitions": 0}
try:
    import mujoco as mj
    import mujoco_warp as mjw
    import warp as wp
    import torch

    result["versions"] = {
        p: importlib.metadata.version(p) for p in ["mujoco", "mujoco-warp", "warp-lang", "torch"]
    }
    result["cuda"] = torch.cuda.is_available()
    result["gpu"] = torch.cuda.get_device_name()
    wp.init()
    model = mj.MjModel.from_xml_path(args.model)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    result["native_profile"] = {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "integrator": int(model.opt.integrator),
        "timestep": model.opt.timestep,
    }
    then = time.monotonic()
    gpu_model = mjw.put_model(model)
    gpu_data = mjw.put_data(model, data, nworld=32, nconmax=64, njmax=256)
    wp.synchronize()
    result["model_data_transfer_seconds"] = time.monotonic() - then
    then = time.monotonic()
    mjw.step(gpu_model, gpu_data)
    wp.synchronize()
    result["first_step_compile_seconds"] = time.monotonic() - then
    result["qpos_dtype"] = str(gpu_data.qpos.dtype)
    result["qpos_shape"] = list(gpu_data.qpos.shape)
    result["step_succeeded"] = True
except Exception as exc:
    result["step_succeeded"] = False
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)
result["elapsed_seconds"] = time.monotonic() - started
(root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
raise SystemExit(0 if result["step_succeeded"] else 1)
