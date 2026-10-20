"""Deterministic offline Japanese HUD composition from genuine camera frames."""

import json
from pathlib import Path
import shutil


def ffmpeg_executable():
    available = shutil.which("ffmpeg")
    if available:
        return available
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def select_runs(root):
    root = Path(root)
    candidates = []
    for path in root.glob("*/mission_summary.json"):
        s = json.loads(path.read_text())
        if s["status"] == "SUCCEEDED" and (path.parent / "raw.mp4").is_file():
            candidates.append((path.parent, s))
    # Select the lowest paired execution seed with three complete comparable runs.
    for seed in sorted({s.get("execution_seed") for _, s in candidates}):
        selected = {s["scenario_id"]: p for p, s in candidates if s.get("execution_seed") == seed}
        if all(name in selected for name in ["normal", "emergency", "cost_shift"]):
            return [selected[name] for name in ["normal", "emergency", "cost_shift"]]
    raise RuntimeError("No paired seed has three authentic complete patrols")


def capture_sample_index(telemetry, frame):
    for index, sample in enumerate(telemetry):
        if int(sample["simulation_step"]) == int(frame["simulation_step"]):
            if float(sample["sim_time_s"]) != float(frame["sim_time_s"]):
                raise ValueError("Camera frame/telemetry timestamp mismatch")
            return index
    raise ValueError("Camera frame has no same-step telemetry")


def compose(root, output, font_path, duration_s=None, fps=30, *, mode="submission", explanations=None):
    """Build explicit 1x delivery; legacy timelines remain read-only evidence."""
    from .video_delivery import compose_delivery

    return compose_delivery(root, output, font_path, mode, fps, duration_s, explanations)
