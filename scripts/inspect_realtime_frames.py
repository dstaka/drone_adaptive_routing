#!/usr/bin/env python
"""Extract real encoded boundary frames for Japanese visual acceptance."""

import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw
from adaptive_drone_patrol.video import ffmpeg_executable
from adaptive_drone_patrol.util import json_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    a = p.parse_args()
    root = Path(a.root)
    out = root / "visual_qa"
    out.mkdir(exist_ok=False)
    selected = json.loads((root / "selection.json").read_text())["selected_runs"]
    records = []
    for run in selected:
        kind = run.rsplit("_seed_", 1)[0]
        clip = root / f"full_{kind}.mp4"
        events = [json.loads(x) for x in (root / "sources" / run / "events.jsonl").read_text().splitlines()]
        timeline = json.loads(clip.with_suffix(".timeline.json").read_text())
        fps = timeline["fps"]
        indices = {0: "first", timeline["frame_count"] - 1: "final"}
        for event in events:
            if event["type"] in ["TARGET_SELECTED", "WAYPOINT_REACHED", "PRIORITY_UPDATED", "COST_UPDATED"]:
                index = min(timeline["frame_count"] - 1, round(event["sim_time_s"] * fps))
                indices[index] = event["type"] + "-" + str(event["event_seq"])
                if event["type"] in ["PRIORITY_UPDATED", "COST_UPDATED"]:
                    for delta in [-2, -1, 1, 2]:
                        indices[max(0, index + delta)] = f"event-{event['event_seq']}-{delta:+}"
        images = []
        for index, label in sorted(indices.items()):
            path = out / f"{kind}-{index:04d}.jpg"
            subprocess.run(
                [
                    ffmpeg_executable(),
                    "-v",
                    "error",
                    "-i",
                    str(clip),
                    "-vf",
                    f"select=eq(n\,{index})",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(path),
                ],
                check=True,
            )
            images.append(path)
            records.append(
                {"clip": str(clip), "frame": index, "pts_s": index / fps, "label": label, "path": str(path)}
            )
        for page in range(0, len(images), 12):
            batch = images[page : page + 12]
            sheet = Image.new("RGB", (1280, 250 * ((len(batch) + 2) // 3)), "#102333")
            for j, path in enumerate(batch):
                x = (j % 3) * 426
                y = (j // 3) * 250
                sheet.paste(Image.open(path).resize((426, 240)), (x, y))
                ImageDraw.Draw(sheet).text((x + 5, y + 239), path.stem, fill="white")
            sheet.save(out / f"{kind}-contact-{page // 12 + 1}.jpg", quality=90)
    demo = root / "demo_full.mp4"
    tl = json.loads(demo.with_suffix(".timeline.json").read_text())
    for i, segment in enumerate(tl["segments"]):
        if segment["segment_type"] == "card":
            index = segment["output_frame_start"]
            path = out / f"card-{i:02d}.jpg"
            subprocess.run(
                [
                    ffmpeg_executable(),
                    "-v",
                    "error",
                    "-ss",
                    str(index / tl["fps"]),
                    "-i",
                    str(demo),
                    "-frames:v",
                    "1",
                    str(path),
                ],
                check=True,
            )
            records.append(
                {"clip": str(demo), "frame": index, "path": str(path), "label": segment["card"]["title"]}
            )
    json_write(out / "index.json", records)


if __name__ == "__main__":
    main()
