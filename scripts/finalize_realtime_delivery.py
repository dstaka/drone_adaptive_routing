#!/usr/bin/env python
"""Publish local metadata only after independent encoded-media acceptance."""

import argparse
import csv
import json
from pathlib import Path
from adaptive_drone_patrol.util import json_write, sha256
from adaptive_drone_patrol.video_check import artifact_entries
from adaptive_drone_patrol.video_delivery import delivery_execution_metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--visual-notes", required=True)
    p.add_argument("--execution-context", help="Verified new execution/transfer metadata; omit for legacy CPU edits")
    a = p.parse_args()
    root = Path(a.root)
    result = json.loads((root / "verification.json").read_text())
    notes = json.loads(Path(a.visual_notes).read_text())
    if result["status"] != "PASS" or notes["frame_review"] != "PASS" or notes["report_review"] != "PASS":
        raise ValueError("Independent media, actual frames and PDF review must precede finalization")
    if not (root / "report/report.pdf").is_file():
        raise ValueError("Missing current report")
    for record in result["outputs"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Video changed after verification")
    for raw in result["raw"]:
        run = raw["run_id"]
        dst = root / "raw" / run
        schedule = json.loads((dst / "timing.json").read_text())
        json_write(
            dst / "raw.timeline.json",
            {
                "schema_version": 2,
                "kind": "raw_vfr",
                "segments": [
                    {
                        "segment_type": "flight",
                        "run_id": run,
                        "source_frame_start": schedule["indices"][0],
                        "source_frame_end": schedule["indices"][-1],
                        "source_simulation_start_s": schedule["start"],
                        "source_simulation_end_s": schedule["end"],
                        "presentation_start_s": 0,
                        "presentation_end_s": raw["physical_end_s"] - schedule["start"],
                        "playback_rate": 1,
                        "source_frame_map": str(dst / "frames.csv"),
                        "cuts": [],
                    }
                ],
                "terminal_codec_packet_s": schedule["terminal_packet_s"],
                "boundary_convention": schedule["boundary_convention"],
            },
        )
        json_write(
            dst / "raw.selection.json",
            {
                "schema_version": 2,
                "run_id": run,
                "selection": str(root / "selection.json"),
                "complete_physical_logs": str(root / "sources" / run),
                "original_capture_index": str(root / "sources" / run / "frames.csv"),
            },
        )
        with (root / "sources" / run / "frames.csv").open() as stream:
            original_frames = csv.DictReader(stream)
            first, second = next(original_frames), next(original_frames)
        scene = json.loads((root / "sources" / run / "scene_manifest.json").read_text())
        json_write(
            dst / "raw.verification.json",
            dict(
                raw,
                sha256=sha256(dst / "raw.mp4"),
                source_video_metadata={
                    "resolution_px": scene["camera"]["res"],
                    "encoded_nominal_fps": 1 / (float(second["pts_s"]) - float(first["pts_s"])),
                    "physical_cadence_fps": 1 / (schedule["dt"] * schedule["cadence_steps"]),
                },
            ),
        )
    for clip in [*root.glob("*.mp4"), *root.glob("raw/*/raw.mp4")]:
        associated = [
            clip,
            clip.with_suffix(".timeline.json"),
            clip.with_suffix(".selection.json"),
            clip.with_suffix(".verification.json"),
        ]
        if not all(p.is_file() for p in associated):
            raise ValueError("Missing per-output delivery metadata")
        json_write(
            clip.with_suffix(".manifest.json"),
            {
                "schema_version": 2,
                "artifacts": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in associated
                ],
            },
        )
    complete = notes["continuous_playback_review"] == "PASS"
    ledger = json.loads(Path("outputs/gpu_ledger.json").read_text())
    context = json.loads(Path(a.execution_context).read_text()) if a.execution_context else None
    execution = delivery_execution_metadata(ledger, context)
    json_write(
        root / "acceptance.json",
        {
            "schema_version": 2,
            "automated_media": "PASS",
            "frame_review": notes["frame_review"],
            "report_review": notes["report_review"],
            "continuous_playback_review": notes["continuous_playback_review"],
            "spec_acceptance": "PASS" if complete else "INCOMPLETE_MANUAL_PLAYBACK_REVIEW",
            "remaining": []
            if complete
            else [
                notes.get("continuous_playback_pending_reason",
                          "Event/response continuous 1x playback review is pending.")
            ],
            **execution,
        },
    )
    entries = artifact_entries(root)
    json_write(root / "manifest.json", {"schema_version": 2, "status": "PASS", "artifacts": entries})
    for entry in entries:
        path = Path(entry["path"])
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("Final manifest integrity failure")
    print(f"Verified local delivery manifest: {len(entries)} files")


if __name__ == "__main__":
    main()
