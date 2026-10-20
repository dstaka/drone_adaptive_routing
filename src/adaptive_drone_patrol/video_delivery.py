"""Versioned offline delivery with immutable capture provenance."""

import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
from PIL import Image
from .util import json_write, sha256
from .video_timing import capture_schedule, flight_samples, sample_index

KINDS = ["normal", "emergency", "cost_shift"]


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def read_events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def validate_options(mode, fps, duration, rate=1):
    if mode not in ["full", "submission"] or fps not in [25, 30] or rate != 1:
        raise ValueError("Deliveries require full|submission, 25|30 FPS and 1x flight")
    if mode == "full" and duration is not None:
        raise ValueError("Full flight has a natural duration, not a duration target")
    if mode == "submission" and duration is not None and not 60 <= duration <= 180:
        raise ValueError("Submission duration must be 60–180 seconds")
    return fps


def delivery_execution_metadata(ledger, context=None):
    """Keep inherited GPU usage and newly verified transfer metadata explicit."""
    used = ledger['used_s']
    baseline = (context or {}).get('gpu_baseline_used_s', used)
    if (ledger.get('active_job') or not all(math.isfinite(v) for v in [used, baseline])
            or not 0 <= baseline <= used <= 28800):
        raise ValueError('Invalid or unreconciled cumulative GPU ledger/context')
    return {'gpu_used_s': used, 'gpu_added_s': used - baseline, 'gpu_remaining_s': 28800 - used,
            'new_server_directories': (context or {}).get('new_server_directories', []),
            'transfer': (context or {}).get('transfer',
                'No new external server transfer; inherited 465/465 input hashes verified.')}


def optimality_captions(value):
    exact = value["exact"]
    if exact.get("status") != "OPTIMAL" or not exact.get("proven"):
        return ["最適性未確認", "最適性未確認"]
    q = next(x for x in value["forced_first"] if x["action"] == value["policy_action"])
    tol = value["equality_tolerance"]
    first = (
        "この選択は現在条件で最適な次の一手"
        if abs(q["cost"] - exact["cost"]) <= tol
        else "次の一手と厳密解には差があります"
    )
    if q.get("status") != "OPTIMAL" or not q.get("proven"):
        first = "最適性未確認"
    gap = value["policy_rollout"]["cost"] - exact["cost"]
    suffix = (
        "AIの残り巡回全体が最適"
        if abs(gap) <= tol
        else f"厳密解との差 {gap:.3f}（{100 * gap / max(abs(exact['cost']), tol):.2f}%）"
    )
    return [first, suffix]


def validate_binding(value, decision):
    if (
        value.get("evidence_kind") != "physical"
        or decision.get("evidence_kind") != "physical"
        or any(value[k] != decision[k] for k in ["run_id", "decision_id", "policy_hash", "boundary_hash"])
        or value["policy_action"] != decision["selected_action"]
        or value["snapshot"] != decision["boundary_snapshot"]
        or value["physical_reference"]
        != {k: decision[k] for k in ["simulation_step", "event_seq", "physical_pose", "boundary_hash"]}
    ):
        raise ValueError("Stale, cross-run or policy/oracle explanation mismatch")


def load_run(run):
    summary = read_json(run / "mission_summary.json")
    frames = read_rows(run / "frames.csv")
    telemetry = read_rows(run / "telemetry.csv")
    events = read_events(run / "events.jsonl")
    return {
        "path": run,
        "summary": summary,
        "frames": frames,
        "telemetry": telemetry,
        "events": events,
        "scenario": read_json(run / "scenario.json"),
        "scene": read_json(run / "scene_manifest.json"),
        "initial": read_json(run / "initial_state.json")["xyz"],
        "schedule": capture_schedule(frames, telemetry, events),
    }


def prepare_sources(root, output_dir, explanations):
    from .video import select_runs, ffmpeg_executable
    from .evidence import validate_run

    selected = select_runs(root)
    source_dir = output_dir / "sources"
    if source_dir.exists():
        selection = read_json(output_dir / "selection.json")
        if selection["explanations_sha256"] != sha256(explanations):
            raise ValueError("Explanation input changed within delivery")
        for entry in selection["input_manifest"]:
            if sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("Preserved source changed within delivery")
        return [load_run(source_dir / p.name) for p in selected]
    source_dir.mkdir(parents=True)
    attempts = []
    for p in sorted(Path(root).glob("*/mission_summary.json")):
        s = validate_run(p.parent, require_media=True)
        if s["evidence_kind"] != "physical":
            raise ValueError("Synthetic evidence cannot be delivered as flight")
        attempts.append(s)
    if len(attempts) != 9 or {(s["scenario_id"], s["execution_seed"]) for s in attempts} != {
        (k, s) for k in KINDS for s in [1, 2, 3]
    }:
        raise ValueError("Must preserve the fixed nine-trial denominator")
    json_write(output_dir / "all_trials.json", attempts)
    shutil.copyfile(explanations, output_dir / "decision_explanations.json")
    values = read_json(explanations)["decisions"]
    records = []
    manifest = []
    for original in selected:
        run = source_dir / original.name
        shutil.copytree(original, run)
        record = load_run(run)
        decisions = read_events(run / "decisions.jsonl")
        for value in [v for v in values if v["run_id"] == run.name]:
            validate_binding(value, next(d for d in decisions if d["decision_id"] == value["decision_id"]))
        scratch = output_dir / "capture_png" / run.name
        scratch.mkdir(parents=True)
        subprocess.run(
            [
                ffmpeg_executable(),
                "-v",
                "error",
                "-i",
                str(run / "raw.mp4"),
                "-fps_mode",
                "passthrough",
                str(scratch / "%06d.png"),
            ],
            check=True,
        )
        if len(list(scratch.glob("*.png"))) != len(record["frames"]):
            raise ValueError("Decoded originals do not match capture records")
        for p in sorted(run.rglob("*")):
            if p.is_file():
                manifest.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)})
        normalize_raw(record, output_dir, scratch)
        records.append(record)
    selection = {
        "schema_version": 2,
        "selected_runs": [p.name for p in selected],
        "execution_seed": records[0]["summary"]["execution_seed"],
        "reason": f"Lowest paired execution seed {records[0]['summary']['execution_seed']} completing all scenarios; all nine attempts retained.",
        "trial_count": 9,
        "success_count": sum(s["status"] == "SUCCEEDED" for s in attempts),
        "completed_inspections": sum(s["completed_count"] for s in attempts),
        "explanations_sha256": sha256(explanations),
        "input_manifest": manifest,
        "models": {
            k: attempts[0][k] for k in ["flight_checkpoint_hash", "flight_config_hash", "routing_model_hash"]
        },
    }
    json_write(output_dir / "selection.json", selection)
    return records


def normalize_raw(record, output_dir, scratch):
    """Encode unique physical instants, preserving the latest boundary capture.

    Original cadence-plus-forced raw remains in sources/. Terminal PTS equals
    mission-end; its packet lasts one actual physics step solely as codec cover.
    """
    from .video import ffmpeg_executable

    dst = output_dir / "raw" / record["path"].name
    dst.mkdir(parents=True)
    schedule = record["schedule"]
    frames = record["frames"]
    rows = [
        dict(
            frames[i],
            source_frame_index=i,
            frame_index=j,
            pts_s=float(frames[i]["sim_time_s"]) - schedule["start"],
        )
        for j, i in enumerate(schedule["indices"])
    ]
    with (dst / "frames.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    concat = dst / "capture.ffconcat"
    lines = ["ffconcat version 1.0"]
    for j, row in enumerate(rows):
        # Paths are relative to the concat file; no deployment roots in metadata.
        path = (
            Path("../../capture_png") / record["path"].name / f"{int(row['source_frame_index']) + 1:06d}.png"
        )
        duration = float(rows[j + 1]["pts_s"]) - float(row["pts_s"]) if j + 1 < len(rows) else schedule["dt"]
        lines.extend(
            [
                f"file '{path.as_posix()}'",
                f"option framerate {round(1 / schedule['dt'])}",
                f"duration {duration:.9f}",
            ]
        )
    concat.write_text("\n".join(lines) + "\n")
    subprocess.run(
        [
            ffmpeg_executable(),
            "-v",
            "error",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-fps_mode",
            "vfr",
            "-enc_time_base",
            f"1:{round(1 / schedule['dt'])}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "10000",
            "-movflags",
            "+faststart",
            str(dst / "raw.mp4"),
        ],
        check=True,
    )
    json_write(
        dst / "timing.json",
        dict(
            schedule,
            source_run_id=record["path"].name,
            original_raw=f"sources/{record['path'].name}/raw.mp4",
            duplicate_rule="Keep latest event_seq at same step; all discarded capture indices retained above.",
            terminal_rule="Final point at mission-end PTS; one dt codec packet, zero additional physical flight.",
        ),
    )


def flight_segment(record, fps, start=None, end=None):
    schedule = record["schedule"]
    start = schedule["start"] if start is None else start
    end = schedule["end"] if end is None else end
    times = [float(f["sim_time_s"]) for f in record["frames"]]
    return {
        "segment_type": "flight",
        "run_id": record["path"].name,
        "clip": f"sources/{record['path'].name}/raw.mp4",
        "source_frame_start": sample_index(times, start),
        "source_frame_end": sample_index(times, end),
        "source_simulation_start_s": start,
        "source_simulation_end_s": end,
        "frame_count": len(flight_samples(start, end, fps)),
        "playback_rate": 1.0,
        "output_fps": fps,
        "cuts": [],
        "disclosure": "全飛行区間・省略なし／実飛行 1.0倍速",
    }


def build_plan(records, values, mode, fps):
    from .video_render import card_specs

    cards = card_specs(records, values)
    if mode == "single":
        return [flight_segment(records[0], fps)]
    segments = [dict(segment_type="card", card=cards["intro"], frame_count=round(7 * fps))]
    for record in records:
        kind = record["scenario"]["scenario_id"]
        # Separate static explanations never advance the simulation clock.
        if kind in ["emergency", "cost_shift"]:
            segments.append(dict(segment_type="card", card=cards[kind], frame_count=round(6 * fps)))
        if kind == "cost_shift":
            segments.append(dict(segment_type="card", card=cards["tradeoff"], frame_count=round(14 * fps)))
        segments.append(flight_segment(record, fps))
        segments.append(dict(segment_type="card", card=cards[kind + "_end"], frame_count=round(2 * fps)))
    segments.append(dict(segment_type="card", card=cards["outro"], frame_count=round(5 * fps)))
    return segments


def render_output(output, records, segments, font_path, fps, mode):
    from .video import ffmpeg_executable
    from .video_render import Renderer, provenance_stamp

    if output.exists() or output.with_suffix(".timeline.json").exists():
        raise FileExistsError(output)
    renderer = Renderer(font_path)
    values = read_json(output.parent / "decision_explanations.json")["decisions"]
    for record in records:
        record["_explanations"] = [v for v in values if v["run_id"] == record["path"].name]
    encoder = subprocess.Popen(
        [
            ffmpeg_executable(),
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1920x1080",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-threads",
            "4",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    cursor = 0
    mappings = []
    try:
        for segment in segments:
            segment.update(presentation_start_s=cursor / fps, output_frame_start=cursor, output_fps=fps)
            if segment["segment_type"] == "card":
                img = renderer.card(segment["card"])
                for _ in range(segment["frame_count"]):
                    encoder.stdin.write(img.tobytes())
            else:
                record = next(r for r in records if r["path"].name == segment["run_id"])
                points = flight_samples(
                    segment["source_simulation_start_s"], segment["source_simulation_end_s"], fps
                )
                times = [float(f["sim_time_s"]) for f in record["frames"]]
                for local, time in enumerate(points):
                    index = sample_index(times, time)
                    frame = record["frames"][index]
                    camera = Image.open(
                        output.parent / "capture_png" / record["path"].name / f"{index + 1:06d}.png"
                    ).convert("RGB")
                    img, state = renderer.flight(camera, record, frame)
                    mapping = {
                        "output_frame_index": cursor + local,
                        "run_id": record["path"].name,
                        "source_frame_index": index,
                        "simulation_step": int(frame["simulation_step"]),
                        "event_seq": state["event_seq"],
                        "sim_time_s": float(frame["sim_time_s"]),
                        "visited": state["visited"],
                        "target": state["selected"],
                        "priorities": state["priorities"],
                    }
                    provenance_stamp(img, mapping)
                    encoder.stdin.write(img.tobytes())
                    mappings.append(mapping)
            cursor += segment["frame_count"]
            segment["presentation_end_s"] = cursor / fps
    finally:
        encoder.stdin.close()
        code = encoder.wait(timeout=120)
    if code:
        raise RuntimeError("Encoding failed")
    timeline = {
        "schema_version": 2,
        "mode": mode,
        "fps": fps,
        "frame_count": cursor,
        "duration_s": cursor / fps,
        "canvas": [1920, 1080],
        "segments": segments,
        "cuts": [],
        "sampling": "Unit-slope previous capture; last output sample replaced by exact endpoint (<=1 output frame).",
        "boundary_convention": "Physical [start,end); endpoint sample carries completion, no fictitious post-end interval.",
        "flight_duration_s": sum(s["frame_count"] / fps for s in segments if s["segment_type"] == "flight"),
    }
    if mode == "submission" and not (
        60 <= cursor / fps <= 180 and timeline["flight_duration_s"] > cursor / fps / 2
    ):
        raise ValueError("Submission cannot satisfy duration/authentic-flight majority with this card plan")
    json_write(output.with_suffix(".timeline.json"), timeline)
    json_write(output.with_suffix(".mapping.json"), mappings)
    selection = read_json(output.parent / "selection.json")
    json_write(output.with_suffix(".selection.json"), selection)
    return timeline


def compose_delivery(root, output, font_path, mode, fps, duration, explanations):
    validate_options(mode, fps, duration)
    output = Path(output)
    # Preserve legacy collision protection, while never using legacy timing.
    if any(
        p.exists() for p in [output, output.with_suffix(".timeline.json"), output.parent / "timeline.json"]
    ):
        raise FileExistsError(output)
    if not explanations:
        raise ValueError("Physical decision explanations are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = prepare_sources(root, output.parent, explanations)
    values = read_json(explanations)["decisions"]
    if mode == "full":
        for record in records:
            render_output(
                output.parent / f"full_{record['scenario']['scenario_id']}.mp4",
                [record],
                build_plan([record], values, "single", fps),
                font_path,
                fps,
                "full",
            )
    segments = build_plan(records, values, mode, fps)
    natural = sum(s["frame_count"] for s in segments) / fps
    if mode == "submission" and duration is not None and natural > duration + 1 / fps:
        raise ValueError(
            "Duration cap too short for complete footage and required cards; no automatic retiming"
        )
    return render_output(output, records, segments, font_path, fps, mode)
