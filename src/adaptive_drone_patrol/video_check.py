"""Independent acceptance from physics records, decoded pixels and encoded PTS.

This module deliberately does not import the compositor, renderer or its timing
helpers. A playback_rate declaration is never sufficient evidence of 1x motion.
"""

import bisect
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .util import sha256, json_write
from .scene import project


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def data(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def probe(path, ffprobe):
    result = json.loads(
        subprocess.check_output(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,codec_name,avg_frame_rate,duration,nb_frames:frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    stream = result["streams"][0]
    if stream["codec_name"] != "h264":
        raise ValueError("Delivery must be H.264")
    return stream, [float(x["best_effort_timestamp_time"]) for x in result["frames"]]


def verify_pts(pts, fps, count):
    if len(pts) != count or not np.isfinite(pts).all() or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Missing, duplicate or nonmonotonic encoded PTS")
    if max(abs(t - i / fps) for i, t in enumerate(pts)) > 1e-5:
        raise ValueError("Encoded PTS disagree with CFR output grid")


def verify_segment(segment, pts, mapping, frames, telemetry, events, *, fps):
    run = segment["run_id"]
    begin = segment["output_frame_start"]
    count = segment["frame_count"]
    times = [float(f["sim_time_s"]) for f in frames]
    indices = [int(f["frame_index"]) for f in frames]
    if indices != list(range(len(frames))) or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Missing/nonmonotonic source frame records")
    if any(t["run_id"] != run for t in telemetry):
        raise ValueError("Cross-run physical source")
    steps = [int(t["simulation_step"]) for t in telemetry]
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError("Missing physical steps")
    dt = (float(telemetry[-1]["sim_time_s"]) - float(telemetry[0]["sim_time_s"])) / (steps[-1] - steps[0])
    origin = float(telemetry[0]["sim_time_s"]) - steps[0] * dt
    if dt <= 0 or any(
        abs(float(t["sim_time_s"]) - origin - int(t["simulation_step"]) * dt) > 1e-8 for t in telemetry
    ):
        raise ValueError("Actual physics timestep mismatch")
    if any(abs(float(f["sim_time_s"]) - origin - int(f["simulation_step"]) * dt) > 1e-8 for f in frames):
        raise ValueError("Source capture is not tied to physical steps")
    start = segment["source_simulation_start_s"]
    end = segment["source_simulation_end_s"]
    actual_start = pts[begin]
    actual_end = pts[begin + count] if begin + count < len(pts) else pts[-1] + 1 / fps
    error = abs((actual_end - actual_start) - (end - start))
    if (
        segment["playback_rate"] != 1
        or error > 1 / fps + 1e-6
        or abs(segment["presentation_start_s"] - actual_start) > 1e-6
        or abs(segment["presentation_end_s"] - actual_end) > 1e-6
    ):
        raise ValueError("Encoded span is not unit-slope physical duration")
    if start < times[0] - 1e-8 or end > events[-1]["sim_time_s"] + 1e-8:
        raise ValueError("Flight extends outside physical mission")
    own = [m for m in mapping if begin <= m["output_frame_index"] < begin + count]
    if len(own) != count:
        raise ValueError("Missing output/source mappings")
    errors = []
    for local, m in enumerate(own):
        physical = start + (pts[begin + local] - actual_start)
        if local == count - 1:
            physical = end
        expected = max(0, bisect.bisect_right(times, physical + 1e-9) - 1)
        source = frames[expected]
        if (
            m["output_frame_index"] != begin + local
            or m["run_id"] != run
            or m["source_frame_index"] != expected
            or m["simulation_step"] != int(source["simulation_step"])
            or abs(m["sim_time_s"] - times[expected]) > 1e-8
        ):
            raise ValueError("Actual source correspondence/run mismatch")
        err = abs(times[expected] - (start + pts[begin + local] - actual_start))
        errors.append(err)
        if err > 1 / fps + 1e-8:
            raise ValueError("Source sampling drifts beyond one output frame")
    boundaries = []
    for event in events:
        if event["type"] not in [
            "TARGET_SELECTED",
            "WAYPOINT_REACHED",
            "PRIORITY_UPDATED",
            "COST_UPDATED",
            "MISSION_FINISHED",
        ]:
            continue
        if not start - 1e-9 <= event["sim_time_s"] <= end + 1e-9:
            continue
        shown = next((m for m in own if m["sim_time_s"] >= event["sim_time_s"] - 1e-9), None)
        if shown is None:
            raise ValueError("Missing event/terminal representation")
        expected = actual_start + event["sim_time_s"] - start
        alignment = pts[shown["output_frame_index"]] - expected
        if abs(alignment) > 1 / fps + 1e-6:
            raise ValueError("HUD event timing exceeds output quantum")
        boundaries.append(
            {
                "type": event["type"],
                "event_seq": event["event_seq"],
                "physical_s": event["sim_time_s"],
                "encoded_pts_s": pts[shown["output_frame_index"]],
                "alignment_error_s": alignment,
            }
        )
    return {
        "run_id": run,
        "actual_encoded_span_s": actual_end - actual_start,
        "physical_span_s": end - start,
        "span_error_s": error,
        "max_source_quantization_s": max(errors),
        "events": boundaries,
        "endpoint_source_frame": own[-1]["source_frame_index"],
        "dt_s": dt,
    }


def recompute_cost(snapshot, order):
    """Replay the objective without the planner's cost implementation."""
    s = snapshot["scenario"]
    coords = np.asarray(s["coords"])
    visited = snapshot["visited"]
    if sorted(order) != [i for i, v in enumerate(visited) if not v]:
        raise ValueError("Incomplete comparison order")
    current = snapshot["current"]
    elapsed = snapshot["elapsed"]
    travel = penalty = 0.0
    for action in order:
        distance = float(np.linalg.norm(coords[action + 1] - coords[current]))
        edge = distance * s["multipliers"][current][action + 1]
        elapsed += distance / s["nominal_speed_mps"] + s["service_duration_s"][action]
        travel += edge
        penalty += s["priorities"][action] * elapsed
        current = action + 1
    if s["return_to_base"]:
        travel += float(np.linalg.norm(coords[current] - coords[0])) * s["multipliers"][current][0]
    result = {
        "cost": s["alpha"] * travel + s["beta"] * penalty,
        "travel_cost": travel,
        "completion_penalty": penalty,
    }
    if s.get('schema_version') == 3:
        result.update(weighted_travel_cost=s['alpha'] * travel,
                      weighted_completion_penalty=s['beta'] * penalty)
    return result


def verify_display_rows(rows, value):
    candidates = {c["id"]: c for c in value["candidate_priorities_distances"]}
    expected = []
    for i, name in enumerate([value["selected_L"], value["selected_H"]]):
        c = candidates[name]
        expected.append(
            {
                "name": name,
                "candidate": c,
                "result": next(q for q in value["forced_first"] if q["action"] == c["action"]),
                "label": f"{'近いL' if i == 0 else '遠い高優先度H'}（地点{name}）を先に訪問した場合の最小コスト",
            }
        )
    expected.append({"label": "AIの残り巡回予測（AIの予測）", "result": value["policy_rollout"]})
    if rows != expected:
        raise ValueError("Displayed alternatives/candidates do not match the authenticated decision")


def checked_priority_text(value, schema_version=2, decimals=2):
    """Independently enforce the integer representation used in encoded text."""
    if schema_version == 3:
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError("Invalid integer priority in media evidence")
        return str(value)
    return f"{value:.{decimals}f}"


def event_lines(event, ids, schema_version=2):
    c = event["update"]
    comparison = event["comparison"]
    update = (
        f"地点C：優先度 {checked_priority_text(c['old'], schema_version, 3)} → {c['value']:g}"
        if event["type"] == "PRIORITY_UPDATED"
        else "有向辺 "
        + ", ".join(f"{a}→{b}" for a, b in c["edges"])
        + f"：倍率 {c['old'][0]:g} → {c['value']:g}"
    )
    return [
        f"物理時刻 {event['sim_time_s']:.2f} 秒、最初の点検完了直後：" + update,
        "更新なしの予測：" + " → ".join(ids[a] for a in comparison["no_event_order"]),
        "更新後の予測：" + " → ".join(ids[a] for a in comparison["post_event_order"]),
        f"更新後条件で再評価：旧順序 {comparison['retained_order_post_event']['cost']:.3f} → 新順序 {comparison['replanned_order_post_event']['cost']:.3f}",
        f"名目目的コストの改善 {comparison['improvement']:.3f}。物理時間・エネルギーの削減ではありません。",
        "同じ境界snapshot・同じ学習済み方策の比較。実飛行は次の映像で確認。",
    ]


def verify_raw_pixels(path, ffmpeg, original_png, indices):
    """Bind every normalized raw picture to independently decoded originals."""
    proc = subprocess.Popen(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(path),
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "1:10000",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    errors = []
    try:
        for index in indices:
            raw = proc.stdout.read(1280 * 720 * 3)
            if len(raw) != 1280 * 720 * 3:
                raise ValueError("Missing normalized raw pixels")
            actual = np.frombuffer(raw, dtype=np.uint8).reshape(720, 1280, 3).astype(np.int16)
            original = np.asarray(Image.open(original_png / f"{index + 1:06d}.png").convert("RGB")).astype(
                np.int16
            )
            diff = np.abs(actual - original)
            # Tight local tiles expose frozen/replaced motion in a mostly-static scene.
            tiles = diff.reshape(45, 16, 80, 16, 3).mean(axis=(1, 3, 4))
            error = float(tiles.max())
            errors.append(error)
            if error > 8:
                raise ValueError("Normalized raw picture differs from original capture")
        if proc.stdout.read(1):
            raise ValueError("Unmapped normalized raw frames")
    finally:
        proc.stdout.close()
        code = proc.wait(timeout=60)
    if code:
        raise ValueError("Normalized raw full decode failed")
    return max(errors)


def verify_card_binding(card, decisions, events):
    if any(k.startswith("source_simulation_") for k in card):
        raise ValueError("Static card cannot advance source time")
    if card.get("kind") == "tradeoff":
        d = next(
            (
                d
                for d in decisions
                if d["run_id"] == card["run_id"] and d["decision_id"] == card["decision_id"]
            ),
            None,
        )
        v = card.get("feature", {})
        if (
            not d
            or d["boundary_hash"] != card["boundary_hash"]
            or v.get("snapshot") != d.get("boundary_snapshot")
        ):
            raise ValueError("Stale comparison card")
        from .physical_evidence import verify_physical_explanation

        verify_physical_explanation(v, d)
        actual_timestamp = next(
            e["sim_time_s"] for e in events if e["run_id"] == d["run_id"] and e["event_seq"] == d["event_seq"]
        )
        if abs(card["timestamp_s"] - actual_timestamp) > 1e-8:
            raise ValueError("Card decision timestamp mismatch")
        verify_display_rows(card["rows"], v)
        for displayed in card["rows"]:
            r = displayed["result"]
            independent = recompute_cost(v["snapshot"], r["order"])
            if any(abs(r[k] - independent[k]) > 1e-6 for k in independent):
                raise ValueError("Displayed component mismatch")
    if card.get("kind") == "event":
        event = next(
            e for e in events if e["run_id"] == card["run_id"] and e["event_seq"] == card["event_seq"]
        )
        if card["comparison"] != event["comparison"] or card["timestamp_s"] != event["sim_time_s"]:
            raise ValueError("Event comparison mismatch")
        from .analytical import snapshot, restore, content_hash

        d = next(
            d for d in decisions if d["run_id"] == card["run_id"] and d["event_seq"] > event["event_seq"]
        )
        if content_hash(d["boundary_snapshot"]) != event["comparison"]["boundary_hash_after"]:
            raise ValueError("Event is not the actual next-decision snapshot")
        if card["lines"] != event_lines(event, d["boundary_snapshot"]["scenario"]["checkpoint_ids"], d["boundary_snapshot"]["scenario"].get("schema_version", 2)):
            raise ValueError("Event card visible numbers/orders differ from original event")
        state = restore(d["boundary_snapshot"])
        frozen = snapshot(state)
        for key, orderkey in [
            ("retained_order_post_event", "no_event_order"),
            ("replanned_order_post_event", "post_event_order"),
        ]:
            calculated = recompute_cost(frozen, event["comparison"][orderkey])
            if any(abs(calculated[k] - event["comparison"][key][k]) > 1e-6 for k in calculated):
                raise ValueError("Event cost comparison does not independently reproduce")


def physical_state(events, time, priorities):
    visited = []
    target = None
    seq = 0
    p = list(priorities)
    update = None
    for e in events:
        if e["sim_time_s"] > time + 1e-9:
            break
        seq = e["event_seq"]
        if e["type"] == "TARGET_SELECTED":
            target = e["target_id"]
        if e["type"] == "WAYPOINT_REACHED":
            visited.append(e["target_id"])
            target = None
        if e["type"] == "PRIORITY_UPDATED":
            p[e["update"]["checkpoint"]] = e["update"]["value"]
            update = e
        if e["type"] == "COST_UPDATED":
            update = e
    return visited, target, p, seq, update


@lru_cache(maxsize=512)
def text_patch(text, font, color, background):
    box = font.getbbox(text)
    width = max(1, int(font.getlength(text)) + 3)
    height = box[3] + 3
    expected = Image.new("RGB", (width, height), background)
    ImageDraw.Draw(expected).text((0, 0), text, font=font, fill=color)
    return np.asarray(expected).astype(float)


def verify_text_region(image, text, position, font, color="white", background="#102333"):
    """Compare a tight visible text region, so erasing a caption cannot pass."""
    expected = text_patch(text, font, color, background)
    height, width = expected.shape[:2]
    x, y = position
    actual = np.asarray(image.crop((x, y, x + width, y + height))).astype(float)
    # Compare luminance: YUV420 chroma subsampling changes saturated text
    # edges without changing their glyphs. Erased text still fails by a wide margin.
    error = float(np.abs((actual - expected) @ [0.2126, 0.7152, 0.0722]).mean())
    if error > 4:
        raise ValueError(f"Encoded visible text differs: {text!r}, MAE={error:.3f}")
    return error


def verify_camera_region(image, source, projected):
    """Compare the principal native camera plane outside explicit overlays."""
    mask = np.ones((720, 1280), dtype=bool)
    for x, y in projected:
        mask[max(0, int(y) - 116) : min(720, int(y) + 64), max(0, int(x) - 100) : min(1280, int(x) + 100)] = (
            False
        )
    mask[:195, 1028:] = False  # same-frame detail and its caption
    a = np.asarray(image.crop((20, 145, 1300, 865))).astype(np.int16)[::4, ::4]
    b = np.asarray(source).astype(np.int16)[::4, ::4]
    error = float(np.abs(a - b)[mask[::4, ::4]].mean())
    if error > 3:
        raise ValueError("Principal camera pixels differ from authenticated original raw")
    return error


def checker_font(record, path, size):
    key = f"_font_{size}"
    if key not in record:
        face = ImageFont.truetype(str(path), size)
        try:
            face.set_variation_by_axes([500])
        except (OSError, AttributeError):
            pass
        record[key] = face
    return record[key]


def verify_card_pixels(image, card, font_path):
    """Independently rasterize numeric text after physical/analytical replay."""
    fonts = {}

    def check(text, xy, size, color="white"):
        return verify_text_region(image, text, xy, checker_font(fonts, font_path, size), color)

    errors = []
    if card.get("kind") == "tradeoff":
        v = card["feature"]
        ids = v["snapshot"]["scenario"]["checkpoint_ids"]
        for i, row in enumerate(card["rows"]):
            y = 266 + i * 165
            r = row["result"]
            s = v["snapshot"]["scenario"]
            errors.append(
                check(
                    (f"総コスト {r['cost']:.3f} ＝ 移動 {r['travel_cost']:.3f} ＋ 0.05 × 重要度時刻 {r['completion_penalty']:.3f}（{.05 * r['completion_penalty']:.3f}）"
                     if s.get('schema_version') == 3 else
                     f"総コスト {r['cost']:.3f} ＝ 移動 {s['alpha'] * r['travel_cost']:.3f} ＋ 優先度・完了時刻 {s['beta'] * r['completion_penalty']:.3f}"),
                    (65, y + 78),
                    28,
                )
            )
            label = "条件付き最適解（比較用）" if i < 2 else "AIの予測"
            errors.append(check(label + "：" + " → ".join(ids[a] for a in r["order"]), (65, y + 119), 25))
            if i < 2:
                c = row["candidate"]
                errors.append(
                    check(
                        f"優先度 {checked_priority_text(c['priority'], s.get('schema_version', 2), 3)} ／計画距離 {c['distance_m']:.3f} m ／実位置から {c['pose_distance_m']:.3f} m",
                        (65, y + 39),
                        25,
                    )
                )
        delta = card["rows"][1]["result"]["cost"] - card["rows"][0]["result"]["cost"]
        errors.append(
            check(f"Q*(H) − Q*(L) = {delta:.3f} ／ J* = {v['exact']['cost']:.3f}", (65, 773), 28, "#ffdc91")
        )
    else:
        for i, line in enumerate(card.get("lines", [])):
            errors.append(check(line, (70, 220 + i * 98), 30))
    return max(errors, default=0.0)


def verify_pixels(image, mapping, record, font_path):
    """Read provenance from encoded pixels and compare actual clock and motion.

    The strip binds pixels to capture/state; independent Japanese text raster
    comparisons test the visible clock, count, target and event banner itself.
    The unannotated drone inset is compared to the original decoded capture.
    """
    arr = np.asarray(image)
    bits = "".join("1" if arr[1067:1073, 21 + i * 6 : 24 + i * 6].mean() > 127 else "0" for i in range(80))
    decoded = [int(bits[i : i + 16], 2) for i in range(0, 80, 16)]
    visited, target, priorities, seq, update = physical_state(
        record["events"], mapping["sim_time_s"], record["scenario"]["priorities"]
    )
    state = {"visited": visited, "target": target, "priorities": priorities}
    digest = int.from_bytes(hashlib.sha256(json.dumps(state, sort_keys=True).encode()).digest()[:2], "big")
    code = ["normal", "emergency", "cost_shift"].index(record["scenario"]["scenario_id"]) + 1
    expected = [code, mapping["source_frame_index"], mapping["simulation_step"], seq, digest]
    if decoded != expected or mapping["event_seq"] != seq or any(mapping[k] != state[k] for k in state):
        raise ValueError("Encoded provenance or HUD state differs from physical events")
    face = checker_font(record, font_path, 30)
    labels = [
        (f"シミュレーション時間 {mapping['sim_time_s']:.2f} 秒", 150, "white"),
        (f"実飛行 1.0倍速 ／ 点検 {len(visited)}/{len(priorities)}", 199, "#86dbe6"),
        ("現在の目標：" + (target or ("完了" if len(visited) == len(priorities) else "待機")), 247, "white"),
    ]
    text_errors = []
    for label, y, color in labels:
        expected_img = Image.new("RGB", (570, 45), "#102333")
        draw = ImageDraw.Draw(expected_img)
        draw.text((0, 0), label, font=face, fill=color)
        actual = np.asarray(image.crop((1340, y, 1910, y + 45))).astype(float)
        err = float(np.abs(actual - np.asarray(expected_img).astype(float)).mean())
        text_errors.append(err)
        if err > 4:
            raise ValueError("Encoded Japanese clock/count/target text mismatch")
    for j, name in enumerate(record["scenario"]["checkpoint_ids"]):
        x = 1340 + (j // 5) * 278
        y = 726 + (j % 5) * 43
        text_errors.append(
            verify_text_region(
                image, f"{name}  優先度 {checked_priority_text(priorities[j], record['scenario'].get('schema_version', 2))}", (x + 37, y), checker_font(record, font_path, 23)
            )
        )
    if update:
        u = update["update"]
        text = (
            f"地点Cの優先度 {checked_priority_text(u['old'], record['scenario'].get('schema_version', 2), 3)} → {u['value']:g}"
            if update["type"] == "PRIORITY_UPDATED"
            else "有向辺 "
            + ", ".join(f"{a}→{b}" for a, b in u["edges"])
            + f" 倍率 {u['old'][0]:g} → {u['value']:g}"
        )
        text_errors.append(
            verify_text_region(
                image,
                text + f"（更新 {update['sim_time_s']:.2f}秒）",
                (25, 930),
                checker_font(record, font_path, 28),
                "#ffdc91",
            )
        )
    index = mapping["source_frame_index"]
    source = Image.open(record["png"] / f"{index + 1:06d}.png").convert("RGB")
    step = mapping["simulation_step"]
    t = record["telemetry"][step - 1] if step else None
    pose = [float(t[k]) for k in ["x", "y", "z"]] if t else record["initial"]
    if t:
        if target:
            goal = record["scenario"]["coords"][record["scenario"]["checkpoint_ids"].index(target) + 1]
            distance = float(np.linalg.norm(np.asarray(goal) - pose))
            label = f"実位置から目標 {distance:.3f} m"
        else:
            label = (
                f"直前の点検位置との距離 {float(t['distance_to_goal_m']):.3f} m"
                if len(visited) == len(priorities)
                else f"待機位置との距離 {float(t['distance_to_goal_m']):.3f} m"
            )
        text_errors.append(verify_text_region(image, label, (1340, 333), checker_font(record, font_path, 25)))
    x, y = project(pose, record["scene"]["camera"])
    left = max(0, min(source.width - 96, int(round(x)) - 48))
    top = max(0, min(source.height - 72, int(round(y)) - 36))
    detail = (
        source.crop((left, top, left + 96, top + 72))
        .resize((240, 180), Image.Resampling.LANCZOS)
        .resize((200, 150))
    )
    actual = np.asarray(image.crop((1070, 180, 1270, 330))).astype(float)
    motion_error = float(np.abs(actual - np.asarray(detail).astype(float)).mean())
    if motion_error > 6:
        raise ValueError("Encoded drone pixels differ from the claimed physical capture")
    projected = [project(m["world_xyz"], record["scene"]["camera"]) for m in record["scene"]["markers"]]
    camera_error = verify_camera_region(image, source, projected)
    return max(text_errors), max(motion_error, camera_error)


def check_delivery(root, ffprobe, ffmpeg, font_path):
    root = Path(root)
    selection = data(root / "selection.json")
    values = data(root / "decision_explanations.json")["decisions"]
    for entry in selection["input_manifest"]:
        p = Path(entry["path"])
        if sha256(p) != entry["sha256"] or p.stat().st_size != entry["bytes"]:
            raise ValueError("Immutable input integrity failure")
    independent_decode = Path(tempfile.mkdtemp(prefix="checker_decode_", dir=root))
    records = {}
    raw_results = []
    decisions = []
    events = []
    for name in selection["selected_runs"]:
        run = root / "sources" / name
        from .evidence import validate_run

        validate_run(run, require_media=True)
        record = {
            "frames": rows(run / "frames.csv"),
            "telemetry": rows(run / "telemetry.csv"),
            "events": lines(run / "events.jsonl"),
            "scenario": data(run / "scenario.json"),
            "scene": data(run / "scene_manifest.json"),
            "initial": data(run / "initial_state.json")["xyz"],
            "png": independent_decode / name,
        }
        record["png"].mkdir()
        subprocess.run(
            [
                str(ffmpeg),
                "-v",
                "error",
                "-i",
                str(run / "raw.mp4"),
                "-fps_mode",
                "passthrough",
                str(record["png"] / "%06d.png"),
            ],
            check=True,
        )
        records[name] = record
        decisions.extend(lines(run / "decisions.jsonl"))
        events.extend(record["events"])
        # First validate immutable legacy PTS against their capture table, then
        # validate new VFR PTS directly against physical steps (not timing.json).
        original_stream, original_pts = probe(run / "raw.mp4", ffprobe)
        capture = record["frames"]
        if (
            len(original_pts) != len(capture)
            or max(abs(t - float(f["pts_s"])) for t, f in zip(original_pts, capture)) > 1e-6
        ):
            raise ValueError("Original camera/capture index mismatch")
        normalized = root / "raw" / name
        stream, pts = probe(normalized / "raw.mp4", ffprobe)
        raw_rows = rows(normalized / "frames.csv")
        unique = {int(f["simulation_step"]): i for i, f in enumerate(capture)}
        expected_indices = list(unique.values())
        if [int(f["source_frame_index"]) for f in raw_rows] != expected_indices or len(pts) != len(
            expected_indices
        ):
            raise ValueError("Normalized raw drops physical instants or reorders captures")
        errors = [
            abs(t - (float(capture[i]["sim_time_s"]) - float(capture[0]["sim_time_s"])))
            for t, i in zip(pts, expected_indices)
        ]
        dt = (float(record["telemetry"][-1]["sim_time_s"]) - float(record["telemetry"][0]["sim_time_s"])) / (
            int(record["telemetry"][-1]["simulation_step"]) - int(record["telemetry"][0]["simulation_step"])
        )
        source_origin = float(capture[0]["sim_time_s"])
        mission_end = record["events"][-1]["sim_time_s"]
        if (
            max(errors) > 1e-6
            or abs(pts[-1] - (mission_end - source_origin)) > 1e-6
            or abs(float(stream["duration"]) - (mission_end - source_origin) - dt) > 1e-5
        ):
            raise ValueError("Raw encoded PTS/terminal coverage do not match physical time")
        raw_pixel_error = verify_raw_pixels(normalized / "raw.mp4", ffmpeg, record["png"], expected_indices)
        raw_results.append(
            {
                "run_id": name,
                "status": "PASS",
                "frames": len(pts),
                "physical_end_s": mission_end,
                "encoded_duration_s": float(stream["duration"]),
                "max_pts_error_s": max(errors),
                "max_raw_tile_pixel_mae": raw_pixel_error,
                "original_end_drift_s": original_pts[-1] - mission_end,
                "full_decode": "PASS",
            }
        )
    from .physical_evidence import verify_physical_explanation, verify_frozen_inputs, verify_event_response

    frozen = []
    for name, record in records.items():
        run = root / "sources" / name
        frozen.append(
            {
                "summary": data(run / "mission_summary.json"),
                "scenario": record["scenario"],
                "scene": record["scene"],
                "initial_pose": data(run / "initial_state.json"),
            }
        )
        verify_event_response(record["events"], lines(run / "decisions.jsonl"))
        for value in [v for v in values if v["run_id"] == name]:
            decision = next(
                d for d in decisions if d["run_id"] == name and d["decision_id"] == value["decision_id"]
            )
            verify_physical_explanation(value, decision)
    verify_frozen_inputs(frozen, selection["models"]["routing_model_hash"])
    results = []
    for path in sorted(root.glob("*.mp4")):
        timeline = data(path.with_suffix(".timeline.json"))
        mapping = data(path.with_suffix(".mapping.json"))
        if timeline["schema_version"] != 2:
            raise ValueError("Legacy timeline cannot pass realtime acceptance")
        stream, pts = probe(path, ffprobe)
        fps = timeline["fps"]
        verify_pts(pts, fps, timeline["frame_count"])
        if (
            stream["width"] < 1280
            or stream["height"] < 720
            or abs(float(stream["duration"]) - timeline["duration_s"]) > 1 / fps
        ):
            raise ValueError("Invalid delivery canvas/duration")
        cursor = 0
        flight = 0
        checks = []
        for segment in timeline["segments"]:
            if (
                segment["output_frame_start"] != cursor
                or abs(segment["presentation_start_s"] - cursor / fps) > 1e-6
            ):
                raise ValueError("Undeclared timeline gap/cut")
            if segment["segment_type"] == "flight":
                record = records[segment["run_id"]]
                checks.append(
                    verify_segment(
                        segment,
                        pts,
                        mapping,
                        record["frames"],
                        record["telemetry"],
                        record["events"],
                        fps=fps,
                    )
                )
                if (
                    abs(segment["source_simulation_start_s"] - float(record["frames"][0]["sim_time_s"]))
                    > 1e-8
                    or segment["source_simulation_end_s"] != record["events"][-1]["sim_time_s"]
                ):
                    raise ValueError("This complete-flight delivery contains an undisclosed omission")
                flight += segment["frame_count"]
            elif segment["segment_type"] == "card":
                verify_card_binding(segment["card"], decisions, events)
            else:
                raise ValueError("Unknown segment type")
            cursor += segment["frame_count"]
        if cursor != len(pts) or timeline["cuts"]:
            raise ValueError("Undeclared output frames/cuts")
        if path.name == "submission.mp4" and not (
            60 <= float(stream["duration"]) <= 180 and flight > cursor / 2
        ):
            raise ValueError("Submission duration or authentic-flight share fails")
        proc = subprocess.Popen(
            [str(ffmpeg), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
        )
        by_index = {m["output_frame_index"]: m for m in mapping}
        card_frames = {
            i: segment["card"]
            for segment in timeline["segments"]
            if segment["segment_type"] == "card"
            for i in range(
                segment["output_frame_start"], segment["output_frame_start"] + segment["frame_count"]
            )
        }
        card_errors = []
        card_references = {}
        hud_errors = []
        motion_errors = []
        decoded = 0
        try:
            for i in range(len(pts)):
                raw = proc.stdout.read(stream["width"] * stream["height"] * 3)
                if len(raw) != stream["width"] * stream["height"] * 3:
                    raise ValueError("Truncated full decode")
                if i in by_index:
                    m = by_index[i]
                    im = Image.frombytes("RGB", (stream["width"], stream["height"]), raw)
                    try:
                        hud, motion = verify_pixels(im, m, records[m["run_id"]], font_path)
                    except ValueError as exc:
                        raise ValueError(
                            f"{path.name}: output frame {i}, {m['run_id']}, source {m['source_frame_index']}: {exc}"
                        ) from exc
                    hud_errors.append(hud)
                    motion_errors.append(motion)
                elif i in card_frames:
                    im = Image.frombytes("RGB", (stream["width"], stream["height"]), raw)
                    card = card_frames[i]
                    key = card["title"]
                    if key not in card_references:
                        card_errors.append(verify_card_pixels(im, card, font_path))
                        card_references[key] = np.asarray(im.resize((480, 270))).astype(np.int16)
                    else:
                        delta = np.abs(
                            np.asarray(im.resize((480, 270))).astype(np.int16) - card_references[key]
                        )
                        if float(delta.reshape(27, 10, 48, 10, 3).mean(axis=(1, 3, 4)).max()) > 5:
                            raise ValueError("Static card changed during presentation")
                decoded += 1
            if proc.stdout.read(1):
                raise ValueError("Unmapped extra decoded frames")
        finally:
            proc.stdout.close()
            code = proc.wait(timeout=60)
        if code:
            raise ValueError("Full decode failed")
        result = {
            "path": str(path),
            "status": "PASS",
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "duration_s": float(stream["duration"]),
            "fps": fps,
            "decoded_frames": decoded,
            "flight_frames": flight,
            "flight_fraction": flight / decoded,
            "max_hud_luma_mae": max(hud_errors),
            "max_card_luma_mae": max(card_errors, default=0),
            "max_motion_pixel_mae": max(motion_errors),
            "segments": checks,
            "full_decode": "PASS",
            "omitted_intervals": [],
        }
        json_write(path.with_suffix(".verification.json"), result)
        results.append(result)
        print(
            f"PASS {path.name}: {decoded} frames, {float(stream['duration']):.3f}s, authentic {flight / decoded:.1%}",
            flush=True,
        )
    if {r["path"].split("/")[-1] for r in results} != {
        "full_normal.mp4",
        "full_emergency.mp4",
        "full_cost_shift.mp4",
        "demo_full.mp4",
        "submission.mp4",
    }:
        raise ValueError("Missing required output videos")
    result = {
        "schema_version": 2,
        "status": "PASS",
        "raw": raw_results,
        "outputs": results,
        "trials": {
            "attempted": 9,
            "succeeded": selection["success_count"],
            "inspections": selection["completed_inspections"],
        },
        "checker_method": "Original physics steps + actual encoded PTS + every flight frame pixel provenance, visible Japanese HUD and original drone crop; independent cost replay and exact re-solve.",
    }
    json_write(root / "verification.json", result)
    artifacts = artifact_entries(root)
    json_write(root / "manifest.json", {"schema_version": 2, "status": "PASS", "artifacts": artifacts})
    return result


def artifact_entries(root):
    entries = []
    for path in sorted(Path(root).rglob("*")):
        if (
            path.is_file()
            and path.suffix in [".mp4", ".json", ".csv", ".jsonl", ".pdf"]
            and "tools" not in path.parts
            and path.name != "manifest.json"
        ):
            entries.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return entries
