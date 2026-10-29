"""Stream verified 100 Hz HUD media; final admission never grants human approval.

The FFV1 RGB master contains unchanged native panels above its Japanese HUD.
Only this master and its MP4 view are encoded. Every decoded master pixel and
native crop is checked against a frame ledger; both decoded timelines are
checked, including the terminal state's normal 0.01 s display. No PNG sequence
or second raw video is retained. Failed attempts keep their diagnostic files.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from .dual_coordination_campaign import VerifiedJointFreeze, _digest, _path
from .dual_coordination_evaluate import file_record
from .dual_coordination_progress import RunJournal, atomic_json
from .util import sha256

ROLES = ("normal", "priority-transfer", "crossing-yield")


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _ffmpeg():
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        path = shutil.which("ffmpeg")
    _require(path is not None, "FFmpeg with decoded frame-duration reporting required")
    return str(Path(path).resolve())


def _rgb_hash(frame):
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _run(arguments, log):
    with Path(log).open("wb") as stream:
        result = subprocess.run(
            arguments, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stream, check=False
        )
    if result.returncode:
        raise ValueError(f"Encoder/decoder failed ({result.returncode}); retained log: {log}")


def _timeline(log, terminal_tick, *, shape=None):
    """Use decoded packet PTS/durations, never an intended fps label alone."""
    text = Path(log).read_text(errors="replace")
    base = re.search(r"config in time_base:\s*(\d+)/(\d+)", text)
    _require(base is not None, "Decoded stream time base missing")
    timebase = Fraction(int(base[1]), int(base[2]))
    rows = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\S+.*?duration:\s*(\d+)", text)
    _require(len(rows) == terminal_tick + 1, "Decoded timestamp/frame count differs")
    if shape is not None:
        sizes = re.findall(r"\bn:\s*\d+\s+pts:.*?\bs:(\d+)x(\d+)", text)
        _require(
            len(sizes) == len(rows)
            and all((int(width), int(height)) == (shape[1], shape[0]) for width, height in sizes),
            "Decoded frame dimensions differ",
        )
    for index, (number, pts, duration) in enumerate(rows):
        _require(
            int(number) == index and int(pts) * timebase == Fraction(index, 100),
            "Missing, reordered or non-1x decoded timestamp",
        )
        _require(int(duration) * timebase == Fraction(1, 100), "Decoded display duration differs from .01 s")
    return dict(
        frames=len(rows),
        fps=100,
        playback_rate=1.0,
        physics_duration_s=terminal_tick / 100,
        displayed_duration_s=len(rows) / 100,
        final_terminal_display_s=0.01,
        every_decoded_timestamp_verified=True,
    )


def _read_frame(stream, size):
    parts = []
    while size:
        part = stream.read(size)
        if not part:
            break
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def verify_video_files(directory, *, terminal_tick, shape, native_height):
    """Re-decode each full frame; the lossy MP4 is checked for timing, not equality."""
    directory = Path(directory)
    executable = _ffmpeg()
    height, width, channels = shape
    _require(channels == 3 and 0 < native_height < height, "Exact RGB/HUD dimensions required")
    size = height * width * channels
    master_log = directory / "decode-master.log"
    with master_log.open("wb") as log, (directory / "frame-ledger.jsonl").open() as ledger:
        process = subprocess.Popen(
            [
                executable,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "info",
                "-i",
                str(directory / "hud-master.mkv"),
                "-map",
                "0:v:0",
                "-vf",
                "showinfo",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=log,
        )
        try:
            count = 0
            for line in ledger:
                expected = json.loads(line)
                raw = _read_frame(process.stdout, size)
                _require(len(raw) == size, "Decoded master RGB frame is missing or truncated")
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
                _require(
                    expected["index"] == count and expected["time_s"] == count / 100,
                    "Frame ledger timestamp differs",
                )
                _require(_rgb_hash(frame) == expected["hud_rgb_sha256"], "Decoded master RGB hash differs")
                _require(
                    _rgb_hash(frame[:native_height]) == expected["native_rgb_sha256"],
                    "Decoded native crop RGB hash differs",
                )
                _require(
                    _rgb_hash(frame[native_height:]) == expected["hud_band_rgb_sha256"],
                    "Decoded HUD band RGB hash differs",
                )
                count += 1
            _require(
                process.stdout.read(1) == b"" and count == terminal_tick + 1,
                "Decoded master has extra or omitted frames",
            )
            _require(process.wait() == 0, "Master decoder failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
    master = _timeline(master_log, terminal_tick, shape=shape)
    master.update(rgb_hashes_match=True, native_crop_hashes_match=True, hud_band_hashes_match=True)
    view_log = directory / "decode-view.log"
    _run(
        [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(directory / "view.mp4"),
            "-map",
            "0:v:0",
            "-vf",
            "showinfo",
            "-fps_mode",
            "passthrough",
            "-f",
            "null",
            "-",
        ],
        view_log,
    )
    view = _timeline(view_log, terminal_tick, shape=shape)
    return dict(master_verified=master, view_verified=view)


def encode_verified_frames(frames, output, *, terminal_tick, heartbeat=None):
    """Consume (native RGB, full HUD RGB, frame receipt) once with bounded memory."""
    _require(type(terminal_tick) is int and 1 <= terminal_tick <= 30000, "Bounded full mission required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = _ffmpeg()
    process = None
    try:
        iterator = iter(frames)
        first = next(iterator)
        native, composed, _ = first
        shape = np.asarray(composed).shape
        _require(
            len(shape) == 3 and shape[2] == 3 and len(np.asarray(native).shape) == 3, "RGB frames required"
        )
        height, width, _ = shape
        native_height = np.asarray(native).shape[0]
        _require(
            width % 2 == 0 and height % 2 == 0 and 0 < native_height < height,
            "Even MP4 dimensions and HUD required",
        )
        from itertools import chain

        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "100",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-coder",
            "1",
            "-context",
            "1",
            "-g",
            "1",
            "-slicecrc",
            "1",
            "-pix_fmt",
            "bgr0",
            str(output / "hud-master.mkv"),
        ]
        with (
            (output / "encode-master.log").open("wb") as log,
            (output / "frame-ledger.jsonl").open("x") as ledger,
        ):
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            count = 0
            last_heartbeat = time.monotonic()
            for native, composed, receipt in chain([first], iterator):
                native, composed = np.asarray(native), np.asarray(composed)
                _require(
                    native.dtype == np.uint8
                    and composed.dtype == np.uint8
                    and composed.shape == shape
                    and native.shape == (native_height, width, 3),
                    "Frame RGB shape/type changed",
                )
                _require(np.array_equal(native, composed[:native_height]), "HUD changed native pixels")
                _require(
                    receipt["index"] == count and receipt["time_s"] == count / 100 and count <= terminal_tick,
                    "Every control timestamp must occur exactly once",
                )
                row = dict(
                    receipt,
                    native_rgb_sha256=_rgb_hash(native),
                    hud_rgb_sha256=_rgb_hash(composed),
                    hud_band_rgb_sha256=_rgb_hash(composed[native_height:]),
                )
                ledger.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                process.stdin.write(np.ascontiguousarray(composed).tobytes())
                count += 1
                if heartbeat and time.monotonic() - last_heartbeat >= 30:
                    heartbeat(count, terminal_tick + 1)
                    last_heartbeat = time.monotonic()
            _require(count == terminal_tick + 1, "Full reset-through-terminal frame sequence required")
            process.stdin.close()
            _require(process.wait() == 0, "FFV1 master encoding failed")
        _run(
            [
                executable,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(output / "hud-master.mkv"),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "passthrough",
                "-video_track_timescale",
                "10000",
                "-movflags",
                "+faststart",
                str(output / "view.mp4"),
            ],
            output / "encode-view.log",
        )
        verified = verify_video_files(
            output, terminal_tick=terminal_tick, shape=shape, native_height=native_height
        )
        version = subprocess.run(
            [executable, "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
        result = dict(
            schema="dual-video-encoding-v1",
            **verified,
            shape=list(shape),
            native_height=native_height,
            master=file_record(output / "hud-master.mkv", output),
            view=file_record(output / "view.mp4", output),
            frame_ledger=file_record(output / "frame-ledger.jsonl", output),
            encoder=dict(path=executable, sha256=sha256(executable), version=version),
            source_native_pixels_retained_in_master=True,
            human_review_performed=False,
        )
        atomic_json(output / "encoding.json", result)
        return result
    except BaseException as error:
        atomic_json(
            output / "failure.json",
            dict(error=f"{type(error).__name__}: {error}", human_review_performed=False),
        )
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass  # Preserve the earlier validation error after killing the encoder.


def run_final_media(*, freeze, original, font, output, journal_root=None):
    _require(type(freeze) is VerifiedJointFreeze, "VerifiedJointFreeze capability required")
    return _run_final_media(
        freeze=freeze, original=original, font=font, output=output, journal_root=journal_root
    )


def declared_video_cases(declaration, challenges):
    from .dual_coordination_schema import validate_case

    expected = [
        "development-canonical-normal",
        "development-video-priority-transfer",
        "development-video-crossing-yield",
    ]
    _require(
        declaration.get("schema") == "dual-campaign-declaration-v1"
        and declaration.get("video_case_ids") == expected,
        "Exact three prospective development video IDs required",
    )
    records = {row["case"]["case_id"]: row for row in challenges}
    _require(len(records) == len(challenges), "Duplicate development declaration IDs")
    cases = {}
    for role, identifier in zip(ROLES, expected, strict=True):
        _require(identifier in records, "Declared video case missing")
        case = validate_case(deepcopy(records[identifier]["case"]))
        _require(case["split"] == 15, "Final videos must use declared development cases")
        _require(
            case["scenario"] == ("priority" if role == "priority-transfer" else "normal"),
            "Declared demonstration scenario differs",
        )
        cases[role] = case
    return cases


def _source_trace(source):
    """Relocation is by recorded basename only, with exact retained size/hash."""
    source = Path(source).resolve()
    metadata = json.loads((source / "environment.json").read_text())
    result = json.loads((source / "result.json").read_text())
    _require(metadata["case"]["split"] == 15, "Development native trace required")
    terminal = result["mission"]["state"]["tick"]
    _require(type(terminal) is int and 1 <= terminal <= 30000, "Bounded terminal native clock required")
    records = result["replay_segments"]
    _require(len(records) == 1, "One uninterrupted full-rate render trace required")
    segment = records[0]
    path = source / Path(segment["path"]).name
    _require(
        path.parent == source
        and not path.is_symlink()
        and path.is_file()
        and path.stat().st_size == segment["bytes"]
        and sha256(path) == segment["sha256"],
        "Native render trace size/hash differs",
    )
    _require(
        segment["start_tick"] == 0 and segment["end_tick"] == terminal and segment["records"] == terminal + 1,
        "Full 100 Hz render trace required; sparse input is not interpolated",
    )
    return metadata, result, path


def behavior_metrics(case, state, audit, *, wait_decisions=0, priority_decisions=0, held_ticks=0):
    """Measured action/service facts; geometric/visual attribution is never fabricated."""
    observed = [
        row["observed_tick"]
        for row in state.get("event_receipts", [])
        if row.get("observed_tick") is not None and not row.get("no_op", False)
    ]
    owners = [None] * 10
    transfers = []
    for row in state.get("action_receipts", []):
        for goal, owner in enumerate(row["owners"]):
            if owner is not None and owners[goal] is not None and owners[goal] != owner:
                done = state["goals"][goal]
                transfers.append(
                    dict(
                        tick=row["tick"],
                        goal=goal,
                        from_drone=owners[goal] + 1,
                        to_drone=owner + 1,
                        post_event=bool(observed and row["tick"] >= min(observed)),
                        successfully_serviced=done["status"] == "DONE"
                        and done["owner"] == owner
                        and done["completed_tick"] >= row["tick"],
                    )
                )
            if owner is not None:
                owners[goal] = owner
    success = state["terminal_reason"] == "success"
    return dict(
        case_id=case["case_id"],
        team_time_s=state["tick"] / 100,
        completed_counts=state["completed_counts"],
        transfers=transfers,
        successful_post_event_transfers=sum(
            t["post_event"] and t["successfully_serviced"] for t in transfers
        ),
        minimum_full_assembly_distance_m=audit.get("minimum_clearance_m"),
        floor_minima_m=audit.get("floor_minima_m"),
        supervisor_held_control_ticks=held_ticks,
        nn_wait_decisions=wait_decisions,
        meaningful_nn_priority_decisions=priority_decisions,
        safety_violations=max(
            audit.get("violation_count", 0),
            len(audit.get("reconstruction_failures", [])),
            int(
                any(
                    audit.get(k) is False for k in ("safe", "retained_points_safe", "reconstructed_path_safe")
                )
            ),
        ),
        settling_passed=success
        and all(g["status"] == "DONE" for g in state["goals"])
        and state["final_dwell_ticks"] == [100, 100],
        terminal_reason=state["terminal_reason"],
    )


def verify_runtime_identity(recorded, current):
    """Compare exact finite JSON properties, normalizing tuple/list arrays only.

    runtime() exposes tuple-valued self exclusions; environment.json necessarily
    saves them as JSON arrays. No numeric rounding, tolerances, field filtering
    or exclusion removal is allowed. Relocated XML bytes retain separate hashes.
    """
    normalized = []
    for value in (recorded, current):
        _require(isinstance(value, dict), "Native runtime property object required")
        try:
            normalized.append(json.loads(json.dumps(value, allow_nan=False)))
        except (TypeError, ValueError) as error:
            raise ValueError("Native runtime must contain finite JSON properties") from error
    left, right = normalized
    for value in normalized:
        digest = value.get("model_xml_sha256")
        _require(
            isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "Native model XML SHA256 required",
        )
    left = {key: value for key, value in left.items() if key != "model_xml_sha256"}
    right = {key: value for key, value in right.items() if key != "model_xml_sha256"}
    differences = []

    def compare(a, b, path):
        if len(differences) >= 32:
            return
        if type(a) is not type(b):
            differences.append(path + " (JSON type differs)")
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append(path + "." + key + " (missing/extra property)")
                else:
                    compare(a[key], b[key], path + "." + key)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(path + " (array length differs)")
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}[{index}]")
        elif a != b:
            differences.append(f"{path} (recorded={a!r}, rendered={b!r})")

    compare(left, right, "$")
    _require(
        not differences,
        "Rendered model physical properties differ from captured native model: " + ", ".join(differences),
    )
    return dict(
        exact_physical_properties_match=True,
        json_array_normalization=True,
        physical_properties_sha256=_digest(left),
        recorded_model_xml_sha256=recorded["model_xml_sha256"],
        render_model_xml_sha256=current["model_xml_sha256"],
    )


def render_development_video(
    source, original, output, font, *, method="DEVELOPMENT_ONLY B-static", journal_root=None
):
    """Render a recorded full-rate development mission; never grants final acceptance."""
    from .dual_coordination_media import (
        DualNativeRenderer,
        camera_profile,
        compose_japanese_hud,
        verify_frame_coverage,
    )
    from .dual_coordination_raw import read_raw
    from .dual_coordination_schema import WAIT, decode_action, encode_action
    from .dual_payload_mujoco import DualMuJoCoFixture
    from .payload_contract import Payload
    from .payload_solver_mujoco import FREE_MASS_KG

    source, output, font = Path(source).resolve(), Path(output).resolve(), Path(font).resolve()
    metadata, result, trace = _source_trace(source)
    _require(font.is_file(), "Explicit Japanese font file required")
    _require(
        sha256(original) == metadata["runtime"]["original_asset_sha256"], "Original asset identity changed"
    )
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        name: file_record(source / name, source) for name in ("environment.json", "result.json", trace.name)
    }
    font_hash = sha256(font)
    journal = RunJournal(journal_root or output)
    attempt = journal.start(
        "native-video-render",
        f"{metadata['case']['case_id']}: full100Hz master/view verification",
        category="video generation and checking",
    )
    renderer = None
    try:
        fixture = DualMuJoCoFixture(
            original,
            output / "display-fixture",
            [Payload(FREE_MASS_KG, load["mass_ratio"], load["side_m"]) for load in metadata["case"]["loads"]],
        )
        runtime = fixture.runtime()
        # Relocated asset paths may alter XML bytes; all compiled properties must
        # remain identical, and both XML hashes are recorded rather than hidden.
        runtime_identity = verify_runtime_identity(metadata["runtime"], runtime)
        renderer = DualNativeRenderer(fixture, metadata["case"]["goals"])
        config = camera_profile()
        receipts = []
        statistics = dict(wait_decisions=0, priority_decisions=0, held_ticks=0)
        action_tick = None
        final = None

        def frames():
            nonlocal action_tick, final
            for index, row in enumerate(read_raw(trace)):
                _require(
                    row["tick"] == index
                    and row["case_id"] == metadata["case"]["case_id"]
                    and row["model_hash"] == result["binding"]["model_sha256"],
                    "Render trace identity/timestamp changed",
                )
                _require((row["kind"] == "RESET") == (index == 0), "Exactly one initial reset required")
                _require(row["mission"]["tick"] == index, "Native/mission clocks differ")
                if row.get("action_tick") is not None and row["action_tick"] != action_tick:
                    action_tick = row["action_tick"]
                    left, right, _ = decode_action(row["action"])
                    mask = row["structural_mask"]
                    _require(
                        len(mask) == 242 and mask[row["action"]], "Actual NN/baseline action must be legal"
                    )
                    statistics["wait_decisions"] += int(
                        (left == WAIT and right != WAIT) or (right == WAIT and left != WAIT)
                    )
                    statistics["priority_decisions"] += int(
                        left != WAIT
                        and right != WAIT
                        and left != right
                        and mask[encode_action(left, right, 1)]
                        and mask[encode_action(left, right, 2)]
                    )
                statistics["held_ticks"] += int(any(row.get("supervisor", {}).get("held", [False, False])))
                native, receipt = renderer.render(row)
                composed = compose_japanese_hud(native, row, font=font, method=method, config=config)
                receipts.append(receipt)
                final = row
                yield native, composed, receipt

        terminal = result["mission"]["state"]["tick"]
        encoding = encode_verified_frames(
            frames(),
            output / "video",
            terminal_tick=terminal,
            heartbeat=lambda n, total: journal.heartbeat(
                attempt,
                completed=n,
                total=total,
                reason="Rendered and streamed to lossless HUD master",
                next_action="Decode every master RGB/frame and MP4 timestamp",
            ),
        )
        coverage = verify_frame_coverage(receipts, config, terminal_tick=terminal)
        state = result["mission"]["state"]
        _require(
            final is not None
            and final["tick"] == terminal
            and all(value == state[key] for key, value in final["mission"].items()),
            "Rendered terminal mission differs from recorded result",
        )
        for name, record in inputs.items():
            _require(file_record(source / name, source) == record, "Render source changed during encoding")
        _require(sha256(font) == font_hash, "Font changed during rendering")
        audit = result["online_audit"].get("latest", result["online_audit"])
        metrics = behavior_metrics(metadata["case"], state, audit, **statistics)
        report = dict(
            schema="dual-render-verification-v1",
            scope="DEVELOPMENT_ONLY",
            method=method,
            case_sha256=_digest(metadata["case"]),
            source=str(source),
            inputs=inputs,
            camera=config,
            camera_sha256=_digest(config),
            font=dict(path=str(font), sha256=font_hash),
            source_model_sha256=result["binding"]["model_sha256"],
            render_model_sha256=runtime["model_xml_sha256"],
            physical_runtime_identity=runtime_identity,
            source_sha256={
                name: sha256(Path(__file__).with_name(name))
                for name in (
                    "dual_coordination_media.py",
                    "dual_coordination_media_run.py",
                    "dual_payload_mujoco.py",
                )
            },
            encoding=encoding,
            frame_coverage=coverage,
            metrics=metrics,
            full_mission_machine_checked=True,
            human_review_performed=False,
            final_acceptance=False,
        )
        atomic_json(output / "render-verification.json", report)
        journal.finish(
            attempt,
            status="passed",
            reason=f"All {terminal + 1} frames, RGB hashes/crops and1x timestamps verified; development only",
        )
        return report
    except BaseException as error:
        atomic_json(
            output / "failure.json", dict(reason=f"{type(error).__name__}: {error}", scope="DEVELOPMENT_ONLY")
        )
        journal.finish(attempt, status="failed", reason=str(error))
        raise
    finally:
        if renderer is not None:
            renderer.close()


def verify_render_capture(capture, source, case, *, auditor_hash):
    """Authenticate full-rate native capture before rendering or retiring its trace."""
    from .dual_coordination_evaluate import build_audit_receipt
    from .dual_coordination_raw import read_raw

    source = Path(source).resolve()
    metadata, result, trace = _source_trace(source)
    _require(
        metadata["case"] == case and not capture["infrastructure_errors"],
        "Capture case/infrastructure differs",
    )
    proposals = [json.loads(line) for line in (source / "proposed-actions.jsonl").read_text().splitlines()]
    _require(proposals == capture["actions"] and proposals, "Durable actual NN proposals required")
    for index, row in enumerate(proposals):
        _require(
            type(row["action"]) is int
            and 0 <= row["action"] < 242
            and row["tick"] == index * 20
            and len(row["mask"]) == 242
            and all(type(v) is bool for v in row["mask"])
            and row["mask"][row["action"]],
            "Invalid or missing fresh NN action",
        )
    reset = json.loads((source / "reset-inputs.json").read_text())
    _require(reset["case"] == case, "Retained reset case differs")
    final = None
    for index, row in enumerate(read_raw(trace)):
        _require(row["tick"] == index and row["case_id"] == case["case_id"], "Missing full-rate native clock")
        _require(abs(row["actual"]["world_time"] - index / 100) <= 1e-8, "Native/control clock differs")
        if index == 0:
            _require(
                row["kind"] == "RESET"
                and all(row["actual"][key] == reset["initial_native"][key] for key in ("qpos", "qvel")),
                "Initial native/reset state differs",
            )
        else:
            proposal = proposals[(index - 1) // 20]
            _require(
                row["kind"] == "STEP"
                and row["action_tick"] == proposal["tick"]
                and row["action"] == proposal["action"]
                and row["structural_mask"] == proposal["mask"],
                "Rendered control differs from durable NN proposal",
            )
        final = row
    _require(
        final is not None
        and final["tick"] == capture["terminal_tick"] == result["mission"]["state"]["tick"]
        and capture["mission"] == result["mission"]["state"],
        "Captured terminal result differs",
    )
    for name in ("qpos", "qvel"):
        a, b = np.asarray(final["actual"][name]), np.asarray(capture["final_backend"][name])
        _require(
            a.shape == b.shape and np.isfinite(b).all() and np.allclose(a, b, rtol=0, atol=1e-8),
            "Final native state differs from retained renderer input",
        )
    audit = build_audit_receipt(
        dict(
            identity=dict(case_sha256=_digest(case)),
            outcome=capture["outcome"],
            terminal_tick=capture["terminal_tick"],
        ),
        capture["audit_report"],
        auditor_hash=auditor_hash,
    )
    _require(audit["complete"], "Complete independent stage/interval audit required")
    return dict(
        schema="dual-full-rate-render-inputs-v1",
        case_sha256=_digest(case),
        controls=capture["terminal_tick"],
        frames=capture["terminal_tick"] + 1,
        decision_count=len(proposals),
        native_reexecution_performed=True,
        fresh_policy_forward_each_decision=True,
        full_rate_inputs_verified=True,
        audit=audit,
        trace=file_record(trace, source),
        reproduction_inputs={
            name: file_record(source / name, source)
            for name in (
                "reset-inputs.json",
                "proposed-actions.jsonl",
                "capture.json",
                "environment.json",
                "result.json",
            )
        },
    )


def _bound_named(freeze, role, name):
    records = [row for row in freeze.binding.files[role] if Path(row["path"]).name == name]
    _require(len(records) == 1, "Exactly one frozen " + name + " required")
    return _path(freeze.binding.root, records[0]["path"])


def _freeze_library(freeze):
    qualification = freeze.receipt["qualification"]
    backend = qualification.get("audit_backend", "python-native")
    if backend == "python-native":
        return None
    _require(backend == "compiled-native", "Unknown frozen audit backend")
    rows = [
        r for r in freeze.binding.files["audit"] if r["sha256"] == qualification.get("audit_library_sha256")
    ]
    _require(len(rows) == 1, "Exactly one qualified compiled audit library required")
    return _path(freeze.binding.root, rows[0]["path"])


def retire_verified_render_trace(source, render, verification, *, freeze_digest):
    """Delete only this new job's exact trace after durable media and inputs verify."""
    source, render = Path(source).resolve(), Path(render).resolve()
    marker = json.loads((source.parent / "media-owned-slot.json").read_text())
    _require(
        marker == dict(schema="dual-media-owned-slot-v1", freeze_sha256=freeze_digest),
        "Only a newly owned final media trace may be retired",
    )
    _require(verification["full_rate_inputs_verified"] is True, "Actual retained input verification required")
    report = json.loads((render / "render-verification.json").read_text())
    encoding = report["encoding"]
    _require(
        report["full_mission_machine_checked"] is True
        and report["frame_coverage"]["every_frame_verified"] is True
        and encoding["master_verified"]["rgb_hashes_match"] is True
        and encoding["master_verified"]["native_crop_hashes_match"] is True
        and encoding["view_verified"]["every_decoded_timestamp_verified"] is True,
        "Verified master/view/camera receipts required before retiring trace",
    )
    for name in ("master", "view", "frame_ledger"):
        record = encoding[name]
        _require(
            file_record(_path(render / "video", record["path"]), render / "video") == record,
            "Final media or frame ledger changed; retain trace",
        )
    retained = {
        name: file_record(source / name, source)
        for name in (
            "reset-inputs.json",
            "proposed-actions.jsonl",
            "capture.json",
            "environment.json",
            "result.json",
        )
    }
    _require(
        retained == verification["reproduction_inputs"],
        "Reproduction inputs changed after native verification; retain trace",
    )
    trace_record = verification["trace"]
    path = _path(source, trace_record["path"])
    _require(
        path.name.startswith("replay-")
        and path.suffix == ".draw"
        and path.parent == source
        and file_record(path, source) == trace_record,
        "Exact owned render trace required",
    )
    receipt = dict(
        schema="dual-render-trace-retention-v1",
        freeze_sha256=freeze_digest,
        retained_reproduction_inputs=retained,
        retired_trace=trace_record,
        verified_master=encoding["master"],
        verified_view=encoding["view"],
        reason="Native pixels retained exactly inside verified FFV1 HUD master; deterministic inputs and online audit retained",
        retired=False,
    )
    atomic_json(source / "retention.json", receipt)
    path.unlink()
    receipt["retired"] = True
    atomic_json(source / "retention.json", receipt)
    return receipt


def _run_final_media(*, freeze, original, font, output, journal_root):
    from .dual_coordination_env import DualCoordinationEnv
    from .dual_coordination_evaluate import _bound_file, _selector, capture_mission
    from .dual_coordination_heldout import AblationSelector
    from .dual_coordination_media import validate_camera
    import torch

    freeze.require_current()
    root = freeze.binding.root
    output = Path(output).resolve()
    _require(output.is_relative_to(root), "Final media must remain inside frozen campaign root")
    original = _bound_file(freeze, "assets", original)
    # Fonts are immutable display assets; they may be frozen with the camera.
    font = Path(font).resolve()
    font_records = [
        r
        for role in ("assets", "camera")
        for r in freeze.binding.files[role]
        if font.is_file() and (font.stat().st_size, sha256(font)) == (r["bytes"], r["sha256"])
    ]
    _require(
        bool(font_records),
        "Frozen Japanese font required",
    )
    for path in Path(__file__).parent.glob("*.py"):
        _bound_file(freeze, "source", path)
    camera_paths = []
    for record in freeze.binding.files["camera"]:
        path = _path(root, record["path"])
        if path.suffix == ".json":
            value = json.loads(path.read_text())
            if value.get("schema") == "dual-camera-v1":
                validate_camera(value)
                camera_paths.append(path)
    _require(len(camera_paths) == 1, "One exact frozen dual-camera-v1 profile required")
    declaration_path = _bound_named(freeze, "declaration", "declaration.json")
    challenge_path = _bound_named(freeze, "declaration", "development-challenges.json")
    declaration = json.loads(declaration_path.read_text())
    _require(
        declaration["files"].get(challenge_path.name)
        == dict(bytes=challenge_path.stat().st_size, sha256=sha256(challenge_path)),
        "Development challenges differ from pre-learning declaration",
    )
    cases = declared_video_cases(declaration, json.loads(challenge_path.read_text()))
    seed = freeze.receipt["selected_seed"]
    terminal_records = freeze.binding.files[f"terminal{seed}"]
    _require(len(terminal_records) == 1, "One frozen selected terminal model required")
    terminal = _path(root, terminal_records[0]["path"])
    design_records = freeze.binding.files["controller"]
    _require(len(design_records) == 1, "One frozen controller required")
    design = json.loads(_path(root, design_records[0]["path"]).read_text())
    backend = freeze.receipt["benchmark"]["selected_backend"]
    _require(backend in ("native-cpu", "native-cuda"), "Qualified native backend required for final video")
    device = "cuda" if backend == "native-cuda" else "cpu"
    _require(device == "cpu" or torch.cuda.is_available(), "Frozen inference device unavailable")
    library = _freeze_library(freeze)
    output.mkdir(parents=True, exist_ok=False)
    journal = RunJournal(journal_root or output)
    attempt = journal.start(
        "final-three-videos",
        "Frozen selected NN; three exact predeclared development cases",
        category="video generation and checking",
    )
    identity = dict(
        freeze_sha256=freeze.digest,
        selected_seed=seed,
        nn_weights=terminal_records[0],
        camera=file_record(camera_paths[0], root),
        font=deepcopy(font_records[0]),
        source_sha256=_digest(freeze.binding.files["source"]),
        development_declaration=file_record(declaration_path, root),
        challenges=file_record(challenge_path, root),
    )
    atomic_json(
        output / "declaration.json",
        dict(
            schema="dual-final-media-run-v1",
            identity=identity,
            cases={role: _digest(case) for role, case in cases.items()},
            method="N-full",
            human_review_performed=False,
        ),
    )
    videos = []
    failures = []
    try:
        for role, case in cases.items():
            freeze.require_current()
            slot = output / role
            slot.mkdir()
            atomic_json(
                slot / "media-owned-slot.json",
                dict(schema="dual-media-owned-slot-v1", freeze_sha256=freeze.digest),
            )
            try:
                choose = AblationSelector(
                    _selector(f"N{seed}", case, terminal, device), "N-full", slot / "nn-proposals.jsonl"
                )
                env = DualCoordinationEnv(
                    original,
                    slot / "capture",
                    case,
                    design,
                    mode="development",
                    retain_replay=True,
                    replay_stride_ticks=1,
                    native_audit_library=library,
                )
                capture = capture_mission(
                    env,
                    choose,
                    case,
                    slot / "capture",
                    heartbeat=lambda tick: journal.heartbeat(
                        attempt, completed=len(videos), total=3, reason=f"{role}: native tick {tick}"
                    ),
                )
                del env
                _require(
                    choose.counters["forward_passes"] == len(capture["actions"]),
                    "Every captured decision must use a fresh policy forward",
                )
                verification = verify_render_capture(
                    capture, slot / "capture", case, auditor_hash=_digest(freeze.binding.files["audit"])
                )
                verification["nn_proposals"] = file_record(slot / "nn-proposals.jsonl", root)
                verification["identity"] = identity
                atomic_json(slot / "native-verification.json", verification)
                render = render_development_video(
                    slot / "capture",
                    original,
                    slot / "render",
                    font,
                    method="N-full",
                    journal_root=journal_root or output,
                )
                metrics = render["metrics"]
                safe = capture["outcome"]["safety_violations"] == 0 and metrics["safety_violations"] == 0
                completed = capture["outcome"]["success"] and metrics["settling_passed"]
                behavior = dict(
                    schema="dual-media-behavior-v1",
                    role=role,
                    case_sha256=_digest(case),
                    identity=identity,
                    native_verification=file_record(slot / "native-verification.json", root),
                    render_verification=file_record(slot / "render" / "render-verification.json", root),
                    metrics=metrics,
                    success=completed,
                    safety_violations=capture["outcome"]["safety_violations"],
                    measured_transfer_passed=completed
                    and safe
                    and metrics["successful_post_event_transfers"] > 0,
                    crossing=dict(
                        predeclared=True,
                        success=completed,
                        safety_violations=capture["outcome"]["safety_violations"],
                        intervened_intervals=capture["outcome"]["intervened_intervals"],
                        nn_priority_yield_visible=False,
                    ),
                    learned_yield_visibility="pending hash-bound human review; action counts alone do not prove attribution",
                    human_review_performed=False,
                    final_acceptance=False,
                )
                atomic_json(slot / "behavior-gate.json", behavior)
                view = slot / "render" / "video" / "view.mp4"
                videos.append(
                    dict(
                        role=role,
                        **file_record(view, output),
                        master=file_record(slot / "render" / "video" / "hud-master.mkv", output),
                        metrics=metrics,
                        behavior_gate=file_record(slot / "behavior-gate.json", output),
                        machine_verified=True,
                    )
                )
                freeze.require_current()
                retire_verified_render_trace(
                    slot / "capture", slot / "render", verification, freeze_digest=freeze.digest
                )
            except Exception as error:
                failure = dict(role=role, case_id=case["case_id"], reason=f"{type(error).__name__}: {error}")
                atomic_json(slot / "failure.json", failure)
                failures.append(failure)
            journal.heartbeat(
                attempt,
                completed=len(videos),
                total=3,
                reason="Completed slots retained; no case/event substitution",
            )
        manifest = dict(
            schema="dual-media-v1",
            finalized_at=datetime.now(timezone.utc).isoformat(),
            identity=identity,
            videos=videos,
            failures=failures,
            complete=len(videos) == 3 and not failures,
            human_review_performed=False,
            final_acceptance=False,
        )
        atomic_json(output / "media-manifest.json", manifest)
        journal.finish(
            attempt,
            status="done" if manifest["complete"] else "failed",
            reason="Three exact files ready for human review"
            if manifest["complete"]
            else "Missing media retained as failures",
        )
        return manifest
    except BaseException as error:
        journal.finish(attempt, status="failed", reason=str(error))
        raise


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("development", "final"):
        sub = commands.add_parser(name)
        sub.add_argument("--original", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--font", required=True)
        sub.add_argument("--journal-root")
        sub.add_argument("--source" if name == "development" else "--freeze", required=True)
    args = parser.parse_args(argv)
    if args.command == "development":
        result = render_development_video(
            args.source, args.original, args.output, args.font, journal_root=args.journal_root
        )
    else:
        result = run_final_media(
            freeze=VerifiedJointFreeze.load(args.freeze),
            original=args.original,
            font=args.font,
            output=args.output,
            journal_root=args.journal_root,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("full_mission_machine_checked", result.get("complete", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
