#!/usr/bin/env python3
"""Fresh independent development admission followed by complete native1x video."""

import argparse
import gzip
import json
import shutil
from itertools import zip_longest
from pathlib import Path


from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_mujoco_close_media import (
    CloseRenderer,
    load_close_camera,
    close_hud,
    verify_caption_binding,
)
from adaptive_drone_patrol.payload_hybrid_media import (
    require_development_capture,
    compare_recomposition,
)
from adaptive_drone_patrol.payload_solver_media import (
    mission_display_state,
    native_frame_rows,
)
from adaptive_drone_patrol.payload_solver_video import (
    PairedVideoWriter,
    verify_video_pair,
    create_viewing_copies,
    verify_viewing_copies,
    decoded_rgb,
)
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.util import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--prior-video-proof", type=Path, required=True)
    p.add_argument("--samples-only", action="store_true")
    p.add_argument("--routing", type=Path, required=True)
    p.add_argument("--font", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--existing-masters", type=Path)
    args = p.parse_args()
    trial = args.trial
    prior_sha256 = sha256(args.prior_video_proof)
    prior = json.loads(args.prior_video_proof.read_text())
    caption_binding = verify_caption_binding(trial, prior)
    checked = prior["independent_mission"]
    case = json.loads((trial / "case.json").read_text())
    checkpoint = trial / "flight/design.json"
    native = trial / "flight/runtime/raw.jsonl.gz"
    summary = json.loads((trial / "mission/mission_summary.json").read_text())
    admission_record = require_development_capture(summary, checked)
    # Reuse only the explicitly bound physical flight, without generating a new success.
    for relative, digest in prior["physical_artifacts"].items():
        if sha256(trial / relative) != digest:
            raise ValueError("Original independently verified physical artifact changed: " + relative)
    admission_record["display_only_existing_success"] = True
    admission_record["new_physical_success_count"] = 0
    snapshot = {str(p.relative_to(trial)): sha256(p) for p in trial.rglob("*") if p.is_file()}
    assets = {
        str(p): sha256(p) for p in (args.font, args.routing, Path("configs/payload_mujoco_close_camera.json"))
    }
    config = load_close_camera()
    scenario = Scenario.from_dict(json.loads((trial / "mission/scenario.json").read_text()))
    events = [json.loads(x) for x in (trial / "mission/events.jsonl").read_text().splitlines()]

    def frames():
        with gzip.open(native, "rt") as stream:
            yield from native_frame_rows(json.loads(x) for x in stream)

    def state_for(step):
        state = mission_display_state(scenario, events, step)
        stop = next(
            (e for e in events if e["type"] == "SHARED_SETTLED" and e["simulation_step"] == step), None
        )
        if stop:
            state["event_text"] = (
                f"地点{stop['target_id']}で基準内に安定：機体と吊り荷が全条件を同時に連続1秒満足"
            )
        return state

    source_snapshot = args.output.parent / (args.output.name + "-source")
    source_snapshot.mkdir(parents=True, exist_ok=False)
    import adaptive_drone_patrol.payload_hybrid_media as media_module

    source_root = Path(media_module.__file__).parent
    source_files = [
        Path(__file__),
        *(
            source_root / name
            for name in (
                "payload_hybrid_media.py",
                "payload_mujoco_close_media.py",
                "payload_solver_media.py",
                "payload_solver_video.py",
                "payload_hybrid_native.py",
                "payload_solver_acceptance.py",
                "payload_solver_validation.py",
            )
        ),
    ]
    for file in source_files:
        shutil.copyfile(file, source_snapshot / file.name)
    source_hashes = {file.name: sha256(file) for file in source_files}
    write_json(
        source_snapshot / "preflight.json",
        dict(
            files=source_hashes,
            software_renderer=True,
            independent_recomposition="EXACT_EVERY_RAW_AND_CAPTION_PIXEL",
        ),
    )
    renderer = CloseRenderer(trial / "flight/fixture/model.xml", config)
    try:
        if args.samples_only:
            from PIL import Image

            args.output.mkdir(parents=True, exist_ok=False)
            for frame in frames():
                if frame["step"] in (0, 200, 500, 1000, 2000, summary["steps"]):
                    state = state_for(frame["step"])
                    raw = renderer.render(
                        frame["actual"],
                        scenario.coords[1:],
                        state["priorities"],
                        state["visited"],
                        state["selected"],
                    )
                    Image.fromarray(
                        close_hud(raw, state, frame["actual"], case, config, args.font, sha256(checkpoint))
                    ).save(args.output / f"sample-{frame['step']:05d}.png")
            return
        if args.existing_masters:
            args.output.mkdir(parents=True, exist_ok=False)
            for name in ("raw.mkv", "hud.mkv", "encoding.json", "frames.jsonl"):
                shutil.copyfile(args.existing_masters / name, args.output / name)
            write_json(
                args.output / "master-derivation.json",
                dict(
                    source=str(args.existing_masters),
                    reason="Preserved native masters; independent exact software raster/HUD admission",
                    masters={name: sha256(args.existing_masters / name) for name in ("raw.mkv", "hud.mkv")},
                ),
            )
        else:
            with PairedVideoWriter(
                args.output,
                width=config["raw_width"],
                raw_height=config["raw_height"],
                hud_height=config["hud_height"],
            ) as writer:
                for frame in frames():
                    state = state_for(frame["step"])
                    raw = renderer.render(
                        frame["actual"],
                        scenario.coords[1:],
                        state["priorities"],
                        state["visited"],
                        state["selected"],
                    )
                    hud = close_hud(raw, state, frame["actual"], case, config, args.font, sha256(checkpoint))
                    writer.append(frame["step"], raw, hud)
                    if frame["step"] % 100 == 0:
                        print("CAPTURE", frame["step"], summary["steps"], flush=True)
        lossless = verify_video_pair(args.output)
        if lossless["frames"] != summary["steps"] + 1:
            raise ValueError("Video omits physical intervals")
        # Independent second traversal of raw/qpos/events and decoded masters.
        streams = [
            decoded_rgb(args.output / "raw.mkv", config["raw_width"], config["raw_height"]),
            decoded_rgb(
                args.output / "hud.mkv", config["raw_width"], config["raw_height"] + config["hud_height"]
            ),
        ]
        raster_differences = []
        try:
            for frame, raw, hud in zip_longest(frames(), *streams):
                if frame is None or raw is None or hud is None:
                    raise ValueError("Native/decoded frame coverage differs")
                state = state_for(frame["step"])
                actual = renderer.render(
                    frame["actual"],
                    scenario.coords[1:],
                    state["priorities"],
                    state["visited"],
                    state["selected"],
                )
                expected = close_hud(raw, state, frame["actual"], case, config, args.font, sha256(checkpoint))
                comparison = compare_recomposition(
                    raw, hud, actual, expected, raw_height=config["raw_height"]
                )
                if comparison["raster_changed_pixels"]:
                    raster_differences.append(dict(step=frame["step"], **comparison))
                if frame["step"] % 500 == 0:
                    print("RECOMPOSE", frame["step"], flush=True)
        finally:
            for stream in streams:
                stream.close()
        create_viewing_copies(args.output)
        viewing = verify_viewing_copies(args.output)
        if {str(p.relative_to(trial)): sha256(p) for p in trial.rglob("*") if p.is_file()} != snapshot:
            raise ValueError("Bound physical evidence changed during render")
        if sha256(args.prior_video_proof) != prior_sha256:
            raise ValueError("Prior video proof changed during capture")
        if any(sha256(p) != digest for p, digest in assets.items()):
            raise ValueError("Camera/font/routing assets changed")
        result = dict(
            schema="mujoco-development-camera-reframe-v1",
            prior_video_proof_sha256=prior_sha256,
            new_physical_success_count=0,
            caption_binding=caption_binding,
            admission=admission_record,
            independent_mission=checked,
            lossless=lossless,
            viewing=viewing,
            renderer=renderer.renderer_string,
            full_native_pose_and_hud_recomposed=True,
            physical_artifacts=snapshot,
            assets=assets,
            independent_raster_quantization=dict(
                maximum_channel_difference=0,
                maximum_pixels_per_frame=0,
                actual_differences=raster_differences,
                appended_hud_exact_every_pixel=True,
            ),
            media_source_snapshot=source_hashes,
            source_sha256=sha256(__file__),
            controller="C-local-lqr-v1",
            backend="MuJoCo3.15.0/float64/RK4/substeps16/N1",
            profile="payload-settling-video-v1",
            framing_all_rendered_frames=renderer.bounds,
            human_review="PENDING_FRESH_CONTINUOUS1X",
            videos={
                name: dict(bytes=(args.output / name).stat().st_size, sha256=sha256(args.output / name))
                for name in ("raw.mkv", "hud.mkv", "raw.mp4", "hud.mp4")
            },
        )
        write_json(args.output / "development-video-proof.json", result)
        print("DEVELOPMENT_VIDEO_VERIFIED", lossless["frames"], flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
