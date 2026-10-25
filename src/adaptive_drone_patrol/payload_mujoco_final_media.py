"""Final3 media replay of already accepted basic seed1 physical recordings.

No flight/reset/optimizer is instantiated. A single typed formal acceptance is
rechecked for each of three100Hz native-pose replays and both complete traversals.
Machine physical acceptance and pending continuous1x human review stay distinct.
"""

import ast
from dataclasses import dataclass
import gzip
import importlib
import importlib.metadata
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from .payload_hybrid_final import FinalAcceptance, digest, require_formal_declaration
from .payload_hybrid_media import compare_recomposition
from .payload_solver_media import native_frame_rows, mission_display_state
from .payload_solver_video import (
    PairedVideoWriter,
    verify_video_pair,
    create_viewing_copies,
    verify_viewing_copies,
    decoded_rgb,
    verify_caption_regions,
)
from .scenario import Scenario
from .util import sha256

SCENARIOS = ("normal", "emergency", "cost_shift")
SOURCE_NAMES = (
    "payload_mujoco_final_media",
    "payload_solver_media",
    "payload_solver_video",
    "payload_hybrid_media",
    "payload_hybrid_final",
    "payload_hybrid_final_audit",
    "payload_hybrid_contract",
    "payload_contract",
    "payload_solver_trace",
    "payload_solver_mujoco",
    "scenario",
    "video",
    "util",
    "jobs",
)


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def snapshot(root):
    root = Path(root).resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in files):
        raise ValueError("Complete immutable regular input files required")
    return {str(p.relative_to(root)): sha256(p) for p in files}


def _asset_entry(path):
    path = Path(path)
    before = path.stat()
    if not path.is_file():
        raise ValueError("Loaded asset must be a regular file")
    value = dict(bytes=before.st_size, sha256=sha256(path))
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(after, key) != getattr(before, key) for key in stable):
        raise ValueError("Loaded asset changed while hashing")
    return value


def loaded_assets(model_path):
    """Bind the filesystem MJCF/include/asset closure without loading a model.

    File resources only: unsupported file-bearing tags and ambiguous relative
    asset resolution fail closed. Includes follow MuJoCo's main-directory first,
    including-directory fallback. No model/asset file is rewritten for rendering.
    """
    model = Path(model_path).resolve(strict=True)
    root_dir = model.parent
    files, documents, visited = {}, [], set()

    def bind(path):
        path = Path(path).resolve(strict=True)
        value = _asset_entry(path)
        if str(path) in files and files[str(path)] != value:
            raise ValueError("Shared loaded asset changed during closure discovery")
        files[str(path)] = value
        return path

    def xml(path):
        path = bind(path)
        if path in visited:
            raise ValueError("Repeated XML include/model dependency")
        visited.add(path)
        document = ET.parse(path).getroot()
        documents.append((document, path.parent))
        for node in document.iter("include"):
            name = node.get("file")
            if not name or list(node):
                raise ValueError("Explicit file-only XML include required")
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = root_dir / name
                if not candidate.is_file():
                    candidate = path.parent / name
            if not candidate.is_file():
                raise ValueError("Missing loaded XML include asset")
            xml(candidate)

    xml(model)
    compiler = {}
    for document, _ in documents:
        for node in document.iter("compiler"):
            for key in ("assetdir", "meshdir", "texturedir", "strippath"):
                if key in node.attrib:
                    if key in compiler and compiler[key] != node.get(key):
                        raise ValueError("Ambiguous compiler asset directories")
                    compiler[key] = node.get(key)
    strip = compiler.get("strippath", "false")
    if strip not in ("false", "true"):
        raise ValueError("Explicit compiler strippath boolean required")
    for document, directory in documents:
        for node in document.iter():
            for key, name in node.attrib.items():
                if key != "file" and key not in (
                    "fileleft",
                    "fileright",
                    "fileup",
                    "filedown",
                    "filefront",
                    "fileback",
                ):
                    continue
                if node.tag == "include":
                    continue
                if node.tag not in ("mesh", "texture", "hfield", "skin", "model"):
                    raise ValueError("Unsupported file-bearing loaded asset element")
                if key != "file" and node.tag != "texture":
                    raise ValueError("Unsupported loaded asset file attribute")
                if not name:
                    continue
                filename = Path(name).name if strip == "true" else name
                candidate = Path(filename)
                if not candidate.is_absolute():
                    option = "texturedir" if node.tag == "texture" else "meshdir"
                    asset_dir = compiler.get(option, compiler.get("assetdir", ""))
                    bases = [root_dir]
                    if directory != root_dir:
                        bases.append(directory)
                    candidates = {(base / asset_dir / filename).resolve() for base in bases}
                    existing = {p for p in candidates if p.is_file()}
                    if len(existing) != 1:
                        raise ValueError("Missing or ambiguous loaded asset file")
                    candidate = existing.pop()
                if not candidate.is_file():
                    raise ValueError("Missing loaded asset file")
                if node.tag == "model":
                    nested = loaded_assets(candidate)
                    for path, value in nested.items():
                        if path in files and files[path] != value:
                            raise ValueError("Shared loaded asset model changed")
                        files[path] = value
                else:
                    bind(candidate)
    for path, value in files.items():
        if _asset_entry(path) != value:
            raise ValueError("Loaded asset changed during closure discovery")
    return dict(sorted(files.items()))


def verify_loaded_assets(model_path, expected):
    """Recompute actual external bytes and reject modified/added dependencies."""
    if loaded_assets(model_path) != expected:
        raise ValueError("Loaded asset closure/bytes differ from bound render inputs")


def require_final_case(case, expected, scenario):
    if (
        scenario not in SCENARIOS
        or case != expected
        or case.get("case_id") != f"basic-{scenario}-1"
        or case.get("suite") != "basic"
        or case.get("scenario") != scenario
        or type(case.get("execution_seed")) is not int
        or case["execution_seed"] != 1
        or case.get("side_m") != 0.045
        or case.get("mass_ratio") != 0.2
    ):
        raise ValueError("Exact accepted formal nominal basic scenario seed1 case required")


def require_controller_binding(path, binding):
    actual = json.loads(Path(path).read_text())
    if actual.get("recipe") != "C-local-lqr-v1" or digest(actual) != binding["design_value_sha256"]:
        raise ValueError("Actual caption controller differs from final frozen controller values")
    return sha256(path)


@dataclass(frozen=True)
class MediaAdapter:
    """Explicit renderer/layout chosen outside the accepted physical campaign."""

    module_name: str
    renderer_factory: object
    load_camera: object
    hud: object
    verify_framing: object
    caption_regions: object
    verify_caption_binding: object

    @classmethod
    def from_module(cls, name):
        if not name.startswith("adaptive_drone_patrol.") or name.count(".") != 1:
            raise ValueError("Explicit project-owned media module required")
        module = importlib.import_module(name)
        expected = Path(__file__).resolve().parent / (name.split(".")[-1] + ".py")
        if not expected.is_file() or Path(getattr(module, "__file__", "")).resolve() != expected:
            raise ValueError("Actual media adapter source must match the executed project file")
        names = (
            "FixedOverviewRenderer",
            "load_overview_camera",
            "final_hud",
            "verify_framing",
            "caption_regions",
            "verify_caption_binding",
        )
        values = [getattr(module, key) for key in names]
        if not all(callable(value) for value in values):
            raise ValueError("Complete renderer/layout/HUD auditing interface required")
        return cls(name, *values)

    def require_current(self):
        if self != type(self).from_module(self.module_name):
            raise ValueError("Actual media callbacks differ from bound project renderer/HUD source")


def verify_frame_coverage(bounds, *, frames):
    """Layout-neutral clock proof; chosen renderer must separately prove bounds."""
    if type(frames) is not int or frames < 1 or len(bounds) != 2 * frames:
        raise ValueError("Two complete framing traversals of every native instant required")
    for index, row in enumerate(bounds):
        if not np.isfinite(row["time_s"]) or abs(row["time_s"] - (index % frames) * 0.01) > 1e-7:
            raise ValueError("Complete chronological framing evidence required")
    return dict(complete_traversals=2, frames_per_traversal=frames)


def _sources(source_root, media):
    root = Path(source_root)
    package = root / "src/adaptive_drone_patrol"
    pending, seen = set(SOURCE_NAMES) | {media.module_name.split(".")[-1]}, set()
    while pending:
        name = pending.pop()
        path = package / (name + ".py")
        if name in seen or not path.is_file():
            continue
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and node.module:
                    pending.add(node.module.split(".")[0])
                elif node.module and node.module.startswith("adaptive_drone_patrol."):
                    pending.add(node.module.split(".")[1])
    paths = ["src/adaptive_drone_patrol/" + name + ".py" for name in sorted(seen)]
    paths += ["scripts/capture_payload_mujoco_final.py"]
    return {name: sha256(root / name) for name in paths}


def _frames(trial):
    with gzip.open(trial / "flight/runtime/raw.jsonl.gz", "rt") as stream:
        yield from native_frame_rows(json.loads(line) for line in stream)


def _state(scenario, events, step):
    result = mission_display_state(scenario, events, step)
    stops = [
        event for event in events if event["type"] == "SHARED_SETTLED" and event["simulation_step"] == step
    ]
    if len(stops) > 1:
        raise ValueError("Multiple physical stops at one native instant")
    if stops:
        result["event_text"] = (
            f"地点{stops[0]['target_id']}で基準内に安定：機体と吊り荷が全条件を同時に連続1秒満足"
        )
    return result


def capture_final_three(*, acceptance, formal_root, output, font, camera, media=None, input_paths=None):
    if not isinstance(acceptance, FinalAcceptance):
        raise ValueError("Typed independently reconstructed FinalAcceptance required before media IO")
    if not isinstance(media, MediaAdapter):
        raise ValueError("Explicit renderer/config/HUD audit adapter required")
    media.require_current()
    acceptance.require_current()
    formal_root, output = Path(formal_root).resolve(), Path(output).resolve()
    if output.is_relative_to(formal_root):
        raise ValueError("Final media must not modify the accepted formal campaign")
    source_root = Path(__file__).resolve().parents[2]
    config = media.load_camera(camera)
    caption_regions = media.caption_regions(config)
    if not caption_regions:
        raise ValueError("Every actual caption field needs a decoded glyph region")
    declaration = require_formal_declaration(formal_root, acceptance.freeze)
    accepted = acceptance.record
    if (
        accepted.get("accepted") is not True
        or accepted.get("branch") != "controller"
        or accepted.get("methods") != 1
        or accepted.get("freeze_sha256") != digest(acceptance.freeze.record)
        or accepted.get("formal_declaration_sha256") != sha256(formal_root / "declaration.json")
    ):
        raise ValueError("Actual accepted formal campaign/freeze/source binding required")
    cases = {c["case_id"]: c for c in declaration["cases"]}
    outcomes = {r["case_id"]: r for r in accepted["outcomes"]}
    roots, physical, case_assets, all_assets = {}, {}, {}, {}
    for scenario in SCENARIOS:
        case_id = f"basic-{scenario}-1"
        trial = formal_root / case_id
        case = json.loads((trial / "case.json").read_text())
        require_final_case(case, cases[case_id], scenario)
        row = outcomes[case_id]
        if (
            row.get("success") is not True
            or row.get("physical_acceptance") is not True
            or row.get("evidence_complete") is not True
            or row.get("native_reconstructed") is not True
            or row.get("completed_targets") != 10
            or type(row.get("steps")) is not int
            or row["steps"] < 1
        ):
            raise ValueError("Accepted complete actual seed1 physical recording required")
        physical[case_id] = snapshot(trial)
        roots[case_id] = trial
        for name, value in physical[case_id].items():
            if name != "independent-check.json" and accepted["artifacts"].get(str(trial / name)) != value:
                raise ValueError("Physical recording is absent from accepted campaign artifact set")
        if json.loads((trial / "binding.json").read_text()) != acceptance.freeze.context.binding:
            raise ValueError("Physical trial source/controller differs from frozen final binding")
        media.verify_caption_binding(trial, dict(physical_artifacts=physical[case_id]))
        case_assets[case_id] = loaded_assets(trial / "flight/fixture/model.xml")
        for path, value in case_assets[case_id].items():
            if path in all_assets and all_assets[path] != value:
                raise ValueError("Shared original visual asset differs between accepted cases")
            all_assets[path] = value
    supplied = {k: Path(v).resolve() for k, v in (input_paths or {}).items()}
    supplied.update(
        camera=Path(camera).resolve(),
        font=Path(font).resolve(),
        formal_declaration=formal_root / "declaration.json",
    )
    supplied.update({f"loaded_asset_{i:04d}": Path(path) for i, path in enumerate(sorted(all_assets))})
    inputs_before = {name: dict(path=str(path), **_asset_entry(path)) for name, path in supplied.items()}
    source_before = _sources(source_root, media)
    output.mkdir(parents=True, exist_ok=False)
    media_source = output / "source"
    media_source.mkdir()
    for name, value in source_before.items():
        destination = media_source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, destination)
        if sha256(destination) != value:
            raise ValueError("Media source changed while snapshotting")
    shutil.copyfile(camera, media_source / "camera.json")
    asset_snapshot = media_source / "loaded-assets"
    asset_snapshot.mkdir()
    for path, value in all_assets.items():
        destination = asset_snapshot / value["sha256"]
        if not destination.exists():
            shutil.copyfile(path, destination)
        if _asset_entry(destination) != value:
            raise ValueError("Loaded asset bytes changed while snapshotting")
    _write(
        output / "inputs.json",
        dict(
            schema="mujoco-final3-media-inputs-v1",
            freeze_record_sha256=digest(acceptance.freeze.record),
            acceptance_record_sha256=digest(accepted),
            selected_cases=[f"basic-{s}-1" for s in SCENARIOS],
            physical_artifacts=physical,
            source_binding=source_before,
            input_binding=inputs_before,
            machine_acceptance=True,
            human_review="PENDING_FRESH_CONTINUOUS1X",
            new_physical_attempts=0,
            case_success_counts_modified=False,
            config=config,
            media_module=media.module_name,
            loaded_assets=case_assets,
            loaded_asset_snapshot="source/loaded-assets/<sha256>",
        ),
    )
    completed = []
    for scenario in SCENARIOS:
        acceptance.require_current()
        media.require_current()
        case_id = f"basic-{scenario}-1"
        trial = roots[case_id]
        target = output / case_id
        case = cases[case_id]
        checked = outcomes[case_id]
        display_scenario = Scenario.from_dict(json.loads((trial / "mission/scenario.json").read_text()))
        if display_scenario.scenario_id != scenario:
            raise ValueError("Actual displayed scenario differs from accepted case")
        events = [json.loads(line) for line in (trial / "mission/events.jsonl").read_text().splitlines()]
        controller_hash = require_controller_binding(
            trial / "flight/design.json", acceptance.freeze.context.binding
        )
        verify_loaded_assets(trial / "flight/fixture/model.xml", case_assets[case_id])
        renderer = media.renderer_factory(trial / "flight/fixture/model.xml", config)
        try:
            with PairedVideoWriter(
                target,
                width=config["raw_width"],
                raw_height=config["raw_height"],
                hud_height=config["hud_height"],
            ) as writer:
                for frame in _frames(trial):
                    state = _state(display_scenario, events, frame["step"])
                    raw = renderer.render(
                        frame["actual"],
                        display_scenario.coords[1:],
                        state["priorities"],
                        state["visited"],
                        state["selected"],
                    )
                    hud = media.hud(raw, state, frame["actual"], case, config, font, controller_hash)
                    writer.append(frame["step"], raw, hud)
                    if frame["step"] % 500 == 0:
                        print("FINAL_CAPTURE", case_id, frame["step"], checked["steps"], flush=True)
            lossless = verify_video_pair(target)
            if lossless["frames"] != checked["steps"] + 1:
                raise ValueError("Final video omitted actual native intervals")
            create_viewing_copies(target)
            viewing = verify_viewing_copies(target)
            streams = [
                decoded_rgb(target / "raw.mkv", config["raw_width"], config["raw_height"]),
                decoded_rgb(
                    target / "hud.mkv", config["raw_width"], config["raw_height"] + config["hud_height"]
                ),
                decoded_rgb(
                    target / "hud.mp4", config["raw_width"], config["raw_height"] + config["hud_height"]
                ),
            ]
            maximum_caption = dict(
                caption_regions_checked=len(caption_regions),
                maximum_region_luminance_mae=0.0,
                maximum_foreground_tile_luminance_mae=0.0,
            )
            recomposed = 0
            try:
                for frame, raw, hud, visible in zip_longest(_frames(trial), *streams):
                    if frame is None or raw is None or hud is None or visible is None:
                        raise ValueError("Native/decoded final frame coverage differs")
                    state = _state(display_scenario, events, frame["step"])
                    expected_raw = renderer.render(
                        frame["actual"],
                        display_scenario.coords[1:],
                        state["priorities"],
                        state["visited"],
                        state["selected"],
                    )
                    expected_hud = media.hud(
                        expected_raw, state, frame["actual"], case, config, font, controller_hash
                    )
                    compare_recomposition(
                        raw, hud, expected_raw, expected_hud, raw_height=config["raw_height"]
                    )
                    caption = verify_caption_regions(hud, visible, caption_regions)
                    for key in ("maximum_region_luminance_mae", "maximum_foreground_tile_luminance_mae"):
                        maximum_caption[key] = max(maximum_caption[key], caption[key])
                    recomposed += 1
                    if frame["step"] % 500 == 0:
                        print("FINAL_RECOMPOSE", case_id, frame["step"], flush=True)
            finally:
                for stream in streams:
                    stream.close()
            coverage = verify_frame_coverage(renderer.bounds, frames=lossless["frames"])
            framing = media.verify_framing(renderer.bounds, frames=lossless["frames"], config=config)
            if recomposed != checked["steps"] + 1:
                raise ValueError("Incomplete independent final recomposition")
            acceptance.require_current()
            if snapshot(trial) != physical[case_id]:
                raise ValueError("Accepted physical trial changed during final capture")
            verify_loaded_assets(trial / "flight/fixture/model.xml", case_assets[case_id])
            if _sources(source_root, media) != source_before or any(
                sha256(row["path"]) != row["sha256"] for row in inputs_before.values()
            ):
                raise ValueError("Actual media source/camera/font/input proof changed during capture")
            proof = dict(
                schema="mujoco-final-accepted-case-media-v1",
                case_id=case_id,
                scenario=scenario,
                physical_attempt_reused=True,
                new_physical_attempts=0,
                machine_acceptance=True,
                accepted_case_audit=checked,
                freeze_record_sha256=digest(acceptance.freeze.record),
                acceptance_record_sha256=digest(accepted),
                full_native_pose_and_hud_recomposed=True,
                actual_frames=recomposed,
                lossless=lossless,
                viewing=viewing,
                decoded_hud_captions=maximum_caption,
                framing_coverage=coverage,
                media_module=media.module_name,
                framing=framing,
                framing_all_rendered_frames=renderer.bounds,
                caption_binding=media.verify_caption_binding(
                    trial, dict(physical_artifacts=physical[case_id])
                ),
                physical_artifacts=physical[case_id],
                source_binding=source_before,
                input_binding=inputs_before,
                loaded_assets=case_assets[case_id],
                renderer=renderer.renderer_string,
                software_device=renderer.device,
                dependencies={
                    name: importlib.metadata.version(name)
                    for name in ("mujoco", "numpy", "Pillow", "PyOpenGL")
                },
                human_review="PENDING_FRESH_CONTINUOUS1X",
                videos={
                    name: dict(bytes=(target / name).stat().st_size, sha256=sha256(target / name))
                    for name in ("raw.mkv", "hud.mkv", "raw.mp4", "hud.mp4")
                },
            )
            _write(target / "final-video-proof.json", proof)
            completed.append(
                dict(
                    case_id=case_id,
                    proof_sha256=sha256(target / "final-video-proof.json"),
                    frames=recomposed,
                    videos=proof["videos"],
                )
            )
        finally:
            renderer.close()
    acceptance.require_current()
    media.require_current()
    for name, assets in case_assets.items():
        verify_loaded_assets(roots[name] / "flight/fixture/model.xml", assets)
    if any(_asset_entry(asset_snapshot / row["sha256"]) != row for row in all_assets.values()):
        raise ValueError("Preserved loaded asset snapshot changed before final3 receipt")
    if (
        any(snapshot(roots[name]) != value for name, value in physical.items())
        or _sources(source_root, media) != source_before
        or any(sha256(row["path"]) != row["sha256"] for row in inputs_before.values())
    ):
        raise ValueError("Accepted source/raw/media inputs changed before final3 receipt")
    result = dict(
        schema="mujoco-final3-accepted-media-v1",
        scenarios=list(SCENARIOS),
        recordings=completed,
        machine_acceptance=True,
        acceptance_record_sha256=digest(accepted),
        freeze_record_sha256=digest(acceptance.freeze.record),
        new_physical_attempts=0,
        case_success_counts_modified=False,
        all_three_complete_native1x=True,
        human_review="PENDING_FRESH_CONTINUOUS1X",
        source_binding=source_before,
        input_binding=inputs_before,
        loaded_assets=case_assets,
    )
    _write(output / "final3-video-proof.json", result)
    return result
