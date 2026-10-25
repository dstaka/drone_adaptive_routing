"""Pure final-media rejection and framing/caption contracts; no flight admission."""

import numpy as np
import pytest

from adaptive_drone_patrol.payload_mujoco_final_media import (
    require_final_case,
    verify_frame_coverage,
    capture_final_three,
)


def test_final_capture_rejects_boolean_or_development_admission_before_io(tmp_path):
    with pytest.raises(ValueError, match="FinalAcceptance"):
        capture_final_three(
            acceptance={"accepted": True},
            formal_root=tmp_path,
            output=tmp_path / "out",
            font=tmp_path / "missing.ttf",
            camera=tmp_path / "missing.json",
        )
    assert not (tmp_path / "out").exists()


def test_only_exact_formal_nominal_basic_scenario_seed1_is_eligible():
    expected = dict(
        case_id="basic-normal-1",
        suite="basic",
        scenario="normal",
        execution_seed=1,
        side_m=0.045,
        mass_ratio=0.2,
        initial={"EXTERNAL_FROZEN_CASE": 1},
    )
    require_final_case(expected, expected, "normal")
    for field, value in [
        ("case_id", "basic-normal-2"),
        ("execution_seed", 2),
        ("suite", "final-development"),
        ("side_m", 0.06),
        ("mass_ratio", 0.1),
        ("initial", {}),
    ]:
        bad = dict(expected)
        bad[field] = value
        with pytest.raises(ValueError):
            require_final_case(bad, expected, "normal")
    with pytest.raises(ValueError):
        require_final_case(expected, expected, "emergency")


def test_controller_binding_accepts_same_values_reformatted_by_physical_recorder(tmp_path):
    import json
    from adaptive_drone_patrol.payload_mujoco_final_media import require_controller_binding
    from adaptive_drone_patrol.payload_hybrid_final import digest
    from adaptive_drone_patrol.util import sha256

    design = dict(recipe="C-local-lqr-v1", gain=[[0.1]], hover_action=[0.2])
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design, indent=2) + "\n")
    binding = dict(design_value_sha256=digest(design))
    assert require_controller_binding(path, binding) == sha256(path)
    design["gain"][0][0] = 0.11
    path.write_text(json.dumps(design))
    with pytest.raises(ValueError):
        require_controller_binding(path, binding)


def test_frame_coverage_requires_every_instant_both_layout_neutral_traversals():
    bounds = [
        dict(time_s=i * 0.01, layout_bounds={"ROOT_OWNS_LAYOUT": True}) for _ in range(2) for i in range(3)
    ]
    assert verify_frame_coverage(bounds, frames=3) == dict(complete_traversals=2, frames_per_traversal=3)
    for bad in (
        bounds[:-1],
        [dict(bounds[0], time_s=0.1), *bounds[1:]],
        [dict(bounds[0], time_s=float("nan")), *bounds[1:]],
    ):
        with pytest.raises(ValueError):
            verify_frame_coverage(bad, frames=3)


def test_caption_audit_rejects_erased_single_status_glyph_in_injected_region():
    from adaptive_drone_patrol.payload_solver_video import verify_caption_regions

    original = np.zeros((1200, 1792, 3), np.uint8)
    original[733:750, 25:36] = 255
    changed = original.copy()
    changed[733:750, 25:36] = 0
    regions = {"actual_status": [12, 726, 1780, 764]}
    with pytest.raises(ValueError):
        verify_caption_regions(original, changed, regions)


def test_media_module_adapter_rejects_nonproject_dynamic_module():
    from adaptive_drone_patrol.payload_mujoco_final_media import MediaAdapter

    with pytest.raises(ValueError):
        MediaAdapter.from_module("untrusted.renderer")


def test_media_adapter_rejects_module_shadowing_actual_project_source(monkeypatch):
    import sys
    from types import ModuleType
    from adaptive_drone_patrol.payload_mujoco_final_media import MediaAdapter

    module = ModuleType("adaptive_drone_patrol.shadow_media")
    module.__file__ = "/tmp/OUTSIDE_PROJECT_MEDIA.py"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="source"):
        MediaAdapter.from_module(module.__name__)


def test_media_adapter_cannot_substitute_unbound_hud_callback():
    from dataclasses import replace
    from adaptive_drone_patrol.payload_mujoco_final_media import MediaAdapter

    actual = MediaAdapter.from_module("adaptive_drone_patrol.payload_mujoco_overview_media")
    forged = replace(actual, hud=lambda *args: None)
    with pytest.raises(ValueError, match="callbacks"):
        forged.require_current()


def test_explicit_overview_adapter_keeps_original_detail_and_checks_all_target_bounds():
    from pathlib import Path
    from adaptive_drone_patrol.payload_mujoco_final_media import MediaAdapter
    from adaptive_drone_patrol.payload_mujoco_overview_media import FixedOverviewRenderer
    from adaptive_drone_patrol.payload_solver_media import load_camera

    root = Path(__file__).resolve().parents[1]
    adapter = MediaAdapter.from_module("adaptive_drone_patrol.payload_mujoco_overview_media")
    assert adapter.renderer_factory is FixedOverviewRenderer
    config = adapter.load_camera(root / "configs/payload_mujoco_overview_camera.json")
    assert config["detail"] == load_camera(root / "configs/payload_solver_camera.json")["detail"]
    assert adapter.caption_regions(config)
    rows = [
        dict(
            time_s=i * 0.01,
            bodies_px=[400.0, 200.0, 800.0, 600.0],
            all_ten_targets_px=[100.0, 100.0, 1100.0, 600.0],
        )
        for _ in range(2)
        for i in range(3)
    ]
    verify_frame_coverage(rows, frames=3)
    assert adapter.verify_framing(rows, frames=3, config=config)["all_ten_destinations_visible"]
    rows[0]["all_ten_targets_px"][2] = 1279.0
    with pytest.raises(ValueError):
        adapter.verify_framing(rows, frames=3, config=config)


@pytest.mark.parametrize("with_receipt", [False, True])
def test_capture_cli_binds_optional_reader_receipt_to_context_and_media_inputs(
    tmp_path, monkeypatch, with_receipt
):
    import importlib.util
    from pathlib import Path
    import sys
    from types import SimpleNamespace

    script = Path(__file__).resolve().parents[1] / "scripts/capture_payload_mujoco_final.py"
    spec = importlib.util.spec_from_file_location("final_media_cli_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    recorded = {}

    def context_factory(reference, **kwargs):
        recorded["context"] = kwargs
        return "context"

    def capture(**kwargs):
        recorded["capture"] = kwargs
        return {"TEST_ONLY": True}

    monkeypatch.setattr(cli, "FinalContext", SimpleNamespace(from_reference=context_factory))
    monkeypatch.setattr(cli, "ReferenceAdmission", SimpleNamespace(from_rebinding=lambda _: "reference"))
    monkeypatch.setattr(cli, "FinalFreeze", SimpleNamespace(load=lambda *args: "freeze"))
    monkeypatch.setattr(cli, "FinalAcceptance", SimpleNamespace(from_campaign=lambda *args: "accepted"))
    monkeypatch.setattr(cli, "MediaAdapter", SimpleNamespace(from_module=lambda _: "media"))
    monkeypatch.setattr(cli, "capture_final_three", capture)
    monkeypatch.setitem(
        sys.modules,
        "adaptive_drone_patrol.routing_agent",
        SimpleNamespace(RoutingAgent=lambda *args, **kwargs: "planner"),
    )
    (tmp_path / "reference-rebinding").write_text("{}")
    names = (
        "reference-rebinding",
        "design",
        "original",
        "routing",
        "freeze",
        "formal-root",
        "font",
        "output",
        "camera",
    )
    argv = [str(script)]
    for name in names:
        argv.extend(["--" + name, str(tmp_path / name)])
    argv.extend(["--media-module", "adaptive_drone_patrol.TEST_ONLY"])
    receipt = tmp_path / "reader-rebinding.json"
    if with_receipt:
        receipt.write_text('{"TEST_ONLY_PINNED_RECEIPT": true}')
        argv.extend(["--reader-rebinding", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert recorded["context"]["reader_rebinding"] == (receipt if with_receipt else None)
    inputs = recorded["capture"]["input_paths"]
    if with_receipt:
        assert inputs["reader_rebinding"] == receipt
    else:
        assert "reader_rebinding" not in inputs


def test_loaded_assets_binds_xml_includes_compiler_mesh_and_cube_textures(tmp_path):
    from adaptive_drone_patrol.payload_mujoco_final_media import loaded_assets, verify_loaded_assets
    from adaptive_drone_patrol.util import sha256

    (tmp_path / "parts").mkdir()
    (tmp_path / "assets").mkdir()
    model = tmp_path / "model.xml"
    model.write_text('<mujoco><compiler assetdir="assets"/><include file="parts/visual.xml"/></mujoco>')
    included = tmp_path / "parts/visual.xml"
    included.write_text(
        '<mujoco><asset><mesh file="body.obj"/><texture fileup="up.png" '
        'filedown="down.png"/></asset></mujoco>'
    )
    for name in ("body.obj", "up.png", "down.png"):
        (tmp_path / "assets" / name).write_bytes(name.encode())
    expected = [model, included, *(tmp_path / "assets" / n for n in ("body.obj", "up.png", "down.png"))]
    bound = loaded_assets(model)
    assert bound == {str(p): dict(sha256=sha256(p), bytes=p.stat().st_size) for p in expected}
    verify_loaded_assets(model, bound)
    (tmp_path / "assets/body.obj").write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="asset"):
        verify_loaded_assets(model, bound)


def test_loaded_assets_nested_include_resolution_and_absolute_mesh(tmp_path):
    from adaptive_drone_patrol.payload_mujoco_final_media import loaded_assets

    sub = tmp_path / "sub"
    sub.mkdir()
    mesh = tmp_path / "body.obj"
    mesh.write_bytes(b"ORIGINAL")
    model = tmp_path / "model.xml"
    model.write_text('<mujoco><include file="sub/first.xml"/></mujoco>')
    (sub / "first.xml").write_text('<mujoco><include file="second.xml"/></mujoco>')
    (sub / "second.xml").write_text(f'<mujoco><asset><mesh file="{mesh}"/></asset></mujoco>')
    assert len(loaded_assets(model)) == 4
    mesh.unlink()
    with pytest.raises(ValueError, match="asset"):
        loaded_assets(model)


def test_loaded_assets_rejects_unknown_file_source_and_include_cycle(tmp_path):
    from adaptive_drone_patrol.payload_mujoco_final_media import loaded_assets

    model = tmp_path / "model.xml"
    model.write_text('<mujoco><plugin file="UNBOUND"/></mujoco>')
    with pytest.raises(ValueError, match="Unsupported"):
        loaded_assets(model)
    model.write_text('<mujoco><include file="model.xml"/></mujoco>')
    with pytest.raises(ValueError, match="Repeated"):
        loaded_assets(model)
