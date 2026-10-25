import copy
import json
import pytest
from adaptive_drone_patrol.payload_mujoco_close_media import load_close_camera
from adaptive_drone_patrol.payload_solver_media import native_frame_rows


def test_close_camera_main_scale_and_inset_bounds(tmp_path):
    c = load_close_camera()
    assert c["main"]["distance"] < 1
    assert c["main"]["width"] == 1280
    c = copy.deepcopy(c)
    c["inset_xy"] = [1200, 16]
    p = tmp_path / "camera.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="fit"):
        load_close_camera(p)


def test_native_camera_cannot_hide_missing_interval_or_retarget_motion():
    actual = {"time_s": 0, "qpos": [0]}
    rows = [
        dict(episode_id=1, kind="RESET", step=0, actual=actual),
        dict(episode_id=1, kind="RETARGET", step=0, actual={**actual, "qpos": [1]}),
    ]
    with pytest.raises(ValueError, match="changed actual"):
        list(native_frame_rows(rows))
    rows[1] = dict(episode_id=1, kind="STEP", step=2, actual={"time_s": 0.02})
    with pytest.raises(ValueError, match="Missing"):
        list(native_frame_rows(rows))


def test_caption_binding_rejects_inaccurate_backend_with_rehashed_metadata(tmp_path):
    from adaptive_drone_patrol.payload_mujoco_close_media import verify_caption_binding
    from adaptive_drone_patrol.payload_hybrid_contract import VIDEO_PROFILE
    from adaptive_drone_patrol.util import sha256

    (tmp_path / "mission").mkdir()
    (tmp_path / "flight").mkdir()
    design = tmp_path / "flight/design.json"
    design.write_text(json.dumps({"recipe": "C-local-lqr-v1"}))
    meta = dict(
        controller="C-local-lqr-v1",
        backend="OTHER",
        flight_nn="NOT_USED",
        settling_profile=VIDEO_PROFILE.id,
        settling_profile_sha256=VIDEO_PROFILE.sha256,
        aggregate_simulated_deadline_s=None,
        leg_deadline_s=30.0,
        mission_wall_deadline_s=600.0,
        checkpoint_sha256=sha256(design),
    )
    (tmp_path / "mission/metadata.json").write_text(json.dumps(meta))
    (tmp_path / "flight/runtime-properties.json").write_text("{}")
    prior = {
        "physical_artifacts": {str(p.relative_to(tmp_path)): sha256(p) for p in tmp_path.rglob("*.json")}
    }
    with pytest.raises(ValueError, match="Fixed captions"):
        verify_caption_binding(tmp_path, prior)


def test_display_proof_rejects_inset_occlusion_even_with_full_canvas_bounds(monkeypatch):
    from adaptive_drone_patrol import payload_mujoco_close_media as media

    monkeypatch.setattr(media, "verify_caption_binding", lambda *_: {})
    prior = dict(independent_mission={}, physical_artifacts={})
    proof = dict(
        prior_video_proof_sha256="KNOWN",
        independent_mission={},
        physical_artifacts={},
        new_physical_success_count=0,
        lossless={"frames": 1},
        framing_all_rendered_frames=[{"time_s": 0, "bounds_px": [900, 10, 1000, 500]}] * 2,
    )
    with pytest.raises(ValueError, match="occludes"):
        media.verify_display_proof(proof, None, prior, "KNOWN", load_close_camera())
    proof["prior_video_proof_sha256"] = "ALTERED"
    with pytest.raises(ValueError, match="SHA differ"):
        media.verify_display_proof(proof, None, prior, "KNOWN", load_close_camera())
