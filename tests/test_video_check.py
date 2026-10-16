import copy
import pytest
from adaptive_drone_patrol.video_check import verify_segment, verify_pts, verify_card_binding


def case():
    frames = [
        {"simulation_step": i * 2, "sim_time_s": i * 0.02, "frame_index": i, "event_seq": 1}
        for i in range(51)
    ]
    telemetry = [
        {"simulation_step": i, "sim_time_s": i * 0.01, "run_id": "normal_seed_1"} for i in range(1, 101)
    ]
    events = [
        {"type": "SCENARIO_START", "sim_time_s": 0, "simulation_step": 0, "event_seq": 1},
        {"type": "MISSION_FINISHED", "sim_time_s": 1, "simulation_step": 100, "event_seq": 2},
    ]
    segment = {
        "run_id": "normal_seed_1",
        "source_simulation_start_s": 0,
        "source_simulation_end_s": 1,
        "presentation_start_s": 0,
        "presentation_end_s": 1,
        "output_frame_start": 0,
        "frame_count": 30,
        "playback_rate": 1,
    }
    pts = [i / 30 for i in range(31)]
    mapping = []
    for i in range(30):
        time = 1 if i == 29 else i / 30
        index = min(50, int((time + 1e-9) / 0.02))
        mapping.append(
            {
                "output_frame_index": i,
                "run_id": "normal_seed_1",
                "source_frame_index": index,
                "simulation_step": index * 2,
                "sim_time_s": index * 0.02,
            }
        )
    return segment, pts, mapping, frames, telemetry, events


def test_checker_reconstructs_unit_slope_instead_of_trusting_rate():
    args = case()
    result = verify_segment(*args, fps=30)
    assert 0 < result["max_source_quantization_s"] <= 1 / 30
    changed = copy.deepcopy(args)
    changed[0]["presentation_end_s"] = 2
    with pytest.raises(ValueError):
        verify_segment(*changed, fps=30)
    changed = copy.deepcopy(args)
    changed[1][12] += 0.2
    with pytest.raises(ValueError):
        verify_segment(*changed, fps=30)
    changed = copy.deepcopy(args)
    changed[2][5]["run_id"] = "other"
    with pytest.raises(ValueError):
        verify_segment(*changed, fps=30)
    changed = copy.deepcopy(args)
    changed[3].pop(8)
    with pytest.raises(ValueError):
        verify_segment(*changed, fps=30)


def test_encoded_pts_reject_missing_repeated_and_scaled_samples():
    verify_pts([i / 30 for i in range(30)], 30, 30)
    for pts in [[0, 0, 0.03], [0, 0.01, 0.1], [i / 15 for i in range(30)]]:
        with pytest.raises(ValueError):
            verify_pts(pts, 30, 30)


def test_static_cards_have_no_simulation_interval_and_reject_stale_binding():
    with pytest.raises(ValueError):
        verify_card_binding({"source_simulation_start_s": 0}, [], [])
    with pytest.raises(ValueError):
        verify_card_binding(
            {"kind": "tradeoff", "run_id": "r", "decision_id": "d", "boundary_hash": "stale", "feature": {}},
            [{"run_id": "r", "decision_id": "d", "boundary_hash": "real"}],
            [],
        )


def test_independent_cost_weights_and_return_do_not_scale_time():
    from adaptive_drone_patrol.video_check import recompute_cost

    s = {
        "scenario": {
            "coords": [[0, 0, 0], [1, 0, 0]],
            "multipliers": [[1, 4], [1, 1]],
            "nominal_speed_mps": 1,
            "service_duration_s": [0],
            "priorities": [1],
            "alpha": 2,
            "beta": 1,
            "return_to_base": True,
        },
        "current": 0,
        "elapsed": 0,
        "visited": [False],
    }
    result = recompute_cost(s, [0])
    assert result == {"cost": 11, "travel_cost": 5, "completion_penalty": 1}


def test_visible_text_and_main_camera_corruption_cannot_pass():
    from PIL import Image, ImageDraw, ImageFont
    from adaptive_drone_patrol.video_check import verify_text_region, verify_camera_region

    image = Image.new("RGB", (1920, 1080), "#102333")
    font = ImageFont.load_default(size=24)
    ImageDraw.Draw(image).text((25, 930), "EVENT C: 2 -> 5", font=font, fill="#ffdc91")
    assert verify_text_region(image, "EVENT C: 2 -> 5", (25, 930), font, "#ffdc91") == 0
    damaged = image.copy()
    ImageDraw.Draw(damaged).rectangle((0, 920, 1000, 980), fill="#102333")
    with pytest.raises(ValueError):
        verify_text_region(damaged, "EVENT C: 2 -> 5", (25, 930), font, "#ffdc91")
    camera = Image.new("RGB", (1280, 720), "#406070")
    image.paste(camera, (20, 145))
    assert verify_camera_region(image, camera, []) == 0
    damaged = image.copy()
    ImageDraw.Draw(damaged).rectangle((20, 340, 1300, 864), fill="black")
    with pytest.raises(ValueError):
        verify_camera_region(damaged, camera, [])


def test_candidate_fields_and_event_lines_are_bound_to_original_records():
    from adaptive_drone_patrol.video_check import verify_display_rows, event_lines

    v = {
        "selected_L": "C",
        "selected_H": "H",
        "snapshot": {"scenario": {"checkpoint_ids": ["C", "H"]}},
        "candidate_priorities_distances": [
            {"id": "C", "action": 0, "priority": 2, "distance_m": 1, "pose_distance_m": 1},
            {"id": "H", "action": 1, "priority": 3, "distance_m": 2, "pose_distance_m": 2},
        ],
        "forced_first": [{"action": 0, "cost": 4}, {"action": 1, "cost": 5}],
        "policy_rollout": {"cost": 4},
    }
    rows = []
    for i, name in enumerate(["C", "H"]):
        rows.append(
            {
                "name": name,
                "candidate": v["candidate_priorities_distances"][i],
                "result": v["forced_first"][i],
                "label": f"{'近いL' if i == 0 else '遠い高優先度H'}（地点{name}）を先に訪問した場合の最小コスト",
            }
        )
    rows.append({"label": "AIの残り巡回予測（AIの予測）", "result": v["policy_rollout"]})
    verify_display_rows(rows, v)
    altered = copy.deepcopy(rows)
    altered[0]["candidate"]["priority"] = 999
    with pytest.raises(ValueError):
        verify_display_rows(altered, v)
    event = {
        "type": "PRIORITY_UPDATED",
        "sim_time_s": 4,
        "update": {"old": 2, "value": 5},
        "comparison": {
            "no_event_order": [0, 1],
            "post_event_order": [1, 0],
            "retained_order_post_event": {"cost": 20},
            "replanned_order_post_event": {"cost": 15},
            "improvement": 5,
        },
    }
    text = event_lines(event, ["A", "C"])
    assert "A → C" in text[1] and "C → A" in text[2]
    assert "20.000 → 新順序 15.000" in text[3]


def test_manifest_excludes_itself_but_covers_per_output_manifests(tmp_path):
    from adaptive_drone_patrol.video_check import artifact_entries

    (tmp_path / "manifest.json").write_text("old manifest")
    (tmp_path / "full.manifest.json").write_text("old output manifest")
    (tmp_path / "verification.json").write_text("{}")
    assert [e["path"] for e in artifact_entries(tmp_path)] == [
        str(tmp_path / "full.manifest.json"),
        str(tmp_path / "verification.json"),
    ]
