"""Pixel checks for real-height badges, independent of flight acceptance."""

import copy
from pathlib import Path
import pytest
from PIL import Image, ImageDraw
import matplotlib
from adaptive_drone_patrol.video_render import Renderer
from adaptive_drone_patrol.analytical import demo_fixture
from adaptive_drone_patrol.dispersed_layout import layout_candidate
from adaptive_drone_patrol.scene import scene_manifest
from adaptive_drone_patrol import video_check


def test_full_size_badges_show_actual_height_and_detect_erasure():
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    scenario = demo_fixture(40029, "integer-100-v1")
    scenario.coords = layout_candidate(51000, 3353)
    record = {"scenario": scenario.to_dict(), "scene": scene_manifest(scenario)}
    state = {"priorities": scenario.priorities, "selected": "C", "visited": ["F"]}
    image = Image.new("RGB", (1920, 1080), "#102333")
    renderer = Renderer(font)
    assert hasattr(renderer, "dispersed_badges")
    renderer.dispersed_badges(ImageDraw.Draw(image), record, state)
    video_check.verify_height_badges(image, record, state["priorities"], font)
    altered = copy.deepcopy(record)
    altered["scene"]["markers"][0]["world_xyz"][2] += 0.1
    with pytest.raises(ValueError):
        video_check.verify_height_badges(image, altered, state["priorities"], font)
    x, y, _, _ = record["scene"]["badge_rects"][0]
    ImageDraw.Draw(image).rectangle((int(x) + 25, int(y) + 175, int(x) + 150, int(y) + 200), fill="black")
    with pytest.raises(ValueError):
        video_check.verify_height_badges(image, record, state["priorities"], font)


def test_height_display_preserves_legacy_fractional_priority_contract():
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    scenario = demo_fixture(40029)
    scenario.coords = layout_candidate(51000, 3353)
    record = {
        "scenario": scenario.to_dict(),
        "scene": scene_manifest(scenario),
        "events": [],
        "telemetry": [],
        "initial": [0, 0, 1],
        "schedule": {"dt": 0.01},
    }
    renderer = Renderer(font)
    image, state = renderer.flight(
        Image.new("RGB", (1280, 720), "#102333"), record, {"sim_time_s": 0, "simulation_step": 0}
    )
    video_check.verify_height_badges(image, record, state["priorities"], font)
