"""Independent boundary examples for the dispersed physical fixture."""

import numpy as np
import pytest
from adaptive_drone_patrol import dispersed_layout as layout
from adaptive_drone_patrol.scenario import draw_integer_priorities
from adaptive_drone_patrol.analytical import demo_fixture
from adaptive_drone_patrol.scene import scene_manifest, validate_scene


def test_candidate_reproduces_without_consuming_priority_stream():
    a = layout.layout_candidate(51000, 0)
    assert a == layout.layout_candidate(51000, 0)
    assert a != layout.layout_candidate(51000, 1)
    assert a[0] == [0.0, 0.0, 1.0]
    assert np.asarray(a).shape == (11, 3)
    assert draw_integer_priorities(40029, 10) == [81, 42, 43, 64, 23, 84, 6, 98, 22, 56]


def test_sliding_triples_straddle_fixed_bins_and_include_closed_boundary():
    assert not layout.band_clear([0.09, 0.14, 0.19, 1.0])
    assert layout.band_clear([0.09, 0.14, 0.190001, 1.0])


def test_height_internal_boundaries_belong_to_upper_third():
    assert layout.height_bands([1.0, 1.0 + 2 / 3, 1.0 + 4 / 3, 3.0]) == [1, 1, 2]


def test_layout_rejects_bad_shape_nonfinite_base_and_spacing():
    valid = next(
        layout.layout_candidate(51000, i)
        for i in range(10000)
        if layout.layout_checks(layout.layout_candidate(51000, i))["passed"]
    )
    for mutation, reason in [
        ("nan", "nonfinite"),
        ("bounds", "bounds"),
        ("spacing", "separation"),
        ("base", "base_overlap"),
    ]:
        coords = np.array(valid)
        if mutation == "nan":
            coords[1, 0] = np.nan
        if mutation == "bounds":
            coords[1, 0] = 2.71
        if mutation == "spacing":
            coords[1] = coords[2] + [0.01, 0, 0]
        if mutation == "base":
            coords[1] = coords[0] + [0.1, 0, 0]
        assert reason in layout.layout_checks(coords.tolist())["rejection_reasons"]
    assert "shape" in layout.layout_checks([[1, 2, 3]])["rejection_reasons"]


def test_geometry_selection_records_every_attempt_and_hash():
    m = layout.generate_layout(51000)
    assert m["checks"]["passed"]
    assert m["coordinate_sha256"] == layout.coordinate_hash(m["coords"])
    assert len(m["attempts"]) == m["selected_attempt"] + 1
    assert all(r["rejection_reasons"] for r in m["attempts"][:-1])
    assert np.all(np.asarray(m["checks"]["spans_m"]) >= [4.86, 3.24, 1.8])
    assert m["checks"]["min_separation_m"] >= 0.5
    assert m["nominal_horizontal_area_ratio"] == 9
    with pytest.raises(RuntimeError):
        layout.generate_layout(51000, max_attempts=0)


def test_dispersed_camera_frames_volume_and_rejects_projection_overlap():
    s = demo_fixture(40029, "integer-100-v1")
    s.coords = layout.generate_layout(51000)["coords"]
    m = scene_manifest(s)
    assert m.get("height_display") == "absolute metres above ground"
    validate_scene(m)
    m["markers"][1]["world_xyz"] = m["markers"][0]["world_xyz"]
    with pytest.raises(ValueError):
        validate_scene(m)


def test_expanded_benchmark_preserves_split_priority_and_objective():
    from adaptive_drone_patrol.scenario import generate_scenario

    original = generate_scenario(20000, "static", num_checkpoints=10, priority_profile="integer-100-v1")
    expanded = layout.expanded_scenario(20000, "static")
    assert expanded.priorities == original.priorities
    assert expanded.multipliers == original.multipliers
    assert expanded.beta == 0.05 and expanded.schema_version == 3
    assert layout.layout_checks(expanded.coords)["passed"]
    assert expanded.coords == layout.expanded_scenario(20000, "static").coords
    with pytest.raises(ValueError):
        layout.expanded_scenario(20000, "validation")


def test_reviewed_layout_golden_hash_and_closed_world_bounds():
    coords = layout.layout_candidate(51000, 3353)
    assert (
        layout.coordinate_hash(coords) == "21dd7cf41bf12c592ccdb4bf9c8817352feeafb9597103a3ba626e51a6d910e0"
    )
    for axis, limit in [(0, 2.7), (1, 1.8), (2, 3.0)]:
        changed = np.array(coords)
        changed[1, axis] = limit
        assert "bounds" not in layout.layout_checks(changed.tolist())["rejection_reasons"]
        changed[1, axis] = limit + 1e-9
        assert "bounds" in layout.layout_checks(changed.tolist())["rejection_reasons"]
