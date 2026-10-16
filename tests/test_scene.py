"""Projection, persistent state and priority encoding contracts."""
import numpy as np
import pytest


def test_priority_palette_and_orthogonal_symbols():
    from adaptive_drone_patrol.scene import priority_color, marker_style
    assert [priority_color(i) for i in range(1, 6)] == ['#fee5d9','#fcae91','#fb6a4a','#de2d26','#a50f15']
    assert priority_color(2.5) == '#fc8c6e'
    for value in [0, 6, float('nan')]:
        with pytest.raises(ValueError):
            priority_color(value)
    assert marker_style(3, True, False, False)['fill'] == marker_style(3, False, True, True)['fill']
    assert marker_style(3, True, True, True)['symbols'] == ['selected', 'visited', 'alert']


def test_camera_contains_every_point_with_separate_projections():
    from adaptive_drone_patrol.scene import scene_manifest, project
    from adaptive_drone_patrol.analytical import demo_fixture
    s = demo_fixture(40004)
    manifest = scene_manifest(s)
    projected = [project(x, manifest['camera']) for x in s.coords]
    assert all(70 < x < 1210 and 100 < y < 620 for x,y in projected)
    assert min(np.linalg.norm(np.array(a)-b) for i,a in enumerate(projected[1:]) for b in projected[i+2:]) > 60
    assert len(manifest['markers']) == 10
    assert all(m['visual_only'] and not m['collision'] and m['fixed'] for m in manifest['markers'])


def test_source_hook_preserves_pinned_source_and_inserts_before_build(tmp_path):
    from adaptive_drone_patrol.target_adapter import prepare_hover_source
    text = 'camera GUI=True,\n        self.scene.build(n_envs=num_envs)\n'
    changed = prepare_hover_source(text, True)
    assert changed.index('install_scene(self)') < changed.index('self.scene.build')
    assert 'GUI=False,' in changed
    assert 'GUI=True,' in text
    with pytest.raises(ValueError):
        prepare_hover_source('unknown source', True)


def test_preflight_rejects_unframed_camera_and_nonfinite_timeout():
    from adaptive_drone_patrol.scene import scene_manifest, validate_scene
    from adaptive_drone_patrol.analytical import demo_fixture
    import copy
    manifest = scene_manifest(demo_fixture(40004))
    validate_scene(manifest)
    bad = copy.deepcopy(manifest)
    bad['camera']['fov'] = float('nan')
    with pytest.raises(ValueError):
        validate_scene(bad)
    bad = copy.deepcopy(manifest)
    bad['camera']['pos'] = [0,0,1.01]
    with pytest.raises(ValueError):
        validate_scene(bad)


def test_priority_label_ink_contrasts_with_fixed_fill():
    from adaptive_drone_patrol.scene import marker_style
    assert marker_style(1)['ink']=='#111111'
    assert marker_style(5)['ink']=='#ffffff'
    assert marker_style(5,True,True,True)['ink']=='#ffffff'
