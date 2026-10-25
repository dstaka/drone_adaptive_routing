import copy
import json
import pytest
from adaptive_drone_patrol.payload_mujoco_overview_media import load_overview_camera, verify_framing
from adaptive_drone_patrol.payload_solver_media import load_camera


def test_original_fixed_overview_enlarged_without_enlarging_individual_detail(tmp_path):
    c = load_overview_camera()
    old = load_camera()
    assert c["overview"]["distance"] == 5 and old["overview"]["distance"] == 10
    assert c["detail"] == old["detail"]
    c["detail"]["distance"] = 0.85
    p = tmp_path / "changed.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="unchanged"):
        load_overview_camera(p)


def test_all_targets_must_remain_visible_in_the_same_fixed_frame():
    c = load_overview_camera()
    row = dict(time_s=0, bodies_px=[100, 100, 300, 400], all_ten_targets_px=[80, 80, 1200, 640])
    assert verify_framing([row, row], frames=1, config=c)["all_ten_destinations_visible"]
    changed = copy.deepcopy(row)
    changed["all_ten_targets_px"][0] = 30
    with pytest.raises(ValueError, match="clips"):
        verify_framing([row, changed], frames=1, config=c)
    changed["all_ten_targets_px"][0] = 90
    with pytest.raises(ValueError, match="moved"):
        verify_framing([row, changed], frames=1, config=c)
