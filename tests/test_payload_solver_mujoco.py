"""Actual MuJoCo articulated mechanics; run in the isolated reference environment."""

import os
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("mujoco")


@pytest.fixture
def original():
    value = os.environ.get("PAYLOAD_SOLVER_CF2X")
    if not value:
        pytest.skip("Actual pinned cf2x asset location required")
    return Path(value)


def test_passive_reference_matches_planar_acceleration_energy_and_geometry(tmp_path, original):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload
    from adaptive_drone_patrol.payload_recovery_planar import PlanarModel

    fixture = MuJoCoFixture(
        original, tmp_path / "fixture", Payload(0.02700047753751278, 0.2, 0.045), fixed=True, substeps=20
    )
    fixture.reset(swing_deg=90.0)
    row = fixture.sample()
    assert np.linalg.norm(np.asarray(row["attachment_b_world"]) - row["attachment_a_world"]) == pytest.approx(
        0.3, abs=1e-12
    )
    r = fixture.runtime()
    assert r["free_drone_mass_kg"] == pytest.approx(0.02700047753751278, rel=1e-12)
    m = PlanarModel(0.0001, fixture.payload.mass, 7.50025e-7, fixture.payload.inertia[1], 0.045)
    assert fixture.planar_acceleration() == pytest.approx(
        m.acceleration([np.pi / 2, np.pi / 2, 0.0, 0.0]), abs=1e-8
    )
    e0 = fixture.energy()
    for _ in range(100):
        fixture.step_rpm(np.zeros(4))
    assert abs(fixture.energy() - e0) / e0 < 1e-5
    assert fixture.sample()["contact"] is False


def test_actual_reset_restores_state_and_properties_without_stepping(tmp_path, original):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload

    fixture = MuJoCoFixture(
        original, tmp_path / "fixture", Payload(0.02700047753751278, 0.2, 0.045), substeps=4
    )
    for i in range(30):
        fixture.step_rpm(np.full(4, 14468.429183500699))
        fixture.reset(swing_deg=float(i % 6), position=[0.0, 0.0, 1.0])
        assert fixture.data.time == 0
        assert np.array_equal(fixture.data.qvel, np.zeros(12))
        assert np.array_equal(fixture.data.qfrc_applied, np.zeros(12))
        assert fixture.runtime()["package_mass_kg"] == pytest.approx(fixture.payload.mass, rel=1e-12)


def test_rk_stages_apply_current_body_rotor_wrench_and_observe_work(tmp_path, original):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload

    fixture = MuJoCoFixture(
        original, tmp_path / "fixture", Payload(0.02700047753751278, 0.2, 0.045), substeps=4
    )
    fixture.reset()
    record = fixture.step_rpm(np.full(4, 14468.429183500699), observe_stages=True)
    assert len(record) == 16
    assert all(np.array_equal(x["rpm"], [14468.429183500699] * 4) for x in record)
    assert all(
        x["motor_power_w"] == pytest.approx(np.dot(x["motor_generalized_force"], x["qvel"]), abs=1e-12)
        for x in record
    )
    assert np.isfinite(fixture.sample()["drone_xyz"]).all()


def test_passive_fast_path_has_no_unobserved_applied_force(tmp_path, original):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload

    fixture = MuJoCoFixture(
        original, tmp_path / "fixture", Payload(0.02700047753751278, 0.2, 0.045), fixed=True, substeps=4
    )
    fixture.reset(swing_deg=170.0)
    fixture.data.qfrc_applied[:] = 1
    fixture.step_rpm(np.zeros(4))
    assert np.array_equal(fixture.data.qfrc_applied, np.zeros(6))


def test_raw_getters_preserve_invalid_quaternion_before_derived_attachment_fields(tmp_path, original):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload

    fixture = MuJoCoFixture(
        original, tmp_path / "fixture", Payload(0.02700047753751278, 0.2, 0.045), fixed=True
    )
    fixture.data.xquat[fixture.ids["payload"]] = np.nan
    raw = fixture.raw_state()
    assert np.isnan(raw["package_quaternion_wxyz"]).all()
    assert "attachment_a_world" not in raw
    with pytest.raises(ValueError):
        fixture.sample()


def test_episode_payload_update_applies_actual_mass_inertia_and_keeps_geometry(tmp_path, original):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload

    f = MuJoCoFixture(original, tmp_path / "fixture", Payload(0.02700047753751278, 0.2, 0.045), substeps=16)
    before = f.model.geom_size.copy()
    draw = Payload(0.02700047753751278, 0.27, 0.045)
    f.apply_payload(draw)
    f.reset()
    idx = f.ids["payload"]
    assert f.model.body_mass[idx] == pytest.approx(draw.mass, rel=1e-12)
    assert f.model.body_inertia[idx] == pytest.approx(draw.inertia, rel=1e-12)
    assert np.array_equal(f.model.geom_size, before)
    assert f.data.time == 0 and np.array_equal(f.rpm, [0.0] * 4)
    with pytest.raises(ValueError):
        f.apply_payload(Payload(0.02700047753751278, 0.2, 0.06))


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3])
def test_damping_matches_observed_legacy_authored_mass_and_episode_updates(tmp_path, original, ratio):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload
    from adaptive_drone_patrol.payload_solver_qualification import physical_errors

    free = 0.02700047753751278
    f = MuJoCoFixture(original, tmp_path / "loaded", Payload(free, ratio, 0.045), substeps=16)
    expected = (0.027 + 0.0001 + ratio * free) * 1e-5
    assert np.allclose(f.model.dof_damping[3:6], expected, rtol=1e-12, atol=0)
    case = dict(substeps=16, mass_ratio=ratio, side_m=0.045)
    assert physical_errors(f.runtime(), case, fixed=False) == []
    f.model.dof_damping[3:6] = (free + 0.0001 + ratio * free) * 1e-5
    assert "ACTUAL_DAMPING" in physical_errors(f.runtime(), case, fixed=False)
    draw = Payload(free, 0.27, 0.045)
    f.apply_payload(draw)
    assert np.allclose(f.model.dof_damping[3:6], (0.027 + 0.0001 + draw.mass) * 1e-5, rtol=1e-12, atol=0)
