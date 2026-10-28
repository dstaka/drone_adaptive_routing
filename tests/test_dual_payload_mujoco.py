"""Actual shared-world mechanics, not inherited single-drone qualification."""

import os
from pathlib import Path
import numpy as np
import pytest


def fixture(tmp_path):
    pytest.importorskip("mujoco", reason="Actual MuJoCo3.15 native fixture tests run on the current server")
    from adaptive_drone_patrol.dual_payload_mujoco import DualMuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload
    from adaptive_drone_patrol.payload_solver_mujoco import FREE_MASS_KG

    original = Path(os.environ.get("DUAL_CF2X", "inputs/cf2x/cf2x.urdf"))
    if not original.exists():
        original = Path("outputs/dual_payload_coordination_20261027T144754Z/inputs/cf2x/cf2x.urdf")
    return DualMuJoCoFixture(
        original, tmp_path / "world", [Payload(FREE_MASS_KG, 0.1, 0.03), Payload(FREE_MASS_KG, 0.3, 0.06)]
    )


def test_one_world_distinct_metadata_and_eight_actuator_channels(tmp_path):
    f = fixture(tmp_path)
    assert f.model.nu == 8 and f.model.nq == 30 and f.model.nv == 24
    assert len(set(f.actuator_ids.ravel())) == 8
    assert f.addresses[0]["free_qpos"] != f.addresses[1]["free_qpos"]
    s = f.sample()
    assert np.allclose(s["drones"][0]["drone_xyz"], [-3.5, -3.5, 1])
    assert np.allclose(s["drones"][1]["drone_xyz"], [3.5, 3.5, 1])
    assert f.model.body_mass[f.body_ids[0]["payload"]] < f.model.body_mass[f.body_ids[1]["payload"]]
    before = f.data.time
    f.step_rpm(np.zeros((2, 4)), observe_stages=True)
    assert f.data.time - before == pytest.approx(0.01)
    assert len(f.stage_rows) == 64 and len(f.substep_rows) == 17
    assert all(len(row["qpos"]) == 30 for row in f.stage_rows)


def test_single_motor_generalized_force_does_not_alias_other_drone(tmp_path):
    f = fixture(tmp_path)
    rpm = np.zeros((2, 4))
    rpm[1, 2] = 10000
    f.set_rpm(rpm)
    f.mj.mj_forward(f.model, f.data)
    a = f.addresses[0]["free_dof"]
    b = f.addresses[1]["free_dof"]
    assert np.allclose(f.data.qfrc_actuator[a : a + 6], 0)
    assert f.data.qfrc_actuator[b + 2] == pytest.approx(0.0316, rel=1e-8)
    assert np.linalg.norm(f.data.qfrc_actuator[b + 3 : b + 6]) > 0
    assert f.data.actuator_force[f.actuator_ids[1, 2]] == pytest.approx(0.0316)


def test_reset_positions_are_independent_and_inputs_are_cleared(tmp_path):
    f = fixture(tmp_path)
    initial = [
        dict(position=[-2, 1, 2], quaternion_wxyz=[1, 0, 0, 0], swing_deg=5, direction_rad=0.7),
        dict(position=[3, -1, 3], quaternion_wxyz=[1, 0, 0, 0], swing_deg=10, direction_rad=1.9),
    ]
    f.step_rpm(np.full((2, 4), 15000.0))
    f.reset(initial)
    s = f.sample()
    assert f.data.time == 0 and not f.data.ctrl.any() and not f.data.qvel.any()
    assert s["drones"][0]["drone_xyz"] == initial[0]["position"]
    assert s["drones"][1]["drone_xyz"] == initial[1]["position"]
    assert s["drones"][0]["package_xyz"] != s["drones"][1]["package_xyz"]


def test_cross_assembly_proxy_contact_activates_and_never_excluded(tmp_path):
    f = fixture(tmp_path)
    f.reset(
        [
            dict(position=[0, 0, 1], quaternion_wxyz=[1, 0, 0, 0], swing_deg=0, direction_rad=0),
            dict(position=[0.015, 0, 1], quaternion_wxyz=[1, 0, 0, 0], swing_deg=0, direction_rad=0),
        ]
    )
    assert f.data.ncon > 0
    pairs = [(f.geom_owner[c.geom1], f.geom_owner[c.geom2]) for c in f.data.contact]
    assert any(set(pair) == {0, 1} for pair in pairs)
    assert all(a.split("/")[0] == b.split("/")[0] for a, b in f.self_exclusions)
    for ids in f.audit_geom_ids:
        assert len(ids) >= 7


def test_original_motor_wrench_and_passive_dynamics_match_two_independent_isolated_models(tmp_path):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture

    dual = fixture(tmp_path)
    original = Path(os.environ.get("DUAL_CF2X", "inputs/cf2x/cf2x.urdf"))
    if not original.exists():
        original = Path("outputs/dual_payload_coordination_20261027T144754Z/inputs/cf2x/cf2x.urdf")
    singles = [
        MuJoCoFixture(original, tmp_path / f"single-{d}", p, substeps=16) for d, p in enumerate(dual.payloads)
    ]
    for d, single in enumerate(singles):
        single.reset(position=([-3.5, -3.5, 1], [3.5, 3.5, 1])[d])
    rpm = np.array([[14000, 15000, 14500, 14700], [15500, 15200, 15400, 15100]])
    for _ in range(20):
        dual.step_rpm(rpm)
        for d, single in enumerate(singles):
            single.step_rpm(rpm[d])
    for d, single in enumerate(singles):
        a = dual.addresses[d]
        assert np.allclose(
            dual.data.qpos[a["free_qpos"] : a["free_qpos"] + 15], single.data.qpos, rtol=0, atol=2e-9
        )
        assert np.allclose(
            dual.data.qvel[a["free_dof"] : a["free_dof"] + 12], single.data.qvel, rtol=0, atol=2e-8
        )
