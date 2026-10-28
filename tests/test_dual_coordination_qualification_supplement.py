"""Measured supplemental reducers; synthetic data never asserts native success."""

import importlib
import pytest


def module():
    return importlib.import_module("adaptive_drone_patrol.dual_coordination_qualification_supplement")


def test_boolean_or_unknown_measurements_rejected(tmp_path):
    m = module()
    for data in ({"passed": True}, {"kind": "mechanics", "address_actuation": True}):
        with pytest.raises(ValueError):
            m.reduce_measurements(data, root=tmp_path)


def test_stopping_requires_all_six_endpoint_loads_and_nominal():
    m = module()
    assert len(m.stopping_cases()) == 7
    assert {(c["side"], c["ratio"]) for c in m.stopping_cases()} == {
        (0.045, 0.2),
        (0.03, 0.1),
        (0.03, 0.3),
        (0.045, 0.1),
        (0.045, 0.3),
        (0.06, 0.1),
        (0.06, 0.3),
    }


def test_tracking_and_geometry_envelopes_are_not_relaxed():
    m = module()
    m.check_envelope_sample(dict(tracking_errors_m=[0.15, 0.15], shape_radii_m=[0.4, 0.4]))
    for field, value in [("tracking_errors_m", 0.150001), ("shape_radii_m", 0.400001)]:
        data = dict(tracking_errors_m=[0.0, 0.0], shape_radii_m=[0.3, 0.3])
        data[field][1] = value
        with pytest.raises(ValueError):
            m.check_envelope_sample(data)


def test_native_mechanics_and_streaming_measurements(tmp_path):
    import os
    from pathlib import Path

    pytest.importorskip("mujoco", reason="Actual native MuJoCo required")
    m = module()
    original = Path(os.environ.get("DUAL_CF2X", "inputs/cf2x/cf2x.urdf"))
    if not original.exists():
        pytest.skip("Original assets unavailable")
    data = m.measure_mechanics(original, tmp_path / "mechanics")
    assert m.reduce_measurements(data, root=tmp_path) == {
        "address_actuation",
        "reset_isolation",
        "collision_detection",
    }
    stream = m.measure_streaming(original, tmp_path / "streaming", artifact_root=tmp_path)
    assert m.reduce_measurements(stream, root=tmp_path) == {"streaming_audit"}


def test_supervisor_horizon_is_recomputed_from_retained_states_and_references():
    m = module()
    refs = [m.ContinuousReference([-3.0, -3.0, 2.0]), m.ContinuousReference([3.0, 3.0, 2.0])]
    states = [m.KinematicState(r.sample(0.0)[0], [0.0, 0.0, 0.0]) for r in refs]
    decision = m.ReferenceSupervisor().decide(refs, states, 0.0, priority=1)
    row = m._decision_record(refs, states, 0.0, 1, decision)
    assert m.check_decision(row) == "clear"
    row["horizon_s"] = 1.99
    with pytest.raises(ValueError, match="Horizon"):
        m.check_decision(row)


def test_missing_stopping_extremes_or_unbound_replay_cannot_pass(tmp_path):
    m = module()
    with pytest.raises(ValueError):
        m.reduce_measurements(dict(kind="stopping", cases=[]), root=tmp_path)
    with pytest.raises((KeyError, ValueError)):
        m.reduce_measurements(dict(kind="deterministic-replay", verified=True), root=tmp_path)


def test_assembly_radius_uses_oriented_surface_not_sum_of_bounding_spheres():
    import numpy as np

    m = module()
    # Largest permitted vertical payload reaches .38m below the drone; its
    # diagonal bounding sphere + center distance is unnecessarily >.40m.
    box = dict(kind="box", size=[0.03, 0.03, 0.03])
    extent = m.surface_extent([0.0, 0.0, 0.0], [0.0, 0.0, -0.35], np.eye(3), box)
    assert extent == pytest.approx(np.sqrt(0.38**2 + 2 * 0.03**2))
    assert extent < 0.4
    cylinder = dict(kind="cylinder", size=[0.1, 0.2, 0.0])
    assert m.surface_extent([0, 0, 0], [0.3, 0, 0.1], np.eye(3), cylinder) == pytest.approx(0.5)


def test_native_four_branch_horizons_use_read_back_initial_states(tmp_path):
    import os
    from pathlib import Path

    pytest.importorskip("mujoco", reason="Actual native read-back states required")
    m = module()
    original = Path(os.environ.get("DUAL_CF2X", "inputs/cf2x/cf2x.urdf"))
    if not original.exists():
        pytest.skip("Original assets unavailable")
    data = m.measure_branch_horizons(original, tmp_path / "branches")
    assert len(data["branches"]) == 100
    reasons = set()
    for row in data["branches"]:
        assert row["decision"]["states"] == [
            dict(position=d["drone_xyz"], velocity=d["drone_velocity"]) for d in row["actual"]["drones"]
        ]
        reasons.add(m.check_decision(row["decision"]))
    assert reasons == {"clear", "priority-yield", "brake-both", "brake-both-unestablished"}


def test_compiled_parity_cannot_be_asserted_by_boolean_or_test_counts(tmp_path):
    m = module()
    for data in [
        dict(kind="compiled-adapter-parity", passed=True, tests_passed=82),
        dict(kind="compiled-adapter-parity", library={"sha256": "a" * 64}, comparison_steps=[]),
    ]:
        with pytest.raises((ValueError, KeyError)):
            m.reduce_measurements(data, root=tmp_path)


def test_native_compiled_supplement_records_actual_parity_checkpoint_and_copyback(tmp_path):
    import os
    import json
    from pathlib import Path

    pytest.importorskip("mujoco")
    from adaptive_drone_patrol.dual_coordination_native_audit import build_native_audit

    m = module()
    library = build_native_audit(tmp_path / "build")
    root = tmp_path
    data = m.measure_compiled_parity(
        os.environ["DUAL_CF2X"],
        json.loads(Path(os.environ["DUAL_CONTROLLER"]).read_text()),
        tmp_path / "compiled-parity",
        artifact_root=root,
        native_audit_library=library,
    )
    assert m.reduce_measurements(data, root=root) == set()
    assert len(data["comparison_steps"]) == 3
    assert data["checkpoint"]["continued"]["backend"] == data["checkpoint"]["resumed"]["backend"]
