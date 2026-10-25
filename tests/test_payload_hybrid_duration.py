"""Pure duration contracts and bounded full-stage storage; no qualification claim."""

import copy
import numpy as np
import pytest

from adaptive_drone_patrol.payload_hybrid_duration import (
    duration_cases,
    duration_declaration,
    validate_duration_case,
    StageWriter,
)
from adaptive_drone_patrol.payload_hybrid_duration_audit import (
    check_duration_boundary,
    check_native_equal,
)
from adaptive_drone_patrol.payload_solver_work import WIDTH


def test_seven_duration_cases_are_development_only_reproducible_and_constraint_consistent():
    from adaptive_drone_patrol.payload_ball_bank import ball_joint_configuration

    cases = duration_cases()
    assert cases == duration_cases()
    assert len(cases) == len({c["case_id"] for c in cases}) == 7
    assert {(c["side_m"], c["mass_ratio"]) for c in cases} == {
        (0.045, 0.2),
        *[(s, r) for s in (0.03, 0.045, 0.06) for r in (0.1, 0.3)],
    }
    for case in cases:
        validate_duration_case(case)
        assert case["duration_s"] == 330.0 and case["intervals"] == 33000
        assert case["target"] == [0.0, 0.0, 1.0]
        assert case["scope"] == "CONTINUOUS_C_DURATION_NOT_MISSION"
        ball_joint_configuration(case["initial"], case["side_m"])
    declaration = duration_declaration(cases)
    assert declaration["reset_count_per_trace"] == 1
    assert declaration["mission_wall_s"] is None and declaration["leg_s"] is None
    assert not declaration["learning_admitted"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_s", 120.0),
        ("intervals", 12000),
        ("target", [0.5, 0.0, 1.0]),
        ("mass_ratio", 0.4),
        ("initial", {}),
        ("seed", 7),
    ],
)
def test_duration_case_rejects_altered_declared_horizon_initial_and_load(field, value):
    case = copy.deepcopy(duration_cases()[0])
    case[field] = value
    with pytest.raises(ValueError):
        validate_duration_case(case)


def test_bounded_stage_writer_retains_every_stage_and_qacc_then_final_flush(tmp_path):
    writer = StageWriter(tmp_path, chunk_intervals=2)
    for i in range(3):
        writer.append(np.full((64, WIDTH), float(i)), np.full((64, 12), i + 0.5))
        assert len(writer.buffer) < 2
    writer.close()
    assert [r["intervals"] for r in writer.index] == [2, 1]
    assert [r["first_step"] for r in writer.index] == [0, 2]
    rows, acc = [], []
    for chunk in writer.index:
        with np.load(tmp_path / chunk["path"], allow_pickle=False) as packed:
            rows.extend(packed["stages"])
            acc.extend(packed["qacc_before_solve"])
    np.testing.assert_array_equal(rows, np.stack([np.full((64, WIDTH), float(i)) for i in range(3)]))
    np.testing.assert_array_equal(acc, np.stack([np.full((64, 12), i + 0.5) for i in range(3)]))
    with pytest.raises((ValueError, RuntimeError)):
        writer.append(np.zeros((64, WIDTH)), np.zeros((64, 12)))


def test_duration_boundary_never_treats_settling_or_30_120_600_as_termination():
    for step in (100, 3000, 12000, 32999):
        assert check_duration_boundary(step, unsafe=False, failed=False) == "CONTINUE"
    assert check_duration_boundary(33000, unsafe=False, failed=False) == "DURATION_COMPLETE"
    assert check_duration_boundary(33000, unsafe=True, failed=False) == "SAFETY_FAILURE"
    assert check_duration_boundary(10, unsafe=False, failed=True) == "RUNTIME_FAILURE"
    for bad in (-1, 33001, True):
        with pytest.raises(ValueError):
            check_duration_boundary(bad, unsafe=False, failed=False)


def test_native_replay_checks_every_native_field_and_shapes():
    row = dict(qpos=[0.0, 1.0], rpm=[100.0] * 4, contact=False, warning_counts=[0] * 7)
    assert check_native_equal(row, copy.deepcopy(row)) == 0
    for changed in (
        dict(row, rpm=[100.0, 100.0, 100.0, 101.0]),
        dict(row, contact=True),
        dict(row, qpos=[[0.0, 1.0]]),
        dict(row, warning_counts=[0] * 6),
        dict(row, qpos=[float("nan"), 1.0]),
    ):
        with pytest.raises(ValueError):
            check_native_equal(row, changed)


def test_observation_adapter_records_same_actual_rk4_without_changing_native_state(
    minimal_cf2x_xml, tmp_path
):
    """Structural API integration only, never original-asset qualification."""
    pytest.importorskip("mujoco")
    from adaptive_drone_patrol.payload_hybrid_duration import ObservedFixture
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG, HOVER_RPM
    from adaptive_drone_patrol.payload_contract import Payload
    from adaptive_drone_patrol.payload_solver_runtime import SolverRuntime
    from adaptive_drone_patrol.payload_solver_work import pack_stages
    from adaptive_drone_patrol.payload_solver_work_check import reconstruct_batch_work

    # Explicit noncoplanar visual meshes for the structural writer input only.
    # These temporary tetrahedra are never original-asset or numerical evidence.
    tetrahedron = "v 0 0 0\nv 0.001 0 0\nv 0 0.001 0\nv 0 0 0.001\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    for name in ("base_link", "prop0_link", "prop1_link", "prop2_link", "prop3_link"):
        (minimal_cf2x_xml.parent / (name + ".obj")).write_text(tetrahedron)

    payload = Payload(FREE_MASS_KG, 0.2, 0.045)
    ordinary = MuJoCoFixture(minimal_cf2x_xml, tmp_path / "ordinary", payload, substeps=16)
    recorded = ObservedFixture(minimal_cf2x_xml, tmp_path / "recorded", payload, substeps=16)
    core = SolverRuntime(recorded, tmp_path / "runtime")
    core.reset_case()
    ordinary.reset()
    try:
        for tick in range(3):
            action = np.array([0.08, 0.09, 0.07, 0.08])
            ordinary.step_rpm((1 + 0.8 * action) * HOVER_RPM)
            core.step(action)
            assert len(recorded.stage_rows) == 64
            check_native_equal(ordinary.raw_state(), recorded.raw_state(), atol=0)
            packed = pack_stages(recorded.stage_rows)
            work = reconstruct_batch_work(
                packed[None],
                substeps=16,
                first_step=tick,
                total_mass_kg=sum(r["mass_kg"] for r in recorded.runtime()["links"]),
            )
            assert np.isfinite(work["motor_work_j"]).all()
            assert np.all(work["dissipative_work_j"] <= 0)
    finally:
        core.close()


def test_complete_stage_reader_rejects_truncation_dtype_and_rehashed_hidden_input(tmp_path):
    from adaptive_drone_patrol.payload_hybrid_duration_audit import _stages
    from adaptive_drone_patrol.util import sha256
    import json

    writer = StageWriter(tmp_path)
    writer.append(np.zeros((64, WIDTH), dtype=np.float64), np.zeros((64, 12), dtype=np.float64))
    writer.close()
    assert len(list(_stages(tmp_path, 1))) == 1
    with pytest.raises(ValueError):
        list(_stages(tmp_path, 2))
    path = tmp_path / writer.index[0]["path"]
    np.savez_compressed(
        path,
        stages=np.zeros((1, 64, WIDTH), dtype=np.float32),
        qacc_before_solve=np.zeros((1, 64, 12), dtype=np.float64),
    )
    writer.index[0]["sha256"] = sha256(path)
    (tmp_path / "stages-index.json").write_text(json.dumps(writer.index))
    with pytest.raises(ValueError):
        list(_stages(tmp_path, 1))


def test_split_host_receipt_requires_exact_external_sha_and_each_canonical_file(tmp_path, monkeypatch):
    import hashlib
    import json
    from adaptive_drone_patrol import payload_hybrid_duration as duration
    from adaptive_drone_patrol.util import sha256

    # Tiny integrity fixture only. Production constants remain pinned to2669/7.7GB.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "one.gz").write_bytes(b"ACTUAL_TEST_BYTES")
    manifest = {"passive/one.gz": dict(bytes=17, sha256=sha256(raw / "one.gz"))}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    old = tmp_path / "old.json"
    old.write_text(json.dumps(dict(raw_manifest=manifest, actual_binding={"PURE_FIXTURE": True})))
    monkeypatch.setattr(duration, "HISTORICAL_RECEIPT_SHA256", sha256(old))
    monkeypatch.setattr(duration, "RAW_MANIFEST_SHA256", digest)
    monkeypatch.setattr(duration, "RAW_FILES", 1)
    monkeypatch.setattr(duration, "RAW_BYTES", 17)
    rebinding = tmp_path / "rebind.json"
    rebinding.write_text(
        json.dumps(
            dict(
                historical_receipt_sha256=sha256(old),
                raw_manifest_sha256=digest,
                raw_files=1,
                raw_bytes=17,
                actual_binding={"PURE_FIXTURE": True},
            )
        )
    )
    receipt = duration.verify_preserved_raw(rebinding, old, {"passive": str(raw)})
    path = tmp_path / "receipt.json"
    duration.write_json(path, receipt)
    assert duration.verify_raw_receipt(path, sha256(path), rebinding, old)["raw_files"] == 1
    with pytest.raises(ValueError):
        duration.verify_raw_receipt(path, "0" * 64, rebinding, old)
    forged = copy.deepcopy(receipt)
    forged["files"]["passive/one.gz"]["sha256"] = "f" * 64
    forged["canonical_receipt_sha256"] = duration.canonical_digest(
        {k: v for k, v in forged.items() if k != "canonical_receipt_sha256"}
    )
    duration.write_json(path, forged)
    with pytest.raises(ValueError):
        duration.verify_raw_receipt(path, sha256(path), rebinding, old)
    (raw / "one.gz").write_bytes(b"CHANGED_TEST_DATA")
    with pytest.raises(ValueError):
        duration.verify_preserved_raw(rebinding, old, {"passive": str(raw)})
