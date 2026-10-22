"""Retry assets and admission must fail closed independently of simulation."""

import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from adaptive_drone_patrol.payload_contract import Payload

ASSET = Path("outputs/suspended_20261021_v1/pinned-source/genesis/assets/urdf/drones/cf2x.urdf")


def test_ball_asset_has_real_two_passive_pivots_and_retains_vehicle(tmp_path):
    from adaptive_drone_patrol.payload_ball_asset import write_ball_mjcf

    path = tmp_path / "candidate.xml"
    receipt = write_ball_mjcf(ASSET, path, Payload(0.02700047753751278, 0.2, 0.045))
    root = ET.parse(path).getroot()
    joints = root.findall(".//joint")
    assert [j.get("type") for j in joints] == ["ball", "ball"]
    assert all(float(j.get("damping")) == 0 and float(j.get("armature")) == 0 for j in joints)
    assert root.find("actuator") is None
    assert len(root.findall(".//body")) == 7
    vehicle = root.find(".//body[@name='base_link']")
    assert vehicle.find("freejoint") is not None
    assert float(vehicle.find("inertial").get("mass")) == 0.027
    assert vehicle.find("inertial").get("diaginertia") == "1.4e-05 1.4e-05 2.17e-05"
    for i, xyz in enumerate(((0.028, -0.028, 0), (-0.028, -0.028, 0), (-0.028, 0.028, 0), (0.028, 0.028, 0))):
        prop = vehicle.find(f"body[@name='prop{i}_link']")
        assert tuple(map(float, prop.find("inertial").get("pos").split())) == xyz
    rod = vehicle.find("body[@name='suspension_rod']")
    assert rod.get("pos") == "0 0 -0.02"
    package = rod.find("body[@name='payload']")
    assert tuple(map(float, package.get("pos").split())) == (0, 0, -0.3225)
    assert tuple(map(float, package.find("joint").get("pos").split())) == (0, 0, 0.0225)
    assert float(package.find("inertial").get("mass")) == pytest.approx(0.02700047753751278 * 0.2)
    assert receipt["status"] == "EXPERIMENTAL_DIAGNOSTICS_ONLY"
    assert receipt["gimbal_count"] == 0
    assert receipt["suspension_length_m"] == 0.3
    with pytest.raises(FileExistsError):
        write_ball_mjcf(ASSET, path, Payload(0.027, 0.2, 0.045))


def test_retry_manifest_covers_long_operating_domain():
    from adaptive_drone_patrol.payload_qualification import validate_declaration

    manifest = json.loads(Path("configs/payload_retry_probe_manifest.json").read_text())
    validate_declaration(manifest)
    cases = manifest["cases"]
    assert len([c for c in cases if c["kind"] == "anchor"]) == 9
    assert len([c for c in cases if c["kind"] == "powered"]) == 18
    assert len([c for c in cases if c["kind"] == "refinement"]) == 12
    for c in cases:
        if c["kind"] in ("anchor", "powered"):
            assert c["duration_s"] >= 120
    for mutate in ("duration", "missing"):
        bad = copy.deepcopy(manifest)
        if mutate == "duration":
            bad["cases"][0]["duration_s"] = 0.6
        else:
            bad["cases"].pop()
        with pytest.raises(ValueError):
            validate_declaration(bad)


def test_old_pass_receipt_and_status_edits_never_admit():
    from adaptive_drone_patrol.payload_qualification import require_retry_admission

    for old in (
        {"status": "PASS", "physical_training_admitted": True},
        {
            "schema": "payload-qualification-v2",
            "status": "QUALIFIED_FOR_TRAINING",
            "physical_training_admitted": True,
        },
    ):
        with pytest.raises(ValueError):
            require_retry_admission(old, {}, {})


def test_raw_checker_rejects_missing_terminal_nonfinite_and_finite_excursion():
    from adaptive_drone_patrol.payload_qualification import check_trace

    case = {"kind": "anchor", "duration_s": 120.0, "sample_dt_s": 0.01}
    limits = {
        "speed_max_m_s": 20.0,
        "angular_speed_max_rad_s": 200.0,
        "quaternion_norm_error": 1e-4,
        "constraint_residual_max_m": 2e-5,
        "passive_positive_energy_fraction": 0.05,
    }
    row = {
        "step": 0,
        "time_s": 0.0,
        "speed_m_s": 0.0,
        "angular_speed_rad_s": 0.0,
        "quaternion_norm_error": 0.0,
        "constraint_residual_m": 0.0,
        "energy_j": 1.0,
        "motor_work_j": 0.0,
    }
    rows = [dict(row, step=i, time_s=i * 0.01) for i in range(12001)]
    assert not check_trace(case, rows, limits)
    assert check_trace(case, rows[:-1], limits)
    for field, value in [
        ("speed_m_s", float("nan")),
        ("speed_m_s", 21.0),
        ("energy_j", 1.06),
        ("quaternion_norm_error", 0.001),
        ("constraint_residual_m", 3e-5),
    ]:
        bad = copy.deepcopy(rows)
        bad[300][field] = value
        assert check_trace(case, bad, limits)
    bad = copy.deepcopy(rows)
    bad[20]["time_s"] = 0.3
    assert check_trace(case, bad, limits)


def test_large_angle_sweep_includes_singular_and_inverted_starts():
    from adaptive_drone_patrol.payload_qualification import validate_declaration

    manifest = json.loads(Path("configs/payload_retry_probe_manifest.json").read_text())
    sweeps = [c for c in manifest["cases"] if c["id"].startswith("sweep_")]
    assert {c["initial_swing_deg"] for c in sweeps} == {5, 45, 80, 89, 91, 120, 170}
    validate_declaration(manifest)
    bad = copy.deepcopy(manifest)
    bad["cases"] = [c for c in bad["cases"] if c["id"] != "sweep_170"]
    with pytest.raises(ValueError):
        validate_declaration(bad)


@pytest.mark.parametrize(
    "field,value", [("initial_swing_deg", 0), ("substeps", 1), ("duration_s", float("nan"))]
)
def test_declaration_rejects_changed_initial_or_numerical_conditions(field, value):
    from adaptive_drone_patrol.payload_qualification import validate_declaration

    manifest = json.loads(Path("configs/payload_retry_probe_manifest.json").read_text())
    manifest["cases"][0][field] = value
    with pytest.raises(ValueError):
        validate_declaration(manifest)


def test_selected_profile_binds_root_alignment_without_relabeling_old_candidate():
    manifest = json.loads(Path("configs/payload_retry_probe_manifest.json").read_text())
    assert manifest["profile"] == "cf2x-two-native-ball-root-aligned-v3"
    assert manifest["settings"]["root_alignment_requested"] is True
