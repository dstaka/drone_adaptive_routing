"""Fail-closed retry declarations and raw trace screening, with no simulator import.

Trace screening is necessary but not sufficient for admission: full independent
motor, property, period, reset, refinement and source binding are also required.
Until that reconstruction is implemented, this module cannot issue admission.
"""

import math

SIZES = (0.03, 0.045, 0.06)
RATIOS = (0.1, 0.2, 0.3)
SCHEMA = "payload-operating-domain-declaration-v2"


def validate_declaration(manifest):
    """Reject missing conditions, shortened runs or undeclared numerical limits."""
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "EXPERIMENTAL_DIAGNOSTICS_ONLY":
        raise ValueError("A new experimental declaration is required")
    expected = {}
    for side in SIZES:
        for ratio in RATIOS:
            expected[f"anchor_{side:.3f}_{ratio:.2f}"] = ("anchor", side, ratio, 120.0)
            expected[f"large_{side:.3f}_{ratio:.2f}"] = ("large_angle", side, ratio, 30.0)
            for swing in (0, 10):
                expected[f"powered_{side:.3f}_{ratio:.2f}_{swing}"] = ("powered", side, ratio, 120.0)
        for ratio in (0.1, 0.3):
            for substeps in (2, 4):
                expected[f"refine_{side:.3f}_{ratio:.2f}_{substeps}"] = ("refinement", side, ratio, 30.0)
    for angle in (5, 45, 80, 89, 91, 120, 170):
        expected[f"sweep_{angle}"] = ("large_angle", 0.045, 0.2, 30.0)
    cases = manifest["cases"]
    if len(cases) != len(expected) or {c["id"] for c in cases} != set(expected):
        raise ValueError("Incomplete or duplicated operating domain")
    for c in cases:
        kind, side, ratio, duration = expected[c["id"]]
        if (
            (c["kind"], c["side_m"], c["mass_ratio"]) != (kind, side, ratio)
            or not math.isfinite(c["duration_s"])
            or c["duration_s"] < duration
        ):
            raise ValueError("Changed physical draw or shortened probe")
        expected_angle = (
            5.0
            if kind == "anchor"
            else 90.0
            if kind == "large_angle"
            else float(c["id"].rsplit("_", 1)[1])
            if kind == "powered"
            else 10.0
        )
        if c["id"].startswith("sweep_"):
            expected_angle = float(c["id"].split("_")[1])
        expected_substeps = int(c["id"].rsplit("_", 1)[1]) if kind == "refinement" else 2
        if (
            c["initial_swing_deg"] != expected_angle
            or c["initial_direction_rad"] != 0.0
            or c["substeps"] != expected_substeps
        ):
            raise ValueError("Changed initial displacement or physics timestep")
        if c["sample_dt_s"] != 0.01 or c["timeout_wall_s"] <= 0 or c["reset_during_run"]:
            raise ValueError("Invalid sample timing, timeout or reset")
    limits = manifest["limits"]
    ceilings = {
        "constraint_residual_max_m": 2e-5,
        "mass_relative_error": 1e-5,
        "inertia_relative_error": 1e-5,
        "collision_extent_error_m": 1e-5,
        "period_relative_error": 0.1,
        "passive_positive_energy_fraction": 0.05,
        "half_dt_position_difference_m": 0.02,
        "half_dt_angle_difference_deg": 2.0,
    }
    for key, value in ceilings.items():
        if not 0 < float(limits[key]) <= value:
            raise ValueError("Numerical tolerance changed")
    if manifest["resets"]["count"] < 30 or manifest["settings"]["genesis_version"] != "1.4.3":
        raise ValueError("Incomplete reset count or changed engine")
    if manifest["settings"]["rigid_dt_s"] != 0.005:
        raise ValueError("Candidate dt must match declared refinement")
    return True


def check_trace(case, rows, limits):
    """Screen complete preclip samples; never trust the runner process status."""
    failures = []
    dt = case["sample_dt_s"]
    n = round(case["duration_s"] / dt)
    if len(rows) != n + 1:
        failures.append("Missing initial, intermediate or terminal sample")
    if not rows:
        return failures
    numeric = (
        "time_s",
        "speed_m_s",
        "angular_speed_rad_s",
        "quaternion_norm_error",
        "constraint_residual_m",
        "energy_j",
        "motor_work_j",
    )
    for i, row in enumerate(rows):
        if any(
            k not in row or not isinstance(row[k], (int, float)) or not math.isfinite(row[k]) for k in numeric
        ):
            failures.append(f"Nonfinite or missing raw state at {i}")
            continue
        if row.get("step") != i or abs(row["time_s"] - i * dt) > 1e-8:
            failures.append(f"Sample timing discontinuity at {i}")
        for field, limit in [
            ("speed_m_s", "speed_max_m_s"),
            ("angular_speed_rad_s", "angular_speed_max_rad_s"),
            ("quaternion_norm_error", "quaternion_norm_error"),
            ("constraint_residual_m", "constraint_residual_max_m"),
        ]:
            if row[field] < 0 or row[field] > limits[limit]:
                failures.append(f"Physical excursion {field} at {i}")
        if case["kind"] == "anchor":
            reference = max(abs(rows[0]["energy_j"]), 1e-12)
            if row["energy_j"] - rows[0]["energy_j"] > limits["passive_positive_energy_fraction"] * reference:
                failures.append(f"Unexplained passive positive energy at {i}")
        # Powered energy must be assessed against measured motor work separately.
    return failures


def require_retry_admission(receipt, actual_binding, reconstructed):
    """No status string, old receipt or incomplete checker can unlock training.

    A future complete checker must supply reconstructed physical evidence, not
    just a self-reported PASS. This experimental implementation deliberately has
    no successful admission path until that checker has been physically tested.
    """
    raise ValueError("Retry backend is diagnostics-only: complete independent qualification is required")
