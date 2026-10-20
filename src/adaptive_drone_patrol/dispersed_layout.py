"""Versioned deterministic geometry, separate from the priority random stream."""

import hashlib
import json
import numpy as np

GENERATOR = "numpy-PCG64-SeedSequence(seed,300,attempt)-uniform-xyz-v1"
BOUNDS = np.array([[-2.7, 2.7], [-1.8, 1.8], [1.0, 3.0]])
MIN_SPANS = np.array([4.86, 3.24, 1.80])


def coordinate_hash(coords):
    return hashlib.sha256(json.dumps(coords, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def layout_candidate(seed, attempt):
    """Sample ten independent continuous points; never edit a sampled coordinate."""
    if type(seed) is not int or type(attempt) is not int or min(seed, attempt) < 0:
        raise ValueError("Nonnegative integer seed and attempt required")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 300, attempt])))
    return [[0.0, 0.0, 1.0], *rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (10, 3)).tolist()]


def band_clear(values):
    ordered = np.sort(values)
    # Closed bands include a triple exactly 0.10 m apart. Tiny arithmetic
    # roundoff must not make an exact boundary pass.
    return bool(np.all(ordered[2:] - ordered[:-2] > 0.10 + 1e-12))


def height_bands(values):
    return np.bincount(
        np.searchsorted([1.0 + 2 / 3, 1.0 + 4 / 3], values, side="right"), minlength=3
    ).tolist()


def hull_area(points):
    """Monotone-chain planar convex hull, with no SciPy dependency."""
    points = sorted(set(map(tuple, points)))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for seq, chain in [(points, lower), (points[::-1], upper)]:
        for p in seq:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
    hull = lower[:-1] + upper[:-1]
    return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(hull, hull[1:] + hull[:1]))) / 2


def layout_checks(coords):
    """Return all geometry rejection reasons, independently of flight outcomes."""
    try:
        a = np.asarray(coords, dtype=float)
    except (TypeError, ValueError):
        return {"passed": False, "rejection_reasons": ["shape"]}
    if a.shape != (11, 3):
        return {"passed": False, "rejection_reasons": ["shape"]}
    if not np.isfinite(a).all():
        return {"passed": False, "rejection_reasons": ["nonfinite"]}
    p = a[1:]
    spans = np.ptp(p, axis=0)
    separation = min(float(np.linalg.norm(x - y)) for i, x in enumerate(p) for y in p[i + 1 :])
    base_distance = float(np.linalg.norm(p - a[0], axis=1).min())
    bands = height_bands(p[:, 2])
    checks = {
        "base_position": np.array_equal(a[0], [0, 0, 1]),
        "bounds": bool(np.all(p >= BOUNDS[:, 0]) and np.all(p <= BOUNDS[:, 1])),
        "spans": bool(np.all(spans >= MIN_SPANS)),
        "x_band": band_clear(p[:, 0]),
        "y_band": band_clear(p[:, 1]),
        "height_thirds": all(bands),
        "separation": separation >= 0.5,
        "base_overlap": base_distance > 0.2,
    }
    reasons = [k for k, v in checks.items() if not v]
    return {
        "passed": not reasons,
        "rejection_reasons": reasons,
        "spans_m": spans.tolist(),
        "height_min_max_m": [float(p[:, 2].min()), float(p[:, 2].max())],
        "height_band_counts": bands,
        "min_separation_m": separation,
        "min_base_distance_m": base_distance,
        "convex_hull_xy_m2": hull_area(p[:, :2]),
    }


def generate_layout(seed, max_attempts=10000):
    attempts = []
    for attempt in range(max_attempts):
        coords = layout_candidate(seed, attempt)
        checks = layout_checks(coords)
        attempts.append(
            {
                "attempt": attempt,
                "coordinate_sha256": coordinate_hash(coords),
                "rejection_reasons": checks["rejection_reasons"],
            }
        )
        if checks["passed"]:
            return {
                "schema_version": 1,
                "generator": GENERATOR,
                "seed": seed,
                "selected_attempt": attempt,
                "coords": coords,
                "checks": checks,
                "coordinate_sha256": coordinate_hash(coords),
                "attempts": attempts,
                "bounds_m": BOUNDS.tolist(),
                "min_spans_m": MIN_SPANS.tolist(),
                "band_width_m": 0.10,
                "nominal_horizontal_area_ratio": 9,
                "operator_review": "PENDING",
                "acceptance_frozen": False,
            }
    raise RuntimeError("No layout passed the frozen geometry checks")


def expanded_scenario(seed, split, kind=None):
    """Keep predecessor split/priority/cost draws and change only the spatial family."""
    from .scenario import generate_scenario, INTEGER_PROFILE

    scenario = generate_scenario(seed, split, kind, 10, INTEGER_PROFILE)
    scenario.coords = generate_layout(seed)["coords"]
    scenario.validate_physical()
    return scenario


def acceptance_prerequisites(layout, probes, checkpoint_hash, config_hash):
    """Keep spatial approval separate from new-model transition success."""
    reasons = []
    review = layout.get("operator_review")
    if not isinstance(review, dict) or review.get("status") != "APPROVED":
        reasons.append("operator spatial review pending")
    seeds = [p.get("execution_seed") for p in probes]
    if len(probes) != 5 or sorted(seeds) != [11, 12, 31, 32, 33]:
        reasons.append("fixed validation/held-out probe denominator")
    if any(
        p.get("status") != "SUCCEEDED" or len(p.get("legs", [])) != 9 or p.get("evidence_kind") != "physical"
        for p in probes
    ):
        reasons.append("incomplete physical transition probes")
    if any(
        p.get("checkpoint_sha256") != checkpoint_hash or p.get("config_sha256") != config_hash for p in probes
    ):
        reasons.append("mixed model/config in transition probes")
    if probes and len({p.get("suite_sha256") for p in probes}) != 1:
        reasons.append("changed transition suite")
    return {"passed": not reasons, "rejection_reasons": reasons}
