#!/usr/bin/env python
"""Check retained 3D coordinates independently using only the standard library."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def verify_layout(value):
    coords = value["coords"]
    if (
        len(coords) != 11
        or any(len(p) != 3 for p in coords)
        or any(type(v) not in (int, float) or not math.isfinite(v) for p in coords for v in p)
    ):
        raise ValueError("Invalid finite 11x3 coordinate data")
    digest = hashlib.sha256(json.dumps(coords, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != value["coordinate_sha256"] or coords[0] != [0, 0, 1]:
        raise ValueError("Coordinate identity/base mismatch")
    points = coords[1:]
    bounds = [(-2.7, 2.7), (-1.8, 1.8), (1, 3)]
    if any(not bounds[j][0] <= p[j] <= bounds[j][1] for p in points for j in range(3)):
        raise ValueError("Destination outside declared bounds")
    axes = [sorted(p[j] for p in points) for j in range(3)]
    spans = [a[-1] - a[0] for a in axes]
    if any(x < minimum for x, minimum in zip(spans, [4.86, 3.24, 1.80])):
        raise ValueError("Insufficient destination spans")
    if any(a[i + 2] - a[i] <= 0.10 + 1e-12 for a in axes[:2] for i in range(8)):
        raise ValueError("Three destinations inside a closed 0.10 m band")
    thirds = [0, 0, 0]
    for p in points:
        z = p[2]
        band = 0 if z < 1 + 2 / 3 else 1 if z < 1 + 4 / 3 else 2
        thirds[band] += 1
    if not all(thirds):
        raise ValueError("Missing height third")
    separation = min(math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1 :])
    if separation < 0.50:
        raise ValueError("Insufficient three-dimensional separation")
    base = min(math.dist(coords[0], p) for p in points)
    if base <= 0.20:
        raise ValueError("Start inside an arrival sphere")
    if (
        value.get("priorities") != [81, 42, 43, 64, 23, 84, 6, 98, 22, 56]
        or value.get("priority_seed") != 40029
    ):
        raise ValueError("Frozen integer priority vector/seed changed")
    return {
        "status": "PASS",
        "spans_m": spans,
        "height_band_counts": thirds,
        "min_separation_m": separation,
        "min_base_distance_m": base,
        "coordinate_sha256": digest,
        "method": "Independent standard-library finite/bounds/sliding-triple/height-third/3D-distance check",
        "scope": "Geometry only; no flight, projected visibility or human approval claim",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    result = verify_layout(json.loads(Path(a.layout).read_text()))
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
