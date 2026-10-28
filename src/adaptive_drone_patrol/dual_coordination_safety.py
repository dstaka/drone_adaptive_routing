"""Conservative geometry diagnostics and latched failures for development.

Native shape distances include massless rotor/link coverage. Sweeps must be
qualified against retained internal poses before a formal continuous-safety
claim; a 100 Hz trace alone is never accepted as that qualification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from .payload_contract import payload_metrics


def swept_sphere_clearance(a0, a1, b0, b1, radius_a, radius_b):
    points = np.asarray([a0, a1, b0, b1], float)
    if points.shape != (4, 3) or not np.isfinite(points).all() or min(radius_a, radius_b) < 0:
        raise ValueError("Finite swept spheres required")
    delta = points[0] - points[2]
    velocity = (points[1] - points[0]) - (points[3] - points[2])
    length = float(velocity @ velocity)
    fraction = float(np.clip(-delta @ velocity / length, 0, 1)) if length else 0.0
    return float(np.linalg.norm(delta + fraction * velocity) - radius_a - radius_b), fraction


def vertical_extent(kind, size, rotation):
    size = np.asarray(size, float)
    z = np.asarray(rotation, float).reshape(3, 3)[2]
    if kind == "box":
        return float(np.abs(z) @ size)
    if kind == "cylinder":
        return float(size[0] * np.linalg.norm(z[:2]) + size[1] * abs(z[2]))
    if kind == "capsule":
        return float(size[0] + size[1] * abs(z[2]))
    if kind == "sphere":
        return float(size[0])
    raise ValueError("Uncovered collision geometry")


def wash_sphere_incursion(rotor_plane, center, radius):
    rotor_plane, center = np.asarray(rotor_plane, float), np.asarray(center, float)
    return bool(
        center[2] - radius <= rotor_plane[2]
        and center[2] + radius >= rotor_plane[2] - 1.0
        and np.linalg.norm(center[:2] - rotor_plane[:2]) - radius < 0.60
    )


def geom_radius(kind, size):
    if kind == "box":
        return float(np.linalg.norm(size))
    if kind == "cylinder":
        return float(np.linalg.norm(size[:2]))
    if kind == "capsule":
        return float(sum(size[:2]))
    if kind == "sphere":
        return float(size[0])
    raise ValueError("Uncovered collision geometry")


def audit_geometry(fixture):
    """Native exact convex distances plus conservative whole-shape wash test."""
    f = fixture
    model, data, mj = f.model, f.data, f.mj
    names = {
        int(mj.mjtGeom.mjGEOM_BOX): "box",
        int(mj.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mj.mjtGeom.mjGEOM_SPHERE): "sphere",
    }
    distance = float("inf")
    closest = None
    for a in f.audit_geom_ids[0]:
        for b in f.audit_geom_ids[1]:
            d = float(mj.mj_geomDistance(model, data, a, b, 100.0, None))
            if d < distance:
                distance, closest = d, [a, b]
    floor = []
    wash = []
    for drone, ids in enumerate(f.audit_geom_ids):
        floor.append(
            min(
                float(data.geom_xpos[g, 2])
                - vertical_extent(names[int(model.geom_type[g])], model.geom_size[g], data.geom_xmat[g])
                for g in ids
            )
        )
        other = 1 - drone
        for rotor in range(4):
            # Projection is world vertical, irrespective of vehicle tilt.
            body = f.body_ids[drone][f"prop{rotor}_link"]
            # Rotor visual plane is offset from original force COM; use proxy center.
            rotor_geoms = [g for g in ids if model.geom_bodyid[g] == body]
            for g in rotor_geoms:
                for h in f.audit_geom_ids[other]:
                    kind = names[int(model.geom_type[h])]
                    if wash_sphere_incursion(
                        data.geom_xpos[g], data.geom_xpos[h], geom_radius(kind, model.geom_size[h])
                    ):
                        wash.append(dict(rotor_geom=g, other_geom=h))
    sample = f.sample()
    metrics = [payload_metrics(row, row["drone_xyz"], p) for row, p in zip(sample["drones"], f.payloads)]
    return dict(
        world_time=float(data.time),
        min_clearance_m=distance,
        closest_pair=closest,
        floor_minima_m=floor,
        contacts=sample["contacts"],
        wash_incursions=wash,
        finite=bool(
            np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()
        ),
        constraint_max_m=max(m["constraint_residual_m"] for m in metrics),
        max_roll_pitch_deg=max(max(abs(m["drone_roll_deg"]), abs(m["drone_pitch_deg"])) for m in metrics),
        max_swing_deg=max(m["swing_angle_deg"] for m in metrics),
        max_spin_deg_s=max(m["package_spin_deg_s"] for m in metrics),
        metrics=metrics,
        wash_method="conservative bounding sphere against vertical cylinder; false positives retained",
    )


@dataclass
class SafetyLedger:
    failures: list = field(default_factory=list)
    minimum_clearance_m: float = float("inf")
    floor_minima_m: list = field(default_factory=lambda: [float("inf"), float("inf")])

    @property
    def safe(self):
        return not self.failures

    def observe(self, row):
        reasons = []
        if not row["finite"]:
            reasons.append("nonfinite state")
        if not math.isfinite(row["min_clearance_m"]) or row["min_clearance_m"] < 0.20:
            reasons.append("assembly margin")
        if any(not math.isfinite(x) or x <= 0.10 for x in row["floor_minima_m"]):
            reasons.append("floor clearance")
        if row["contacts"]:
            reasons.append("forbidden contact")
        if row["wash_incursions"]:
            reasons.append("rotor-wash exclusion")
        if not math.isfinite(row["constraint_max_m"]) or row["constraint_max_m"] > 2e-5:
            reasons.append("constraint residual")
        if not math.isfinite(row["max_roll_pitch_deg"]) or row["max_roll_pitch_deg"] > 60:
            reasons.append("tilt")
        self.minimum_clearance_m = min(self.minimum_clearance_m, row["min_clearance_m"])
        self.floor_minima_m = [min(a, b) for a, b in zip(self.floor_minima_m, row["floor_minima_m"])]
        if reasons:
            self.failures.append(
                dict(world_time=row["world_time"], reasons=reasons, closest_pair=row.get("closest_pair"))
            )
        return self.safe


@dataclass(frozen=True)
class SupervisorDecision:
    references: tuple
    held: tuple[bool, bool]
    reason: str
    horizon_s: float
    predicted_clearance_m: float
    failure: str | None = None
    qualification: str = "DEVELOPMENT_ONLY"


class ReferenceSupervisor:
    """Development whole-assembly envelopes; no hidden assignment/waypoint search.

    A 0.40 m sphere centered on the vehicle contains the .30 m rigid suspension,
    largest oriented box and original rotors. Tracking inflation is prospective,
    not a measured qualification. Production admission must bind a stopping and
    tracking receipt; this class intentionally makes no formal safety claim.
    """

    def __init__(self, *, tracking_inflation=0.15):
        if not math.isfinite(tracking_inflation) or tracking_inflation < 0:
            raise ValueError("Finite nonnegative tracking envelope required")
        self.tracking_inflation = tracking_inflation
        self.assembly_radius = 0.40

    def predict(self, references, time_s, horizon_s, radii):
        # For each .1s interval the Bezier speed certificate encloses departure
        # from the linear center chord by <=vmax*dt. Inflate each sphere by that
        # bound; this checks all time between predictions, not just samples.
        samples = max(1, math.ceil(horizon_s / 0.1))
        times = np.linspace(time_s, time_s + horizon_s, samples + 1)
        positions = [reference.positions(times) for reference in references]
        horizontal_positions = [position.copy() for position in positions]
        for position in horizontal_positions:
            position[:,2] = 0.
        speeds = [reference.bounds().speed for reference in references]
        minimum = float("inf")
        for index,(start, end) in enumerate(zip(times, times[1:])):
            a0, b0 = (p[index] for p in positions)
            a1, b1 = (p[index+1] for p in positions)
            inflate = [speed * (end - start) for speed in speeds]
            rr = [radii[i] + inflate[i] for i in range(2)]
            distance, _ = swept_sphere_clearance(a0, a1, b0, b1, *rr)
            minimum = min(minimum, distance)
            if distance < 0.20:
                return True, minimum
            # Conservative swept wash: sphere XY traces within .60+both shape
            # envelopes, and vertical intervals overlap either downward zone.
            h0,h1 = horizontal_positions
            horizontal, _ = swept_sphere_clearance(
                h0[index], h0[index+1], h1[index], h1[index+1], rr[0], rr[1]
            )
            low_a, high_a = min(a0[2], a1[2]) - rr[0], max(a0[2], a1[2]) + rr[0]
            low_b, high_b = min(b0[2], b1[2]) - rr[1], max(b0[2], b1[2]) + rr[1]
            if horizontal < 0.60 and (
                low_b <= high_a and high_b >= low_a - 1 or low_a <= high_b and high_a >= low_b - 1
            ):
                return True, minimum
        return False, minimum

    def decide(self, references, states, time_s, *, priority):
        if len(references) != 2 or len(states) != 2 or priority not in (1, 2):
            raise ValueError("Two references and measured states, priority1/2 required")
        brakes = tuple(r.brake(time_s, actual_state=s) for r, s in zip(references, states))
        horizon = max(2.0, max(b.duration for b in brakes) + 0.20)
        radii = [
            self.assembly_radius
            + self.tracking_inflation
            + np.linalg.norm(r.sample(time_s)[0] - np.asarray(s.position))
            for r, s in zip(references, states)
        ]
        conflict, minimum = self.predict(references, time_s, horizon, radii)
        if not conflict:
            return SupervisorDecision(tuple(references), (False, False), "clear", horizon, minimum)
        yielding = 2 - priority
        candidate = list(references)
        candidate[yielding] = brakes[yielding]
        # Include a terminal hover for the yielding assembly while the other
        # follows its remaining finite path. No global detour or reassignment.
        terminal_horizon = max(horizon, max(r.end_time - time_s for r in candidate) + 0.20)
        conflict, minimum = self.predict(candidate, time_s, terminal_horizon, radii)
        if not conflict:
            held = [False, False]
            held[yielding] = True
            return SupervisorDecision(tuple(candidate), tuple(held), "priority-yield", horizon, minimum)
        conflict, minimum = self.predict(brakes, time_s, horizon, radii)
        if conflict:
            return SupervisorDecision(
                brakes,
                (True, True),
                "brake-both-unestablished",
                horizon,
                minimum,
                failure="safe braking/terminal hold not established",
            )
        return SupervisorDecision(brakes, (True, True), "brake-both", horizon, minimum)
