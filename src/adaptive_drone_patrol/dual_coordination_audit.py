"""Independent raw structure, native point replay, and declared swept reconstruction.

No producer geometry verdict is trusted. Native replay uses saved qpos/qvel and
commands in a separately loaded model. Reconstruction connects native substep
geometry centers linearly and orientations by shortest rotation; RK trial states
are point checks, never a purported physical trajectory. This numerical path
contract is explicit and is not a theorem about unsaved true continuous physics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _array(value, shape=None):
    result = np.asarray(value, float)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("invalid shape or nonfinite state")
    return result


def _same(a, b, name, *, atol=1e-10):
    a, b = _array(a), _array(b)
    if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=0):
        raise ValueError(f"{name} mismatch")


def _time(row, expected):
    value = row["world_time"]
    if not math.isfinite(value) or abs(value - expected) > 5e-9:
        raise ValueError(f"world time mismatch: expected {expected}, got {value}")


def _state(row, expected, *, geometry=False, actual=False, stage=False):
    _time(row, expected)
    _array(row["qpos"], (30,))
    _array(row["qvel"], (24,))
    if geometry or actual:
        _array(row["qacc"], (24,))
        _array(row["qfrc_actuator"], (24,))
        for contact in row["contacts"]:
            if any(type(contact[k]) is not int or contact[k] < 0 for k in ("geom1", "geom2")):
                raise ValueError("invalid contact geometry identity")
            if not math.isfinite(contact["distance"]):
                raise ValueError("nonfinite contact distance")
    if actual or stage:
        _array(row["ctrl"], (8,))
        _array(row["qfrc_passive"], (24,))
        warnings = row["warning_counts"]
        if len(warnings) != 7 or any(type(x) is not int or x < 0 for x in warnings):
            raise ValueError("invalid numerical warning counters")
    if geometry:
        points = _array(row["geom_xpos"])
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("invalid geometry positions")
        _array(row["geom_xmat"], (len(points), 9))
        _array(row["actuator_force"], (8,))
    if stage:
        matrices = _array(row["xmat"])
        if matrices.ndim != 2 or matrices.shape[1] != 9:
            raise ValueError("invalid stage body orientations")
        _array(row["xipos"], (len(matrices), 3))
    if actual:
        if row["schema"] != "dual-raw-v1" or [d["drone_id"] for d in row["drones"]] != [1, 2]:
            raise ValueError("exactly two ordered drone identities required")
        _array(row["rpm"], (2, 4))
        for drone in row["drones"]:
            for body in ("drone", "rod", "package"):
                for name in ("xyz", "velocity", "angular_velocity"):
                    _array(drone[f"{body}_{name}"], (3,))
                q = _array(drone[f"{body}_quaternion_wxyz"], (4,))
                if abs(np.linalg.norm(q) - 1) > 1e-5:
                    raise ValueError("invalid raw body quaternion")


class StructureVerifier:
    """Streaming verifier; an error remains latched while later rows are counted."""

    def __init__(self, expected_steps=None):
        self.expected_steps = expected_steps
        self.records = self.steps = self.rk_states = self.native_intervals = 0
        self.previous = None
        self.failures = []

    def observe(self, row):
        location = f"raw-row:{self.records + 1}"
        self.records += 1
        if row.get("kind") == "STEP":
            self.steps += 1
        try:
            if self.records == 1:
                if row["kind"] != "RESET":
                    raise ValueError("first row must be RESET")
                _state(row["actual"], 0.0, actual=True)
                _state(row["geometry"], 0.0, geometry=True)
                for name in ("qpos", "qvel", "qacc", "qfrc_actuator"):
                    _same(row["geometry"][name], row["actual"][name], f"RESET {name}", atol=0.0)
            else:
                if row["kind"] != "STEP" or row["tick"] != self.records - 1:
                    raise ValueError("expected sequential 100 Hz STEP after RESET")
                tick = row["tick"]
                _state(row["actual"], tick * 0.01, actual=True)
                if len(row["substeps"]) != 17 or len(row["stages"]) != 64:
                    raise ValueError("expected 17 native poses and 64 RK states")
                if len(row["controls"]) != 2 or len(row["reference"]) != 2:
                    raise ValueError("two controls and references required")
                for reference in row["reference"]:
                    for name in ("p", "v", "a"):
                        _array(reference[name], (3,))
                start = (tick - 1) * 0.01
                for index, substep in enumerate(row["substeps"]):
                    _state(substep, start + index * 0.01 / 16, geometry=True)
                for index, stage in enumerate(row["stages"]):
                    _state(stage, start + (index // 4 + (0, 0.5, 0.5, 1)[index % 4]) * 0.01 / 16, stage=True)
                    _same(stage["ctrl"], row["actual"]["ctrl"], "constant step actuator command", atol=0.0)
                    if index % 4 == 0:
                        for name in ("qpos", "qvel"):
                            _same(
                                stage[name], row["substeps"][index // 4][name], f"RK initial {name}", atol=0.0
                            )
                for name in ("qpos", "qvel"):
                    _same(row["substeps"][0][name], self.previous[name], f"substep initial {name}", atol=0.0)
                    _same(row["substeps"][-1][name], row["actual"][name], f"substep final {name}", atol=0.0)
                self.rk_states += 64
                self.native_intervals += 16
            self.previous = row["actual"]
            return True
        except (KeyError, TypeError, ValueError, IndexError) as error:
            self.failures.append(dict(location=location, reason=str(error)))
            if isinstance(row.get("actual"), dict):
                self.previous = row["actual"]
            return False

    def report(self):
        failures = list(self.failures)
        if self.records == 0:
            failures.append(dict(location="trace", reason="expected RESET, empty trace"))
        if self.expected_steps is not None and self.steps != self.expected_steps:
            failures.append(
                dict(location="trace", reason=f"expected {self.expected_steps} STEP rows, got {self.steps}")
            )
        return dict(
            structural_valid=not failures,
            failures=failures,
            raw_records=self.records,
            control_steps=self.steps,
            rk_states=self.rk_states,
            native_substep_intervals=self.native_intervals,
            continuous_safety_qualified=False,
        )


def verify_structure(rows, *, expected_steps=None):
    verifier = StructureVerifier(expected_steps)
    for row in rows:
        verifier.observe(row)
    return verifier.report()


@dataclass
class PointAuditLedger:
    points: int = 0
    violations: list = field(default_factory=list)
    minimum: dict | None = None
    floors: list = field(default_factory=lambda: [math.inf, math.inf])
    max_swing: float = 0.0
    max_spin: float = 0.0

    def observe(self, row, *, time, location):
        self.points += 1
        clearance = row["min_clearance_m"]
        if self.minimum is None or clearance < self.minimum["distance_m"]:
            self.minimum = dict(
                distance_m=clearance, time=time, location=location, closest_pair=row["closest_pair"]
            )
        self.floors = [min(a, b) for a, b in zip(self.floors, row["floor_minima_m"])]
        self.max_swing = max(self.max_swing, row["max_swing_deg"])
        self.max_spin = max(self.max_spin, row["max_spin_deg_s"])
        reasons = []
        if not row["finite"]:
            reasons.append("nonfinite native state")
        if not math.isfinite(clearance) or clearance < 0.2:
            reasons.append("assembly margin")
        if any(not math.isfinite(v) or v <= 0.1 for v in row["floor_minima_m"]):
            reasons.append("floor clearance")
        if row["contacts"]:
            reasons.append("forbidden contact")
        if row["wash_incursions"]:
            reasons.append("rotor-wash exclusion")
        if not math.isfinite(row["constraint_max_m"]) or row["constraint_max_m"] > 2e-5:
            reasons.append("constraint residual")
        if not math.isfinite(row["max_roll_pitch_deg"]) or row["max_roll_pitch_deg"] > 60:
            reasons.append("tilt")
        if any(row["warning_counts"]):
            reasons.append("native numerical warning")
        for reason in reasons:
            self.violations.append(
                dict(
                    time=time,
                    location=location,
                    reason=reason,
                    min_clearance_m=clearance,
                    closest_pair=row["closest_pair"],
                )
            )

    def report(self):
        return dict(
            retained_points_safe=not self.violations and self.points > 0,
            audited_points=self.points,
            violation_count=len(self.violations),
            violations=self.violations,
            clearance_minimum=self.minimum,
            minimum_clearance_m=None if self.minimum is None else self.minimum["distance_m"],
            floor_minima_m=[v if math.isfinite(v) else None for v in self.floors],
            max_swing_deg=self.max_swing,
            max_spin_deg_s=self.max_spin,
        )


def conservative_interval_bound(*, clearance_a, clearance_b, dt, speed_bounds):
    """Nearest-endpoint Lipschitz bound, only with externally justified speeds.

    Measured endpoint speeds alone are NOT bounds on the intervening trajectory.
    The reconstruction below derives displacement from its declared path instead.
    """
    if len(speed_bounds) != 2 or any(v is None for v in speed_bounds):
        return dict(qualified=False, lower_bound_m=None, reason="interval speed envelope unavailable")
    values = [clearance_a, clearance_b, dt, *speed_bounds]
    if not all(math.isfinite(v) for v in values) or dt <= 0 or min(speed_bounds) < 0:
        raise ValueError("finite positive interval and nonnegative speed bounds required")
    return dict(qualified=True, lower_bound_m=min(clearance_a, clearance_b) - sum(speed_bounds) * dt / 2)


def _radius(shape):
    size, kind = np.asarray(shape["size"]), shape["kind"]
    if kind == "box":
        return float(np.linalg.norm(size))
    if kind == "cylinder":
        return float(np.linalg.norm(size[:2]))
    if kind == "capsule":
        return float(size[0] + size[1])
    if kind == "sphere":
        return float(size[0])
    raise ValueError("uncovered collision shape")


def _floor(position, rotation, shape):
    size, kind, z = np.asarray(shape["size"]), shape["kind"], rotation[2]
    if kind == "box":
        support = abs(z) @ size
    elif kind == "cylinder":
        support = size[0] * np.linalg.norm(z[:2]) + size[1] * abs(z[2])
    elif kind == "capsule":
        support = size[0] + size[1] * abs(z[2])
    elif kind == "sphere":
        support = size[0]
    else:
        raise ValueError("uncovered collision shape")
    return float(position[2] - support)


def _angle(a, b):
    relative = a.T @ b
    skew = (
        np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        / 2
    )
    return float(math.atan2(np.linalg.norm(skew), np.clip((np.trace(relative) - 1) / 2, -1, 1)))


def _rotation_midpoint(a, b):
    relative = a.T @ b
    angle = _angle(a, b)
    if angle < 1e-14:
        return a.copy()
    if abs(math.pi - angle) < 1e-7:
        _, axes = np.linalg.eigh((relative + relative.T + 2 * np.eye(3)) / 4)
        axis = axes[:, -1]
    else:
        axis = np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return a @ (np.eye(3) + math.sin(angle / 2) * cross + (1 - math.cos(angle / 2)) * (cross @ cross))


def _pair_bounds_cache(shapes):
    pairs = [
        (a, b)
        for a, x in enumerate(shapes)
        for b, y in enumerate(shapes)
        if x["owner"] == 0 and y["owner"] == 1
    ]
    left, right = np.asarray(pairs, dtype=int).T
    radii = np.asarray([_radius(shapes[a]) + _radius(shapes[b]) for a, b in pairs])
    return left, right, radii


def minimum_shape_distance(positions, shapes, pair_distance, *, bounds_cache=None):
    """Exact nearest pair, pruned only by conservative enclosing spheres."""
    left, right, radii = _pair_bounds_cache(shapes) if bounds_cache is None else bounds_cache
    lower = np.linalg.norm(np.asarray(positions)[left] - np.asarray(positions)[right], axis=1) - radii - 1e-12
    best = (math.inf, -1, -1)
    calls = 0
    for index in np.argsort(lower, kind="stable"):
        if lower[index] > best[0]:
            break
        a, b = int(left[index]), int(right[index])
        value = float(pair_distance(a, b))
        calls += 1
        if not math.isfinite(value):
            raise ValueError("nonfinite native shape distance")
        best = min(best, (value, a, b))
    return dict(distance_m=best[0], closest_pair=[best[1], best[2]], exact_evaluations=calls)


def _wash_cache(shapes):
    pairs = [
        (a, b)
        for a, x in enumerate(shapes)
        for b, y in enumerate(shapes)
        if x.get("rotor", False) and y["owner"] is not None and x["owner"] != y["owner"]
    ]
    if not pairs:
        return np.array([], int), np.array([], int), np.array([], float)
    left, right = np.asarray(pairs, dtype=int).T
    return left, right, np.asarray([_radius(shapes[b]) for _, b in pairs])


def _point_wash_possible(positions, cache):
    """The stationary special case of the same sphere/cylinder predicate."""
    rotor, other, radius = cache
    if not len(rotor):
        return []
    delta = positions[other] - positions[rotor]
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    selected = np.flatnonzero(
        (horizontal - radius < 0.60)
        & (delta[:, 2] - radius <= 0)
        & (delta[:, 2] + radius >= -1)
    )
    return [dict(rotor_geom=int(rotor[i]), other_geom=int(other[i])) for i in selected]


def _wash_possible(start, end, shapes, *, cache=None):
    """Conservative sphere/vertical-cylinder overlap along linear center paths."""
    rotor, other, radius = _wash_cache(shapes) if cache is None else cache
    if not len(rotor):
        return []
    if start is end:
        return _point_wash_possible(start,(rotor,other,radius))
    a, b = start[other] - start[rotor], end[other] - end[rotor]
    delta = b[:, :2] - a[:, :2]
    squared = np.sum(delta * delta, axis=1)
    fraction = np.clip(
        np.divide(-np.sum(a[:, :2] * delta, axis=1), squared, out=np.zeros_like(squared), where=squared > 0),
        0,
        1,
    )
    horizontal = np.linalg.norm(a[:, :2] + fraction[:, None] * delta, axis=1)
    selected = np.flatnonzero(
        (horizontal - radius < 0.60)
        & (np.minimum(a[:, 2], b[:, 2]) - radius <= 0)
        & (np.maximum(a[:, 2], b[:, 2]) + radius >= -1)
    )
    return [dict(rotor_geom=int(rotor[i]), other_geom=int(other[i])) for i in selected]


def self_collision_pairs(*, shapes, geom_bodyid, body_weldid, body_parentid,
                         contype, conaffinity, exclusions, filter_parent=True):
    """Enabled same-assembly pairs under the native dynamic-pair filters.

    Explicit geom pairs are rejected by the model adapter. Weld groups and
    welded parents follow MuJoCo's collision-driver rules; exclusions use the
    actual named bodies, never an inferred whole-assembly exclusion.
    """
    excluded={tuple(sorted(map(int,pair))) for pair in exclusions}
    pairs=[]
    for left,a in enumerate(shapes):
        if a['owner'] is None:
            continue
        for right in range(left+1,len(shapes)):
            if shapes[right]['owner'] != a['owner']:
                continue
            ba,bb=int(geom_bodyid[left]),int(geom_bodyid[right])
            wa,wb=int(body_weldid[ba]),int(body_weldid[bb])
            pa,pb=int(body_weldid[body_parentid[wa]]),int(body_weldid[body_parentid[wb]])
            if wa==wb or tuple(sorted((ba,bb))) in excluded:
                continue
            if filter_parent and wa and wb and (wa==pb or wb==pa):
                continue
            if not ((int(contype[left]) & int(conaffinity[right]))
                    or (int(contype[right]) & int(conaffinity[left]))):
                continue
            pairs.append((left,right))
    return pairs


def swept_reconstruction(start, end, *, shapes, pair_distance, max_depth=12, wash_cache=None,
                         self_pairs=()):
    """Certify the declared interpolation with adaptive conservative enclosures.

    Each surface point moves at most center displacement + radius*rotation angle.
    The nearest endpoint lies at most half that displacement away on the declared
    constant-speed interpolation. Subdivision tightens bounds, never assumes the
    measured endpoint velocity bounds an unseen physical acceleration.
    """
    if type(max_depth) is not int or not 0 <= max_depth <= 16:
        raise ValueError("bounded subdivision depth required")
    count = len(shapes)
    wash_cache = _wash_cache(shapes) if wash_cache is None else wash_cache
    pa, pb = (_array(row["geom_xpos"], (count, 3)) for row in (start, end))
    ra, rb = (_array(row["geom_xmat"]).reshape(count, 3, 3) for row in (start, end))
    for rotations in (ra, rb):
        if not np.allclose(
            np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=1e-8, rtol=0
        ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-8, rtol=0):
            raise ValueError("geometry matrices must be proper rotations")
    ids = [g for g, s in enumerate(shapes) if s["owner"] is not None]
    cross_pairs = [(a, b) for a in ids for b in ids if shapes[a]["owner"] == 0 and shapes[b]["owner"] == 1]
    if not cross_pairs:
        raise ValueError("two covered assemblies required")
    self_pairs=[tuple(pair) for pair in self_pairs]
    if any(len(pair)!=2 or pair[0]>=pair[1] or any(g not in ids for g in pair)
           or shapes[pair[0]]['owner']!=shapes[pair[1]]['owner'] for pair in self_pairs):
        raise ValueError('valid distinct same-assembly geometry pairs required')
    pairs=cross_pairs+self_pairs
    margins={pair:(.2 if pair in cross_pairs else 0.) for pair in pairs}
    radii = np.array([_radius(s) if s["owner"] is not None else 0 for s in shapes])
    sampled_min = math.inf
    witness = None
    checked = 0
    lower_clearance = math.inf
    self_lower_clearance = math.inf
    unresolved = 0

    def point(p, r, fraction):
        nonlocal sampled_min, witness, checked
        checked += 1
        distances = {pair: float(pair_distance(p, r, *pair)) for pair in pairs}
        closest = min(cross_pairs, key=distances.get)
        distance = distances[closest]
        self_closest=min(self_pairs,key=distances.get) if self_pairs else None
        floors = {g: _floor(p[g], r[g], shapes[g]) for g in ids}
        reason = (
            "assembly margin"
            if distance < 0.2
            else "self-contact"
            if self_closest is not None and distances[self_closest] < 0.
            else "floor clearance"
            if min(floors.values()) <= 0.1
            else "rotor-wash exclusion"
            if _wash_possible(p, p, shapes, cache=wash_cache)
            else None
        )
        if not math.isfinite(distance) or not all(math.isfinite(v) for v in floors.values()):
            reason = "nonfinite reconstructed geometry"
        if distance < sampled_min:
            sampled_min = distance
        if reason and witness is None:
            if reason=='self-contact':
                closest=self_closest
                distance=distances[closest]
            witness = dict(fraction=fraction, reason=reason, closest_pair=list(closest), distance_m=distance)
        return distances, floors

    sa, sb = point(pa, ra, 0.0), point(pb, rb, 1.0)
    pending = [(pa, ra, pb, rb, 0.0, 1.0, sa, sb, 0)]
    while pending and witness is None:
        a, ar, b, br, fa, fb, aa, bb, depth = pending.pop()
        relative = np.swapaxes(ar, 1, 2) @ br
        skew = (
            np.stack(
                (
                    relative[:, 2, 1] - relative[:, 1, 2],
                    relative[:, 0, 2] - relative[:, 2, 0],
                    relative[:, 1, 0] - relative[:, 0, 1],
                ),
                axis=1,
            )
            / 2
        )
        angles = np.arctan2(
            np.linalg.norm(skew, axis=1), np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
        )
        movement = np.linalg.norm(b - a, axis=1) + radii * angles
        bounds={pair:min(aa[0][pair],bb[0][pair])-(movement[pair[0]]+movement[pair[1]])/2-1e-12
                for pair in pairs}
        clear=min(bounds[pair] for pair in cross_pairs)
        floor_safe = all(min(aa[1][g], bb[1][g]) - movement[g] / 2 - 1e-12 > 0.1 for g in ids)
        if all(bounds[pair]>=margins[pair] for pair in pairs) and floor_safe and not _wash_possible(a, b, shapes, cache=wash_cache):
            lower_clearance = min(lower_clearance, clear)
            if self_pairs:
                self_lower_clearance=min(self_lower_clearance,min(bounds[pair] for pair in self_pairs))
            continue
        if depth == max_depth:
            unresolved += 1
            continue
        mid, rm, fm = (a + b) / 2, np.array([_rotation_midpoint(x, y) for x, y in zip(ar, br)]), (fa + fb) / 2
        mm = point(mid, rm, fm)
        pending.extend(
            [(mid, rm, b, br, fm, fb, mm, bb, depth + 1), (a, ar, mid, rm, fa, fm, aa, mm, depth + 1)]
        )
    status = "violation" if witness else "unresolved" if unresolved else "certified"
    return dict(
        status=status,
        contract="linear-centers-shortest-rotation-v1",
        lower_clearance_m=float(lower_clearance) if math.isfinite(lower_clearance) else None,
        self_lower_clearance_m=float(self_lower_clearance) if math.isfinite(self_lower_clearance) else None,
        self_pairs_checked=len(self_pairs),
        minimum_sampled_clearance_m=sampled_min,
        witness=witness,
        witness_fraction=None if witness is None else witness["fraction"],
        sampled_poses=checked,
        unresolved_leaves=unresolved,
    )


def body_interval_certificate(start, end, *, package_offsets, max_depth=12):
    """Certify attitude and rigid-link residual on the declared body path.

    Body centers interpolate linearly and body rotations follow shortest SO(3)
    arcs. For attachment offset r and rotation arc theta, the departure from
    its endpoint chord is at most |r|*theta**2/8: the interpolation's second
    derivative norm is bounded by |r|*theta**2. This quadratic bound avoids
    treating the vehicle's common translational motion as constraint error.

    Pitch is 1-Lipschitz in rotation angle. Away from the pitch singularity,
    roll's local angular rate is bounded by angular speed/cos(pitch_bound).
    Unproved intervals are subdivided; exhaustion is a failure, never approval.
    Numerical guards add 1e-12 m/rad to bounds, without relaxing either limit.
    """
    if type(max_depth) is not int or not 0<=max_depth<=16:
        raise ValueError('bounded subdivision depth required')
    offsets=_array(package_offsets)
    if offsets.ndim!=2 or offsets.shape[1]!=3 or not len(offsets):
        raise ValueError('one finite package attachment offset per assembly required')
    count=len(offsets)
    pa,pb=(_array(row['assembly_xpos'],(count,2,3)) for row in (start,end))
    ra,rb=(_array(row['assembly_xmat'],(count,2,3,3)) for row in (start,end))
    for rotations in (ra,rb):
        if not np.allclose(np.swapaxes(rotations,-1,-2)@rotations,np.eye(3),atol=1e-8,rtol=0) or not np.allclose(np.linalg.det(rotations),1.,atol=1e-8,rtol=0):
            raise ValueError('proper body rotations required')
    base_offset=np.array([0.,0.,-.020])
    witness=None
    checked=unresolved=0
    max_residual=max_tilt=0.

    def vector(p,r,offset):
        return p[1]+r[1]@offset-p[0]-r[0]@base_offset

    def point(p,r,offset,drone,fraction):
        nonlocal witness,max_residual,max_tilt
        roll=abs(math.atan2(r[0,2,1],r[0,2,2]))
        pitch=abs(math.asin(float(np.clip(-r[0,2,0],-1,1))))
        tilt=math.degrees(max(roll,pitch))
        residual=abs(float(np.linalg.norm(vector(p,r,offset)))-.30)
        max_residual=max(max_residual,residual)
        max_tilt=max(max_tilt,tilt)
        reason='tilt' if tilt>60. else 'suspension constraint' if residual>2e-5 else None
        if reason is not None and witness is None:
            witness=dict(assembly=drone,fraction=fraction,reason=reason,
                         roll_pitch_max_deg=tilt,constraint_residual_m=residual)

    for drone,offset in enumerate(offsets):
        point(pa[drone],ra[drone],offset,drone,0.)
        point(pb[drone],rb[drone],offset,drone,1.)
    pending=[(pa[d],ra[d],pb[d],rb[d],0.,1.,0,d) for d in range(count)]
    while pending and witness is None:
        a,ar,b,br,fa,fb,depth,drone=pending.pop()
        checked+=1
        offset=offsets[drone]
        mid=(a+b)/2
        rm=np.array([_rotation_midpoint(x,y) for x,y in zip(ar,br)])
        fm=(fa+fb)/2
        point(mid,rm,offset,drone,fm)
        if witness is not None:
            break
        angles=[_angle(x,y)+1e-12 for x,y in zip(ar,br)]
        half=angles[0]/2
        pitch=abs(math.asin(float(np.clip(-rm[0,2,0],-1,1))))+half+1e-12
        roll=(abs(math.atan2(rm[0,2,1],rm[0,2,2]))+half/math.cos(pitch)+1e-12
              if pitch<math.pi/2 else math.inf)
        tilt_bound=math.degrees(max(roll,pitch))
        va,vb=vector(a,ar,offset),vector(b,br,offset)
        delta=vb-va
        squared=float(delta@delta)
        fraction=float(np.clip(-(va@delta)/squared,0.,1.)) if squared else 0.
        deviation=(np.linalg.norm(base_offset)*angles[0]**2+np.linalg.norm(offset)*angles[1]**2)/8
        lower=max(0.,float(np.linalg.norm(va+fraction*delta)-deviation-1e-12))
        upper=float(max(np.linalg.norm(va),np.linalg.norm(vb))+deviation+1e-12)
        residual_bound=max(abs(lower-.30),abs(upper-.30))
        if tilt_bound<=60. and residual_bound<=2e-5:
            max_tilt=max(max_tilt,tilt_bound)
            max_residual=max(max_residual,residual_bound)
            continue
        if depth==max_depth:
            unresolved+=1
            max_tilt=max(max_tilt,tilt_bound)
            max_residual=max(max_residual,residual_bound)
            continue
        pending.extend(((mid,rm,b,br,fm,fb,depth+1,drone),(a,ar,mid,rm,fa,fm,depth+1,drone)))
    return dict(status='violation' if witness else 'unresolved' if unresolved else 'certified',
                contract='linear-body-centers-shortest-rotation-attachment-chord-v1',
                witness=witness,checked_subintervals=checked,unresolved_leaves=unresolved,
                constraint_upper_bound_m=float(max_residual),
                tilt_upper_bound_deg=float(max_tilt) if math.isfinite(max_tilt) else None)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON token {value}")

    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line, parse_constant=reject)


class NativePointAuditor:
    """Independent named-metadata replay; never calls the producing fixture."""

    def __init__(self, model_path, properties, *, asset_root=None):
        import mujoco as mj

        if mj.__version__ != "3.15.0":
            raise ValueError("independent native replay requires MuJoCo 3.15.0")
        self.mj = mj
        document = ET.parse(model_path)
        self.asset_rebases = []
        for asset in document.getroot().findall("asset/*"):
            filename = asset.get("file")
            if filename is None:
                continue
            original = Path(filename)
            candidate = original if original.is_absolute() else Path(model_path).parent / original
            if not candidate.is_file():
                if asset_root is None:
                    raise ValueError(f"missing asset and no asset_root: {filename}")
                found = list(Path(asset_root).rglob(original.name))
                if len(found) != 1:
                    raise ValueError(f"asset rebase is missing or ambiguous: {filename}")
                candidate = found[0]
            asset.set("file", str(candidate.resolve()))
            self.asset_rebases.append(
                dict(original=filename, local=str(candidate.resolve()), sha256=_hash(candidate))
            )
        self.mj.set_mjcb_control(None)
        self.model = mj.MjModel.from_xml_string(ET.tostring(document.getroot(), encoding="unicode"))
        model = self.model
        if (
            (model.nq, model.nv, model.nu, model.na) != (30, 24, 8, 0)
            or model.opt.integrator != int(mj.mjtIntegrator.mjINT_RK4)
            or abs(model.opt.timestep - 0.01 / 16) > 1e-15
        ):
            raise ValueError("unexpected world dimensions, activation dynamics, or native integrator")
        self.body_ids, self.addresses, self.actuator_ids = [], [], []

        def identity(kind, name):
            value = mj.mj_name2id(model, kind, name)
            if value < 0:
                raise ValueError(f"missing model name {name}")
            return value

        for drone in (1, 2):
            prefix = f"drone-{drone}/"
            bodies = {
                name: identity(mj.mjtObj.mjOBJ_BODY, prefix + name)
                for name in ("base_link", "suspension_rod", "payload", *(f"prop{i}_link" for i in range(4)))
            }
            joints = {
                name: identity(mj.mjtObj.mjOBJ_JOINT, prefix + name)
                for name in ("vehicle_free", "suspension_ball", "payload_ball")
            }
            addresses = dict(
                free_qpos=int(model.jnt_qposadr[joints["vehicle_free"]]),
                free_dof=int(model.jnt_dofadr[joints["vehicle_free"]]),
                link_qpos=int(model.jnt_qposadr[joints["suspension_ball"]]),
                package_qpos=int(model.jnt_qposadr[joints["payload_ball"]]),
            )
            # Runtime-only authored damping is reconstructed from model masses,
            # rather than copied from the producer's property array.
            damping = (
                0.027 + model.body_mass[bodies["suspension_rod"]] + model.body_mass[bodies["payload"]]
            ) * 1e-5
            dof = addresses["free_dof"]
            model.dof_damping[dof + 3 : dof + 6] = damping
            self.body_ids.append(bodies)
            self.addresses.append(addresses)
            self.actuator_ids.append(
                [identity(mj.mjtObj.mjOBJ_ACTUATOR, prefix + f"motor{i}") for i in range(4)]
            )
        for name, value in [
            ("body_ids", self.body_ids),
            ("addresses", self.addresses),
            ("actuator_ids", self.actuator_ids),
        ]:
            if properties[name] != value:
                raise ValueError(f"properties metadata {name} mismatch")
        for name in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "geom_size",
            "geom_type",
            "geom_bodyid",
            "actuator_gear",
        ):
            _same(getattr(model, name), properties[name], f"properties {name}", atol=1e-12)
        _same(model.dof_damping, properties["damping"], "properties damping", atol=1e-14)
        exclusion_names=[]
        excluded_bodies=[]
        for exclusion in document.getroot().findall("contact/exclude"):
            if exclusion.get("body1").split("/")[0] != exclusion.get("body2").split("/")[0]:
                raise ValueError("cross-assembly contact exclusion is forbidden")
            names=(exclusion.get('body1'),exclusion.get('body2'))
            exclusion_names.append(names)
            excluded_bodies.append(tuple(identity(mj.mjtObj.mjOBJ_BODY,name) for name in names))
        if sorted(exclusion_names)!=sorted(tuple(pair) for pair in properties['self_exclusions']):
            raise ValueError('explicit self-exclusion metadata mismatch')
        if model.npair:
            raise ValueError('explicit geom collision pairs are outside this declared profile')
        kinds = {
            int(mj.mjtGeom.mjGEOM_BOX): "box",
            int(mj.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mj.mjtGeom.mjGEOM_SPHERE): "sphere",
        }
        self.shapes, audit_ids = [], [[], []]
        for geom in range(model.ngeom):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom) or ""
            owner = next((d for d in range(2) if name.startswith(f"drone-{d + 1}/")), None)
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                owner = None
            kind = kinds.get(int(model.geom_type[geom]))
            if owner is not None:
                if kind is None:
                    raise ValueError("uncovered native collision geometry")
                audit_ids[owner].append(geom)
            rotor = owner is not None and model.geom_bodyid[geom] in [
                self.body_ids[owner][f"prop{i}_link"] for i in range(4)
            ]
            self.shapes.append(dict(kind=kind, size=model.geom_size[geom].copy(), owner=owner, rotor=rotor))
        if audit_ids != properties["audit_geom_ids"]:
            raise ValueError("collision coverage metadata mismatch")
        self.audit_ids = audit_ids
        self._distance_cache = _pair_bounds_cache(self.shapes)
        self._wash_cache = _wash_cache(self.shapes)
        self.self_pairs=self_collision_pairs(shapes=self.shapes,geom_bodyid=model.geom_bodyid,
            body_weldid=model.body_weldid,body_parentid=model.body_parentid,
            contype=model.geom_contype,conaffinity=model.geom_conaffinity,exclusions=excluded_bodies,
            filter_parent=not bool(model.opt.disableflags & int(mj.mjtDisableBit.mjDSBL_FILTERPARENT)))
        cross=list(zip(self._distance_cache[0].tolist(),self._distance_cache[1].tolist()))
        all_pairs=cross+self.self_pairs
        self._sweep_margins={pair:(.2 if pair in cross else 0.) for pair in all_pairs}
        left,right=np.asarray(all_pairs,dtype=int).T
        self._sweep_bounds_cache=(left,right,np.array([_radius(self.shapes[a])+_radius(self.shapes[b])
                                                      for a,b in all_pairs]))
        self._assembly_body_ids=np.array([[ids['base_link'],ids['payload']] for ids in self.body_ids])
        self._package_offsets = []
        for ids in self.body_ids:
            package_geoms = [g for g in range(model.ngeom)
                             if model.geom_bodyid[g] == ids['payload'] and self.shapes[g]['owner'] is not None]
            if len(package_geoms) != 1:
                raise ValueError('exact package collision geometry required')
            self._package_offsets.append(np.array([0,0,model.geom_size[package_geoms[0],2]]))
        self._drone_attachment = np.array([0,0,-0.020])
        self.exact_distance_evaluations = 0
        self.data = mj.MjData(model)
        self.sweep_data = mj.MjData(model)
        self.kinematics_data=mj.MjData(model)
        self.ledger = PointAuditLedger()
        self.previous_ctrl = np.zeros(8)
        self.sweeps = 0
        self.sweep_failures = []
        self.sweep_lower = math.inf
        self.self_sweep_lower=math.inf
        self.body_intervals=self.body_subintervals=0
        self.body_constraint_upper=self.body_tilt_upper=0.
        self.maximum_replay_error = 0.0
        self.proof_gaps = []
        self.limitations = [
            "qualification is for saved native states and declared linear-centers-shortest-rotation reconstruction; unsaved true continuous physics is not proved",
        ]

    def _compare(self, observed, expected, label, tolerance=1e-8):
        a, b = _array(observed), _array(expected)
        if a.shape == b.shape:
            self.maximum_replay_error = max(
                self.maximum_replay_error, float(np.max(abs(a - b))) if a.size else 0.0
            )
        _same(a, b, label, atol=tolerance)

    def _body(self, body):
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self.mj.mj_jacBody(self.model, self.data, jp, jr, body)
        return dict(
            xyz=self.data.xpos[body].copy(),
            quaternion_wxyz=self.data.xquat[body].copy(),
            velocity=jp @ self.data.qvel,
            angular_velocity=jr @ self.data.qvel,
        )

    def _contacts(self):
        # Accessing MuJoCo's contact view materializes wrappers for its allocated
        # capacity even when no contacts are active. ncon is the authoritative
        # active count; do not cache a view across native state updates.
        if self.data.ncon == 0:
            return []
        return [
            dict(geom1=int(c.geom1), geom2=int(c.geom2), distance=float(c.dist)) for c in self.data.contact
        ]

    def _contact_compare(self, expected):
        def ordered(rows):
            return sorted((int(c["geom1"]), int(c["geom2"]), float(c["distance"])) for c in rows)

        left, right = ordered(self._contacts()), ordered(expected)
        if len(left) != len(right):
            raise ValueError("native contact count mismatch")
        for a, b in zip(left, right):
            if a[:2] != b[:2] or abs(a[2] - b[2]) > 1e-8:
                raise ValueError("native contact geometry/distance mismatch")

    def _diagnostics(self, warning_counts):
        model, data, mj = self.model, self.data, self.mj
        positions = data.geom_xpos
        nearest = minimum_shape_distance(
            positions,
            self.shapes,
            lambda a, b: mj.mj_geomDistance(model, data, a, b, 100.0, None),
            bounds_cache=self._distance_cache,
        )
        self.exact_distance_evaluations += nearest["exact_evaluations"]
        distance, (left, right) = nearest["distance_m"], nearest["closest_pair"]
        floors = [
            min(_floor(positions[g], data.geom_xmat[g].reshape(3, 3), self.shapes[g]) for g in ids)
            for ids in self.audit_ids
        ]
        residuals, tilts, swings, spins = [], [], [], []
        for ids,package_offset in zip(self.body_ids,self._package_offsets):
            package = self._body(ids["payload"])
            rd, rp = data.xmat[ids["base_link"]].reshape(3, 3), data.xmat[ids["payload"]].reshape(3, 3)
            suspension = (
                package["xyz"]
                + rp @ package_offset
                - data.xpos[ids["base_link"]]
                - rd @ self._drone_attachment
            )
            length = np.linalg.norm(suspension)
            residuals.append(abs(length - 0.30))
            swings.append(math.degrees(math.acos(float(np.clip(-suspension[2] / length, -1, 1)))))
            spins.append(float(np.degrees(np.linalg.norm(package["angular_velocity"]))))
            tilts.extend(
                (
                    abs(math.degrees(math.atan2(rd[2, 1], rd[2, 2]))),
                    abs(math.degrees(math.asin(float(np.clip(-rd[2, 0], -1, 1))))),
                )
            )
        return dict(
            min_clearance_m=distance,
            closest_pair=[left, right],
            floor_minima_m=floors,
            contacts=self._contacts(),
            wash_incursions=_point_wash_possible(positions,self._wash_cache),
            constraint_max_m=float(max(residuals)),
            max_roll_pitch_deg=max(tilts),
            max_swing_deg=max(swings),
            max_spin_deg_s=max(spins),
            finite=bool(
                np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()
            ),
            warning_counts=list(warning_counts),
        )

    def _point(self, row, ctrl, location, *, actual=False, geometry=False, stage=False):
        mj, model, data = self.mj, self.model, self.data
        mj.set_mjcb_control(None)
        mj.mj_resetData(model, data)
        data.qpos[:], data.qvel[:], data.ctrl[:] = row["qpos"], row["qvel"], ctrl
        data.time = row["world_time"]
        mj.mj_forward(model, data)
        if actual or geometry:
            for key in ("qacc", "qfrc_actuator"):
                self._compare(getattr(data, key), row[key], f"{location}/{key}")
            self._contact_compare(row["contacts"])
        if stage:
            for key in ("xmat", "xipos", "qfrc_passive"):
                self._compare(getattr(data, key), row[key], f"{location}/{key}")
        if geometry:
            for key in ("geom_xpos", "geom_xmat", "actuator_force"):
                self._compare(getattr(data, key), row[key], f"{location}/{key}")
        if actual:
            self._compare(data.qfrc_passive, row["qfrc_passive"], f"{location}/qfrc_passive")
            for raw, ids in zip(row["drones"], self.body_ids):
                for label, name in [
                    ("drone", "base_link"),
                    ("rod", "suspension_rod"),
                    ("package", "payload"),
                ]:
                    for field, value in self._body(ids[name]).items():
                        self._compare(value, raw[f"{label}_{field}"], f"{location}/{label}_{field}")
            expected_ctrl = np.zeros(8)
            expected_ctrl[np.asarray(self.actuator_ids)] = 3.16e-10 * np.asarray(row["rpm"]) ** 2
            self._compare(ctrl, expected_ctrl, f"{location}/rpm command map", tolerance=1e-12)
        warnings = np.maximum(
            row.get("warning_counts", [0] * len(data.warning)), [int(w.number) for w in data.warning]
        )
        self.ledger.observe(self._diagnostics(warnings), time=data.time, location=location)

    def _assembly_snapshot(self,data):
        return dict(assembly_xpos=data.xpos[self._assembly_body_ids].tolist(),
                    assembly_xmat=data.xmat[self._assembly_body_ids].reshape(2,2,3,3).tolist())

    def _reconstructed_body_state(self,row):
        self.kinematics_data.qpos[:]=_array(row['qpos'],(self.model.nq,))
        self.mj.mj_kinematics(self.model,self.kinematics_data)
        return self._assembly_snapshot(self.kinematics_data)

    def _sweep(self, start, end, location, *, live=False):
        last_positions = None
        lower_bounds = np.empty((self.model.ngeom, self.model.ngeom))

        def distance(positions, rotations, left, right):
            nonlocal last_positions
            if positions is not last_positions:
                self.sweep_data.geom_xpos[:] = positions
                self.sweep_data.geom_xmat[:] = rotations.reshape(-1, 9)
                a, b, radii = self._sweep_bounds_cache
                lower_bounds[a, b] = np.linalg.norm(positions[a] - positions[b], axis=1) - radii - 1e-12
                last_positions = positions
            if lower_bounds[left, right] >= self._sweep_margins[(left,right)]:
                return float(lower_bounds[left, right])
            self.exact_distance_evaluations += 1
            return float(self.mj.mj_geomDistance(self.model, self.sweep_data, left, right, 100.0, None))

        result = swept_reconstruction(
            start, end, shapes=self.shapes, pair_distance=distance, wash_cache=self._wash_cache,
            self_pairs=self.self_pairs,
        )
        body=body_interval_certificate(start if live else self._reconstructed_body_state(start),
            end if live else self._reconstructed_body_state(end),package_offsets=self._package_offsets)
        self.body_intervals+=1
        self.body_subintervals+=body['checked_subintervals']
        self.body_constraint_upper=max(self.body_constraint_upper,body['constraint_upper_bound_m'])
        if body['tilt_upper_bound_deg'] is not None:
            self.body_tilt_upper=max(self.body_tilt_upper,body['tilt_upper_bound_deg'])
        result["distance_mode"] = "conservative-sphere-bound-or-exact-native-near-threshold"
        self.sweeps += 1
        if result["status"] != "certified" or body['status']!='certified':
            self.sweep_failures.append(
                dict(location=location, start_time=start["world_time"], end_time=end["world_time"],
                     body_certificate=body, **result)
            )
        if result["lower_clearance_m"] is not None:
            self.sweep_lower = min(self.sweep_lower, result["lower_clearance_m"])
        if result['self_lower_clearance_m'] is not None:
            self.self_sweep_lower=min(self.self_sweep_lower,result['self_lower_clearance_m'])

    def observe(self, row):
        actual = row["actual"]
        if row["kind"] == "RESET":
            self.previous_ctrl = np.asarray(actual["ctrl"])
            self._point(actual, self.previous_ctrl, "reset/actual", actual=True)
            self._point(row["geometry"], self.previous_ctrl, "reset/geometry", geometry=True)
            return
        prefix, ctrl = f"step:{row['tick']}", np.asarray(actual["ctrl"])
        self._point(actual, ctrl, prefix + "/actual", actual=True)
        for index, state in enumerate(row["substeps"]):
            self._point(
                state, self.previous_ctrl if index == 0 else ctrl, f"{prefix}/substep:{index}", geometry=True
            )
        for index, state in enumerate(row["stages"]):
            self._point(state, ctrl, f"{prefix}/rk:{index}", stage=True)
        for index, (a, b) in enumerate(zip(row["substeps"], row["substeps"][1:])):
            self._sweep(a, b, f"{prefix}/interval:{index}")
        self.previous_ctrl = ctrl.copy()

    def report(self):
        return dict(
            **self.ledger.report(),
            reconstruction_contract="linear-centers-shortest-rotation-v1",
            reconstructed_path_safe=self.sweeps > 0 and self.body_intervals==self.sweeps and not self.sweep_failures,
            reconstructed_intervals=self.sweeps,
            reconstruction_failures=self.sweep_failures,
            reconstructed_clearance_lower_bound_m=self.sweep_lower
            if math.isfinite(self.sweep_lower)
            else None,
            maximum_replay_absolute_error=self.maximum_replay_error,
            asset_rebases=self.asset_rebases,
            source_model_bytes_preserved=True,
            native_step_reintegration_performed=False,
            exact_native_distance_evaluations=self.exact_distance_evaluations,
            self_collision_pairs=[list(pair) for pair in self.self_pairs],
            reconstructed_self_clearance_lower_bound_m=self.self_sweep_lower if math.isfinite(self.self_sweep_lower) else None,
            body_reconstructed_intervals=self.body_intervals,
            body_checked_subintervals=self.body_subintervals,
            reconstructed_constraint_upper_bound_m=self.body_constraint_upper,
            reconstructed_tilt_upper_bound_deg=self.body_tilt_upper,
            body_reconstruction_contract='linear-body-centers-shortest-rotation-attachment-chord-v1',
            limitations=list(self.limitations),
        )


class OnlineNativeAudit:
    """Compact, checkpointable online audit; no persistent raw trace is needed.

    Use either the live RK/native hooks or the transient-row replay adapter in a
    given instance. The live hook reads the actual native model/data, with no
    forward call inside MuJoCo's callback. It checks geometry before returning.
    ``safe`` means the checks performed so far passed, not full qualification.
    """

    def __init__(self, fixture):
        self._live_model = fixture.model
        self._auditor = NativePointAuditor(fixture.model_path, fixture.runtime())
        for name in ('geom_contype','geom_conaffinity','body_weldid','body_parentid','exclude_signature'):
            _same(getattr(fixture.model,name),getattr(self._auditor.model,name),
                  f'live collision-filter metadata {name}',atol=0.)
        if fixture.model.opt.disableflags!=self._auditor.model.opt.disableflags:
            raise ValueError('live/native collision-filter flags disagree')
        self.model_sha256 = _hash(fixture.model_path)
        self._auditor.observe(dict(kind="RESET", actual=fixture.sample(), geometry=fixture.geometry_state()))
        self._previous_native = self._geometry_snapshot(fixture.data)
        self._origin_time = float(fixture.data.time)
        self._mode = None
        self.rk_states = 0
        self.native_poses = 1
        self.control_steps = 0
        self.errors = []

    def _geometry_snapshot(self,data):
        return dict(
            world_time=float(data.time),
            qpos=data.qpos.tolist(),
            qvel=data.qvel.tolist(),
            geom_xpos=data.geom_xpos.tolist(),
            geom_xmat=data.geom_xmat.tolist(),
            **self._auditor._assembly_snapshot(data),
        )

    def observe_native_state(self, model, data, *, phase, location):
        """Call at RK control hook, and at reset/post-forward native boundaries."""
        try:
            if (
                self._mode not in (None, "live")
                or model is not self._live_model
                or phase not in ("rk", "native")
            ):
                raise ValueError("one live native model and one observation mode required")
            self._mode = "live"
            now = float(data.time)
            if phase == "rk":
                index = self.rk_states
                expected = self._origin_time + (index // 4 + (0, 0.5, 0.5, 1)[index % 4]) * 0.01 / 16
                if abs(now - expected) > 5e-9:
                    raise ValueError("online RK callback ordering mismatch")
                self.rk_states += 1
            else:
                # A control boundary is also the previous substep endpoint.
                if abs(now - self._previous_native["world_time"]) <= 1e-12:
                    _same(
                        data.qpos, self._previous_native["qpos"], "duplicate native boundary qpos", atol=0.0
                    )
                    _same(
                        data.qvel, self._previous_native["qvel"], "duplicate native boundary qvel", atol=0.0
                    )
                    return self.report()
                expected = self._origin_time + self.native_poses * 0.01 / 16
                if abs(now - expected) > 5e-9 or self.rk_states != 4 * self.native_poses:
                    raise ValueError("online native pose or RK-state count mismatch")
                current = self._geometry_snapshot(data)
                self._auditor._sweep(self._previous_native, current, str(location),live=True)
                self._previous_native = current
                self.native_poses += 1
                self.control_steps = (self.native_poses - 1) // 16
            if (
                not np.isfinite(data.ctrl).all()
                or np.any(data.ctrl < model.actuator_ctrlrange[:, 0])
                or np.any(data.ctrl > model.actuator_ctrlrange[:, 1])
            ):
                raise ValueError("online actuator command outside original authority")
            old_model, old_data = self._auditor.model, self._auditor.data
            try:
                self._auditor.model, self._auditor.data = model, data
                diagnostics = self._auditor._diagnostics([int(w.number) for w in data.warning])
                self._auditor.ledger.observe(diagnostics, time=now, location=str(location))
            finally:
                self._auditor.model, self._auditor.data = old_model, old_data
            return self.report()
        except (ValueError, RuntimeError, KeyError) as error:
            self.errors.append(dict(location=str(location), reason=str(error)))
            raise

    def observe_fixture_step(self, fixture, tick=None, *, stages=None, substeps=None):
        """Slower fallback: independently replay one transient captured interval."""
        try:
            if self._mode not in (None, "replay") or fixture.model is not self._live_model:
                raise ValueError("one fixture and one observation mode required")
            self._mode = "replay"
            tick = self.control_steps + 1 if tick is None else tick
            if tick != self.control_steps + 1:
                raise ValueError("online replay requires consecutive control steps")
            stages = fixture.stage_rows if stages is None else stages
            substeps = fixture.substep_rows if substeps is None else substeps
            if len(stages) != 64 or len(substeps) != 17:
                raise ValueError("online replay requires 64 RK states and 17 native poses")
            actual = fixture.sample()
            end = self._origin_time + tick * 0.01
            _state(actual, end, actual=True)
            for index, row in enumerate(substeps):
                _state(row, end - 0.01 + index * 0.01 / 16, geometry=True)
            for index, row in enumerate(stages):
                _state(row, end - 0.01 + (index // 4 + (0, 0.5, 0.5, 1)[index % 4]) * 0.01 / 16, stage=True)
                _same(row["ctrl"], actual["ctrl"], "online constant actuator command", atol=0.0)
            for name in ("qpos", "qvel"):
                _same(substeps[0][name], self._previous_native[name], f"online boundary {name}", atol=0.0)
            self._auditor.observe(
                dict(kind="STEP", tick=tick, actual=actual, stages=stages, substeps=substeps)
            )
            self._previous_native = self._geometry_snapshot(fixture.data)
            self.control_steps += 1
            self.rk_states += 64
            self.native_poses += 16
            return self.report()
        except (ValueError, RuntimeError, KeyError) as error:
            self.errors.append(dict(location=f"control-step:{tick}", reason=str(error)))
            raise

    def __call__(self, *, fixture, stages=None, substeps=None):
        return self.observe_fixture_step(fixture, stages=stages, substeps=substeps)

    def report(self):
        result = self._auditor.report()
        scoped_safe = bool(
            result["retained_points_safe"] and result["reconstructed_path_safe"] and not self.errors
        )
        result.update(
            safe=bool(
                result["retained_points_safe"] and not self._auditor.sweep_failures and not self.errors
            ),
            continuous_safety_qualified=scoped_safe,
            full_physical_acceptance_qualified=False,
            qualification_scope="retained-points-and-cross-self-floor-wash-attitude-constraint-reconstruction",
            proof_gaps=list(self._auditor.proof_gaps),
            errors=deepcopy(self.errors),
            rk_states=self.rk_states,
            native_poses=self.native_poses,
            control_steps=self.control_steps,
            evidence_mode="online-" + str(self._mode),
            permanent_raw_required=False,
        )
        return result

    def to_dict(self):
        auditor = self._auditor
        return deepcopy(
            dict(
                schema="dual-online-native-audit-v2",
                model_sha256=self.model_sha256,
                origin_time=self._origin_time,
                mode=self._mode,
                previous_native=self._previous_native,
                rk_states=self.rk_states,
                native_poses=self.native_poses,
                control_steps=self.control_steps,
                errors=self.errors,
                ledger=asdict(auditor.ledger),
                previous_ctrl=auditor.previous_ctrl.tolist(),
                sweeps=auditor.sweeps,
                sweep_failures=auditor.sweep_failures,
                sweep_lower=None if not math.isfinite(auditor.sweep_lower) else auditor.sweep_lower,
                self_sweep_lower=None if not math.isfinite(auditor.self_sweep_lower) else auditor.self_sweep_lower,
                body_intervals=auditor.body_intervals,body_subintervals=auditor.body_subintervals,
                body_constraint_upper=auditor.body_constraint_upper,body_tilt_upper=auditor.body_tilt_upper,
                maximum_replay_error=auditor.maximum_replay_error,
                exact_distance_evaluations=auditor.exact_distance_evaluations,
            )
        )

    def restore(self, snapshot):
        """Restore audit continuity only; caller restores matching physics state."""
        state = deepcopy(snapshot)
        json.dumps(state, allow_nan=False)
        if state["schema"] != "dual-online-native-audit-v2" or state["model_sha256"] != self.model_sha256:
            raise ValueError("audit checkpoint model/schema mismatch")
        if state["mode"] not in (None, "live", "replay"):
            raise ValueError("invalid online audit mode")
        for name in ("rk_states", "native_poses", "control_steps", "sweeps",'body_intervals','body_subintervals'):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("invalid audit counters")
        if state["native_poses"] != state["sweeps"] + 1 or state["rk_states"] < 4 * state["sweeps"]:
            raise ValueError("inconsistent audit checkpoint counters")
        if state['body_intervals']!=state['sweeps']:
            raise ValueError('body interval audit coverage missing from checkpoint')
        for name in ("qpos", "qvel", "geom_xpos", "geom_xmat",'assembly_xpos','assembly_xmat'):
            _same(
                state["previous_native"][name],
                self._previous_native[name],
                f"checkpoint current physical {name}",
                atol=1e-10,
            )
        self._origin_time, self._mode = state["origin_time"], state["mode"]
        self._previous_native = state["previous_native"]
        self.rk_states, self.native_poses, self.control_steps = (
            state["rk_states"],
            state["native_poses"],
            state["control_steps"],
        )
        self.errors = state["errors"]
        auditor = self._auditor
        auditor.ledger = PointAuditLedger(**state["ledger"])
        auditor.previous_ctrl = _array(state["previous_ctrl"], (8,))
        auditor.sweeps, auditor.sweep_failures = state["sweeps"], state["sweep_failures"]
        auditor.sweep_lower = math.inf if state["sweep_lower"] is None else state["sweep_lower"]
        auditor.self_sweep_lower=math.inf if state['self_sweep_lower'] is None else state['self_sweep_lower']
        for name in ('body_intervals','body_subintervals','body_constraint_upper','body_tilt_upper'):
            setattr(auditor,name,state[name])
        auditor.maximum_replay_error = state["maximum_replay_error"]
        auditor.exact_distance_evaluations = state["exact_distance_evaluations"]


def audit_probe(root, *, expected_hashes=None, native=False, asset_root=None):
    """Audit recovered capture, retaining failures instead of trusting its verdict."""
    root = Path(root)
    properties = json.loads((root / "properties.json").read_text())
    capture = json.loads((root / "result.json").read_text())
    paths = ("native-control.jsonl.gz", "properties.json", "fixture/model.xml")
    hashes = {path: _hash(root / path) for path in paths}
    failures, gaps = [], []
    expected_hashes = expected_hashes or {}
    bound = all(path in expected_hashes for path in paths)
    if not bound:
        gaps.append("independent raw/properties/model source hash inventory not supplied in full")
    for path in paths:
        if path in expected_hashes and hashes[path] != expected_hashes[path]:
            bound = False
            failures.append(dict(location="source", reason=f"{path} independent SHA-256 mismatch"))
    for path, expected in [
        ("native-control.jsonl.gz", capture["raw_sha256"]),
        ("fixture/model.xml", properties["model_xml_sha256"]),
    ]:
        if hashes[path] != expected:
            bound = False
            failures.append(dict(location="source", reason=f"{path} producer SHA-256 mismatch"))
    for name, value in [
        ("schema", "dual-native-properties-v1"),
        ("profile", "dual-payload-v1"),
        ("version", "3.15.0"),
        ("precision", "float64"),
        ("integrator", "RK4"),
        ("substeps", 16),
    ]:
        if properties.get(name) != value:
            failures.append(dict(location="properties", reason=f"unexpected native property {name}"))
    verifier = StructureVerifier(capture["completed_control_steps"])
    auditor = None
    if native and not failures:
        try:
            auditor = NativePointAuditor(root / "fixture/model.xml", properties, asset_root=asset_root)
        except (ImportError, ValueError, RuntimeError, OSError) as error:
            failures.append(dict(location="native-model", reason=str(error)))
    try:
        for row in _records(root / "native-control.jsonl.gz"):
            valid = verifier.observe(row)
            if valid and auditor is not None:
                try:
                    auditor.observe(row)
                except (ValueError, RuntimeError, KeyError) as error:
                    failures.append(dict(location=f"native-row:{verifier.records}", reason=str(error)))
    except (ValueError, OSError, EOFError) as error:
        failures.append(dict(location="trace-read", reason=str(error)))
    report = verifier.report()
    failures.extend(report["failures"])
    completed = auditor is not None and not failures
    if auditor is None:
        gaps.append("native retained-state replay and swept reconstruction not performed")
    if auditor is not None:
        report.update(auditor.report())
        gaps.extend(auditor.proof_gaps)
    report.update(
        source_binding_verified=bound,
        source_hashes=hashes,
        failures=failures,
        structural_valid=report["structural_valid"]
        and not any(f["location"] == "trace-read" for f in failures),
        native_replay_completed=completed,
        proof_gaps=gaps,
        retained_and_reconstructed_geometry_safe=bool(
            completed
            and bound
            and report.get("reconstructed_path_safe", False)
            and report.get("retained_points_safe", False)
        ),
        continuous_safety_qualified=bool(
            completed
            and bound
            and report.get("reconstructed_path_safe", False)
            and report.get("retained_points_safe", False)
        ),
        continuous_safety_scope="retained-points-and-cross-self-floor-wash-attitude-constraint-reconstruction",
        full_physical_acceptance_qualified=False,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-root")
    parser.add_argument(
        "--expected-hashes", help="JSON mapping relative captured filenames to independent SHA-256"
    )
    parser.add_argument("--native", action="store_true")
    options = parser.parse_args()
    hashes = (
        None if options.expected_hashes is None else json.loads(Path(options.expected_hashes).read_text())
    )
    result = audit_probe(
        options.root, expected_hashes=hashes, native=options.native, asset_root=options.asset_root
    )
    Path(options.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "structural_valid",
                    "control_steps",
                    "native_replay_completed",
                    "continuous_safety_qualified",
                    "retained_points_safe",
                    "reconstructed_intervals",
                    "violation_count",
                )
            }
        )
    )
    raise SystemExit(
        1
        if result["failures"] or (options.native and not result["retained_and_reconstructed_geometry_safe"])
        else 0
    )
