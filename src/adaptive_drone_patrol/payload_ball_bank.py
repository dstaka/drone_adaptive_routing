"""Shared native-ball diagnostics backend; construction never grants admission.

Genesis initialization belongs to the caller. Geometry scenes are isolated and
only the active episode scene is stepped. No controller or optimizer is loaded.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .payload_ball_asset import ROD_MASS_KG, write_ball_mjcf
from .payload_contract import DRONE_ATTACHMENT, LINK_M, SIZES_M, Payload, rotation
from .util import sha256


def _vector(value, size=3):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError("Expected finite physical state vector")
    return value


def _quaternion(matrix):
    """Stable matrix-to-wxyz conversion, including rotations near pi."""
    matrix = np.asarray(matrix, dtype=float)
    trace = np.trace(matrix)
    if trace > 0:
        scale = 2 * math.sqrt(trace + 1)
        q = np.array(
            [
                scale / 4,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        i = int(np.argmax(np.diag(matrix)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = 2 * math.sqrt(max(0.0, 1 + matrix[i, i] - matrix[j, j] - matrix[k, k]))
        q = np.zeros(4)
        q[0] = (matrix[k, j] - matrix[j, k]) / scale
        q[i + 1] = scale / 4
        q[j + 1] = (matrix[j, i] + matrix[i, j]) / scale
        q[k + 1] = (matrix[k, i] + matrix[i, k]) / scale
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def ball_joint_configuration(initial, side_m):
    """Return (upper_wxyz, lower_wxyz) from actual attachment-consistent poses.

    The rod world orientation is the minimum rotation from world down to the
    physical attachment direction. Native ball coordinates are relative body
    rotations, with no Euler angles or singular-coordinate decomposition.
    Only a zero-velocity episode reset is supported by this interface.
    """
    if side_m not in SIZES_M:
        raise ValueError("Unsupported cube collision size")
    drone = _vector(initial["drone_xyz"])
    package = _vector(initial["package_xyz"])
    rd = rotation(initial["drone_quaternion_wxyz"])
    rp = rotation(initial["package_quaternion_wxyz"])
    for body in ("drone", "package"):
        for field in ("velocity", "angular_velocity"):
            if np.any(_vector(initial.get(f"{body}_{field}", [0.0, 0.0, 0.0])) != 0):
                raise ValueError("Native-ball reset requires zero initial velocities")
    anchor = drone + rd @ DRONE_ATTACHMENT
    end = package + rp @ np.array([0.0, 0.0, side_m / 2])
    delta = end - anchor
    length = float(np.linalg.norm(delta))
    if not math.isclose(length, LINK_M, rel_tol=0, abs_tol=1e-7):
        raise ValueError("Initial package center violates attachment length")
    direction = delta / length
    down = np.array([0.0, 0.0, -1.0])
    cosine = float(np.clip(np.dot(down, direction), -1, 1))
    cross = np.cross(down, direction)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-15:
        rod_quaternion = np.array([0.0, 1.0, 0.0, 0.0]) if cosine < 0 else np.array([1.0, 0.0, 0.0, 0.0])
    else:
        half_angle = math.atan2(sine, cosine) / 2
        rod_quaternion = np.r_[math.cos(half_angle), cross / sine * math.sin(half_angle)]
    rod = rotation(rod_quaternion)
    return _quaternion(rd.T @ rod).tolist(), _quaternion(rod.T @ rp).tolist()


def _options(dt, substeps, fixture, align_root=False):
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (float, int))
        or not math.isfinite(dt)
        or dt <= 0
        or type(substeps) is not int
        or substeps <= 0
        or fixture not in ("original", "loaded", "unloaded", "anchor")
        or type(align_root) is not bool
    ):
        raise ValueError("Invalid declared native-ball construction options")


def _damping(entity, authored_mass, *, fixed=False):
    """Retain only the Drone importer's authored base angular damping."""
    import genesis as gs
    import torch

    damping = torch.zeros(entity.n_dofs, device=gs.device, dtype=gs.tc_float)
    if not fixed:
        roots = [j for j in entity.joints if j.n_qs == 7 and j.n_dofs == 6]
        if len(roots) != 1:
            raise ValueError("Expected one original free vehicle root")
        start = roots[0].dof_start - entity.dof_start
        damping[start + 3 : start + 6] = authored_mass * 1e-5
    entity.set_dofs_damping(damping)


def build_ball_fixture(output_dir, payload, *, fixture="loaded", dt=0.01, substeps=2, align_root=False):
    """Build the same pinned scene for both probes and episode geometry banks.

    Returns scene/entity/ground/asset_manifest/damping_before/original_mass_kg.
    This assumes the caller already initialized Genesis. It performs no steps.
    """
    _options(dt, substeps, fixture, align_root)
    if not isinstance(payload, Payload):
        raise ValueError("A validated payload draw is required")
    import genesis as gs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = Path(gs.__file__).parent / "assets/urdf/drones/cf2x.urdf"
    authored_reference = sum(float(x.get("value")) for x in ET.parse(original).findall(".//inertial/mass"))
    if fixture == "original":
        manifest = dict(
            schema="payload-ball-original-asset-v2",
            status="EXPERIMENTAL_DIAGNOSTICS_ONLY",
            original_asset_sha256=sha256(original),
            loaded=False,
            fixed=False,
        )
    else:
        manifest = write_ball_mjcf(
            original,
            output_dir / "candidate.xml",
            payload,
            fixed=fixture == "anchor",
            loaded=fixture in ("loaded", "anchor"),
        )
    manifest["align_root"] = None if fixture == "original" else align_root
    manifest["root_alignment_policy"] = (
        "pinned Drone default" if fixture == "original" else "explicit MJCF align flag"
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt),
        rigid_options=gs.options.RigidOptions(
            dt=dt / substeps,
            enable_collision=True,
            enable_joint_limit=True,
            batch_links_info=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        show_viewer=False,
    )
    ground = scene.add_entity(gs.morphs.Plane())
    if fixture == "original":
        morph = gs.morphs.Drone(file=str(original), pos=(0, 0, 1), default_armature=0.0, collision=True)
    else:
        morph = gs.morphs.MJCF(
            file=str((output_dir / "candidate.xml").resolve()),
            pos=(0, 0, 1),
            default_armature=0.0,
            align=align_root,
            collision=True,
        )
    entity = scene.add_entity(morph)
    scene.build(n_envs=1)
    masses = entity.get_links_mass()[0].detach().cpu().tolist()
    names = ("base_link", "prop0_link", "prop1_link", "prop2_link", "prop3_link")
    original_mass = sum(masses[entity.get_link(name).idx - entity.link_start] for name in names)
    before = entity.get_dofs_damping().detach().cpu().tolist()
    total = authored_reference + (payload.mass + ROD_MASS_KG if fixture == "loaded" else 0.0)
    _damping(entity, total, fixed=fixture == "anchor")
    return dict(
        scene=scene,
        entity=entity,
        ground=ground,
        asset_manifest=manifest,
        damping_before=before,
        original_mass_kg=original_mass,
        authored_reference_mass_kg=authored_reference,
        align_root=manifest["align_root"],
    )


class BallSceneBank:
    """Three physical scenes; only the selected size advances or receives motors."""

    def __init__(self, output, dt=0.01, substeps=2, align_root=False):
        _options(dt, substeps, "loaded", align_root)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.dt, self.substeps, self.align_root = dt, substeps, align_root
        self.banks, self.fixtures = {}, {}
        measured = None
        for side in SIZES_M:
            draw = Payload(0.02700047753751278, 0.2, side)
            fixture = build_ball_fixture(
                self.output / f"load_{side:.3f}",
                draw,
                fixture="loaded",
                dt=dt,
                substeps=substeps,
                align_root=align_root,
            )
            mass = fixture["original_mass_kg"]
            if (
                not math.isfinite(mass)
                or abs(mass - 0.02700047753751278) > 1e-8
                or (measured is not None and abs(mass - measured) > 1e-12)
            ):
                raise ValueError("Original free vehicle mass differs; physical audit required")
            measured = mass
            scene, entity, ground = fixture["scene"], fixture["entity"], fixture["ground"]
            self.banks[side] = (scene, entity, entity.get_link("payload"), ground)
            self.fixtures[side] = fixture
        self.reference_mass = measured
        self.active = None

    def reset(self, payload, initial):
        """Apply one physical draw and native coordinates without a solver step."""
        import torch

        if payload.reference_mass_kg != self.reference_mass:
            raise ValueError("Payload reference differs from actual original free mass")
        upper, lower = ball_joint_configuration(initial, payload.side_m)
        self.active = self.banks[payload.side_m]
        self.scene, self.drone, self.package, self.ground = self.active
        self.payload = payload
        self.package.set_mass(payload.mass)
        self.package.set_inertia(np.eye(3) * payload.inertia[0])
        # Public authored-frame setters include any principal-frame alignment.
        self.drone.set_pos(initial["drone_xyz"], zero_velocity=True, relative=True)
        self.drone.set_quat(initial["drone_quaternion_wxyz"], zero_velocity=True, relative=True)
        q = self.drone.get_qpos().clone()
        for name, values in [("suspension_ball", upper), ("payload_ball", lower)]:
            joint = self.drone.get_joint(name)
            start = joint.q_start - self.drone.q_start
            q[:, start : start + 4] = torch.as_tensor(values, device=q.device, dtype=q.dtype)
        self.drone.set_qpos(q, zero_velocity=True)
        self.drone.zero_all_dofs_velocity()
        authored = self.fixtures[payload.side_m]["authored_reference_mass_kg"]
        _damping(self.drone, authored + payload.mass + ROD_MASS_KG)
        return self.sample()

    def sample(self):
        if self.active is None:
            raise ValueError("Select a physical payload episode before sampling")
        sample = {}
        for label, body in [("drone", self.drone), ("package", self.package)]:
            for field, method in [
                ("xyz", "get_pos"),
                ("quaternion_wxyz", "get_quat"),
                ("velocity", "get_vel"),
                ("angular_velocity", "get_ang"),
            ]:
                sample[f"{label}_{field}"] = getattr(body, method)()[0].detach().cpu().tolist()
        rd, rp = rotation(sample["drone_quaternion_wxyz"]), rotation(sample["package_quaternion_wxyz"])
        for label, offset, body in [
            ("a", rd @ DRONE_ATTACHMENT, "drone"),
            ("b", rp @ np.array([0, 0, self.payload.side_m / 2]), "package"),
        ]:
            sample[f"attachment_{label}_world"] = (np.asarray(sample[f"{body}_xyz"]) + offset).tolist()
            sample[f"attachment_{label}_velocity"] = (
                np.asarray(sample[f"{body}_velocity"]) + np.cross(sample[f"{body}_angular_velocity"], offset)
            ).tolist()
        return sample

    def step(self, actions):
        """Same four rotor COM wrench kernel; never act directly on the load."""
        actions = _vector(actions, 4)
        if self.active is None:
            raise ValueError("Select a physical payload episode before stepping")
        import genesis as gs
        import torch

        props = [self.drone.get_link(f"prop{i}_link") for i in range(4)]
        indices = torch.tensor([p.idx for p in props], dtype=gs.tc_int, device=gs.device)
        spins = torch.tensor([-1, 1, -1, 1], dtype=gs.tc_float, device=gs.device)
        rpm = torch.as_tensor(
            (1 + 0.8 * np.clip(actions, -1, 1)) * 14468.429183500699, dtype=gs.tc_float, device=gs.device
        ).reshape(1, 4)
        self.drone.solver.set_drone_rpm(indices, 3.16e-10, 7.94e-12, rpm, spins, False)
        self.scene.step()
        return self.sample()
