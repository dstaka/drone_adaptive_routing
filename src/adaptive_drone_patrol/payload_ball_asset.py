"""Experimental passive two-ball MJCF asset; no operational admission implied."""

from pathlib import Path
import xml.etree.ElementTree as ET

from .payload_contract import LINK_M
from .util import sha256

ROD_MASS_KG = 0.0001
ROD_RADIUS_M = 0.001


def _values(values):
    return " ".join(str(float(v)) if float(v) else "0" for v in values)


def write_ball_mjcf(original, output, payload, *, fixed=False, loaded=True):
    """Retain cf2x authored inertials/frames and replace only the suspension.

    Zero-mass fixed rotor inertials are intentionally left as authored. Runtime
    epsilon regularization MUST be audited against the original Drone importer.
    Assets are private generated outputs, not relocatable public evidence.
    """
    original, output = Path(original), Path(output)
    if output.exists():
        raise FileExistsError(output)
    urdf = ET.parse(original).getroot()
    base = urdf.find("link[@name='base_link']")
    if base is None:
        raise ValueError("Expected pinned cf2x base_link")
    root = ET.Element("mujoco", model="cf2x_passive_ball_experimental")
    ET.SubElement(root, "compiler", angle="radian", fusestatic="false", inertiafromgeom="false")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="base_link")
    if not fixed:
        ET.SubElement(body, "freejoint", name="vehicle_free")

    def copy_link(source, destination):
        inertial = source.find("inertial")
        inertia = inertial.find("inertia")
        if any(float(inertia.get(k, "0")) != 0 for k in ("ixy", "ixz", "iyz")):
            raise ValueError("cf2x audit expects diagonal original inertia")
        pos = inertial.find("origin").get("xyz", "0 0 0")
        ET.SubElement(
            destination,
            "inertial",
            pos=pos,
            mass=inertial.find("mass").get("value"),
            diaginertia=_values(float(inertia.get(k)) for k in ("ixx", "iyy", "izz")),
        )
        for visual in source.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Expected pinned mesh visualization")
            name = source.get("name") + "_mesh"
            mesh_path = original.parent / mesh.get("filename")
            ET.SubElement(
                assets, "mesh", name=name, file=str(mesh_path.resolve()), scale=mesh.get("scale", "1 1 1")
            )
            ET.SubElement(
                destination,
                "geom",
                type="mesh",
                mesh=name,
                pos=visual.find("origin").get("xyz", "0 0 0"),
                contype="0",
                conaffinity="0",
                group="2",
                density="0",
            )

    copy_link(base, body)
    cylinder = base.find("collision/geometry/cylinder")
    ET.SubElement(
        body,
        "geom",
        name="vehicle_collision",
        type="cylinder",
        size=_values([float(cylinder.get("radius")), float(cylinder.get("length")) / 2]),
        pos=base.find("collision/origin").get("xyz"),
        density="0",
        group="0",
    )
    for i in range(4):
        name = f"prop{i}_link"
        source = urdf.find(f"link[@name='{name}']")
        joint = urdf.find(f"joint[@name='prop{i}_joint']")
        if source is None or joint is None or joint.get("type") != "fixed":
            raise ValueError("Original rotor identity mismatch")
        origin = joint.find("origin")
        if origin is not None and origin.get("rpy", "0 0 0") != "0 0 0":
            raise ValueError("Unexpected rotor rotation")
        rotor = ET.SubElement(
            body, "body", name=name, pos=origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        )
        copy_link(source, rotor)
    if loaded:
        rod = ET.SubElement(body, "body", name="suspension_rod", pos="0 0 -0.02")
        ET.SubElement(
            rod,
            "joint",
            name="suspension_ball",
            type="ball",
            pos="0 0 0",
            damping="0",
            armature="0",
            frictionloss="0",
        )
        transverse = ROD_MASS_KG * (3 * ROD_RADIUS_M**2 + LINK_M**2) / 12
        axial = ROD_MASS_KG * ROD_RADIUS_M**2 / 2
        ET.SubElement(
            rod,
            "inertial",
            pos=_values([0, 0, -LINK_M / 2]),
            mass=str(ROD_MASS_KG),
            diaginertia=_values([transverse, transverse, axial]),
        )
        # The rod has visual geometry only, matching the original approximation.
        ET.SubElement(
            rod,
            "geom",
            type="cylinder",
            size=_values([ROD_RADIUS_M, LINK_M / 2]),
            pos=_values([0, 0, -LINK_M / 2]),
            contype="0",
            conaffinity="0",
            group="2",
            density="0",
            rgba="1 .6 .1 1",
        )
        package = ET.SubElement(
            rod, "body", name="payload", pos=_values([0, 0, -LINK_M - payload.side_m / 2])
        )
        ET.SubElement(
            package,
            "joint",
            name="payload_ball",
            type="ball",
            pos=_values([0, 0, payload.side_m / 2]),
            damping="0",
            armature="0",
            frictionloss="0",
        )
        ET.SubElement(
            package, "inertial", pos="0 0 0", mass=str(payload.mass), diaginertia=_values(payload.inertia)
        )
        ET.SubElement(
            package,
            "geom",
            name="payload_collision",
            type="box",
            size=_values([payload.side_m / 2] * 3),
            density="0",
            rgba=".95 .45 .08 1",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return {
        "schema": "payload-ball-asset-v2",
        "status": "EXPERIMENTAL_DIAGNOSTICS_ONLY",
        "original_asset_sha256": sha256(original),
        "generated_asset_sha256": sha256(output),
        "suspension_length_m": LINK_M,
        "rod_mass_kg": ROD_MASS_KG,
        "gimbal_count": 0,
        "joint_damping": 0.0,
        "armature": 0.0,
        "loaded": loaded,
        "fixed": fixed,
        "approximation": "compression-capable rigid link; not tension-only rope",
        "reference_mass_kg": payload.reference_mass_kg,
        "mass_ratio": payload.mass_ratio,
        "side_m": payload.side_m,
        "motor_api": "EXPERIMENTAL: solver.set_drone_rpm on original four link COM frames; equivalence unverified",
    }
