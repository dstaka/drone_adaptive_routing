"""Two independently loaded physical assemblies, one native world and clock.

Site motors express the original COM thrust/yaw wrench, evaluated by MuJoCo at
all RK4 stages. This adapter is DEVELOPMENT ONLY until new joint qualification.
No singleton qpos/qvel/body indices or previous qualification are inherited.
"""

from __future__ import annotations
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .payload_ball_asset import write_ball_mjcf, ROD_MASS_KG
from .payload_contract import Payload, rotation
from .payload_ball_bank import _quaternion
from .payload_solver_mujoco import BASE_MASS_KG, ROTOR_MASS_KG, FREE_MASS_KG, KF, KM, HOVER_RPM, SPIN
from .util import sha256


def _vec(values):
    return " ".join(str(float(x)) for x in values)


def _mesh_bounds(original, mesh):
    vertices = []
    for line in (Path(original).parent / mesh.get("file")).read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
    a = np.asarray(vertices) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
    return a.min(axis=0), a.max(axis=0)


class DualMuJoCoFixture:
    dt = 0.01
    substeps = 16

    def __init__(self, original, output, payloads):
        import mujoco as mj

        if mj.__version__ != "3.15.0":
            raise ValueError("Native reference requires MuJoCo 3.15.0")
        if len(payloads) != 2 or any(
            not isinstance(p, Payload) or p.reference_mass_kg != FREE_MASS_KG for p in payloads
        ):
            raise ValueError("Two original-reference independent payloads required")
        self.mj, self.payloads = mj, tuple(payloads)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        original = Path(original).resolve()
        self.original_sha256 = sha256(original)
        root = ET.Element("mujoco", model="dual-payload-v1")
        ET.SubElement(root, "compiler", angle="radian", fusestatic="false", inertiafromgeom="false")
        option = ET.SubElement(
            root,
            "option",
            timestep=str(self.dt / self.substeps),
            gravity="0 0 -9.81",
            integrator="RK4",
            solver="Newton",
            tolerance="1e-12",
            iterations="100",
        )
        ET.SubElement(option, "flag", energy="enable")
        assets = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 .1", rgba=".16 .18 .20 1")
        actuator = ET.SubElement(root, "actuator")
        contacts = ET.SubElement(root, "contact")
        self.self_exclusions = []
        self.proxy_coverage = []
        for d, payload in enumerate(payloads):
            prefix = f"drone-{d + 1}/"
            authored = self.output / f"authored-{d + 1}.xml"
            write_ball_mjcf(original, authored, payload)
            aroot = ET.parse(authored).getroot()
            body = aroot.find("worldbody/body")
            body.set("pos", "-3.5 -3.5 1" if d == 0 else "3.5 3.5 1")
            body.find("inertial").set("mass", str(BASE_MASS_KG))
            meshes = {m.get("name"): m for m in aroot.find("asset")}
            # All visual geoms have a primitive cover preserving all inertials.
            # Bounding boxes of body mesh plus swept disks for rotor mesh vertices.
            counter = 0
            for b in body.iter("body"):
                for geom in list(b.findall("geom")):
                    if not geom.get("name"):
                        geom.set("name", f"visual-{counter}")
                        counter += 1
                    if geom.get("type") == "mesh":
                        mesh = meshes[geom.get("mesh")]
                        lo, hi = _mesh_bounds(original, mesh)
                        pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
                        center = (lo + hi) / 2 + pos
                        size = (hi - lo) / 2
                        if b.get("name", "").startswith("prop"):
                            # Sweep around visual rotor center; conservative for mesh bbox.
                            radius = float(np.linalg.norm(np.maximum(np.abs(lo[:2]), np.abs(hi[:2]))))
                            kind = "cylinder"
                            proxy_size = [max(radius, 0.0231348), max(size[2], 0.001)]
                        else:
                            kind = "box"
                            proxy_size = np.maximum(size, 1e-5).tolist()
                        ET.SubElement(
                            b,
                            "geom",
                            name=geom.get("name") + "-proxy",
                            type=kind,
                            size=_vec(proxy_size),
                            pos=_vec(center),
                            density="0",
                            group="3",
                            rgba=".2 .5 .9 .12",
                        )
                        self.proxy_coverage.append(
                            dict(
                                drone_id=d + 1,
                                visual=prefix + geom.get("name"),
                                proxy=prefix + geom.get("name") + "-proxy",
                                mesh_lo=lo.tolist(),
                                mesh_hi=hi.tolist(),
                            )
                        )
                    elif b.get("name") == "suspension_rod":
                        geom.set("contype", "1")
                        geom.set("conaffinity", "1")
                        geom.set("name", "link-collision")
            for i in range(4):
                rotor = body.find(f"body[@name='prop{i}_link']")
                inertial = rotor.find("inertial")
                inertial.set("mass", str(ROTOR_MASS_KG))
                ET.SubElement(
                    rotor, "site", name=f"motor{i}-com", pos=inertial.get("pos", "0 0 0"), size=".001"
                )
                ET.SubElement(
                    actuator,
                    "motor",
                    name=prefix + f"motor{i}",
                    site=prefix + f"motor{i}-com",
                    gear=_vec([0, 0, 1, 0, 0, KM / KF * SPIN[i]]),
                    ctrllimited="true",
                    ctrlrange=_vec([0, KF * (1.8 * HOVER_RPM) ** 2]),
                )
            for a, b in [
                ("base_link", "suspension_rod"),
                ("suspension_rod", "payload"),
                *[("base_link", f"prop{i}_link") for i in range(4)],
            ]:
                self.self_exclusions.append((prefix + a, prefix + b))
                ET.SubElement(
                    contacts, "exclude", name=prefix + a + "-" + b, body1=prefix + a, body2=prefix + b
                )
            for node in aroot.iter():
                for key in ("name", "mesh", "material"):
                    if key in node.attrib:
                        node.set(key, prefix + node.get(key))
            for mesh in aroot.find("asset"):
                assets.append(copy.deepcopy(mesh))
            world.append(copy.deepcopy(body))
        self.model_path = self.output / "model.xml"
        ET.ElementTree(root).write(self.model_path, encoding="utf-8", xml_declaration=True)
        mj.set_mjcb_control(None)
        self.model = mj.MjModel.from_xml_path(str(self.model_path))
        self.data = mj.MjData(self.model)
        self.body_ids = []
        self.addresses = []
        aids = []
        self.audit_geom_ids = []
        self.geom_owner = {i: None for i in range(self.model.ngeom)}
        for d in range(2):
            prefix = f"drone-{d + 1}/"
            self.body_ids.append(
                {
                    name: self._id(mj.mjtObj.mjOBJ_BODY, prefix + name)
                    for name in [
                        "base_link",
                        "suspension_rod",
                        "payload",
                        *[f"prop{i}_link" for i in range(4)],
                    ]
                }
            )
            joints = {
                name: self._id(mj.mjtObj.mjOBJ_JOINT, prefix + name)
                for name in ["vehicle_free", "suspension_ball", "payload_ball"]
            }
            address = dict(
                free_qpos=int(self.model.jnt_qposadr[joints["vehicle_free"]]),
                free_dof=int(self.model.jnt_dofadr[joints["vehicle_free"]]),
                link_qpos=int(self.model.jnt_qposadr[joints["suspension_ball"]]),
                package_qpos=int(self.model.jnt_qposadr[joints["payload_ball"]]),
            )
            self.addresses.append(address)
            self.model.dof_damping[address["free_dof"] + 3 : address["free_dof"] + 6] = (
                0.027 + ROD_MASS_KG + payloads[d].mass
            ) * 1e-5
            aids.append([self._id(mj.mjtObj.mjOBJ_ACTUATOR, prefix + f"motor{i}") for i in range(4)])
            geom_ids = []
            for g in range(self.model.ngeom):
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, g)
                if name.startswith(prefix):
                    self.geom_owner[g] = d
                    if self.model.geom_contype[g] or self.model.geom_conaffinity[g]:
                        geom_ids.append(g)
            self.audit_geom_ids.append(geom_ids)
        self.actuator_ids = np.array(aids, dtype=int)
        self.stage_rows = []
        self.substep_rows = []
        self.reset()

    def _id(self, kind, name):
        result = self.mj.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Missing namespaced model metadata: {name}")
        return result

    def reset(self, initials=None):
        if initials is None:
            initials = [
                dict(position=p, quaternion_wxyz=[1, 0, 0, 0], swing_deg=0, direction_rad=0)
                for p in [[-3.5, -3.5, 1], [3.5, 3.5, 1]]
            ]
        if len(initials) != 2:
            raise ValueError("Two explicit initial states required")
        self.mj.set_mjcb_control(None)
        self.mj.mj_resetData(self.model, self.data)
        for d, initial in enumerate(initials):
            q = np.asarray(initial["quaternion_wxyz"], float)
            rd = rotation(q)
            position = np.asarray(initial["position"], float)
            angle = float(initial["swing_deg"])
            direction = float(initial["direction_rad"])
            if (
                position.shape != (3,)
                or not np.isfinite(position).all()
                or not 0 <= angle <= 10
                or not math.isfinite(direction)
            ):
                raise ValueError("Finite declared initial state required")
            half = math.radians(angle) / 2
            qr = np.array(
                [
                    math.cos(half),
                    math.sin(direction) * math.sin(half),
                    -math.cos(direction) * math.sin(half),
                    0,
                ]
            )
            rr = rotation(qr)
            a = self.addresses[d]
            self.data.qpos[a["free_qpos"] : a["free_qpos"] + 3] = position
            self.data.qpos[a["free_qpos"] + 3 : a["free_qpos"] + 7] = q
            self.data.qpos[a["link_qpos"] : a["link_qpos"] + 4] = _quaternion(rd.T @ rr)
            # Original attachment-consistent package is initially world-upright.
            self.data.qpos[a["package_qpos"] : a["package_qpos"] + 4] = _quaternion(rr.T)
        self.rpm = np.zeros((2, 4))
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.stage_rows = []
        self.substep_rows = []
        self.mj.mj_forward(self.model, self.data)
        return self.sample()

    def _body(self, idx):
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        self.mj.mj_jacBody(self.model, self.data, jp, jr, idx)
        return dict(
            xyz=self.data.xpos[idx].tolist(),
            quaternion_wxyz=self.data.xquat[idx].tolist(),
            velocity=(jp @ self.data.qvel).tolist(),
            angular_velocity=(jr @ self.data.qvel).tolist(),
        )

    def sample(self):
        drones = []
        for d, ids in enumerate(self.body_ids):
            row = {"drone_id": d + 1}
            for label, name in [("drone", "base_link"), ("rod", "suspension_rod"), ("package", "payload")]:
                row.update({f"{label}_{key}": value for key, value in self._body(ids[name]).items()})
            drones.append(row)
        return dict(
            schema="dual-raw-v1",
            world_time=float(self.data.time),
            drones=drones,
            qpos=self.data.qpos.tolist(),
            qvel=self.data.qvel.tolist(),
            qacc=self.data.qacc.tolist(),
            rpm=self.rpm.tolist(),
            ctrl=self.data.ctrl.tolist(),
            qfrc_actuator=self.data.qfrc_actuator.tolist(),
            qfrc_passive=self.data.qfrc_passive.tolist(),
            warning_counts=[int(w.number) for w in self.data.warning],
            contacts=[
                dict(geom1=int(c.geom1), geom2=int(c.geom2), distance=float(c.dist))
                for c in self.data.contact
            ],
        )

    def set_rpm(self, rpm):
        rpm = np.asarray(rpm, float)
        if (
            rpm.shape != (2, 4)
            or not np.isfinite(rpm).all()
            or np.any(rpm < 0)
            or np.any(rpm > 1.8 * HOVER_RPM)
        ):
            raise ValueError("Eight channels within original motor authority required")
        self.rpm = rpm.copy()
        self.data.ctrl[self.actuator_ids] = KF * rpm**2

    def _stage(self, model, data):
        self.stage_rows.append(
            dict(
                world_time=float(data.time),
                qpos=data.qpos.tolist(),
                qvel=data.qvel.tolist(),
                ctrl=data.ctrl.tolist(),
                xmat=data.xmat.tolist(),
                xipos=data.xipos.tolist(),
                qfrc_passive=data.qfrc_passive.tolist(),
                warning_counts=[int(w.number) for w in data.warning],
            )
        )

    def geometry_state(self):
        return dict(
            world_time=float(self.data.time),
            qpos=self.data.qpos.tolist(),
            qvel=self.data.qvel.tolist(),
            qacc=self.data.qacc.tolist(),
            geom_xpos=self.data.geom_xpos.tolist(),
            geom_xmat=self.data.geom_xmat.tolist(),
            actuator_force=self.data.actuator_force.tolist(),
            qfrc_actuator=self.data.qfrc_actuator.tolist(),
            contacts=[
                dict(geom1=int(c.geom1), geom2=int(c.geom2), distance=float(c.dist))
                for c in self.data.contact
            ],
        )

    def step_rpm(self, rpm, *, observe_stages=False):
        self.set_rpm(rpm)
        self.stage_rows = []
        self.substep_rows = [self.geometry_state()] if observe_stages else []
        self.mj.set_mjcb_control(self._stage if observe_stages else None)
        try:
            for _ in range(self.substeps):
                self.mj.mj_step(self.model, self.data)
                if observe_stages:
                    self.mj.set_mjcb_control(None)
                    self.mj.mj_forward(self.model, self.data)
                    self.substep_rows.append(self.geometry_state())
                    self.mj.set_mjcb_control(self._stage)
        finally:
            self.mj.set_mjcb_control(None)
        self.mj.mj_forward(self.model, self.data)
        return self.stage_rows

    def runtime(self):
        return dict(
            schema="dual-native-properties-v1",
            profile="dual-payload-v1",
            qualification="UNQUALIFIED",
            backend="mujoco",
            version=self.mj.__version__,
            precision="float64",
            integrator="RK4",
            substeps=16,
            original_asset_sha256=self.original_sha256,
            model_xml_sha256=sha256(self.model_path),
            body_ids=self.body_ids,
            addresses=self.addresses,
            actuator_ids=self.actuator_ids.tolist(),
            body_mass=self.model.body_mass.tolist(),
            body_inertia=self.model.body_inertia.tolist(),
            body_ipos=self.model.body_ipos.tolist(),
            body_iquat=self.model.body_iquat.tolist(),
            geom_size=self.model.geom_size.tolist(),
            geom_type=self.model.geom_type.tolist(),
            geom_bodyid=self.model.geom_bodyid.tolist(),
            damping=self.model.dof_damping.tolist(),
            actuator_gear=self.model.actuator_gear.tolist(),
            audit_geom_ids=self.audit_geom_ids,
            self_exclusions=self.self_exclusions,
            proxy_coverage=self.proxy_coverage,
            motor_mapping="ctrl=KF*rpm^2; site at original rotor COM; gear yaw=SPIN*KM/KF",
        )
