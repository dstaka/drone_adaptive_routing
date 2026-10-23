"""Double precision RK4 reference for the unchanged articulated payload model.

Only the original rotor link COMs receive input. The same held RPM is evaluated
in each RK stage's current body frame. This experimental adapter grants no
qualification. It must not be used by an optimizer without independent gates.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import math
import numpy as np

from .payload_ball_asset import write_ball_mjcf, ROD_MASS_KG
from .payload_contract import Payload, rotation
from .payload_passive_check import mechanical_energy
from .util import sha256

AUTHORED_MASS_KG = 0.027
FREE_MASS_KG = 0.02700047753751278
BASE_MASS_KG = 0.027000000700354576
ROTOR_MASS_KG = 1.1920928955078125e-7
KF = 3.16e-10
KM = 7.94e-12
HOVER_RPM = 14468.429183500699
SPIN = np.array([-1.0, 1.0, -1.0, 1.0])


class MuJoCoFixture:
    """One physical scene with explicit reset and zero implicit resampling."""

    dt = 0.01

    def __init__(self, original, output, payload, *, fixed=False, loaded=True, substeps=20, integrator="RK4"):
        import mujoco as mj

        if (
            not isinstance(payload, Payload)
            or payload.reference_mass_kg != FREE_MASS_KG
            or type(substeps) is not int
            or not 1 <= substeps <= 1024
            or type(fixed) is not bool
            or type(loaded) is not bool
            or integrator not in ("RK4", "implicitfast")
        ):
            raise ValueError("Explicit physical model and supported numerical settings required")
        self.mj = mj
        self.payload = payload
        self.fixed = fixed
        self.loaded = loaded
        self.substeps = substeps
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.asset = write_ball_mjcf(
            original, self.output / "authored.xml", payload, fixed=fixed, loaded=loaded
        )
        tree = ET.parse(self.output / "authored.xml")
        root = tree.getroot()
        option = ET.SubElement(
            root,
            "option",
            timestep=str(self.dt / substeps),
            gravity="0 0 -9.81",
            integrator=integrator,
            solver="Newton",
            tolerance="1e-12",
            iterations="100",
        )
        ET.SubElement(option, "flag", energy="enable")
        world = root.find("worldbody")
        ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 .1")
        base = world.find("body")
        base.set("pos", "0 0 1")
        # Preserve measured original Drone-importer mass regularization, rather
        # than silently losing four finite rotor masses in a different importer.
        base.find("inertial").set("mass", str(BASE_MASS_KG))
        for i in range(4):
            base.find(f"body[@name='prop{i}_link']/inertial").set("mass", str(ROTOR_MASS_KG))
        tree.write(self.output / "model.xml", encoding="utf-8", xml_declaration=True)
        mj.set_mjcb_control(None)
        self.model = mj.MjModel.from_xml_path(str(self.output / "model.xml"))
        self.data = mj.MjData(self.model)
        self.ids = {
            name: mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            for name in [
                "base_link",
                *[f"prop{i}_link" for i in range(4)],
                *(["suspension_rod", "payload"] if loaded else []),
            ]
        }
        if not fixed:
            self.model.dof_damping[3:6] = (
                AUTHORED_MASS_KG + (payload.mass + ROD_MASS_KG if loaded else 0)
            ) * 1e-5
        self.rpm = np.zeros(4)
        self.stage_rows = []
        self._observe = False
        self.reset()

    def apply_payload(self, payload):
        """Apply an episode mass/inertia draw to the same actual cube geometry.

        This is an explicit reset operation. Callers reset the episode after it;
        target changes and rollout boundaries must never call this method.
        """
        if (
            not self.loaded
            or not isinstance(payload, Payload)
            or payload.reference_mass_kg != FREE_MASS_KG
            or payload.side_m != self.payload.side_m
        ):
            raise ValueError("Payload draw requires the same qualified cube geometry")
        self.mj.set_mjcb_control(None)
        idx = self.ids["payload"]
        self.model.body_mass[idx] = payload.mass
        self.model.body_inertia[idx] = payload.inertia
        if not self.fixed:
            self.model.dof_damping[3:6] = (AUTHORED_MASS_KG + ROD_MASS_KG + payload.mass) * 1e-5
        self.mj.mj_setConst(self.model, self.data)
        self.payload = payload

    def reset(
        self, *, swing_deg=0.0, direction_rad=0.0, position=(0.0, 0.0, 1.0), quaternion=(1.0, 0.0, 0.0, 0.0)
    ):
        """Reset actual bodies and all input buffers without advancing simulation."""
        mj = self.mj
        mj.set_mjcb_control(None)
        mj.mj_resetData(self.model, self.data)
        if not math.isfinite(swing_deg) or not 0 <= swing_deg <= 180 or not math.isfinite(direction_rad):
            raise ValueError("Finite unrestricted passive initial displacement required")
        if not self.fixed:
            self.data.qpos[:3] = position
            self.data.qpos[3:7] = quaternion
        else:
            self.model.body_pos[self.ids["base_link"]] = position
        if self.loaded:
            joint = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "suspension_ball")
            start = self.model.jnt_qposadr[joint]
            half = math.radians(swing_deg) / 2
            self.data.qpos[start : start + 4] = [
                math.cos(half),
                math.sin(direction_rad) * math.sin(half),
                -math.cos(direction_rad) * math.sin(half),
                0.0,
            ]
        self.rpm = np.zeros(4)
        self.stage_rows = []
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        mj.mj_forward(self.model, self.data)
        return self.sample()

    def _body(self, name):
        idx = self.ids[name]
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        self.mj.mj_jacBody(self.model, self.data, jp, jr, idx)
        return dict(
            xyz=self.data.xpos[idx].copy(),
            quaternion_wxyz=self.data.xquat[idx].copy(),
            velocity=jp @ self.data.qvel,
            angular_velocity=jr @ self.data.qvel,
        )

    def raw_state(self):
        """Read untransformed actual getters, including invalid values on failure."""
        result = {}
        for label, name in [
            ("drone", "base_link"),
            *([("rod", "suspension_rod"), ("package", "payload")] if self.loaded else []),
        ]:
            for field, value in self._body(name).items():
                result[f"{label}_{field}"] = value.tolist()
        result.update(
            time_s=float(self.data.time),
            qpos=self.data.qpos.tolist(),
            qvel=self.data.qvel.tolist(),
            qacc=self.data.qacc.tolist(),
            contact=bool(self.data.ncon),
            rpm=self.rpm.tolist(),
            actual_applied_generalized_force=self.data.qfrc_applied.tolist(),
            actual_passive_generalized_force=self.data.qfrc_passive.tolist(),
            actual_xfrc_applied=self.data.xfrc_applied.tolist(),
            warning_counts=[int(w.number) for w in self.data.warning],
        )
        return result

    def sample(self):
        """Derive attachment positions only after callers can preserve raw getters."""
        result = self.raw_state()
        if self.loaded:
            for key, label, offset in [
                ("a", "drone", [0, 0, -0.02]),
                ("b", "package", [0, 0, self.payload.side_m / 2]),
            ]:
                arm = rotation(result[f"{label}_quaternion_wxyz"]) @ offset
                result[f"attachment_{key}_world"] = (np.array(result[f"{label}_xyz"]) + arm).tolist()
                result[f"attachment_{key}_velocity"] = (
                    np.array(result[f"{label}_velocity"]) + np.cross(result[f"{label}_angular_velocity"], arm)
                ).tolist()
        return result

    def runtime(self):
        """Expose actual compiled mass, inertia, COM and collision measurements."""
        result = dict(
            links=[],
            inertia_kg_m2=[],
            links_com_body_m=[],
            links_inertial_quaternion_wxyz=[],
            links_frame_offset_quaternion_wxyz=[],
            damping_after=self.model.dof_damping.tolist(),
            armature=self.model.dof_armature.tolist(),
        )
        for name, idx in self.ids.items():
            result["links"].append(dict(name=name, mass_kg=float(self.model.body_mass[idx])))
            transform = rotation(self.model.body_iquat[idx])
            inertia = transform @ np.diag(self.model.body_inertia[idx]) @ transform.T
            result["inertia_kg_m2"].append(inertia.tolist())
            result["links_com_body_m"].append(self.model.body_ipos[idx].tolist())
            result["links_inertial_quaternion_wxyz"].append([1.0, 0.0, 0.0, 0.0])
            result["links_frame_offset_quaternion_wxyz"].append([1.0, 0.0, 0.0, 0.0])
        result.update(
            free_drone_mass_kg=sum(
                float(self.model.body_mass[self.ids[name]])
                for name in ["base_link", *[f"prop{i}_link" for i in range(4)]]
            ),
            package_mass_kg=float(self.model.body_mass[self.ids["payload"]]) if self.loaded else None,
            actual_dt_s=float(self.model.opt.timestep),
            substeps=self.substeps,
            integrator=int(self.model.opt.integrator),
            model_xml_sha256=sha256(self.output / "model.xml"),
            backend="mujoco",
            version=self.mj.__version__,
            precision="64",
            body_ipos=self.model.body_ipos.tolist(),
            body_iquat=self.model.body_iquat.tolist(),
            geom_size=self.model.geom_size.tolist(),
            geom_type=self.model.geom_type.tolist(),
            geom_bodyid=self.model.geom_bodyid.tolist(),
            joint_type=self.model.jnt_type.tolist(),
            joint_pos=self.model.jnt_pos.tolist(),
            dof_frictionloss=self.model.dof_frictionloss.tolist(),
            gravity_m_s2=self.model.opt.gravity.tolist(),
            joint_limited=self.model.jnt_limited.tolist(),
            collision_geoms=[
                dict(
                    name=self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, i),
                    type=int(self.model.geom_type[i]),
                    size=self.model.geom_size[i].tolist(),
                    pos=self.model.geom_pos[i].tolist(),
                    quat=self.model.geom_quat[i].tolist(),
                )
                for i in range(self.model.ngeom)
                if self.model.geom_contype[i] or self.model.geom_conaffinity[i]
            ],
        )
        return result

    def planar_acceleration(self):
        if not self.fixed or not self.loaded:
            raise ValueError("Fixed planar model required")
        return np.array([-self.data.qacc[1], -self.data.qacc[1] - self.data.qacc[4]])

    def energy(self):
        row = self.sample()
        anchor = np.array(row["drone_xyz"]) + rotation(row["drone_quaternion_wxyz"]) @ [0, 0, -0.02]
        return mechanical_energy(row, self.runtime(), self.payload.side_m, float(anchor[2]))

    def _control(self, model, data):
        data.qfrc_applied[:] = 0
        forces = []
        torques = []
        points = []
        for i in range(4):
            idx = self.ids[f"prop{i}_link"]
            axis = data.xmat[idx].reshape(3, 3)[:, 2]
            force = axis * KF * self.rpm[i] ** 2
            torque = axis * KM * SPIN[i] * self.rpm[i] ** 2
            point = data.xipos[idx]
            self.mj.mj_applyFT(model, data, force, torque, point, idx, data.qfrc_applied)
            forces.append(force.tolist())
            torques.append(torque.tolist())
            points.append(point.tolist())
        if self._observe:
            self.stage_rows.append(
                dict(
                    time_s=float(data.time),
                    qpos=data.qpos.tolist(),
                    qvel=data.qvel.tolist(),
                    qacc_before_solve=data.qacc.tolist(),
                    rpm=self.rpm.tolist(),
                    rotor_force_world_n=forces,
                    rotor_torque_world_nm=torques,
                    rotor_com_world=points,
                    motor_generalized_force=data.qfrc_applied.tolist(),
                    motor_power_w=float(data.qfrc_applied @ data.qvel),
                    dissipative_power_w=float(data.qfrc_passive @ data.qvel),
                )
            )

    def step_rpm(self, rpm, *, observe_stages=False):
        """Hold current RPM through four actual RK stages per numerical step."""
        rpm = np.asarray(rpm, dtype=float)
        if (
            rpm.shape != (4,)
            or not np.isfinite(rpm).all()
            or np.any(rpm < 0)
            or np.any(rpm > 1.8 * HOVER_RPM + 1e-8)
        ):
            raise ValueError("Original finite rotor authority required")
        self.rpm = rpm.copy()
        self.stage_rows = []
        self._observe = observe_stages
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.mj.set_mjcb_control(self._control if (observe_stages or np.any(rpm)) else None)
        try:
            for _ in range(self.substeps):
                self.mj.mj_step(self.model, self.data)
        finally:
            self.mj.set_mjcb_control(None)
        # Refresh final getters after RK4; this evaluation is not an RK stage.
        self._observe = False
        self.mj.mj_forward(self.model, self.data)
        return self.stage_rows
