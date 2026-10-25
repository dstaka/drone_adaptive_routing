"""Isolated single-environment scenes for episode-level collision geometry selection.

Only the active scene is stepped. Other sizes live in different solvers, so no
inactive collision geometry, inertia or wrench participates in the episode.
The controller is shared across sizes; scene selection is never an actor input.
"""

from pathlib import Path
import numpy as np
from .payload_asset import write_payload_urdf
from .payload_contract import Payload, DRONE_ATTACHMENT, rotation


class PayloadSceneBank:
    def __init__(self, output, dt=.01, substeps=2):
        import genesis as gs
        import xml.etree.ElementTree as ET
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.dt = dt
        asset = Path(gs.__file__).parent/'assets/urdf/drones/cf2x.urdf'
        self.reference_mass = sum(float(x.get('value')) for x in ET.parse(asset).findall('.//inertial/mass'))
        authored_reference_mass = self.reference_mass
        measured_reference_mass = None
        self.banks = {}
        for size in (.03, .045, .06):
            path = self.output/f'load_{size}.urdf'
            write_payload_urdf(asset, path, Payload(authored_reference_mass, .2, size))
            scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt),
                             rigid_options=gs.options.RigidOptions(dt=dt/substeps, enable_collision=True,
                                 enable_joint_limit=True, batch_links_info=True,
                                 constraint_solver=gs.constraint_solver.Newton), show_viewer=False)
            ground = scene.add_entity(gs.morphs.Plane())
            drone = scene.add_entity(gs.morphs.Drone(file=str(path.resolve()), pos=(0,0,1),
                                     default_armature=0., collision=True))
            scene.build(n_envs=1)
            masses = drone.get_links_mass()[0].detach().cpu().tolist()
            original_mass = sum(value for link,value in zip(drone.links,masses)
                                if link.name in ('base_link','prop0_link','prop1_link','prop2_link','prop3_link'))
            if measured_reference_mass is not None and original_mass != measured_reference_mass:
                raise ValueError('Original runtime drone masses differ across geometry scenes')
            measured_reference_mass = original_mass
            self.banks[size] = (scene, drone, drone.get_link('payload'), ground)
        self.reference_mass = measured_reference_mass
        self.active = None

    def reset(self, payload, initial):
        """Switch whole isolated scenes only at an episode boundary."""
        import torch
        from scipy.spatial.transform import Rotation
        if payload.reference_mass_kg != self.reference_mass:
            raise ValueError('Original drone mass differs from payload reference')
        self.active = self.banks[payload.side_m]
        self.scene, self.drone, self.package, self.ground = self.active
        self.payload = payload
        self.package.set_mass(payload.mass)
        self.package.set_inertia(np.eye(3)*payload.inertia[0])
        d = np.array(initial['drone_xyz'])
        rd = rotation(initial['drone_quaternion_wxyz'])
        rp = rotation(initial['package_quaternion_wxyz'])
        end = np.array(initial['package_xyz']) + rp @ [0,0,payload.side_m/2]
        anchor = d + rd @ DRONE_ATTACHMENT
        direction = rd.T @ ((end-anchor)/.3)
        if not np.isclose(np.linalg.norm(end-anchor),.3,atol=1e-7):
            raise ValueError('Initial state violates the fixed suspension length')
        pitch = -np.arcsin(np.clip(direction[0],-1,1))
        roll = np.arctan2(direction[1],-direction[2])
        rod_rotation = rd @ Rotation.from_euler('XY',[roll,pitch]).as_matrix()
        bottom_angles = Rotation.from_matrix(rod_rotation.T @ rp).as_euler('XYZ')
        q = self.drone.get_qpos().clone()
        q[:,:3] = torch.as_tensor(d,device=q.device,dtype=q.dtype)
        q[:,3:7] = torch.as_tensor(initial['drone_quaternion_wxyz'],device=q.device,dtype=q.dtype)
        angles = dict(suspension_roll=roll, suspension_pitch=pitch,
                      payload_roll=bottom_angles[0], payload_pitch=bottom_angles[1], payload_yaw=bottom_angles[2])
        for name, value in angles.items():
            joint = self.drone.get_joint(name)
            q[:,joint.q_start-self.drone.q_start] = float(value)
        self.drone.set_qpos(q,zero_velocity=True)
        self.drone.zero_all_dofs_velocity()
        # No physical advancement and no policy call occurs during reset.
        return self.sample()

    def sample(self):
        sample = {}
        for label, body in [('drone',self.drone),('package',self.package)]:
            for field, method in [('xyz','get_pos'),('quaternion_wxyz','get_quat'),
                                  ('velocity','get_vel'),('angular_velocity','get_ang')]:
                sample[f'{label}_{field}'] = getattr(body,method)()[0].detach().cpu().tolist()
        rd, rp = rotation(sample['drone_quaternion_wxyz']), rotation(sample['package_quaternion_wxyz'])
        a, b = rd @ DRONE_ATTACHMENT, rp @ np.array([0,0,self.payload.side_m/2])
        for label, offset, prefix in [('a',a,'drone'),('b',b,'package')]:
            sample[f'attachment_{label}_world'] = (np.array(sample[f'{prefix}_xyz'])+offset).tolist()
            sample[f'attachment_{label}_velocity'] = (np.array(sample[f'{prefix}_velocity'])+
                    np.cross(sample[f'{prefix}_angular_velocity'],offset)).tolist()
        return sample

    def step(self, actions):
        """Four motor commands; no pose corrections or swing controllers."""
        import torch
        motor = torch.as_tensor(actions, device=self.drone.get_pos().device).reshape(1,4).clamp(-1,1)
        self.drone.set_propellers_rpm((1+.8*motor)*14468.429183500699)
        self.scene.step()
        return self.sample()
