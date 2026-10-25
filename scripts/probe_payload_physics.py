#!/usr/bin/env python
"""Measure the pinned articulated payload; never report this as mission success."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback
import numpy as np
from adaptive_drone_patrol.payload_asset import write_payload_urdf
from adaptive_drone_patrol.payload_contract import Payload
from adaptive_drone_patrol.util import json_write, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--side', type=float, default=.045)
    parser.add_argument('--dt', type=float, default=.01)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--anchor-fixed', action='store_true')
    parser.add_argument('--reference-mass-kg', type=float)
    parser.add_argument('--compensate-weight', action='store_true')
    parser.add_argument('--single-rotor', type=int, choices=range(4))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='STARTED', evidence_kind='physical_feasibility_probe', mission_success=False,
                  source_sha256=sha256(__file__), args=vars(args))
    # Store only relative output paths in the public receipt.
    result['args']['out'] = str(out)
    try:
        if importlib.metadata.version('genesis-world') != '1.4.3':
            raise ValueError('Pinned Genesis 1.4.3 is required')
        import genesis as gs
        import torch
        asset = Path(gs.__file__).parent/'assets/urdf/drones/cf2x.urdf'
        import xml.etree.ElementTree as ET
        tree = ET.parse(asset)
        reference = sum(float(x.get('value')) for x in tree.findall('.//inertial/mass'))
        payload = Payload(reference, .2, args.side)
        manifest = write_payload_urdf(asset, out/'loaded.urdf', payload)
        manifest.update(reference_mass_kg=reference, original_asset_sha256=sha256(asset),
                        generated_asset_sha256=sha256(out/'loaded.urdf'))
        json_write(out/'asset.json', manifest)
        gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=7)
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=args.dt),
                         rigid_options=gs.options.RigidOptions(dt=args.dt/2, enable_collision=True,
                             enable_joint_limit=True, batch_links_info=True,
                             constraint_solver=gs.constraint_solver.Newton), show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        morph_type = gs.morphs.URDF if args.anchor_fixed else gs.morphs.Drone
        extra = {'fixed': True} if args.anchor_fixed else {}
        drone = scene.add_entity(morph_type(file=str((out/'loaded.urdf').resolve()),
                                 pos=(0, 0, 1), default_armature=0., collision=True, **extra))
        scene.build(n_envs=3)
        authored_reference = reference
        masses = drone.get_links_mass()[0].detach().cpu().tolist()
        reference = sum(v for link,v in zip(drone.links,masses)
                        if link.name in ('base_link','prop0_link','prop1_link','prop2_link','prop3_link'))
        if args.anchor_fixed:
            if args.reference_mass_kg is None or args.reference_mass_kg <= 0:
                raise ValueError('Fixed-anchor diagnostic requires the measured free-drone reference mass')
            reference = args.reference_mass_kg
        manifest.update(authored_reference_mass_kg=authored_reference, reference_mass_kg=reference)
        json_write(out/'asset.json',manifest)
        package = drone.get_link('payload')
        ratios = torch.tensor([.1, .2, .3], device=gs.device)
        package.set_mass(reference*ratios)
        inertia = torch.eye(3, device=gs.device)[None] * (reference*ratios*args.side**2/6)[:, None, None]
        package.set_inertia(inertia)
        result.update(reference_mass_kg=reference, root_xyz=drone.get_pos().cpu().tolist(),
                      links=[dict(name=x.name, idx=x.idx) for x in drone.links],
                      rotor_idx=drone.propellers_idx.cpu().tolist() if not args.anchor_fixed else None,
                      rotor_spin=drone.propellers_spin.cpu().tolist() if not args.anchor_fixed else None,
                      kf=float(tree.getroot().find('properties').get('kf')),
                      km=float(tree.getroot().find('properties').get('km')),
                      effective_mass=package.get_mass().cpu().tolist(),
                      effective_inertia=drone.get_links_inertia([package.idx-drone.link_start]).cpu().tolist(),
                      dofs=drone.n_dofs, qs=drone.n_qs,
                      payload_collision_geoms=len(package.geoms))
        json_write(out/'runtime-initial.json', result)
        # The excitation is a declared initial joint displacement, not a state correction during motion.
        q = drone.get_qpos().clone()
        result['initial_qpos'] = q.cpu().tolist()
        joint = drone.get_joint('suspension_roll')
        q[:, joint.q_start-drone.q_start] = np.radians(5)
        drone.set_qpos(q, zero_velocity=True)
        base_before = drone.get_pos().clone()
        rows = []
        for i in range(args.steps):
            rpm = torch.full((3, 4), 14468.429183500699, device=gs.device)
            if args.compensate_weight:
                rpm *= torch.sqrt(1+ratios[:, None]+.000103/reference)
            if args.single_rotor is not None:
                rpm[:, [j for j in range(4) if j != args.single_rotor]] = 0
            if not args.anchor_fixed:
                drone.set_propellers_rpm(rpm)
            scene.step()
            row = dict(step=i+1, time_s=(i+1)*args.dt,
                       drone_xyz=drone.get_pos().cpu().tolist(), drone_velocity=drone.get_vel().cpu().tolist(),
                       drone_angular_velocity=drone.get_ang().cpu().tolist(),
                       joint_qpos=drone.get_qpos().cpu().tolist(),
                       joint_velocity=drone.get_dofs_velocity().cpu().tolist(),
                       rod_xyz=drone.get_link('suspension_rod').get_pos().cpu().tolist(),
                       rod_quaternion_wxyz=drone.get_link('suspension_rod').get_quat().cpu().tolist(),
                       rod_velocity=drone.get_link('suspension_rod').get_vel().cpu().tolist(),
                       rod_angular_velocity=drone.get_link('suspension_rod').get_ang().cpu().tolist(),
                       drone_quaternion_wxyz=drone.get_quat().cpu().tolist(),
                       package_xyz=package.get_pos().cpu().tolist(),
                       package_quaternion_wxyz=package.get_quat().cpu().tolist(),
                       package_velocity=package.get_vel().cpu().tolist(),
                       package_angular_velocity=package.get_ang().cpu().tolist())
            rows.append(row)
            with (out/'samples.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            if not all(np.isfinite(np.asarray(row[k])).all() for k in row):
                raise RuntimeError('Nonfinite physical probe')
        result.update(status='PROBE_RECORDED', steps=len(rows),
                      telemetry_sha256=sha256(out/'samples.jsonl'),
                      initial_root=base_before.cpu().tolist(),
                      caveat='Open-loop excitation, not a flight policy or acceptance result')
    except Exception as exc:
        result.update(status='FAILED', error=repr(exc))
        (out/'traceback.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        json_write(out/'summary.json', result)
    raise SystemExit(0 if result['status'] == 'PROBE_RECORDED' else 2)


if __name__ == '__main__':
    main()
