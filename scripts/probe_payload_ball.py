#!/usr/bin/env python
"""Record native-ball construction/stepping before operational qualification.

This experimental script never imports a learned controller or an optimizer.
All motor input uses the pinned four-rotor wrench kernel on rotor link COMs.
"""

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import math
import shutil
from pathlib import Path
import time
import traceback

import numpy as np

from adaptive_drone_patrol.payload_ball_bank import build_ball_fixture
from adaptive_drone_patrol.payload_contract import Payload, rotation
from adaptive_drone_patrol.payload_qualification import validate_declaration
from adaptive_drone_patrol.payload_substep_probe import METHOD, SubstepRecorder
from adaptive_drone_patrol.util import sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def initial_suspension_quaternion(angle_deg, direction_rad):
    """Rotate world down toward the declared XY azimuth without Euler angles."""
    half = math.radians(angle_deg) / 2
    return [math.cos(half), math.sin(direction_rad) * math.sin(half),
            -math.cos(direction_rad) * math.sin(half), 0.0]


def physical_speed_extrema(row):
    """Use individual world vectors, including each rotor, never a matrix norm."""
    linear, angular = [0.0], [0.0]
    for key, value in row.items():
        if key.endswith('_angular_velocity'):
            angular.extend(np.linalg.norm(np.asarray(value).reshape(-1, 3), axis=1).tolist())
        elif key.endswith('_velocity') and key != 'dofs_velocity':
            linear.extend(np.linalg.norm(np.asarray(value).reshape(-1, 3), axis=1).tolist())
    return max(linear), max(angular)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixture", choices=("original", "unloaded", "loaded", "anchor"), default="loaded")
    parser.add_argument("--case")
    parser.add_argument("--align-root", action="store_true")
    parser.add_argument("--rotor", type=int, choices=range(4))
    parser.add_argument("--reference-audit", default="outputs/construction-loaded-v1/runtime-initial.json")
    parser.add_argument("--manifest", default="configs/payload_retry_probe_manifest.json")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    declaration = json.loads(Path(args.manifest).read_text())
    validate_declaration(declaration)
    case = next((c for c in declaration["cases"] if c["id"] == args.case), None)
    if args.case and case is None:
        raise ValueError("Unknown predeclared case")
    if case and ((args.fixture == "anchor") != (case["kind"] in ("anchor", "large_angle"))):
        raise ValueError("Case/fixture mismatch")
    settings = declaration["settings"]
    if settings.get("root_alignment_requested") and args.fixture != "original" and not args.align_root:
        raise ValueError("Selected retry profile requires explicitly requested root alignment")
    result = {
        "schema": "payload-ball-probe-v2",
        "status": "STARTED",
        "physical_training_admitted": False,
        "fixture": args.fixture,
        "case": case,
        "manifest_sha256": sha256(args.manifest),
        "source_binding": {
            p: sha256(p)
            for p in (
                "scripts/probe_payload_ball.py",
                "src/adaptive_drone_patrol/payload_ball_asset.py",
                "src/adaptive_drone_patrol/payload_qualification.py",
                "src/adaptive_drone_patrol/payload_contract.py",
            )
        },
        "settings": settings,
        "steps": 0,
        "rotor_excitation": args.rotor,
        "controller_source_sha256": sha256("src/adaptive_drone_patrol/payload_probe_controller.py"),
        "motor_plan_sha256": sha256("configs/payload_retry_motor_audit.json"),
        "backend_source_sha256": sha256("src/adaptive_drone_patrol/payload_ball_bank.py"),
        "align_root": args.align_root,
        "substep_work_method": METHOD,
        "substep_work_source_binding": {
            p: sha256(p) for p in ("scripts/probe_payload_ball.py",
                "src/adaptive_drone_patrol/payload_substep_probe.py")
        },
    }
    if args.fixture == "anchor":
        reference_path = Path(args.reference_audit)
        free = json.loads(reference_path.read_text())
        if free.get("fixture") not in ("original", "loaded", "unloaded") or not free.get("runtime"):
            raise ValueError("Fixed diagnostic requires an actual free-vehicle runtime audit")
        names = {"base_link", "prop0_link", "prop1_link", "prop2_link", "prop3_link"}
        reference_mass = sum(link["mass_kg"] for link in free["runtime"]["links"] if link["name"] in names)
        if abs(reference_mass - 0.02700047753751278) > 1e-8:
            raise ValueError("Original free-vehicle reference mass changed")
        shutil.copyfile(reference_path, out / "reference-free-runtime.json")
        result["reference_mass_audit"] = {
            "path": "reference-free-runtime.json",
            "sha256": sha256(reference_path),
        }
    else:
        reference_mass = 0.02700047753751278
    write(out / "declaration.json", result)
    started = time.monotonic()
    try:
        if importlib.metadata.version("genesis-world") != "1.4.3":
            raise ValueError("Pinned Genesis required")
        import genesis as gs
        import torch

        genesis_root = Path(gs.__file__).parent
        asset = genesis_root / "assets/urdf/drones/cf2x.urdf"
        result["original_asset_sha256"] = sha256(asset)
        # The old measured mass is a provisional payload reference until audited.
        payload = Payload(
            reference_mass, case["mass_ratio"] if case else 0.2, case["side_m"] if case else 0.045
        )
        result["engine_source"] = {}
        for pattern in (
            "utils/*mj*.py",
            "engine/entities/rigid_entity/*.py",
            "engine/solvers/rigid/abd/*.py",
            "options/morphs.py",
        ):
            for source in genesis_root.glob(pattern):
                result["engine_source"][str(source.relative_to(genesis_root))] = sha256(source)
        for name in ("engine/solvers/rigid/rigid_solver.py", "engine/simulator.py", "engine/scene.py"):
            result["engine_source"][name] = sha256(genesis_root / name)
        write(out / "pre-init.json", result)
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=7)
        dt = 0.01
        substeps = case["substeps"] if case else 2
        fixture = build_ball_fixture(
            out, payload, fixture=args.fixture, dt=dt, substeps=substeps, align_root=args.align_root
        )
        scene, entity, ground = fixture["scene"], fixture["entity"], fixture["ground"]
        result["asset"] = fixture["asset_manifest"]
        base = entity.get_link("base_link")
        props = [entity.get_link(f"prop{i}_link") for i in range(4)]
        masses = entity.get_links_mass()[0].cpu().tolist()
        original_mass = sum(masses[link.idx - entity.link_start] for link in [base] + props)
        result["runtime"] = {
            "reference_mass_kg": reference_mass if args.fixture == "anchor" else original_mass,
            "fixture_original_mass_kg": original_mass,
            "links": [
                {"name": link.name, "idx": link.idx, "mass_kg": masses[link.idx - entity.link_start]}
                for link in entity.links
            ],
            "inertia_kg_m2": entity.get_links_inertia()[0].cpu().tolist(),
            "links_com_body_m": entity.get_links_COM()[0].cpu().tolist(),
            "n_qs": entity.n_qs,
            "n_dofs": entity.n_dofs,
            "joints": [
                {
                    "name": j.name,
                    "type": str(j.type),
                    "n_qs": j.n_qs,
                    "n_dofs": j.n_dofs,
                    "q_start": j.q_start,
                    "dof_start": j.dof_start,
                }
                for j in entity.joints
            ],
            "initial_entity_pos": entity.get_pos()[0].cpu().tolist(),
            "initial_base_pos": base.get_pos()[0].cpu().tolist(),
            "initial_links_pos": entity.get_links_pos()[0].cpu().tolist(),
            "initial_links_quat": entity.get_links_quat()[0].cpu().tolist(),
            "initial_links_internal_quaternion_wxyz": entity.get_links_quat(relative=False)[0].cpu().tolist(),
            "links_inertial_quaternion_wxyz": [
                None if link.desc.inertial_quat is None else np.asarray(link.desc.inertial_quat).tolist()
                for link in entity.links
            ],
            "links_frame_offset_quaternion_wxyz": [
                np.asarray(link.desc.offset_quat).tolist() for link in entity.links
            ],
            "actual_solver_settings": {
                name: str(getattr(entity.solver, name, "UNAVAILABLE"))
                for name in ("dt", "substep_dt", "_substep_dt", "_integrator")
            },
            "rotor_indices": [p.idx for p in props],
            "rotor_spin": [-1, 1, -1, 1],
            "kf": 3.16e-10,
            "km": 7.94e-12,
            "rpm_mapping": "(1+0.8*action)*14468.429183500699",
            "motor_application": "same pinned solver.set_drone_rpm; rotor local +z force/torque at original link COM",
            "damping_before": fixture["damping_before"],
            "root_aligned": base.aligned,
            "root_qpos": entity.get_qpos()[0].cpu().tolist(),
            "armature": entity.get_dofs_armature().cpu().tolist(),
        }
        result["runtime"]["damping_after"] = entity.get_dofs_damping().cpu().tolist()
        if args.fixture in ("loaded", "anchor"):
            package = entity.get_link("payload")
            result["runtime"]["package_aabb"] = package.get_AABB()[0].cpu().tolist()
            q = entity.get_qpos().clone()
            joint = entity.get_joint("suspension_ball")
            angle = case["initial_swing_deg"] if case else 5.0
            direction = case["initial_direction_rad"] if case else 0.0
            q[:, joint.q_start - entity.q_start : joint.q_start - entity.q_start + 4] = torch.tensor(
                initial_suspension_quaternion(angle, direction), device=gs.device
            )
            entity.set_qpos(q, zero_velocity=True)
        write(out / "runtime-initial.json", result)
        indices = torch.tensor([p.idx for p in props], dtype=gs.tc_int, device=gs.device)
        spin = torch.tensor([-1, 1, -1, 1], dtype=gs.tc_float, device=gs.device)
        duration = case["duration_s"] if case else (0.05 if args.rotor is not None else 1.0)
        n = round(duration / dt)
        total_mass = sum(masses)
        work = 0.0
        previous_rpm = np.zeros(4)
        failures = []
        with (
            (out / "raw-states.jsonl").open("x") as raw_stream,
            (out / "samples.jsonl").open("x") as stream,
            (out / "motor-inputs.jsonl").open("x") as motor_stream,
            (out / "substep-work.jsonl").open("x") as work_stream,
            ExitStack() as stack,
        ):
            work_recorder = (stack.enter_context(SubstepRecorder(entity.solver, entity, work_stream))
                if case and case["kind"] in ("powered", "refinement") else None)
            for i in range(n + 1):
                row = {"step": i, "time_s": i * dt, "motor_work_j": work, "rpm": previous_rpm.tolist()}
                values = {
                    "xyz": entity.get_links_pos()[0].cpu().numpy(),
                    "quaternion_wxyz": entity.get_links_quat()[0].cpu().numpy(),
                    "velocity": entity.get_links_vel()[0].cpu().numpy(),
                    "angular_velocity": entity.get_links_ang()[0].cpu().numpy(),
                    "internal_quaternion_wxyz": entity.get_links_quat(relative=False)[0].cpu().numpy(),
                }
                bodies = [("drone", base)]
                if args.fixture in ("loaded", "anchor"):
                    rod = entity.get_link("suspension_rod")
                    bodies += [("package", package), ("rod", rod)]
                for label, link in bodies:
                    local = link.idx - entity.link_start
                    for key, array in values.items():
                        row[f"{label}_{key}"] = array[local].tolist()
                row["qpos"] = entity.get_qpos()[0].cpu().tolist()
                row["dofs_velocity"] = entity.get_dofs_velocity()[0].cpu().tolist()
                raw_stream.write(json.dumps(row, allow_nan=True) + "\n")
                raw_stream.flush()
                if args.fixture in ("loaded", "anchor"):
                    a = np.array(row["drone_xyz"]) + rotation(row["drone_quaternion_wxyz"]) @ np.array(
                        [0, 0, -0.02]
                    )
                    b = np.array(row["package_xyz"]) + rotation(row["package_quaternion_wxyz"]) @ np.array(
                        [0, 0, payload.side_m / 2]
                    )
                    row["attachment_a_world"] = a.tolist()
                    row["attachment_b_world"] = b.tolist()
                    row["constraint_residual_m"] = abs(float(np.linalg.norm(b - a)) - 0.3)
                    row["package_floor_m"] = float(
                        row["package_xyz"][2]
                        - payload.side_m / 2 * sum(abs(rotation(row["package_quaternion_wxyz"])[2]))
                    )
                R = rotation(row["drone_quaternion_wxyz"])
                row["drone_floor_m"] = float(
                    row["drone_xyz"][2] - 0.0125 * abs(R[2, 2]) - 0.06 * np.linalg.norm(R[2, :2])
                )
                row["roll_deg"] = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                row["pitch_deg"] = math.degrees(math.asin(np.clip(-R[2, 0], -1.0, 1.0)))
                contacts = entity.get_contacts(ground)
                row["contact"] = bool(contacts["valid_mask"].any().item())
                row["rotor_com_world"] = (
                    entity.get_links_pos(
                        [p.idx - entity.link_start for p in props], ref=gs.link_ref_frame.link_COM
                    )[0]
                    .cpu()
                    .tolist()
                )
                row["rotor_com_velocity"] = (
                    entity.get_links_vel(
                        [p.idx - entity.link_start for p in props], ref=gs.link_ref_frame.link_COM
                    )[0]
                    .cpu()
                    .tolist()
                )
                row["rotor_angular_velocity"] = (
                    entity.get_links_ang([p.idx - entity.link_start for p in props])[0].cpu().tolist()
                )
                axis = rotation(row["drone_quaternion_wxyz"])[:, 2]
                forces = previous_rpm[:, None] ** 2 * 3.16e-10 * axis
                torques = previous_rpm[:, None] ** 2 * 7.94e-12 * np.array([-1, 1, -1, 1])[:, None] * axis
                row["rotor_force_world_n"] = forces.tolist()
                row["rotor_torque_world_nm"] = torques.tolist()
                row["motor_power_w"] = float(
                    np.sum(forces * np.array(row["rotor_com_velocity"]))
                    + np.sum(torques * np.array(row["rotor_angular_velocity"]))
                )
                work += dt * row["motor_power_w"] if i else 0.0
                row["motor_work_j"] = work
                # Write the raw offending state before all screening and before step/reset/clip.
                stream.write(json.dumps(row, allow_nan=True) + "\n")
                stream.flush()
                result["steps"] = i
                vals = [v for k, v in row.items() if isinstance(v, (list, int, float))]
                finite = all(np.isfinite(np.asarray(v)).all() for v in vals)
                if not finite:
                    failures.append("NUMERICAL_NONFINITE")
                    break
                speed, angular = physical_speed_extrema(row)
                if speed > 20 or angular > 200:
                    failures.append("FINITE_NUMERICAL_EXCURSION")
                    break
                if row.get("constraint_residual_m", 0) > 2e-5:
                    failures.append("CONSTRAINT_RESIDUAL")
                    break
                if args.fixture not in ("anchor",) and (
                    row["drone_floor_m"] <= 0.1 or row.get("package_floor_m", 1) <= 0.1
                ):
                    failures.append("PHYSICAL_FLOOR_FAILURE")
                    break
                if args.fixture != "anchor" and (
                    abs(row["roll_deg"]) > 60 or abs(row["pitch_deg"]) > 60 or row["contact"]
                ):
                    failures.append("PHYSICAL_ATTITUDE_OR_CONTACT_FAILURE")
                    break
                if i == n:
                    break
                rpm = np.full(4, 14468.429183500699 * math.sqrt(total_mass / original_mass))
                if args.fixture == "anchor":
                    rpm[:] = 0
                if args.fixture != "anchor":
                    if case and case["kind"] in ("powered", "refinement"):
                        from adaptive_drone_patrol.payload_probe_controller import diagnostic_rpm

                        commands = declaration["commands"]
                        target = commands[int(i * dt / declaration["command_interval_s"]) % len(commands)]
                        rpm = diagnostic_rpm(
                            row["drone_xyz"],
                            row["drone_quaternion_wxyz"],
                            row["drone_velocity"],
                            row["drone_angular_velocity"],
                            target,
                            total_mass,
                        )
                    if args.rotor is not None:
                        rpm = np.array([14468.429183500699 if j == args.rotor else 0.0 for j in range(4)])
                    previous_rpm = rpm.copy()
                    tensor = torch.tensor(rpm[None], dtype=gs.tc_float, device=gs.device)
                    if args.fixture == "original":
                        entity.set_propellers_rpm(tensor)
                    else:
                        entity.solver.set_drone_rpm(indices, 3.16e-10, 7.94e-12, tensor, spin, False)
                if args.fixture != "anchor":
                    from genesis.utils.misc import qd_to_torch

                    state = entity.solver.dyn_state.links
                    motor_input = {
                        "step": i,
                        "time_s": i * dt,
                        "rpm": rpm.tolist(),
                        "rotor_indices": [p.idx for p in props],
                        "negative_external_force_field": qd_to_torch(state.cfrc_applied_vel, copy=True)
                        .cpu()
                        .tolist(),
                        "negative_external_torque_about_root_com_field": qd_to_torch(
                            state.cfrc_applied_ang, copy=True
                        )
                        .cpu()
                        .tolist(),
                        "root_com_world_field": qd_to_torch(state.root_COM, copy=True).cpu().tolist(),
                        "field_convention": "Global link index, environment index, xyz; source negative sign; actual solver external wrench before integration, not constraint force",
                    }
                    motor_stream.write(json.dumps(motor_input, allow_nan=True) + "\n")
                    motor_stream.flush()
                if work_recorder:
                    work_recorder.begin_control(i)
                scene.step()
        result.update(
            status="CONSTRUCTION_RECORDED" if not failures else "PHYSICAL_FAILURE",
            failures=failures,
            samples_sha256=sha256(out / "samples.jsonl"),
            raw_state_sha256=sha256(out / "raw-states.jsonl"),
            motor_inputs_sha256=sha256(out / "motor-inputs.jsonl"),
            substep_work_sha256=sha256(out / "substep-work.jsonl"),
            substep_work_records=work_recorder.records if work_recorder else 0,
            measured_duration_s=result["steps"] * dt,
            motor_work_observability="Actual pre-step solver external wrench fields plus control-step COM endpoint work estimate; substep work unobserved, no exact energy balance claim",
            constraints_observability="kinematic endpoint residual; no measured rope tension",
            operational_qualification="NOT_RUN",
            policy_admitted=False,
        )
    except Exception as exc:
        result.update(status="CONSTRUCTION_FAILED", error=repr(exc))
        (out / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        result["elapsed_s"] = time.monotonic() - started
        write(out / "summary.json", result)
        print(result["status"], result.get("error", result.get("failures")))
    raise SystemExit(0 if result["status"] == "CONSTRUCTION_RECORDED" else 2)


if __name__ == "__main__":
    main()
