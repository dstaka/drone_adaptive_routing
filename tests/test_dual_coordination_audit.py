"""Independent raw structural checks do not imply continuous physical safety."""

from copy import deepcopy
import gzip
import hashlib
import json

import pytest

from adaptive_drone_patrol.dual_coordination_audit import (
    verify_structure,
    audit_probe,
    PointAuditLedger,
    conservative_interval_bound,
)


def records():
    q, v = [0.0] * 30, [0.0] * 24

    def geometry(t):
        return dict(
            world_time=t,
            qpos=q.copy(),
            qvel=v.copy(),
            qacc=v.copy(),
            geom_xpos=[[0, 0, 1], [3, 0, 1]],
            geom_xmat=[[1, 0, 0, 0, 1, 0, 0, 0, 1]] * 2,
            actuator_force=[0.0] * 8,
            qfrc_actuator=v.copy(),
            contacts=[],
        )

    def actual(t):
        drones = []
        for d in (1, 2):
            drone = dict(drone_id=d)
            for body in ("drone", "rod", "package"):
                for field in ("xyz", "velocity", "angular_velocity"):
                    drone[f"{body}_{field}"] = [0.0, 0.0, 1.0] if field == "xyz" else [0.0] * 3
                drone[f"{body}_quaternion_wxyz"] = [1.0, 0.0, 0.0, 0.0]
            drones.append(drone)
        return dict(
            schema="dual-raw-v1",
            world_time=t,
            qpos=q.copy(),
            qvel=v.copy(),
            qacc=v.copy(),
            drones=drones,
            ctrl=[0.0] * 8,
            rpm=[[0.0] * 4] * 2,
            qfrc_actuator=v.copy(),
            qfrc_passive=v.copy(),
            warning_counts=[0] * 7,
            contacts=[],
        )

    stages = []
    for i in range(16):
        for offset in (0, 0.5, 0.5, 1):
            stages.append(
                dict(
                    world_time=(i + offset) * 0.01 / 16,
                    qpos=q.copy(),
                    qvel=v.copy(),
                    ctrl=[0.0] * 8,
                    xmat=[[1, 0, 0, 0, 1, 0, 0, 0, 1]] * 3,
                    xipos=[[0.0, 0.0, 0.0]] * 3,
                    qfrc_passive=v.copy(),
                    warning_counts=[0] * 7,
                )
            )
    return [
        dict(kind="RESET", actual=actual(0.0), geometry=geometry(0.0)),
        dict(
            kind="STEP",
            tick=1,
            actual=actual(0.01),
            stages=stages,
            substeps=[geometry(i * 0.01 / 16) for i in range(17)],
            reference=[dict(p=[0, 0, 1], v=[0, 0, 0], a=[0, 0, 0])] * 2,
            controls=[{}, {}],
            audit={},
        ),
    ]


def test_exact_reset_step_and_duplicate_rk_stage_times_are_structurally_valid():
    report = verify_structure(iter(records()), expected_steps=1)
    assert report["structural_valid"]
    assert report["control_steps"] == 1
    assert report["native_substep_intervals"] == 16
    assert report["rk_states"] == 64
    assert not report["continuous_safety_qualified"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_reset",
        "skip_tick",
        "duplicate_reset",
        "missing_substep",
        "missing_stage",
        "wrong_stage_time",
        "nonfinite",
        "wrong_drone",
        "discontinuous_boundary",
        "wrong_control_shape",
    ],
)
def test_structural_corruption_fails_closed(change):
    rows = records()
    if change == "missing_reset":
        rows.pop(0)
    if change == "skip_tick":
        rows[1]["tick"] = 2
    if change == "duplicate_reset":
        rows.append(deepcopy(rows[0]))
    if change == "missing_substep":
        rows[1]["substeps"].pop(8)
    if change == "missing_stage":
        rows[1]["stages"].pop(8)
    if change == "wrong_stage_time":
        rows[1]["stages"][2]["world_time"] += 0.0001
    if change == "nonfinite":
        rows[1]["stages"][3]["qvel"][0] = float("nan")
    if change == "wrong_drone":
        rows[1]["actual"]["drones"][1]["drone_id"] = 1
    if change == "discontinuous_boundary":
        rows[1]["substeps"][0]["qpos"][0] = 1.0
    if change == "wrong_control_shape":
        rows[1]["actual"]["ctrl"] = [0.0] * 4
    report = verify_structure(rows, expected_steps=1)
    assert not report["structural_valid"]
    assert report["failures"]


def test_missing_final_steps_cannot_be_reported_complete():
    report = verify_structure(records(), expected_steps=2)
    assert not report["structural_valid"]
    assert any("expected" in row["reason"] for row in report["failures"])


def test_point_ledger_latches_exact_violations_and_their_locations():
    ledger = PointAuditLedger()
    clear = dict(
        min_clearance_m=1.0,
        closest_pair=[2, 7],
        floor_minima_m=[0.7, 0.6],
        contacts=[],
        wash_incursions=[],
        constraint_max_m=0.0,
        max_roll_pitch_deg=0.0,
        finite=True,
        warning_counts=[0] * 7,
        max_swing_deg=0.0,
        max_spin_deg_s=0.0,
    )
    bad = dict(
        clear,
        min_clearance_m=0.19,
        floor_minima_m=[0.09, 0.6],
        contacts=[{"geom1": 2, "geom2": 7}],
        wash_incursions=[{"rotor_geom": 3, "other_geom": 8}],
    )
    ledger.observe(bad, time=0.0003125, location="step:1/rk:2")
    ledger.observe(clear, time=0.01, location="step:1/actual")
    result = ledger.report()
    assert not result["retained_points_safe"]
    assert result["minimum_clearance_m"] == 0.19
    assert result["clearance_minimum"]["location"] == "step:1/rk:2"
    assert result["floor_minima_m"] == [0.09, 0.6]
    assert result["violation_count"] == 4
    assert result["audited_points"] == 2


def test_unknown_between_state_acceleration_never_becomes_a_certificate():
    result = conservative_interval_bound(
        clearance_a=1.0, clearance_b=0.9, dt=0.001, speed_bounds=(None, None)
    )
    assert result["qualified"] is False
    assert result["lower_bound_m"] is None
    certified = conservative_interval_bound(
        clearance_a=0.5, clearance_b=0.6, dt=0.01, speed_bounds=(2.0, 3.0)
    )
    assert certified["qualified"]
    assert certified["lower_bound_m"] == pytest.approx(0.475)


def write_probe(tmp_path, rows=None):
    root = tmp_path / "probe"
    root.mkdir()
    (root / "fixture").mkdir()
    (root / "fixture/model.xml").write_text("<mujoco/>")

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    properties = dict(
        schema="dual-native-properties-v1",
        profile="dual-payload-v1",
        version="3.15.0",
        precision="float64",
        integrator="RK4",
        substeps=16,
        model_xml_sha256=digest(root / "fixture/model.xml"),
    )
    (root / "properties.json").write_text(json.dumps(properties))
    with gzip.open(root / "native-control.jsonl.gz", "wt") as stream:
        for row in rows or records():
            stream.write(json.dumps(row) + "\n")
    result = dict(completed_control_steps=1, raw_sha256=digest(root / "native-control.jsonl.gz"))
    (root / "result.json").write_text(json.dumps(result))
    expected = {
        p: digest(root / p) for p in ("fixture/model.xml", "properties.json", "native-control.jsonl.gz")
    }
    return root, expected


def test_probe_binds_hashes_but_structural_only_never_claims_native_safety(tmp_path):
    root, expected = write_probe(tmp_path)
    report = audit_probe(root, expected_hashes=expected, native=False)
    assert report["source_binding_verified"]
    assert report["structural_valid"]
    assert report["native_replay_completed"] is False
    assert report["continuous_safety_qualified"] is False
    assert report["proof_gaps"]


def test_probe_rejects_model_or_properties_tampering(tmp_path):
    root, expected = write_probe(tmp_path)
    (root / "properties.json").write_text((root / "properties.json").read_text() + " ")
    report = audit_probe(root, expected_hashes=expected, native=False)
    assert not report["source_binding_verified"]
    assert any("properties.json" in failure["reason"] for failure in report["failures"])


def test_self_reported_hash_without_independent_inventory_is_explicitly_unbound(tmp_path):
    root, _ = write_probe(tmp_path)
    report = audit_probe(root, native=False)
    assert not report["source_binding_verified"]
    assert any("independent" in gap for gap in report["proof_gaps"])


def test_swept_reconstruction_catches_endpoint_clear_fast_crossing():
    from adaptive_drone_patrol.dual_coordination_audit import swept_reconstruction
    import numpy as np

    shapes = [dict(kind="sphere", size=(0.1, 0, 0), owner=i, rotor=False) for i in range(2)]
    a = dict(geom_xpos=[[-1, 0, 1], [1, 0, 1]], geom_xmat=[np.eye(3)] * 2)
    b = dict(geom_xpos=[[1, 0, 1], [-1, 0, 1]], geom_xmat=[np.eye(3)] * 2)

    def distance(positions, rotations, left, right):
        return np.linalg.norm(positions[left] - positions[right]) - 0.2

    result = swept_reconstruction(a, b, shapes=shapes, pair_distance=distance)
    assert result["status"] == "violation"
    assert result["minimum_sampled_clearance_m"] == pytest.approx(-0.2)
    assert result["witness_fraction"] == 0.5


def test_swept_reconstruction_catches_rotating_link_despite_clear_endpoints():
    from adaptive_drone_patrol.dual_coordination_audit import swept_reconstruction
    import numpy as np

    def rotation(angle):
        # Capsule longitudinal local z axis rotates around world y.
        c, s = np.cos(angle), np.sin(angle)
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]

    shapes = [
        dict(kind="capsule", size=(0.02, 1, 0), owner=0, rotor=False),
        dict(kind="sphere", size=(0.02, 0, 0), owner=1, rotor=False),
    ]
    a = dict(geom_xpos=[[0, 0, 2], [0.8, 0, 2]], geom_xmat=[rotation(np.pi / 4), np.eye(3)])
    b = dict(geom_xpos=[[0, 0, 2], [0.8, 0, 2]], geom_xmat=[rotation(3 * np.pi / 4), np.eye(3)])

    def distance(positions, rotations, left, right):
        axis = rotations[0][:, 2]
        delta = positions[1] - positions[0]
        return np.linalg.norm(delta - axis * np.clip(delta @ axis, -1, 1)) - 0.04

    result = swept_reconstruction(a, b, shapes=shapes, pair_distance=distance)
    assert result["status"] == "violation"
    assert result["minimum_sampled_clearance_m"] < 0


def test_swept_reconstruction_certifies_small_clear_motion_and_labels_contract():
    from adaptive_drone_patrol.dual_coordination_audit import swept_reconstruction
    import numpy as np

    shapes = [dict(kind="sphere", size=(0.1, 0, 0), owner=i, rotor=False) for i in range(2)]
    a = dict(geom_xpos=[[-2, 0, 1], [2, 0, 1]], geom_xmat=[np.eye(3)] * 2)
    b = dict(geom_xpos=[[-1.9, 0, 1], [1.9, 0, 1]], geom_xmat=[np.eye(3)] * 2)
    result = swept_reconstruction(
        a, b, shapes=shapes, pair_distance=lambda p, r, i, j: np.linalg.norm(p[i] - p[j]) - 0.2
    )
    assert result["status"] == "certified"
    assert type(result["lower_clearance_m"]) is float
    assert result["lower_clearance_m"] >= 0.2
    assert result["contract"] == "linear-centers-shortest-rotation-v1"


def test_native_replay_checks_real_probe_when_reference_engine_available(tmp_path):
    mj = pytest.importorskip("mujoco")
    if mj.__version__ != "3.15.0":
        pytest.skip("native reference version required")
    from pathlib import Path
    import os
    from adaptive_drone_patrol.dual_payload_mujoco import DualMuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload
    from adaptive_drone_patrol.payload_solver_mujoco import FREE_MASS_KG, HOVER_RPM
    from adaptive_drone_patrol.dual_coordination_audit import NativePointAuditor
    from adaptive_drone_patrol.dual_coordination_audit import OnlineNativeAudit
    import numpy as np

    original = Path(os.environ.get("DUAL_CF2X", "inputs/cf2x/cf2x.urdf"))
    if not original.exists():
        original = Path("outputs/dual_payload_coordination_20261027T144754Z/inputs/cf2x/cf2x.urdf")
    if not original.exists():
        pytest.skip("original fixture input required")
    fixture = DualMuJoCoFixture(original, tmp_path / "fixture", [Payload(FREE_MASS_KG, 0.2, 0.045)] * 2)
    auditor = NativePointAuditor(fixture.model_path, fixture.runtime())
    auditor.observe(dict(kind="RESET", actual=fixture.sample(), geometry=fixture.geometry_state()))
    assert auditor.data.ncon == 0 and auditor._contacts() == []
    saved_qpos = auditor.data.qpos.copy()
    auditor.data.qpos[auditor.addresses[0]['free_qpos']+2] = .01
    mj.mj_forward(auditor.model,auditor.data)
    assert auditor.data.ncon > 0
    expected_contacts = [dict(geom1=int(c.geom1),geom2=int(c.geom2),distance=float(c.dist))
                         for c in auditor.data.contact]
    assert auditor._contacts() == expected_contacts
    auditor.data.qpos[:] = saved_qpos
    mj.mj_forward(auditor.model,auditor.data)
    assert auditor.data.ncon == 0 and auditor._contacts() == []
    fixture.step_rpm(np.full((2, 4), HOVER_RPM), observe_stages=True)
    row = dict(
        kind="STEP", tick=1, actual=fixture.sample(), stages=fixture.stage_rows, substeps=fixture.substep_rows
    )
    auditor.observe(row)
    assert auditor.report()["audited_points"] == 84
    row["actual"]["drones"][0]["drone_velocity"][0] += 0.01
    with pytest.raises(ValueError):
        auditor.observe(row)
    fixture.reset()
    online = OnlineNativeAudit(fixture)
    fixture.step_rpm(np.full((2, 4), HOVER_RPM), observe_stages=True)
    online.observe_fixture_step(fixture, tick=1)
    saved = json.loads(json.dumps(online.to_dict()))
    resumed = OnlineNativeAudit(fixture)
    resumed.restore(saved)
    fixture.step_rpm(np.full((2, 4), HOVER_RPM), observe_stages=True)
    online.observe_fixture_step(fixture, tick=2)
    resumed.observe_fixture_step(fixture, tick=2)
    assert resumed.to_dict() == online.to_dict()
    assert resumed.report()["safe"]
    assert resumed.report()["continuous_safety_qualified"]
    assert not resumed.report()["full_physical_acceptance_qualified"]
    assert resumed.report()['proof_gaps'] == []
    assert resumed.report()['limitations']
    assert resumed.report()['body_reconstructed_intervals'] == 32
    assert resumed.report()['self_collision_pairs']
    assert resumed.report()['reconstructed_constraint_upper_bound_m'] <= 2e-5
    assert resumed.report()['reconstructed_tilt_upper_bound_deg'] <= 60.
    incomplete=json.loads(json.dumps(resumed.to_dict()))
    incomplete['body_intervals']-=1
    with pytest.raises(ValueError):
        OnlineNativeAudit(fixture).restore(incomplete)


def test_exhausted_sweep_subdivision_budget_is_unresolved_not_safe():
    from adaptive_drone_patrol.dual_coordination_audit import swept_reconstruction
    import numpy as np

    shapes = [dict(kind="sphere", size=(0.1, 0, 0), owner=i, rotor=False) for i in range(2)]
    a = dict(geom_xpos=[[-1, 0, 1], [1, 0, 1]], geom_xmat=[np.eye(3)] * 2)
    b = dict(geom_xpos=[[1, 0, 1], [-1, 0, 1]], geom_xmat=[np.eye(3)] * 2)
    result = swept_reconstruction(
        a, b, shapes=shapes, pair_distance=lambda p, r, i, j: np.linalg.norm(p[i] - p[j]) - 0.2, max_depth=0
    )
    assert result["status"] == "unresolved"
    assert result["unresolved_leaves"] == 1


def test_sweep_rejects_nonrotation_geometry_matrices():
    from adaptive_drone_patrol.dual_coordination_audit import swept_reconstruction
    import numpy as np

    shapes = [dict(kind="box", size=(0.1, 0.1, 0.1), owner=i, rotor=False) for i in range(2)]
    a = dict(geom_xpos=[[-2, 0, 1], [2, 0, 1]], geom_xmat=[np.zeros((3, 3))] * 2)
    with pytest.raises(ValueError):
        swept_reconstruction(a, a, shapes=shapes, pair_distance=lambda p, r, i, j: 3.8)


def test_sphere_broadphase_preserves_exact_minimum_and_skips_distant_pairs():
    from adaptive_drone_patrol.dual_coordination_audit import minimum_shape_distance
    import numpy as np

    positions = np.array([[0, 0, 1], [100, 0, 1], [1, 0, 1], [200, 0, 1]], float)
    shapes = [dict(kind="sphere", size=(0.1, 0, 0), owner=owner, rotor=False) for owner in (0, 0, 1, 1)]

    def distance(left, right):
        return float(np.linalg.norm(positions[left] - positions[right]) - 0.2)

    result = minimum_shape_distance(positions, shapes, distance)
    assert result["distance_m"] == pytest.approx(0.8)
    assert result["closest_pair"] == [0, 2]
    assert result["exact_evaluations"] == 1


def test_point_wash_fast_path_exactly_matches_stationary_swept_predicate():
    import numpy as np
    from adaptive_drone_patrol.dual_coordination_audit import _wash_cache, _wash_possible, _point_wash_possible

    shapes = [dict(kind='sphere',size=(.1,0,0),owner=0,rotor=True),
              dict(kind='box',size=(.05,.05,.05),owner=1,rotor=False),
              dict(kind='capsule',size=(.05,.1,0),owner=1,rotor=False)]
    cache = _wash_cache(shapes)
    radius = np.linalg.norm(shapes[1]['size'])
    coordinates = [[0.,0.,1.],[-0.,0.,0.], [.6+radius,0.,1.],
                   [np.nextafter(.6+radius,0),0.,1.],
                   [.2,0.,1.+radius],[.2,0.,-radius],[9.,9.,9.]]
    rng = np.random.default_rng(9127)
    coordinates.extend(rng.uniform(-2,2,(200,3)).tolist())
    for point in coordinates:
        positions = np.array([[0.,0.,1.],point,[0.3,0.1,.5]])
        # Distinct arrays force the general linear-interval path, even once the
        # implementation adds an identity-based stationary shortcut.
        expected = _wash_possible(positions,positions.copy(),shapes,cache=cache)
        assert _point_wash_possible(positions,cache) == expected
        assert _wash_possible(positions,positions,shapes,cache=cache) == expected


def _body_interval_row(base_rotation=None, package_position=(0.,0.,.9575), package_rotation=None):
    import numpy as np
    return dict(assembly_xpos=np.array([[[0.,0.,1.3],package_position]]),
                assembly_xmat=np.array([[np.eye(3) if base_rotation is None else base_rotation,
                                         np.eye(3) if package_rotation is None else package_rotation]]))


def test_self_collision_filter_matches_weld_parent_masks_and_explicit_exclusions():
    from adaptive_drone_patrol.dual_coordination_audit import self_collision_pairs
    shapes=[dict(owner=0)]*4+[dict(owner=1)]
    args=dict(shapes=shapes,geom_bodyid=[1,2,3,4,5],body_weldid=[0,1,1,3,4,5],
              body_parentid=[0,0,1,1,3,0],contype=[1]*5,conaffinity=[1]*5,exclusions=[(2,4)])
    assert self_collision_pairs(**args) == [(0,3)]
    assert self_collision_pairs(**args,filter_parent=False) == [(0,2),(0,3),(1,2),(2,3)]
    args['conaffinity']=[0]*5
    assert self_collision_pairs(**args) == []


def test_self_sweep_detects_between_endpoint_penetration_without_cross_margin_change():
    import numpy as np
    from adaptive_drone_patrol.dual_coordination_audit import swept_reconstruction
    shapes=[dict(kind='sphere',size=(.1,0,0),owner=i,rotor=False) for i in (0,0,1)]
    a=dict(geom_xpos=[[-1,0,1],[1,0,1],[10,0,1]],geom_xmat=[np.eye(3)]*3)
    b=dict(geom_xpos=[[1,0,1],[-1,0,1],[10,0,1]],geom_xmat=[np.eye(3)]*3)
    def distance(p,r,i,j):
        return np.linalg.norm(p[i]-p[j])-.2
    report=swept_reconstruction(a,b,shapes=shapes,pair_distance=distance,self_pairs=[(0,1)])
    assert report['status']=='violation'
    assert report['witness']['reason']=='self-contact'
    assert report['witness']['closest_pair']==[0,1]


def test_body_interval_certifies_small_attachment_motion_and_fails_unresolved_budget():
    import numpy as np
    from adaptive_drone_patrol.dual_coordination_audit import body_interval_certificate
    a=_body_interval_row()
    b=_body_interval_row()
    b['assembly_xpos'] += [1e-4,0,0]
    report=body_interval_certificate(a,b,package_offsets=[[0,0,.0225]])
    assert report['status']=='certified'
    assert report['constraint_upper_bound_m']<=2e-5
    assert report['tilt_upper_bound_deg']<=60.
    # Identical valid rod length at both endpoints, but their linear chord
    # shortens at the midpoint. The native endpoint predicate alone misses it.
    angle=.1
    a=_body_interval_row(package_position=(.3*np.sin(angle),0,1.28-.3*np.cos(angle)-.0225))
    b=_body_interval_row(package_position=(-.3*np.sin(angle),0,1.28-.3*np.cos(angle)-.0225))
    report=body_interval_certificate(a,b,package_offsets=[[0,0,.0225]])
    assert report['status']=='violation'
    assert 'constraint' in report['witness']['reason']
    limited=body_interval_certificate(a,b,package_offsets=[[0,0,.0225]],max_depth=0)
    assert limited['status'] in ('violation','unresolved')


def test_body_interval_catches_tilt_violation_between_safe_euler_endpoints():
    import math
    import numpy as np
    from adaptive_drone_patrol.dual_coordination_audit import body_interval_certificate
    def rotation(r,p,y):
        cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
        return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                         [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
    ra=rotation(-.806747494352217,.9520203375003908,2.6076569857458427)
    rb=rotation(.44387601455128256,.9704681868268568,-1.1595544752609923)
    a,b=_body_interval_row(ra),_body_interval_row(rb)
    for row,rot in ((a,ra),(b,rb)):
        row['assembly_xpos'][0,1]=row['assembly_xpos'][0,0]+rot@[0,0,-.02]-[0,0,.3225]
    report=body_interval_certificate(a,b,package_offsets=[[0,0,.0225]])
    assert report['status']=='violation'
    assert 'tilt' in report['witness']['reason']


def test_body_interval_budget_exhaustion_is_unresolved_even_when_samples_are_safe():
    import math
    import numpy as np
    from adaptive_drone_patrol.dual_coordination_audit import body_interval_certificate
    angle=.2
    rotation=np.array([[math.cos(angle),-math.sin(angle),0],
                       [math.sin(angle),math.cos(angle),0],[0,0,1]])
    a=_body_interval_row()
    b=_body_interval_row(rotation,package_rotation=rotation)
    limited=body_interval_certificate(a,b,package_offsets=[[0,0,.0225]],max_depth=0)
    assert limited['status']=='unresolved'
    assert limited['witness'] is None
    assert limited['unresolved_leaves']>0
    resolved=body_interval_certificate(a,b,package_offsets=[[0,0,.0225]])
    assert resolved['status']=='certified'
    assert resolved['checked_subintervals']>1
    assert resolved['constraint_upper_bound_m']<=2e-5


def test_self_interval_uses_zero_margin_while_cross_assembly_keeps_point_two():
    import numpy as np
    from adaptive_drone_patrol.dual_coordination_audit import swept_reconstruction
    shapes=[dict(kind='sphere',size=(.1,0,0),owner=i,rotor=False) for i in (0,0,1)]
    row=dict(geom_xpos=[[0,0,1],[.25,0,1],[10,0,1]],geom_xmat=[np.eye(3)]*3)
    def distance(p,r,i,j):
        return np.linalg.norm(p[i]-p[j])-.2
    report=swept_reconstruction(row,row,shapes=shapes,pair_distance=distance,self_pairs=[(0,1)])
    assert report['status']=='certified'
    assert 0<report['self_lower_clearance_m']<.2
    row['geom_xpos'][2]=[.6,0,1]
    report=swept_reconstruction(row,row,shapes=shapes,pair_distance=distance,self_pairs=[(0,1)])
    assert report['status']=='violation'
    assert report['witness']['reason']=='assembly margin'
