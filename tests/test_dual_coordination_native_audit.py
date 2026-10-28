"""Compiled shadow candidate tests; synthetic scenes never qualify production."""

import ctypes
import json
import os
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    from adaptive_drone_patrol.dual_coordination_native_audit import build_native_audit

    root = tmp_path_factory.mktemp("compiled-native-audit")
    sdk = Path("/private/tmp/dual-mujoco-315/mujoco.framework")
    if sdk.exists():
        return build_native_audit(
            root, headers=sdk / "Headers", library=sdk / "Versions/A/libmujoco.3.15.0.dylib"
        )
    pytest.importorskip("mujoco")
    return build_native_audit(root)


def synthetic_xml(path):
    """Same dimensions/names/actuator authority, deliberately synthetic shapes."""
    from adaptive_drone_patrol.payload_solver_mujoco import KF, KM, HOVER_RPM, SPIN

    bodies, motors, exclusions = [], [], []
    for drone in (1, 2):
        p = f"drone-{drone}/"
        props = []
        for i, (x, y) in enumerate(((0.06, 0.06), (0.06, -0.06), (-0.06, -0.06), (-0.06, 0.06))):
            props.append(
                f'<body name="{p}prop{i}_link" pos="{x} {y} 0"><geom name="{p}rotor{i}" type="cylinder" size=".023 .001" mass=".00025"/><site name="{p}motor{i}-com"/></body>'
            )
            motors.append(
                f'<motor name="{p}motor{i}" site="{p}motor{i}-com" gear="0 0 1 0 0 {KM / KF * SPIN[i]}" ctrllimited="true" ctrlrange="0 {KF * (1.8 * HOVER_RPM) ** 2}"/>'
            )
            exclusions.append(f'<exclude body1="{p}base_link" body2="{p}prop{i}_link"/>')
        bodies.append(f'''<body name="{p}base_link" pos="{(-2 if drone == 1 else 2)} 0 2">
          <joint type="free" name="{p}vehicle_free"/><geom name="{p}base" type="box" size=".025 .025 .01" mass=".027"/>
          {"".join(props)}<body name="{p}suspension_rod" pos="0 0 -.020"><joint name="{p}suspension_ball" type="ball"/>
          <geom name="{p}link" type="capsule" fromto="0 0 0 0 0 -.30" size=".001" mass=".00001"/>
          <body name="{p}payload" pos="0 0 -.315"><joint name="{p}payload_ball" type="ball"/>
          <geom name="{p}package" type="box" size=".015 .015 .015" mass=".005"/></body></body></body>''')
        exclusions.extend(
            f'<exclude body1="{p}{a}" body2="{p}{b}"/>'
            for a, b in [("base_link", "suspension_rod"), ("suspension_rod", "payload")]
        )
    path.write_text(
        f'<mujoco><option timestep=".000625" integrator="RK4" tolerance="1e-12" iterations="100"/><worldbody><geom name="ground" type="plane" size="20 20 .1"/>{"".join(bodies)}</worldbody><actuator>{"".join(motors)}</actuator><contact>{"".join(exclusions)}</contact></mujoco>'
    )
    return path


def test_build_receipt_records_dependencies_and_candidate_scope(compiled):
    receipt = json.loads(Path(compiled).with_suffix(".json").read_text())
    assert receipt["qualification"] == "UNQUALIFIED_SHADOW_ONLY"
    assert receipt["mujoco_version"] == "3.15.0"
    assert receipt["source"]["sha256"] and receipt["headers"] and receipt["library"]["sha256"]
    assert receipt["compiler"]["version"] and receipt["compiler"]["sha256"]
    assert receipt["dependencies"]
    assert Path(receipt["source"]["path"]).parent == Path(compiled).parent


def test_compiled_step_counts_state_and_safety_are_reproducible(tmp_path, compiled):
    from adaptive_drone_patrol.dual_coordination_native_audit import CompiledAuditShadow

    path = synthetic_xml(tmp_path / "synthetic.xml")
    with (
        CompiledAuditShadow.from_xml(path, compiled) as left,
        CompiledAuditShadow.from_xml(path, compiled) as right,
    ):
        original = left.inspect()
        a = left.step_rpm(np.full((2, 4), 14000.0))
        b = right.step_rpm(np.full((2, 4), 14000.0))
        assert len(a["rk_points"]) == 64 and len(a["boundaries"]) == 17
        assert a["boundaries"][-1]["world_time"] == pytest.approx(0.01)
        assert np.allclose(a["boundaries"][-1]["qpos"], b["boundaries"][-1]["qpos"], atol=0, rtol=0)
        assert all(p["flags"] == 0 for p in a["rk_points"])
        assert original["min_clearance_m"] > 0.2
        assert a["production_qualified"] is False
        assert a["scratch_forward_recomputes_acceleration"] is True
        assert all(p["live_acceleration_finite"] for p in a["rk_points"] + a["boundaries"])
        assert np.asarray(a["boundaries"][0]["geom_xpos"]).shape == (left.ngeom, 3)


def test_compiled_point_detects_cross_contact_floor_and_authority(tmp_path, compiled):
    from adaptive_drone_patrol.dual_coordination_native_audit import CompiledAuditShadow

    with CompiledAuditShadow.from_xml(synthetic_xml(tmp_path / "synthetic.xml"), compiled) as shadow:
        initial = shadow.inspect()
        qpos = np.asarray(initial["qpos"])
        qpos[15:18] = qpos[:3] + [0.01, 0, 0]
        shadow.reset_state(qpos, np.zeros(24))
        point = shadow.inspect()
        assert point["flags"] & 2  # assembly margin
        assert point["contacts"] > 0
        qpos[2] = qpos[17] = 0.2
        shadow.reset_state(qpos, np.zeros(24))
        assert shadow.inspect()["flags"] & 4  # floor
        before = shadow.inspect()["qpos"]
        with pytest.raises(ValueError):
            shadow.step_rpm(np.full((2, 4), 30000.0))
        assert shadow.inspect()["qpos"] == before


def test_compiled_rejects_changed_authority_and_cross_exclusion(tmp_path, compiled):
    from adaptive_drone_patrol.dual_coordination_native_audit import CompiledAuditShadow

    path = synthetic_xml(tmp_path / "synthetic.xml")
    path.write_text(path.read_text().replace('gear="0 0 1', 'gear="0 0 2', 1))
    with pytest.raises(ValueError, match="authority"):
        CompiledAuditShadow.from_xml(path, compiled)
    synthetic_xml(path)
    path.write_text(
        path.read_text().replace(
            "<contact>", '<contact><exclude body1="drone-1/base_link" body2="drone-2/base_link"/>'
        )
    )
    with pytest.raises(ValueError, match="exclusion"):
        CompiledAuditShadow.from_xml(path, compiled)


def test_compiled_clone_matches_bare_c_api_steps_and_preserves_original(tmp_path, compiled):
    from adaptive_drone_patrol.dual_coordination_native_audit import CompiledAuditShadow
    from adaptive_drone_patrol.payload_solver_mujoco import KF

    receipt = json.loads(Path(compiled).with_suffix(".json").read_text())
    lib = ctypes.PyDLL(receipt["library"]["path"])
    p, numbers = ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)
    signatures = dict(
        mj_loadXML=([ctypes.c_char_p, p, ctypes.c_char_p, ctypes.c_int], p),
        mj_makeData=([p], p),
        mj_deleteData=([p], None),
        mj_deleteModel=([p], None),
        mj_forward=([p, p], None),
        mj_step=([p, p], None),
        mj_getState=([p, p, numbers, ctypes.c_int], None),
        mj_setState=([p, p, numbers, ctypes.c_int], None),
    )
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    msg = ctypes.create_string_buffer(2048)
    model = lib.mj_loadXML(str(synthetic_xml(tmp_path / "synthetic.xml")).encode(), None, msg, 2048)
    assert model, msg.value
    data = lib.mj_makeData(model)

    def state():
        row = np.zeros(55)
        lib.mj_getState(model, data, row.ctypes.data_as(numbers), 7)  # TIME|QPOS|QVEL in fixed3.15 header
        return row

    shadow = None
    try:
        lib.mj_forward(model, data)
        shadow = CompiledAuditShadow(compiled)
        shadow.handle = shadow.lib.da_create(model, data)
        shadow._initialize()
        for rpm in (np.full((2, 4), 14000.0), np.arange(8).reshape(2, 4) * 500 + 12000.0, np.zeros((2, 4))):
            before = state()
            result = shadow.step_rpm(rpm)
            assert np.array_equal(before, state())
            ctrl = np.ascontiguousarray(KF * rpm.reshape(8) ** 2)
            lib.mj_setState(model, data, ctrl.ctypes.data_as(numbers), 1 << 6)
            for i in range(16):
                lib.mj_step(model, data)
                lib.mj_forward(model, data)
                expected = state()
                actual = result["boundaries"][i + 1]
                assert np.allclose(
                    expected, np.r_[actual["world_time"], actual["qpos"], actual["qvel"]], rtol=0, atol=1e-12
                )
            lib.mj_forward(model, data)
        callback = ctypes.CFUNCTYPE(None, p, p)(lambda _m, _d: None)
        current = ctypes.c_void_p.in_dll(lib, "mjcb_control")
        current.value = ctypes.cast(callback, p).value
        try:
            with pytest.raises(ValueError, match="callback"):
                shadow.step_rpm(np.zeros((2, 4)))
            assert current.value == ctypes.cast(callback, p).value
        finally:
            current.value = None
        # Explicit candidate publication copies the complete native integration
        # state only after all point/interval audits, never by qpos/qvel alone.
        from types import SimpleNamespace

        target = SimpleNamespace(
            model=SimpleNamespace(_address=model),
            data=SimpleNamespace(_address=data),
            mj=SimpleNamespace(__version__="3.15.0"),
            rpm=np.zeros((2, 4)),
            audit_geom_ids=[[i for i, s in enumerate(shadow.shapes) if s["owner"] == d] for d in (0, 1)],
        )
        with CompiledAuditShadow.from_fixture(target, compiled) as publisher:
            publisher.step_audited(np.full((2, 4), 14000.0))
            publisher.copy_to_fixture(target)
            lib.mj_stateSize.argtypes, lib.mj_stateSize.restype = [p, ctypes.c_int], ctypes.c_int
            full = np.zeros(lib.mj_stateSize(model, 16383))
            lib.mj_getState(model, data, full.ctypes.data_as(numbers), 16383)
            assert np.array_equal(full, np.asarray(publisher.snapshot()["state"]))
            assert np.array_equal(target.rpm, np.full((2, 4), 14000.0))
            changed_model_path = synthetic_xml(tmp_path / "changed-mass.xml")
            changed_model_path.write_text(
                changed_model_path.read_text().replace('mass=".027"', 'mass=".028"', 1)
            )
            changed_model = lib.mj_loadXML(str(changed_model_path).encode(), None, msg, 2048)
            assert changed_model, msg.value
            changed_data = lib.mj_makeData(changed_model)
            try:
                target.model._address, target.data._address = changed_model, changed_data
                with pytest.raises(ValueError, match="model"):
                    publisher.copy_to_fixture(target)
            finally:
                lib.mj_deleteData(changed_data)
                lib.mj_deleteModel(changed_model)
    finally:
        if shadow is not None:
            shadow.close()
        lib.mj_deleteData(data)
        lib.mj_deleteModel(model)


def test_native_fixture_shadow_matches_python_auditor(tmp_path, compiled):
    pytest.importorskip(
        "mujoco", reason="Real fixture/Python shadow comparison requires MuJoCo3.15 Python package"
    )
    from adaptive_drone_patrol.dual_coordination_native_audit import (
        CompiledAuditShadow,
        compare_python_shadow,
    )
    from adaptive_drone_patrol.dual_payload_mujoco import DualMuJoCoFixture
    from adaptive_drone_patrol.payload_contract import Payload
    from adaptive_drone_patrol.payload_solver_mujoco import FREE_MASS_KG

    original = Path(
        os.environ.get(
            "DUAL_CF2X", "outputs/dual_payload_coordination_20261027T144754Z/inputs/cf2x/cf2x.urdf"
        )
    )
    if not original.is_file():
        pytest.skip("Original cf2x assets unavailable")
    f = DualMuJoCoFixture(
        original, tmp_path / "fixture", [Payload(FREE_MASS_KG, 0.1, 0.03), Payload(FREE_MASS_KG, 0.3, 0.06)]
    )
    with CompiledAuditShadow.from_fixture(f, compiled) as shadow:
        receipt = compare_python_shadow(f, shadow, np.full((2, 4), 14000.0))
        assert receipt["matched"] is True
        assert receipt["rk_points"] == 64 and receipt["native_intervals"] == 16
        assert receipt["production_qualified"] is False
    with (
        CompiledAuditShadow.from_fixture(f, compiled) as slow,
        CompiledAuditShadow.from_fixture(f, compiled) as fast,
        CompiledAuditShadow.from_fixture(f, compiled) as resumed,
    ):
        command = np.arange(8).reshape(2, 4) * 100 + 14000.0
        original = slow.step_audited(command, fast_path=False)
        candidate = fast.step_audited(command, fast_path=True)
        assert original["rk_points"] == candidate["rk_points"]
        assert original["boundaries"] == candidate["boundaries"]
        assert (
            original["interval_audit"]["certified_intervals"]
            == candidate["interval_audit"]["certified_intervals"]
        )
        for a, b in zip(original["interval_audit"]["reports"], candidate["interval_audit"]["reports"]):
            assert a["sweep"]["status"] == b["sweep"]["status"]
            assert a["body"]["status"] == b["body"]["status"]
        assert slow.snapshot()["state"] == fast.snapshot()["state"]
        resumed.restore(fast.snapshot())
        continued = fast.step_audited(command)
        recovered = resumed.step_audited(command)
        assert continued["rk_points"] == recovered["rk_points"]
        assert continued["boundaries"] == recovered["boundaries"]
        fast.copy_to_fixture(f)
        state_spec = f.mj.mjtState.mjSTATE_INTEGRATION
        native_state = np.zeros(f.mj.mj_stateSize(f.model, state_spec))
        f.mj.mj_getState(f.model, f.data, native_state, state_spec)
        assert np.array_equal(native_state, fast.snapshot()["state"])
    f.data.qacc[0] = float("nan")
    with CompiledAuditShadow.from_fixture(f, compiled) as shadow:
        point = shadow.inspect()
        assert point["live_acceleration_finite"] is False
        assert point["finite"] is False and point["flags"] & 1


def test_sdk_shadow_supplies_latest_python_interval_audit(tmp_path, compiled):
    from adaptive_drone_patrol.dual_coordination_native_audit import CompiledAuditShadow

    with CompiledAuditShadow.from_xml(synthetic_xml(tmp_path / "synthetic.xml"), compiled) as shadow:
        assert hasattr(shadow, "audit_intervals"), "SDK interval comparison API missing"
        result = shadow.step_rpm(np.full((2, 4), 14000.0))
        assert np.asarray(result["boundaries"][0]["assembly_xpos"]).shape == (2, 2, 3)
        audit = shadow.audit_intervals(result, fast_path=False)
        assert audit["native_intervals"] == 16
        assert audit["body_intervals"] == 16
        assert audit["certified_intervals"] == 16
        assert audit["python_fallback_intervals"] == 16
        assert audit["production_qualified"] is False


def test_conservative_fast_intervals_match_python_or_fall_back(tmp_path, compiled):
    from adaptive_drone_patrol.dual_coordination_native_audit import CompiledAuditShadow

    with CompiledAuditShadow.from_xml(synthetic_xml(tmp_path / "synthetic.xml"), compiled) as shadow:
        assert hasattr(shadow, "fast_interval"), "Conservative compiled interval kernel missing"
        result = shadow.step_rpm(np.arange(8).reshape(2, 4) * 200 + 14000.0)
        original = shadow.audit_intervals(result, fast_path=False)
        fast = shadow.audit_intervals(result, fast_path=True)
        assert fast["certified_intervals"] == original["certified_intervals"] == 16
        assert fast["python_fallback_intervals"] == 0
        for a, b in zip(original["reports"], fast["reports"]):
            for key in ("lower_clearance_m", "self_lower_clearance_m"):
                assert b["sweep"][key] == pytest.approx(a["sweep"][key], abs=1e-10)
            for key in ("constraint_upper_bound_m", "tilt_upper_bound_deg"):
                assert b["body"][key] == pytest.approx(a["body"][key], abs=1e-8)
        # Endpoint-safe cross-through cannot be approved by a root bound.
        import copy

        crossing = copy.deepcopy(result)
        for row in crossing["boundaries"]:
            for g, shape in enumerate(shadow.shapes):
                if shape["owner"] == 1:
                    row["geom_xpos"][g][0] -= 8.0 * row["world_time"] / 0.01
            for body in row["assembly_xpos"][1]:
                body[0] -= 8.0 * row["world_time"] / 0.01
        slow_cross = shadow.audit_intervals(crossing, fast_path=False)
        fast_cross = shadow.audit_intervals(crossing, fast_path=True)
        assert fast_cross["python_fallback_intervals"] > 0
        assert fast_cross["certified_intervals"] == slow_cross["certified_intervals"] < 16
        assert [r["sweep"]["status"] for r in fast_cross["reports"]] == [
            r["sweep"]["status"] for r in slow_cross["reports"]
        ]


def test_candidate_checkpoint_restores_exact_native_continuation_and_audit(tmp_path, compiled):
    from adaptive_drone_patrol.dual_coordination_native_audit import CompiledAuditShadow

    path = synthetic_xml(tmp_path / "synthetic.xml")
    with (
        CompiledAuditShadow.from_xml(path, compiled) as left,
        CompiledAuditShadow.from_xml(path, compiled) as right,
    ):
        assert hasattr(left, "snapshot"), "Full native integration checkpoint missing"
        left.step_audited(np.full((2, 4), 14000.0))
        saved = left.snapshot()
        right.restore(saved)
        command = np.arange(8).reshape(2, 4) * 300 + 13000.0
        a, b = left.step_audited(command), right.step_audited(command)
        assert a["boundaries"] == b["boundaries"]
        assert a["rk_points"] == b["rk_points"]
        assert a["interval_audit"]["certified_intervals"] == 16
        assert b["interval_audit"]["certified_intervals"] == 16
        assert left.snapshot() == right.snapshot()
        import copy

        changed = copy.deepcopy(saved)
        changed["model_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="model"):
            right.restore(changed)
        changed = copy.deepcopy(saved)
        changed["audit"]["native_intervals"] -= 1
        with pytest.raises(ValueError, match="audit"):
            right.restore(changed)
        changed = copy.deepcopy(saved)
        changed["last_rpm"][0][0] = 0.0
        before = right.snapshot()
        with pytest.raises(ValueError, match="RPM"):
            right.restore(changed)
        assert right.snapshot() == before


def test_fast_path_near_floor_threshold_uses_exact_python_fallback(tmp_path, compiled):
    """Stricter numerical slack may cause fallback, never relax the floor limit."""
    import copy
    from adaptive_drone_patrol.dual_coordination_native_audit import CompiledAuditShadow

    with CompiledAuditShadow.from_xml(synthetic_xml(tmp_path / "synthetic.xml"), compiled) as shadow:
        first = shadow.step_rpm(np.zeros((2, 4)))["boundaries"][0]
        box = next(i for i, s in enumerate(shadow.shapes) if s["owner"] == 0 and s["kind"] == "box")
        first["geom_xpos"][box][2] = shadow.shapes[box]["size"][2] + 0.1 + 1e-11
        rows = []
        for i in range(17):
            row = copy.deepcopy(first)
            row["world_time"] = i * 0.01 / 16
            rows.append(row)
        slow = shadow.audit_intervals(dict(boundaries=rows), fast_path=False)
        fast = shadow.audit_intervals(dict(boundaries=rows), fast_path=True)
        assert slow["certified_intervals"] == fast["certified_intervals"] == 16
        assert fast["python_fallback_intervals"] == 16
