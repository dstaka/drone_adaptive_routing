"""Prospective qualification software tests; no synthetic physics pass receipt."""

import importlib
import json

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module("adaptive_drone_patrol.dual_coordination_qualify")
    except ModuleNotFoundError:
        pytest.fail("Native qualification scenario driver is not implemented")


def test_declaration_preserves_all_seven_330s_and_36_mixed_cases():
    m = module()
    cases = m.qualification_cases()
    duration = [c for c in cases if c["group"] == "duration"]
    assert len(duration) == 7 and all(c["duration_s"] == 330 for c in duration)
    assert {(c["sides"][0], c["ratios"][0]) for c in duration} == {
        (0.045, 0.2),
        (0.03, 0.1),
        (0.03, 0.3),
        (0.045, 0.1),
        (0.045, 0.3),
        (0.06, 0.1),
        (0.06, 0.3),
    }
    mixed = [c for c in cases if c["group"] == "mixed"]
    assert len(mixed) == 36
    for kind in ("hover", "vertical", "crossing", "brake"):
        rows = [c for c in mixed if c["kind"] == kind]
        assert len({tuple(c["sides"]) for c in rows}) == 9
        assert all(c["duration_s"] == 30 for c in rows)
    passive = [c for c in cases if c["group"] == "passive"]
    assert len(passive) == 32
    assert sum(c["duration_s"] == 120 for c in passive) == 9
    refinement = [c for c in cases if c["group"] == "refinement"]
    assert len(refinement) == 6 and all(c["substeps_pair"] == [16, 32] for c in refinement)
    assert len({c["case_id"] for c in cases}) == len(cases)


def test_passive_loss_is_failure_even_without_positive_energy_growth():
    m = module()
    result = m.passive_energy_report([1.0, 0.99, 0.87])
    assert result["passed"] is False
    assert result["maximum_absolute_energy_fraction"] == pytest.approx(0.13)
    assert m.passive_energy_report([1.0, 1.05, 0.95])["passed"] is True
    with pytest.raises(ValueError):
        m.passive_energy_report([0.0, 0.0])


def test_refinement_checks_every_body_and_ignores_quaternion_sign():
    m = module()

    def row():
        return {
            "drones": [
                dict(
                    drone_xyz=[0.0, 0.0, 1.0],
                    package_xyz=[0.0, 0.0, 0.6],
                    rod_xyz=[0.0, 0.0, 0.98],
                    drone_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
                    package_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
                    rod_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
                )
                for _ in range(2)
            ]
        }

    a, b = row(), row()
    b["drones"][1]["package_quaternion_wxyz"] = [-1.0, 0.0, 0.0, 0.0]
    assert m.refinement_difference(a, b)["within_limits"]
    b["drones"][1]["package_xyz"][0] = 0.021
    assert not m.refinement_difference(a, b)["within_limits"]
    b = row()
    b["drones"][0]["rod_quaternion_wxyz"] = [np.cos(np.deg2rad(1.1)), np.sin(np.deg2rad(1.1)), 0.0, 0.0]
    assert not m.refinement_difference(a, b)["within_limits"]


def test_anchored_variant_removes_only_free_joints_and_preserves_other_physics(tmp_path):
    m = module()
    original = tmp_path / "model.xml"
    original.write_text(
        '<mujoco><worldbody><body name="drone-1/base_link"><freejoint name="drone-1/vehicle_free"/><inertial mass="1"/><body name="rod1"><joint type="ball"/></body></body><body name="drone-2/base_link"><freejoint name="drone-2/vehicle_free"/><inertial mass="2"/><body name="rod2"><joint type="ball"/></body></body></worldbody></mujoco>'
    )
    target = tmp_path / "anchored.xml"
    m.write_anchored_variant(original, target)
    import xml.etree.ElementTree as ET

    tree = ET.parse(target)
    assert not tree.findall(".//freejoint")
    assert [n.get("mass") for n in tree.findall(".//inertial")] == ["1", "2"]
    assert len(tree.findall(".//joint")) == 2
    assert len(ET.parse(original).findall(".//freejoint")) == 2


def test_invalid_or_overwritten_run_rejected_before_native_construction(tmp_path):
    m = module()
    case = m.qualification_cases()[0]
    with pytest.raises(ValueError):
        m.run_case(original="missing", design={}, output=tmp_path / "new", case=case, pilot_duration_s=331)
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        m.run_case(original="missing", design={}, output=existing, case=case, pilot_duration_s=0.01)
    assert not (tmp_path / "new").exists()


def test_setup_failure_is_persisted_as_failure_not_qualification(tmp_path):
    m = module()
    output = tmp_path / "failed"
    with pytest.raises((FileNotFoundError, ImportError, ValueError)):
        m.run_case(
            original=tmp_path / "missing.xml",
            design={},
            output=output,
            case=m.qualification_cases()[0],
            pilot_duration_s=0.01,
        )
    result = json.loads((output / "result.json").read_text())
    assert result["passed"] is False and result["admission_granted"] is False
    assert result["pilot"] is True and result["failure"]


def test_qualification_has_no_mission_wall_cap_but_watchdog_requires_native_progress():
    m = module()
    import time

    with pytest.raises(TimeoutError, match="native progress"):
        with m.NativeProgressWatchdog(0.03):
            time.sleep(0.15)
    with m.NativeProgressWatchdog(0.08) as watchdog:
        for _ in range(3):
            time.sleep(0.03)
            watchdog.progress()


def test_refinement_rejects_missing_drone_poses():
    m = module()
    with pytest.raises(ValueError):
        m.refinement_difference({"drones": []}, {"drones": []})


def test_numerical_native_failure_is_latched_for_failure_diagnostics():
    from types import SimpleNamespace

    m = module()
    monitor = m.NumericalMonitor.__new__(m.NumericalMonitor)
    monitor.f = SimpleNamespace(data=None)
    monitor.failures = []

    def fail(data):
        raise ValueError("Native attachment constraint residual exceeded2e-5m")

    monitor._check = fail
    with pytest.raises(ValueError, match="constraint"):
        monitor.native()
    assert monitor.failures == ["Native attachment constraint residual exceeded2e-5m"]


def test_special_probe_properties_do_not_claim_standard_free_16_profile():
    from types import SimpleNamespace

    m = module()
    fixture = SimpleNamespace(
        runtime=lambda: {"profile": "dual-payload-v1", "substeps": 16, "addresses": [{"free_qpos": 0}]},
        model=SimpleNamespace(nq=16, nv=12, opt=SimpleNamespace(timestep=0.01 / 32)),
    )
    actual = m.probe_runtime(fixture, substeps=32, anchored=True)
    assert actual["profile"] == "dual-anchored-passive-probe-v1"
    assert actual["substeps"] == 32 and actual["nq"] == 16 and actual["nv"] == 12
    assert "addresses" not in actual
    assert actual["timestep_s"] == 0.01 / 32


def test_crossing_v2_is_explicit_perpendicular_qualification_only():
    m = module()
    crossing = [c for c in m.qualification_cases() if c["kind"] == "crossing"]
    assert len(crossing) == 9
    for case in crossing:
        assert case["layout"] == "perpendicular-qualification-v2"
        assert case["starts"] == [[-2.0, 0.0, 3.0], [0.0, -2.0, 3.0]]
        assert case["targets"] == [[2.0, 0.0, 3.0], [0.0, 2.0, 3.0]]
        assert [r["position"] for r in m.qualification_initials(case)] == case["starts"]
    assert all(
        m.qualification_initials(c) is None for c in m.qualification_cases() if c["kind"] != "crossing"
    )


def test_crossing_resume_preserves_goal_and_c2_at_hold_release():
    m = module()
    starts = [[-2.0, 0.0, 3.0], [0.0, -2.0, 3.0]]
    targets = [[2.0, 0.0, 3.0], [0.0, 2.0, 3.0]]
    refs = [
        m.ContinuousReference(p).retarget(q, 0.0, actual_state=m.KinematicState(p, [0.0, 0.0, 0.0]))
        for p, q in zip(starts, targets)
    ]
    supervisor = m.ReferenceSupervisor()
    states = [m.KinematicState(*r.sample(0.0)[:2]) for r in refs]
    held = supervisor.decide(refs, states, 0.0, priority=1)
    assert held.held == (False, True) and held.failure is None
    now = max(refs[0].end_time, held.references[1].end_time) + 0.01
    states = [m.KinematicState(*r.sample(now)[:2]) for r in held.references]
    requested = m.crossing_requests(held.references, targets, held.held, states, now)
    assert requested[0] is held.references[0]
    assert list(requested[1].target) == targets[1]
    for old, new in zip(held.references[1].sample(now), requested[1].sample(now)):
        np.testing.assert_allclose(new, old, atol=1e-13, rtol=0)
    released = supervisor.decide(requested, states, now, priority=1)
    assert released.held == (False, False) and released.failure is None
    assert list(released.references[1].target) == targets[1]
    np.testing.assert_allclose(released.references[1].sample(released.references[1].end_time)[0], targets[1])


def test_crossing_resume_from_moving_brake_preserves_acceleration():
    m = module()
    moving = m.ContinuousReference([0.0, -2.0, 3.0]).retarget(
        [0.0, 2.0, 3.0], 0.0, actual_state=m.KinematicState([0.0, -2.0, 3.0], [0.0, 0.0, 0.0])
    )
    state = m.KinematicState(*moving.sample(0.4)[:2])
    braking = moving.brake(0.4, actual_state=state)
    now = 0.5
    other = m.ContinuousReference([2.0, 0.0, 3.0])
    refs = [other, braking]
    states = [m.KinematicState(*r.sample(now)[:2]) for r in refs]
    requested = m.crossing_requests(refs, [[2.0, 0.0, 3.0], [0.0, 2.0, 3.0]], (False, True), states, now)
    assert np.linalg.norm(braking.sample(now)[2]) > 0
    for before, after in zip(braking.sample(now), requested[1].sample(now)):
        np.testing.assert_allclose(after, before, atol=1e-12, rtol=0)


def test_v2_declaration_is_prospective_and_refuses_overwrite(tmp_path):
    m = module()
    target = tmp_path / "v2"
    assert m.main(["--declare-only", "--output", str(target)]) == 0
    declaration = json.loads((target / "declaration.json").read_text())
    assert declaration["schema"] == "dual-native-qualification-declaration-v2"
    assert declaration["admission_granted"] is False
    assert declaration["cases"] == m.qualification_cases()
    before = (target / "declaration.json").read_bytes()
    with pytest.raises(FileExistsError):
        m.main(["--declare-only", "--output", str(target)])
    assert (target / "declaration.json").read_bytes() == before


def test_native_crossing_reset_and_reference_match_declared_v2(tmp_path):
    import os
    from pathlib import Path

    pytest.importorskip("mujoco", reason="Actual native MuJoCo required; no synthetic physics")
    m = module()
    original = Path(os.environ.get("DUAL_CF2X", "inputs/cf2x/cf2x.urdf"))
    if not original.is_file():
        pytest.skip("Original CF2X native asset unavailable")
    case = next(c for c in m.qualification_cases() if c["kind"] == "crossing")
    payloads = [m.Payload(m.FREE_MASS_KG, r, s) for r, s in zip(case["ratios"], case["sides"])]
    fixture = m.DualMuJoCoFixture(original, tmp_path / "fixture", payloads)
    fixture.reset(m.qualification_initials(case))
    rows = fixture.sample()["drones"]
    refs = m._references(fixture, case)
    for row, ref, start, target in zip(rows, refs, case["starts"], case["targets"]):
        np.testing.assert_array_equal(row["drone_xyz"], start)
        np.testing.assert_array_equal(row["drone_velocity"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(ref.sample(0.0)[0], start)
        np.testing.assert_array_equal(ref.target, target)
    assert m.OnlineNativeAudit(fixture).report()["safe"]


def test_source_freeze_covers_lazy_step_dependencies_without_unimported_modules():
    import os
    from pathlib import Path
    import subprocess
    import sys

    # The contract applies to a dedicated qualifier process. In a full test
    # session other tests legitimately load train/evaluate/media first; the
    # production freeze must continue to bind those already-imported modules.
    source_root = Path(module().__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(source_root), PYTHONDONTWRITEBYTECODE="1")
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from adaptive_drone_patrol import dual_coordination_qualify as m
frozen = m.qualification_source_freeze()
sources = frozen["source_hashes"]
for name in ("env", "audit", "compiled_audit", "native_audit"):
    assert f"src/adaptive_drone_patrol/dual_coordination_{name}.py" in sources
assert "src/adaptive_drone_patrol/payload_hybrid_controller.py" in sources
for name in ("train", "evaluate", "media"):
    assert f"src/adaptive_drone_patrol/dual_coordination_{name}.py" not in sources
assert "numpy" in frozen["dependency_versions"]
assert m.verify_qualification_source_freeze(frozen) == frozen
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_source_freeze_detects_changed_imported_source_and_late_repo_import(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    m = module()
    # Isolated fake repository exercises sys.modules/file filtering without
    # modifying the shared checkout used by other agents.
    root = tmp_path / "repo"
    src = root / "src"
    src.mkdir(parents=True)
    source = src / "executed.py"
    source.write_text("answer=42\n")
    outside = tmp_path / "outside.py"
    outside.write_text("answer=1\n")
    executed = ModuleType("qualified_executed")
    executed.__file__ = str(source)
    ignored = ModuleType("qualified_outside")
    ignored.__file__ = str(outside)
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setitem(sys.modules, "qualified_executed", executed)
    monkeypatch.setitem(sys.modules, "qualified_outside", ignored)
    frozen = m.qualification_source_freeze()
    assert set(frozen["source_hashes"]) == {"src/executed.py"}
    source.write_text("answer=43\n")
    with pytest.raises(ValueError, match="source"):
        m.verify_qualification_source_freeze(frozen)
    source.write_text("answer=42\n")
    late = src / "late.py"
    late.write_text("answer=0\n")
    late_module = ModuleType("qualified_late")
    late_module.__file__ = str(late)
    monkeypatch.setitem(sys.modules, "qualified_late", late_module)
    with pytest.raises(ValueError, match="source"):
        m.verify_qualification_source_freeze(frozen)


def batch_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/run_dual_qualification_batch.py"
    assert path.is_file(), "Finite qualification batch runner missing"
    spec = importlib.util.spec_from_file_location("qualification_batch_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def stub_batch(tmp_path, monkeypatch, *, first_fails=False):
    import sys

    m = batch_module()
    cases = module().qualification_cases()[:3]
    monkeypatch.setattr(m, "qualification_cases", lambda: cases)
    original = tmp_path / "original.urdf"
    original.write_text("<robot/>")
    design = tmp_path / "design.json"
    design.write_text("{}")
    child = tmp_path / "case_child.py"
    child.write_text("""import json,sys,time
from pathlib import Path
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=False)
case=json.loads(sys.argv[2]); frozen=json.loads(Path(sys.argv[3]).read_text())
time.sleep(float(sys.argv[4]))
passed=sys.argv[5]=='pass'
(out/'inputs.json').write_text(json.dumps(dict(source_hashes=frozen['source_hashes'],dependency_versions=frozen['dependency_versions'])))
(out/'result.json').write_text(json.dumps(dict(schema='dual-qualification-case-v1',case=case,pilot=False,admission_granted=False,passed=passed,completed_declared_duration=passed,completed_control_steps=round(case['duration_s']*100),failure=None if passed else 'deliberate synthetic failure',source_integrity_passed=True,source_hashes_end=frozen['source_hashes'],dependency_versions_end=frozen['dependency_versions'])))
print('synthetic child completed',case['case_id'],flush=True)
raise SystemExit(0 if passed else 1)
""")

    def command(*, case, output, source_freeze, **kwargs):
        i = cases.index(case)
        return [
            sys.executable,
            str(child),
            str(output),
            json.dumps(case),
            str(source_freeze),
            ".20" if i == 0 else ".50",
            "fail" if i == 0 and first_fails else "pass",
        ]

    preflight_child = tmp_path / "preflight_child.py"
    preflight_child.write_text("""import json,sys
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
case=json.loads(sys.argv[2]);frozen=json.loads(Path(sys.argv[3]).read_text())
(out/'inputs.json').write_text(json.dumps(dict(source_hashes=frozen['source_hashes'],dependency_versions=frozen['dependency_versions'])))
(out/'result.json').write_text(json.dumps(dict(schema='dual-qualification-case-v1',case=case,pilot=True,passed=False,admission_granted=False,failure=None,completed_declared_duration=False,completed_requested_duration=True,completed_control_steps=1,source_integrity_passed=True,source_hashes_end=frozen['source_hashes'],dependency_versions_end=frozen['dependency_versions'],online_audit=dict(safe=True),numerical=dict(passed=True))))
print('synthetic preflight fixture completed',flush=True)
""")

    def preflight_command(*, case, output, source_freeze, **kwargs):
        return [sys.executable, str(preflight_child), str(output), json.dumps(case), str(source_freeze)]

    monkeypatch.setattr(m, "_preflight_command", preflight_command)
    monkeypatch.setattr(m, "_case_command", command)
    return m, dict(
        original=original,
        design_path=design,
        output=tmp_path / "batch",
        journal_root=tmp_path / "journal",
        workers=2,
    )


def test_batch_waits_running_after_failure_and_never_launches_pending(tmp_path, monkeypatch):
    m, args = stub_batch(tmp_path, monkeypatch, first_fails=True)
    result = m.run_batch(**args)
    assert result["status"] == "failed" and result["admission_granted"] is False
    assert [r["status"] for r in result["cases"]] == ["failed", "passed", "skipped"]
    assert result["cases"][2]["started_at"] is None
    assert result["cases"][2]["reason"] == "not launched after an earlier failure"
    assert (
        "synthetic child completed"
        in (args["output"] / "logs" / f"{result['cases'][1]['case']['case_id']}.log").read_text()
    )
    assert not (args["output"] / "cases" / result["cases"][2]["case"]["case_id"]).exists()


def test_batch_complete_results_are_distinct_from_overall_physics_admission_and_refuse_overwrite(
    tmp_path, monkeypatch
):
    m, args = stub_batch(tmp_path, monkeypatch)
    result = m.run_batch(**args)
    assert result["status"] == "completed" and all(r["status"] == "passed" for r in result["cases"])
    assert result["admission_granted"] is False
    before = (args["output"] / "batch.json").read_bytes()
    with pytest.raises(FileExistsError):
        m.run_batch(**args)
    assert (args["output"] / "batch.json").read_bytes() == before


def test_batch_worker_cap_rejects_before_creating_output(tmp_path):
    m = batch_module()
    with pytest.raises(ValueError, match="1.*7"):
        m.run_batch(
            original="missing",
            design_path="missing",
            output=tmp_path / "batch",
            journal_root=tmp_path / "journal",
            workers=8,
        )
    assert not (tmp_path / "batch").exists()


def test_batch_rejects_passing_flags_with_truncated_control_count(tmp_path):
    m = batch_module()
    case = module().qualification_cases()[0]
    frozen = module().qualification_source_freeze()
    (tmp_path / "inputs.json").write_text(
        json.dumps(
            dict(source_hashes=frozen["source_hashes"], dependency_versions=frozen["dependency_versions"])
        )
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            dict(
                schema="dual-qualification-case-v1",
                case=case,
                pilot=False,
                admission_granted=False,
                passed=True,
                completed_declared_duration=True,
                completed_control_steps=1,
                failure=None,
                source_integrity_passed=True,
                source_hashes_end=frozen["source_hashes"],
                dependency_versions_end=frozen["dependency_versions"],
            )
        )
    )
    with pytest.raises(ValueError, match="duration|control"):
        m._case_result(dict(case=case, output=str(tmp_path)), 0, frozen)


def test_freeze_is_stable_across_executed_native_step_helper():
    # Execute the real helper with a software C-API surface; this reproduces
    # its unconditional lazy adapter import without claiming native physics.
    from types import SimpleNamespace

    m = module()
    from adaptive_drone_patrol.dual_coordination_env import step_native_with_audit

    frozen = m.qualification_source_freeze()

    class Audit(m.OnlineNativeAudit):
        def __init__(self):
            self.rk = self.native = 0

        def observe_native_state(self, model, data, *, phase, location):
            self.rk += phase == "rk"
            self.native += phase == "native"

        def report(self):
            return {"safe": True}

    class MJ:
        callback = None

        def set_mjcb_control(self, callback):
            self.callback = callback

        def mj_step(self, model, data):
            for _ in range(4):
                self.callback(model, data)

        def mj_forward(self, model, data):
            pass

    fixture = SimpleNamespace(mj=MJ(), model=object(), data=object(), substeps=16, set_rpm=lambda rpm: None)
    audit = Audit()
    step_native_with_audit(fixture, np.zeros((2, 4)), audit)
    assert audit.rk == 64 and audit.native == 17
    assert m.verify_qualification_source_freeze(frozen) == frozen
    for name in ("compiled_audit", "native_audit"):
        assert f"src/adaptive_drone_patrol/dual_coordination_{name}.py" in frozen["source_hashes"]


def test_native_one_tick_keeps_prospective_dependency_closure(tmp_path):
    import os
    from pathlib import Path

    pytest.importorskip(
        "mujoco", reason="Actual MuJoCo required; local software fixture is not native evidence"
    )
    m = module()
    from adaptive_drone_patrol.dual_coordination_env import step_native_with_audit

    original = Path(os.environ.get("DUAL_CF2X", "inputs/cf2x/cf2x.urdf"))
    if not original.is_file():
        pytest.skip("Original native CF2X asset unavailable")
    frozen = m.qualification_source_freeze()
    fixture = m.DualMuJoCoFixture(
        original, tmp_path / "one-tick", [m.Payload(m.FREE_MASS_KG, 0.2, 0.045)] * 2
    )
    audit = m.OnlineNativeAudit(fixture)
    result = step_native_with_audit(fixture, np.full((2, 4), m.HOVER_RPM), audit)
    assert result["safe"] and fixture.data.time == pytest.approx(0.01)
    assert m.verify_qualification_source_freeze(frozen) == frozen


def test_batch_preflight_failure_launches_no_formal_case(tmp_path, monkeypatch):
    import sys

    m, args = stub_batch(tmp_path, monkeypatch)
    monkeypatch.setattr(
        m,
        "_preflight_command",
        lambda **kwargs: [sys.executable, "-c", "print('preflight failed');raise SystemExit(3)"],
    )
    launches = []
    monkeypatch.setattr(m, "_case_command", lambda **kwargs: launches.append(kwargs))
    result = m.run_batch(**args)
    assert result["status"] == "failed" and not launches
    assert all(row["status"] == "skipped" and row["started_at"] is None for row in result["cases"])
    assert "preflight" in " ".join(result["failures"]).lower()
    assert "preflight failed" in (args["output"] / "logs/preflight.log").read_text()


@pytest.mark.parametrize("damage", ["formal_pass", "truncated", "late_source", "unsafe"])
def test_preflight_receipt_rejects_formal_or_incomplete_or_changed_execution(tmp_path, monkeypatch, damage):
    import subprocess

    m, args = stub_batch(tmp_path, monkeypatch)
    frozen = module().qualification_source_freeze()
    freeze_path = tmp_path / "frozen.json"
    freeze_path.write_text(json.dumps(frozen))
    case = module().qualification_cases()[0]
    output = tmp_path / "unit-preflight"
    command = m._preflight_command(case=case, output=output, source_freeze=freeze_path)
    subprocess.run(command, check=True, capture_output=True)
    assert m._preflight_result(output, 0, frozen, case)["sha256"]
    result = json.loads((output / "result.json").read_text())
    if damage == "formal_pass":
        result["passed"] = True
    elif damage == "truncated":
        result["completed_control_steps"] = 0
    elif damage == "late_source":
        result["source_hashes_end"]["src/late.py"] = "f" * 64
    else:
        result["online_audit"]["safe"] = False
    (output / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="preflight"):
        m._preflight_result(output, 0, frozen, case)


def test_source_freeze_clean_probe_with_preimported_evaluator():
    # A full regression legitimately imports evaluator before qualification.
    importlib.import_module("adaptive_drone_patrol.dual_coordination_evaluate")
    test_source_freeze_covers_lazy_step_dependencies_without_unimported_modules()
    assert (
        "src/adaptive_drone_patrol/dual_coordination_evaluate.py"
        in module().qualification_source_freeze()["source_hashes"]
    )
