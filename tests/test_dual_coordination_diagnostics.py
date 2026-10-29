"""Software tests for post-selection diagnostics; no flight or heldout generation."""

import importlib
import json
from pathlib import Path

import pytest


def module():
    return importlib.import_module("adaptive_drone_patrol.dual_coordination_diagnostics")


def test_missing_learning_lock_cannot_create_diagnostics(tmp_path):
    with pytest.raises((OSError, ValueError)):
        module().run_diagnostics(
            learning_lock=tmp_path / "missing",
            selection=tmp_path / "selection",
            terminal=tmp_path / "terminal",
            original="missing",
            design_path="missing",
            output=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_diagnostic_methods_are_exact_and_do_not_enter_selection():
    from adaptive_drone_patrol.dual_coordination_evaluate import METHODS

    assert module().METHODS == ("N-no-transfer", "N-fixed-priority")
    assert set(module().METHODS).isdisjoint(METHODS)


def test_validation_no_transfer_reaches_admission_without_physics(monkeypatch, tmp_path):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    import adaptive_drone_patrol.dual_coordination_env as env

    def qualified(cap):
        raise RuntimeError("qualified capability checked")

    monkeypatch.setattr(env, "qualified_audit_library", qualified)
    with pytest.raises(RuntimeError, match="qualified capability checked"):
        env.DualCoordinationEnv(
            "missing",
            tmp_path / "out",
            generate_case(20, 0, "normal"),
            {},
            mode="validation",
            no_transfer=True,
        )
    assert not (tmp_path / "out").exists()


def test_wrong_modes_still_reject_no_transfer_before_physics(tmp_path):
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    from adaptive_drone_patrol.dual_coordination_env import DualCoordinationEnv

    for mode in ("training", "physical-heldout"):
        with pytest.raises(ValueError, match="No-transfer"):
            DualCoordinationEnv(
                "missing", tmp_path / "out", generate_case(20, 0, "normal"), {}, mode=mode, no_transfer=True
            )


def test_selection_pin_rejects_reselection_and_changed_terminal(tmp_path):
    m = module()
    terminal = tmp_path / "terminal"
    terminal.write_bytes(b"fixed model")
    from adaptive_drone_patrol.util import sha256

    selection = {
        "schema": "dual-selection-bound-v1",
        "selected_seed": 42,
        "checkpoint_sha256": {"42": sha256(terminal)},
    }
    m.check_selected_terminal(selection, terminal, {"selected": 42})
    with pytest.raises(ValueError, match="selection"):
        m.check_selected_terminal(selection, terminal, {"selected": 43})
    with pytest.raises(ValueError, match="selection"):
        m.check_selected_terminal(selection, terminal, {"selected": None})
    terminal.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="terminal"):
        m.check_selected_terminal(selection, terminal, {"selected": 42})


def test_summary_retains_all_missing_slots_without_selection(tmp_path):
    m = module()
    cases = {
        f"case-{i}": {"case_id": f"case-{i}", "scenario": s}
        for i, s in enumerate(["normal"] * 100 + ["priority"] * 100 + ["availability-delay"] * 100)
    }
    result = m.summarize(
        tmp_path,
        cases,
        lambda method, case: {"method": method, "case_id": case["case_id"]},
        lambda document, identity, case: None,
    )
    assert result["expected_missions"] == 600 and result["unstarted_missions"] == 600
    assert result["diagnostic_only"] and not result["selection_modified"]
    assert not result["joint_freeze_depends_on_diagnostics"]
    for method in m.METHODS:
        assert len(result["outcomes"][method]) == 300
        assert result["metrics"][method]["mean_score"] == 600
        assert result["metrics"][method]["safety_violations"] == 300
    assert not (tmp_path / "selection.json").exists()


def test_failed_claim_is_preserved_and_summary_cannot_drop_case(tmp_path):
    m = module()
    cases = {"one": {"case_id": "one", "scenario": "normal"}}

    def identity(method, case):
        return dict(method=method, case_id=case["case_id"])

    from adaptive_drone_patrol.dual_coordination_heldout import claim_slot

    slot = m.slot_path(tmp_path, m.METHODS[0], "one")
    assert claim_slot(slot, identity(m.METHODS[0], cases["one"]))
    assert claim_slot(slot, identity(m.METHODS[0], cases["one"])) is None
    result = m.summarize(tmp_path, cases, identity, lambda *args: None)
    assert result["attempted_missions"] == 1 and result["verified_missions"] == 0
    assert result["outcomes"][m.METHODS[0]][0]["safety_violations"] == 1
    assert json.loads((slot / "slot.json").read_text())["status"] == "started"


@pytest.mark.parametrize("method", ["N-no-transfer", "N-fixed-priority"])
def test_one_diagnostic_uses_fresh_forward_and_real_retained_evidence_then_skips(
    tmp_path, monkeypatch, method
):
    """Only physics/admission are synthetic; proposal, raw, bundle and verification are real."""
    from types import SimpleNamespace
    from test_dual_coordination_evaluate import SyntheticEnv
    import adaptive_drone_patrol.dual_coordination_env as env_module
    from adaptive_drone_patrol.dual_coordination_raw import RawWriter

    m = module()
    selection = tmp_path / "selection.json"
    selection.write_text('{"original_selection":42}')
    terminal = tmp_path / "terminal.pt"
    terminal.write_bytes(b"synthetic selected weights")
    original = tmp_path / "original"
    original.write_text("synthetic")
    design = tmp_path / "design.json"
    design.write_text("{}")
    reference_path = tmp_path / "development.json"
    reference_path.write_text("{}")
    reference = m.file_record(reference_path, tmp_path)
    cases = {
        f"case-{i}": {"case_id": f"case-{i}", "scenario": s}
        for i, s in enumerate(["normal"] * 100 + ["priority"] * 100 + ["availability-delay"] * 100)
    }
    lock = SimpleNamespace(
        digest="synthetic",
        receipt={"qualification": {}},
        selected_backend="native-cpu",
        require_current=lambda: None,
        verify_case=lambda case, mode: dict(case),
        binding=SimpleNamespace(root=tmp_path, files={r: [] for r in m.EXPERIMENT_ROLES}),
    )
    monkeypatch.setattr(m.VerifiedLearningLock, "load", lambda path: lock)
    monkeypatch.setattr(m, "_bound_file", lambda lock, role, path: Path(path))
    monkeypatch.setattr(
        m,
        "authenticate_selection",
        lambda *args: (42, cases, SimpleNamespace(verify=lambda: None), m.file_record(selection, tmp_path)),
    )
    monkeypatch.setattr(m, "verified_development_replay_reference", lambda *args: reference)
    forwards = []

    def selector(name, case, model, device):
        assert name == "N42" and model == terminal

        def choose(env):
            forwards.append((env.mission.tick, env.state["no_transfer"]))
            return 240

        return choose

    monkeypatch.setattr(m, "_selector", selector)
    admissions = []

    class CaptureEnv(SyntheticEnv):
        def __init__(self, original, output, case, design, **kwargs):
            admissions.append(kwargs)
            super().__init__(output)
            self.case = case
            self.state["no_transfer"] = kwargs["no_transfer"]
            self.writer = RawWriter(output / "replay.draw")
            self.writer.append(
                dict(
                    kind="RESET",
                    tick=0,
                    actual=dict(schema="dual-raw-v1", world_time=0.0, qpos=[1.0], qvel=[0.0]),
                    case_id=case["case_id"],
                    replay_stride_ticks=20,
                )
            )

        def close(self):
            for tick in (1, 20):
                self.writer.append(
                    dict(
                        kind="STEP",
                        tick=tick,
                        actual=dict(schema="dual-raw-v1", world_time=tick * 0.01, qpos=[1.0], qvel=[0.0]),
                        case_id=self.case["case_id"],
                        replay_stride_ticks=20,
                        action=240,
                        action_tick=0,
                        structural_mask=self.legal_mask().tolist(),
                    )
                )
            self.writer.close()
            record = m.file_record(self.output / "replay.draw", tmp_path)
            record["path"] = str(self.output / "replay.draw")
            self.replay_segments = [record]

    monkeypatch.setattr(env_module, "DualCoordinationEnv", CaptureEnv)
    args = dict(
        learning_lock="synthetic",
        selection=selection,
        terminal=terminal,
        original=original,
        design_path=design,
        output=tmp_path / "out",
        case_id="case-0",
        method=method,
    )
    first = m.run_diagnostics(**args)
    assert first["verified_missions"] == 1 and first["unstarted_missions"] == 599
    assert forwards == [(0, method == "N-no-transfer")]
    assert admissions[0]["mode"] == "validation" and admissions[0]["admission"] is lock
    assert admissions[0]["retain_replay"] and admissions[0]["replay_stride_ticks"] == 20
    slot = m.slot_path(tmp_path / "out", method, "case-0")
    assert not (slot / "capture").exists() and (slot / "capture.zip").is_file()
    assert (
        json.loads((slot / "diagnostic.json").read_text())["receipts"]["replay"][
            "native_reexecution_performed"
        ]
        is False
    )
    second = m.run_diagnostics(**args)
    assert second["verified_missions"] == 1 and len(forwards) == 1 and len(admissions) == 1
    assert selection.read_text() == '{"original_selection":42}'
    assert terminal.read_bytes() == b"synthetic selected weights"
    assert not (tmp_path / "out" / "selection.json").exists()
    assert not (tmp_path / "out" / "freeze.json").exists()
    # Evidence corruption keeps the slot in the denominator and is never rerun.
    (slot / "capture.zip").write_bytes(b"corrupt")
    failed = m.run_diagnostics(**args)
    assert failed["verified_missions"] == 0 and len(forwards) == 1
    assert failed["outcomes"][method][0]["safety_violations"] == 1
