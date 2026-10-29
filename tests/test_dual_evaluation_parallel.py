"""Administrative parallelism must not permit partial acceptance or duplicate slots."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).parents[1] / "scripts/run_dual_evaluation_parallel.py"
    spec = importlib.util.spec_from_file_location("parallel_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_command(tmp_path, *, fail=None):
    script = tmp_path / "child.py"
    script.write_text("""import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);method=sys.argv[2];fail=sys.argv[3]
if method=='a':
 root.mkdir();time.sleep(.08)
 p=root/'tmp';p.write_text('{}');p.rename(root/'declaration.json')
elif not (root/'declaration.json').exists():
 raise RuntimeError('declaration race')
(root/(method+'.start')).write_text(str(time.time_ns()))
time.sleep(.12)
if method==fail:sys.exit(7)
(root/(method+'.end')).write_text(str(time.time_ns()))
""")
    return lambda method: [sys.executable, str(script), str(tmp_path / "out"), method, fail or ""]


def test_initialization_bounded_overlap_and_final_reduce_after_every_child(driver, tmp_path):
    methods = ("a", "b", "c", "d", "e")

    def finalize():
        assert all((tmp_path / "out" / f"{m}.end").exists() for m in methods)
        return {"status": "complete", "verified_missions": 1500}

    state = driver.coordinate(
        methods,
        fake_command(tmp_path),
        output=tmp_path / "out",
        execution=tmp_path / "execution",
        max_workers=3,
        effective_cpus=3.8,
        finalize=finalize,
        poll_seconds=0.01,
    )
    assert state["status"] == "complete"
    assert [r["method"] for r in state["children"]] == list(methods)
    assert state["global_reduction_calls"] == 1
    events = []
    for m in methods:
        for edge, delta in [("start", 1), ("end", -1)]:
            events.append((int((tmp_path / "out" / f"{m}.{edge}").read_text()), delta))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    assert 2 <= peak <= 3
    assert json.loads((tmp_path / "execution/state.json").read_text())["status"] == "complete"


def test_failure_is_preserved_without_retry_or_final_selection(driver, tmp_path):
    def forbidden():
        pytest.fail("partial campaign must never reduce/select")

    with pytest.raises(RuntimeError, match="child"):
        driver.coordinate(
            ("a", "b", "c"),
            fake_command(tmp_path, fail="b"),
            output=tmp_path / "out",
            execution=tmp_path / "execution",
            max_workers=2,
            effective_cpus=7.68,
            finalize=forbidden,
            poll_seconds=0.01,
        )
    state = json.loads((tmp_path / "execution/state.json").read_text())
    assert state["status"] == "failed" and state["global_reduction_calls"] == 0
    assert len([r for r in state["children"] if r["method"] == "b"]) == 1
    assert (tmp_path / "execution/b.log").exists()


@pytest.mark.parametrize("workers,quota", [(0, 7.68), (6, 7.68), (5, 4.9)])
def test_refuse_oversubscription_before_any_output(driver, tmp_path, workers, quota):
    with pytest.raises(ValueError):
        driver.coordinate(
            ("a", "b"),
            lambda _: [],
            output=tmp_path / "out",
            execution=tmp_path / "execution",
            max_workers=workers,
            effective_cpus=quota,
            finalize=lambda: None,
        )
    assert not (tmp_path / "execution").exists()


def test_duplicate_methods_and_existing_outputs_are_rejected(driver, tmp_path):
    with pytest.raises(ValueError, match="unique"):
        driver.coordinate(
            ("a", "a"),
            lambda _: [],
            output=tmp_path / "out",
            execution=tmp_path / "execution",
            max_workers=2,
            effective_cpus=7.68,
            finalize=lambda: None,
        )
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        driver.coordinate(
            ("a",),
            lambda _: [],
            output=tmp_path / "out",
            execution=tmp_path / "execution",
            max_workers=2,
            effective_cpus=7.68,
            finalize=lambda: None,
        )


def test_deferred_reduction_has_no_acceptance_or_selection(driver):
    value = driver.deferred_reduction()
    assert value["status"] == "deferred-to-coordinator"
    assert value["acceptance_verified"] is False
    assert "selected_seed" not in value


def test_partial_canonical_result_cannot_be_reported_complete(driver, tmp_path):
    with pytest.raises(RuntimeError, match="Canonical"):
        driver.coordinate(
            ("a",),
            fake_command(tmp_path),
            output=tmp_path / "out",
            execution=tmp_path / "execution",
            max_workers=1,
            effective_cpus=7.68,
            finalize=lambda: {"status": "partial"},
            poll_seconds=0.01,
        )
    state = json.loads((tmp_path / "execution/state.json").read_text())
    assert state["status"] == "failed" and state["acceptance_verified"] is False
    assert state["global_reduction_calls"] == 1


def test_child_validation_defers_selection_and_restores_original_reducer(driver, monkeypatch, tmp_path):
    from types import SimpleNamespace
    import adaptive_drone_patrol

    calls = []

    def original(*a, **k):
        pytest.fail("child must not run canonical reducer or select")

    module = SimpleNamespace(METHODS=driver.METHODS["validation"], aggregate_validation=original)

    def run(**kw):
        calls.append(kw)
        assert not (tmp_path / "selection.json").exists()
        return module.aggregate_validation()

    module.run_validation = run
    monkeypatch.setattr(adaptive_drone_patrol, "dual_coordination_evaluate", module, raising=False)
    forwarded = [
        "--learning-lock",
        "lock",
        "--terminal42",
        "t42",
        "--terminal43",
        "t43",
        "--terminal44",
        "t44",
        "--original",
        "asset",
        "--design",
        "design",
        "--output",
        str(tmp_path),
        "--journal-root",
        "journal",
    ]
    assert driver.main(["--phase", "validation", "--child-method", "N42", "--", *forwarded]) == 0
    assert calls[0]["method"] == "N42" and calls[0]["terminals"] == {42: "t42", 43: "t43", 44: "t44"}
    assert "case_id" not in calls[0] and "native_reexecute" not in calls[0]
    assert module.aggregate_validation is original


def test_parent_uses_original_validation_reducer_and_complete_manifest(driver, monkeypatch, tmp_path):
    from types import SimpleNamespace
    import adaptive_drone_patrol

    calls = []
    (tmp_path / "declaration.json").write_text('{"case_sha256":{"a":"b"}}')
    lock = SimpleNamespace(
        binding=SimpleNamespace(root=tmp_path), require_current=lambda: calls.append("current")
    )
    module = SimpleNamespace(
        METHODS=driver.METHODS["validation"],
        run_validation=lambda: None,
        VerifiedLearningLock=SimpleNamespace(load=lambda _: lock),
        aggregate_validation=lambda *a, **k: calls.append((a, k)) or {"status": "complete"},
    )
    monkeypatch.setattr(adaptive_drone_patrol, "dual_coordination_evaluate", module, raising=False)
    _, _, _, _, _, reduce = driver.phase_api(
        "validation",
        [
            "--learning-lock",
            "lock",
            "--terminal42",
            "t42",
            "--terminal43",
            "t43",
            "--terminal44",
            "t44",
            "--original",
            "asset",
            "--design",
            "design",
            "--output",
            str(tmp_path),
            "--journal-root",
            "journal",
        ],
    )
    assert reduce() == {"status": "complete"}
    assert calls[0] == "current"
    assert calls[1][0][1] == {"case_sha256": {"a": "b"}}


def test_reject_case_override_and_physical_parallelism(driver):
    with pytest.raises(SystemExit):
        driver.phase_api("validation", ["--case-id", "subset"])
