"""Progress must report completed updates without touching the live trainer."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def observer():
    path = Path(__file__).parents[1] / "scripts/observe_dual_training.py"
    spec = importlib.util.spec_from_file_location("dual_training_observer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_curves(root, seed, count):
    folder = root / f"seed-{seed}"
    folder.mkdir(exist_ok=True)
    rows = [
        dict(update=i, transitions=i * 4096, stage=1 if i <= 64 else 2 if i <= 240 else 3)
        for i in range(1, count + 1)
    ]
    path = folder / "curves.json"
    path.write_text(json.dumps(dict(seed=seed, selectable=False, updates=rows)))
    return path


def test_reports_only_complete_update_lower_bound_and_does_not_modify_inputs(observer, tmp_path):
    path = write_curves(tmp_path, 42, 6)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = observer.read_training_progress(tmp_path)
    assert result["completed_updates"] == 6
    assert result["completed_transitions_lower_bound"] == 24576
    assert result["partial_rollout_transitions"] == "not recorded"
    assert [row["completed_updates"] for row in result["seeds"]] == [6, 0, 0]
    assert result["total_transitions"] == 3145728
    assert result["terminal_models_verified"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("field,value", [("update", 9), ("transitions", 5000), ("stage", 2)])
def test_refuses_inconsistent_completed_update_counters(observer, tmp_path, field, value):
    path = write_curves(tmp_path, 42, 1)
    data = json.loads(path.read_text())
    data["updates"][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="counter"):
        observer.read_training_progress(tmp_path)


def test_uses_predeclared_stage_boundaries_and_accumulates_all_seeds(observer, tmp_path):
    write_curves(tmp_path, 42, 256)
    write_curves(tmp_path, 43, 65)
    result = observer.read_training_progress(tmp_path)
    assert result["completed_updates"] == 321
    assert result["completed_transitions_lower_bound"] == 321 * 4096
    assert [x["last_completed_stage"] for x in result["seeds"]] == [3, 2, None]


def test_refuses_regression_in_durable_curves(observer, tmp_path):
    write_curves(tmp_path, 42, 6)
    prior = observer.read_training_progress(tmp_path)
    write_curves(tmp_path, 42, 5)
    with pytest.raises(ValueError, match="regress"):
        observer.read_training_progress(tmp_path, previous=prior)


def test_refuses_wrong_seed_or_more_than_finite_budget(observer, tmp_path):
    path = write_curves(tmp_path, 42, 257)
    with pytest.raises(ValueError, match="budget"):
        observer.read_training_progress(tmp_path)
    path = write_curves(tmp_path, 42, 1)
    data = json.loads(path.read_text())
    data["seed"] = 43
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seed"):
        observer.read_training_progress(tmp_path)


def test_pid_reuse_and_shell_command_text_are_not_live_training_proof(observer, tmp_path):
    proc = tmp_path / "123"
    proc.mkdir()
    stat = ["S", *(["0"] * 18), "999", "0"]
    (proc / "stat").write_text("123 (python worker) " + " ".join(stat))
    (proc / "cmdline").write_bytes(b"python\0-m\0adaptive_drone_patrol.dual_coordination_train\0--run\0")
    assert observer.training_process_matches(123, "999", proc_root=tmp_path)
    assert not observer.training_process_matches(123, "998", proc_root=tmp_path)
    (proc / "cmdline").write_bytes(b"bash\0-c\0python -m adaptive_drone_patrol.dual_coordination_train\0")
    assert not observer.training_process_matches(123, "999", proc_root=tmp_path)
    assert not observer.training_process_matches(124, "999", proc_root=tmp_path)
