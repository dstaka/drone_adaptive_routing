"""Software-only storage checks: synthetic inputs do not qualify physical runs."""

import gzip
import importlib.util
import json
from pathlib import Path
import pickle
import struct

import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "measure_dual_storage", Path("scripts/measure_dual_storage.py")
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_shared_training_archive_is_exact_multimember_gzip_and_durable(tmp_path):
    from adaptive_drone_patrol.dual_coordination_train import append_training_episode_receipt

    path = tmp_path / "episodes.gz"
    records = [
        dict(schema="dual-training-episode-v1", synthetic_test_only=True, actions=[dict(tick=0, action=240)]),
        dict(schema="dual-training-episode-v1", synthetic_test_only=True, actions=[]),
    ]
    counts = [append_training_episode_receipt(path, row) for row in records]
    assert [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()] == records
    assert sum(counts) == path.stat().st_size
    before = path.read_bytes()
    with pytest.raises(ValueError):
        append_training_episode_receipt(path, dict(schema="dual-training-episode-v1", invalid=float("nan")))
    assert path.read_bytes() == before


def test_history_padding_serializes_independent_records_without_alias_credit():
    m = module()
    padding = m.history_padding(assignment_rows=8, curve_rows=3, case_id_width=64)
    for key, expected in [("assignments", 8), ("curves", 3)]:
        data = padding[key]
        rows = []
        while data:
            length = struct.unpack("<Q", data[:8])[0]
            rows.append(pickle.loads(data[8 : 8 + length]))
            data = data[8 + length :]
        assert len(rows) == expected and len({id(row) for row in rows}) == expected
        assert len({row["episode" if key == "assignments" else "update"] for row in rows}) == expected
    assert padding["serialization"] == "independent-pickle-records-with-length-prefix-no-alias-credit"


def episode_fixture(tmp_path):
    from types import SimpleNamespace
    from adaptive_drone_patrol.dual_coordination_campaign import _digest
    from adaptive_drone_patrol.dual_coordination_raw import RawWriter
    from adaptive_drone_patrol.util import sha256

    directory = tmp_path / "episode"
    directory.mkdir()
    case = dict(case_id="synthetic-software", split=15, goals=[], scenario="normal")
    binding = dict(
        mode="development",
        admission_sha256=None,
        case_sha256=_digest(case),
        controller_sha256=_digest({}),
        profile_sha256="a" * 64,
        model_sha256="b" * 64,
        audit_library_sha256=None,
    )
    env = dict(
        schema="dual-development-environment-v1",
        mode="development",
        production_admitted=False,
        scope="DEVELOPMENT_ONLY",
        case=case,
        design={},
        binding=binding,
    )
    actual = dict(schema="dual-raw-v1", world_time=0.0, qpos=[0.0] * 30, qvel=[0.0] * 24)
    path = directory / "replay.draw"
    with RawWriter(path) as writer:
        writer.append(
            dict(
                kind="RESET",
                tick=0,
                actual=actual,
                case_id=case["case_id"],
                model_hash="b" * 64,
                profile_hash="a" * 64,
            )
        )
        for tick in (1, 20):
            writer.append(
                dict(
                    kind="STEP",
                    tick=tick,
                    actual=dict(actual, world_time=tick * 0.01),
                    action=240,
                    action_tick=0,
                    case_id=case["case_id"],
                    model_hash="b" * 64,
                    profile_hash="a" * 64,
                )
            )
    result = dict(
        schema="dual-development-result-v1",
        mode="development",
        production_admitted=False,
        binding=binding,
        terminal_reason="success",
        final_native_state=dict(actual, world_time=0.2),
        mission=dict(state=dict(tick=20, terminal_reason="success")),
        online_audit={},
        replay_segments=[dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))],
    )
    (directory / "environment.json").write_text(json.dumps(env))
    (directory / "result.json").write_text(json.dumps(result))
    bound = SimpleNamespace(files=dict(controller=[], source=[], assets=[]))
    return directory, bound, "a" * 64


def test_episode_conversion_preserves_actual_actions_and_has_zero_cleanup_credit(tmp_path):
    directory, bound, profile = episode_fixture(tmp_path)
    record, units, evidence = module().episode_record(directory, binding=bound, profile_sha256=profile)
    assert units == 1 and record["actions"] == [dict(tick=0, action=240)]
    assert record["schema"] == "dual-training-episode-v1"
    assert record["derived_directory_retired"] is False
    assert record["retirement_policy"] == "DEVELOPMENT_PROBE_NO_DELETION_NO_CLEANUP_CREDIT"
    assert record["mission"]["state"]["tick"] == 20 and len(evidence) == 3
    assert (directory / "replay.draw").is_file()


@pytest.mark.parametrize(
    "damage", ["raw_bytes", "profile", "nonterminal", "final_state", "missing_final_proposal"]
)
def test_episode_conversion_rejects_unbound_or_incomplete_measurement(tmp_path, damage):
    directory, bound, profile = episode_fixture(tmp_path)
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    if damage == "raw_bytes":
        (directory / "replay.draw").write_bytes(b"changed")
    elif damage == "profile":
        profile = "c" * 64
    elif damage == "nonterminal":
        result["terminal_reason"] = None
    elif damage == "final_state":
        result["final_native_state"]["qpos"][0] = 1.0
    elif damage == "missing_final_proposal":
        result["terminal_reason"] = "reference:failed"
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        module().episode_record(directory, binding=bound, profile_sha256=profile)


def test_incomplete_benchmark_is_refused_before_unpickling_or_output(tmp_path, monkeypatch):
    import torch

    m = module()
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(dict(completed=False)))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Must not unpickle incomplete benchmark"))
    with pytest.raises(ValueError, match="Completed"):
        m.measure(
            benchmark=benchmark,
            episode=tmp_path,
            experiment_binding=tmp_path / "missing",
            execution_profile=tmp_path / "missing",
            output=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_checkpoint_probe_rejects_missing_actual_rollout_values(tmp_path):
    import numpy as np
    import torch

    m = module()
    row = dict(
        worker=0,
        observation=dict(goals=np.zeros((10, 11)), drones=np.zeros((2, 66)), global_features=np.zeros(11)),
        mask=np.ones(242, dtype=bool),
        action=240,
        log_prob=0.0,
        value=0.0,
        reward=0.0,
        terminal=False,
        elapsed=0.2,
    )
    native = dict(
        binding=dict(profile_sha256="a" * 64),
        mode="development",
        backend=dict(integration=[0.0], qpos=[0.0] * 30, qvel=[0.0] * 24),
    )
    state = dict(
        schema="dual-policy-v1",
        selectable=False,
        physical_resume_available=False,
        backend_state=None,
        budget=dict(seed=10272799, transitions=16384, updates=4),
        environment_state=dict(
            schema="dual-development-checkpoint-size-v1",
            partial_rollout=dict(count=4096, workers=1, rows=[row.copy() for _ in range(4096)]),
            workers=[dict(state=native, torch_rng=b"synthetic", python_rng=(1,), numpy_rng=(1,))],
        ),
        optimizer=dict(state={1: {}}),
        policy={},
        torch_rng=None,
        numpy_rng=None,
        python_rng=None,
        cuda_rng=None,
        versions={},
        policy_config={},
        ppo_config={},
    )
    path = tmp_path / "current-size-probe.pt"
    torch.save(state, path)
    assert m._checkpoint(path, m._file_record(path), 1, "a" * 64)["budget"]["updates"] == 4
    del state["environment_state"]["partial_rollout"]["rows"][0]["observation"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="rollout"):
        m._checkpoint(path, m._file_record(path), 1, "a" * 64)


def test_external_failed_final_proposal_is_preserved(tmp_path):
    directory, bound, profile = episode_fixture(tmp_path)
    path = directory / "result.json"
    result = json.loads(path.read_text())
    result["terminal_reason"] = "reference:failed"
    result["mission"]["state"]["terminal_reason"] = "reference:failed"
    path.write_text(json.dumps(result))
    actions = directory / "proposed-actions.json"
    actions.write_text(
        json.dumps([dict(tick=0, decision=dict(action=240)), dict(tick=20, decision=dict(action=240))])
    )
    record, units, _ = module().episode_record(
        directory, binding=bound, profile_sha256=profile, actions_path=actions
    )
    assert units == 2 and record["actions"][-1] == dict(tick=20, action=240)
