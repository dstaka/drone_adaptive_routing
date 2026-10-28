"""Measure development archive I/O and a conservative finite-history size envelope.

No flight, production admission, cleanup, or selectable policy is performed.
The checkpoint is trusted local Torch data linked by a completed benchmark.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import pickle
import struct
import tempfile
import time
import uuid

import numpy as np
import torch

from adaptive_drone_patrol.dual_coordination_benchmark import (
    _assert_same_payload,
    _file_record,
    _load_experiment_binding,
    _require_identity_binding,
    benchmark_summary,
    machine_inventory,
    validate_phase_coverage,
    worker_candidates,
)
from adaptive_drone_patrol.dual_coordination_campaign import _digest
from adaptive_drone_patrol.dual_coordination_env import OBSERVATION_SCALES
from adaptive_drone_patrol.dual_coordination_progress import atomic_json
from adaptive_drone_patrol.dual_coordination_raw import read_raw
from adaptive_drone_patrol.dual_coordination_train import append_training_episode_receipt
from adaptive_drone_patrol.util import sha256


def _read(path):
    return json.loads(Path(path).read_text())


def _native(sample):
    result = {k: sample[k] for k in ("world_time", "qpos", "qvel")}
    if not math.isfinite(result["world_time"]):
        raise ValueError("Finite native time required")
    for key, size in (("qpos", 30), ("qvel", 24)):
        array = np.asarray(result[key], dtype=float)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError("Complete finite native state required")
    return result


def episode_record(directory, *, binding, profile_sha256, actions_path=None):
    """Convert verified development bytes to the production archive record shape."""
    directory = Path(directory).resolve()
    metadata, result = (_read(directory / name) for name in ("environment.json", "result.json"))
    case = metadata["case"]
    physical = metadata["binding"]
    if (
        metadata.get("schema") != "dual-development-environment-v1"
        or result.get("schema") != "dual-development-result-v1"
        or any(
            row.get("mode") != "development" or row.get("production_admitted") is not False
            for row in (metadata, result)
        )
        or case["split"] != 15
        or physical != result["binding"]
        or physical.get("admission_sha256") is not None
        or physical["profile_sha256"] != profile_sha256
        or physical["case_sha256"] != _digest(case)
        or not result.get("terminal_reason")
    ):
        raise ValueError("Completed source-bound development episode required")
    tick = result["mission"]["state"]["tick"]
    if type(tick) is not int or tick <= 0 or tick > 30000:
        raise ValueError("Completed positive finite mission ticks required")
    segments = result["replay_segments"]
    if len(segments) != 1:
        raise ValueError("One uninterrupted actual development episode required")
    segment = segments[0]
    path = directory / Path(segment["path"]).name
    if path.is_symlink() or path.stat().st_size != segment["bytes"] or sha256(path) != segment["sha256"]:
        raise ValueError("Actual development replay bytes changed")
    initial = last = None
    inferred = {}
    records = 0
    for row in read_raw(path):
        records += 1
        if (
            row.get("case_id") != case["case_id"]
            or row.get("profile_hash") != profile_sha256
            or row.get("model_hash") != physical["model_sha256"]
        ):
            raise ValueError("Retained physical case/model/profile mismatch")
        if row["kind"] == "RESET":
            if initial is not None or row["tick"] != 0:
                raise ValueError("Exactly one initial reset required")
            initial = _native(row["actual"])
        else:
            pair = dict(tick=row["action_tick"], action=row["action"])
            if pair["tick"] in inferred and inferred[pair["tick"]] != pair:
                raise ValueError("Inconsistent retained action")
            inferred[pair["tick"]] = pair
        last = row
    if initial is None or last["tick"] != tick:
        raise ValueError("Missing native reset/terminal boundary")
    final = _native(result["final_native_state"])
    for key in ("world_time", "qpos", "qvel"):
        if not np.allclose(final[key], last["actual"][key], rtol=0, atol=1e-8):
            raise ValueError("Retained terminal native state differs")
    if actions_path is None:
        if result["terminal_reason"] != "success":
            raise ValueError("Failed episode requires durable final proposed actions")
        actions = list(inferred.values())
    else:
        actions_path = Path(actions_path)
        rows = (
            _read(actions_path)
            if actions_path.suffix == ".json"
            else [json.loads(line) for line in actions_path.read_text().splitlines()]
        )
        actions = [
            dict(tick=row["tick"], action=row.get("action", row.get("decision", {}).get("action")))
            for row in rows
        ]
    for index, row in enumerate(actions):
        if (
            type(row["tick"]) is not int
            or row["tick"] != index * 20
            or row["tick"] > tick
            or type(row["action"]) is not int
            or not 0 <= row["action"] < 242
            or (row["tick"] in inferred and inferred[row["tick"]] != row)
        ):
            raise ValueError("Canonical contiguous actual proposal history required")
    if not actions or set(inferred) - {row["tick"] for row in actions} or actions[-1]["tick"] + 20 < tick:
        raise ValueError("Missing actual proposal interval")
    evidence = [_file_record(directory / name) for name in ("environment.json", "result.json")] + [
        _file_record(path)
    ]
    if actions_path is not None:
        evidence.append(_file_record(actions_path))
    provenance = {
        role + "_sha256": _digest(binding.files[role]) for role in ("source", "controller", "assets")
    }
    provenance.update(learning_lock_sha256=None, development_binding_sha256=_digest(binding.files))
    record = dict(
        schema="dual-training-episode-v1",
        case=case,
        provenance=provenance,
        physical_binding=physical,
        actions=actions,
        initial_native=initial,
        final_native=final,
        mission=result["mission"],
        audit=result["online_audit"].get("latest", result["online_audit"]),
        restored_from=None,
        episode_token=uuid.uuid4().hex,
        execution_error=None,
        close_error=None,
        derived_directory_retired=False,
        retirement_policy="DEVELOPMENT_PROBE_NO_DELETION_NO_CLEANUP_CREDIT",
    )
    return record, len(actions), evidence


def history_padding(*, assignment_rows, curve_rows, case_id_width):
    """Independent serialized records are conservative versus pickle memo sharing."""
    if any(type(v) is not int or v < 1 for v in (assignment_rows, curve_rows, case_id_width)):
        raise ValueError("Positive finite history shape required")

    def serialize(rows):
        stream = io.BytesIO()
        for row in rows:
            value = pickle.dumps(row, protocol=5)
            stream.write(struct.pack("<Q", len(value)))
            stream.write(value)
        return stream.getvalue()

    assignments = serialize(
        dict(
            episode=i,
            case_id=("DECLARED_SHAPE_ONLY_" + str(i)).ljust(case_id_width, "x"),
            scenario="availability-delay",
            worker=7,
            stage=3,
        )
        for i in range(assignment_rows)
    )
    curves = serialize(
        dict(
            update=i,
            stage=3,
            transitions=1048576,
            rollout_reward=-1e300,
            terminal_episodes=4096,
            simulated_seconds=819.2,
            update_wall_seconds=1e300,
            ppo=dict(total=1e300, policy=1e300, value=1e300, entropy=1e300, minibatches=40),
        )
        for i in range(1, curve_rows + 1)
    )
    return dict(
        serialization="independent-pickle-records-with-length-prefix-no-alias-credit",
        assignments=assignments,
        curves=curves,
    )


def _profile_hash(binding):
    names = (
        "dual_coordination_env.py",
        "dual_coordination_mission.py",
        "dual_coordination_audit.py",
        "dual_coordination_reference.py",
        "dual_coordination_safety.py",
        "dual_payload_mujoco.py",
        "payload_hybrid_controller.py",
    )
    hashes = {}
    for name in names:
        entries = [r for r in binding.files["source"] if Path(r["path"]).name == name]
        if len(entries) != 1:
            raise ValueError("Exact frozen executable profile required: " + name)
        hashes[name] = entries[0]["sha256"]
    return _digest(dict(scales=OBSERVATION_SCALES, source_sha256=hashes, profile="dual-payload-v1"))


def _checkpoint(path, record, workers, profile):
    if _file_record(path) != record:
        raise ValueError("Actual completed checkpoint hash differs")
    state = torch.load(path, map_location="cpu", weights_only=False)
    environment = state["environment_state"]
    rollout = environment["partial_rollout"]
    if (
        state.get("schema") != "dual-policy-v1"
        or state.get("selectable") is not False
        or state.get("physical_resume_available") is not False
        or state["backend_state"] is not None
        or state["budget"] != {"seed": 10272799, "transitions": 16384, "updates": 4}
        or environment["schema"] != "dual-development-checkpoint-size-v1"
        or rollout["count"] != 4096
        or len(rollout["rows"]) != 4096
        or rollout["workers"] != workers
        or len(environment["workers"]) != workers
        or not state["optimizer"]["state"]
    ):
        raise ValueError("Complete disposable 4096 rollout/Adam/native checkpoint required")
    for row in rollout["rows"]:
        try:
            if (
                type(row["worker"]) is not int
                or not 0 <= row["worker"] < workers
                or type(row["action"]) is not int
                or not 0 <= row["action"] < 242
                or type(row["terminal"]) is not bool
            ):
                raise ValueError("Invalid actual rollout action/worker/terminal")
            mask = np.asarray(row["mask"])
            if mask.shape != (242,) or mask.dtype != np.bool_ or not mask[row["action"]]:
                raise ValueError("Invalid actual rollout mask")
            for key, shape in (("goals", (10, 11)), ("drones", (2, 66)), ("global_features", (11,))):
                value = np.asarray(row["observation"][key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("Incomplete finite actual rollout observation")
            if any(not math.isfinite(row[key]) for key in ("log_prob", "value", "reward", "elapsed")):
                raise ValueError("Nonfinite actual rollout scalar")
        except (KeyError, TypeError) as error:
            raise ValueError("Incomplete actual rollout values") from error
    for key in (
        "policy",
        "torch_rng",
        "numpy_rng",
        "python_rng",
        "cuda_rng",
        "versions",
        "policy_config",
        "ppo_config",
    ):
        if key not in state:
            raise ValueError("Complete NN and process RNG state required")
    for worker in environment["workers"]:
        native = worker["state"]
        if (
            native["binding"]["profile_sha256"] != profile
            or native["mode"] != "development"
            or not native["backend"]["integration"]
            or not worker["torch_rng"]
            or not worker["python_rng"]
            or not worker["numpy_rng"]
        ):
            raise ValueError("Complete source-bound native worker/RNG state required")
    return state


def measure(*, benchmark, episode, experiment_binding, execution_profile, output, actions=None):
    """Actual storage measurement only; all original development inputs survive."""
    benchmark, episode, output = Path(benchmark), Path(episode), Path(output)
    row = _read(benchmark)
    if row.get("schema") != "dual-development-benchmark-v1" or row.get("completed") is not True:
        raise ValueError("Completed native development benchmark required")
    binding = _load_experiment_binding(experiment_binding)
    if row.get("experiment_inputs") != binding.files or row.get("physics_backend") != "native-mujoco":
        raise ValueError("Completed benchmark experiment identity differs")
    _require_identity_binding(row["input_identity"], binding)
    device = str(row["policy_device"]).split(":")[0]
    summary = benchmark_summary(row["measurements"], workers=row["workers"], device=device)
    for key in (
        "repetitions",
        "transitions_per_repetition",
        "epochs_per_update",
        "measured_transitions",
        "warmup_transitions",
        "total_pilot_transitions",
    ):
        if row.get(key) != summary[key]:
            raise ValueError("Incomplete benchmark repetitions")
    for measurement in row["measurements"]:
        validate_phase_coverage(measurement, workers=row["workers"])
    profile = _read(execution_profile)
    profile_file = _file_record(execution_profile)
    if (
        profile.get("schema") != "dual-execution-profile-v1"
        or profile.get("checkpoint_rounds") != 64
        or not any(
            (r["bytes"], r["sha256"]) == (profile_file["bytes"], profile_file["sha256"])
            for r in binding.files["configuration"]
        )
    ):
        raise ValueError("Frozen actual execution profile required")
    for source in (Path(__file__), Path(append_training_episode_receipt.__code__.co_filename)):
        if not any(
            (r["bytes"], r["sha256"]) == (source.stat().st_size, sha256(source))
            for r in binding.files["source"]
        ):
            raise ValueError("Storage measurement executable must be source-bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    observed = machine_inventory(output.parent)
    for phase in ("machine_before", "machine_after"):
        if (
            observed["machine_id"] != row[phase]["machine_id"]
            or observed["dependencies"] != row[phase]["dependencies"]
        ):
            raise ValueError("Storage measurement machine/dependencies differ from completed benchmark")
    candidates = worker_candidates(observed["cpu"]["effective_quota"])
    physical_profile = _profile_hash(binding)
    archive_record, units, episode_evidence = episode_record(
        episode, binding=binding, profile_sha256=physical_profile, actions_path=actions
    )
    if archive_record["physical_binding"].get("audit_library_sha256") != (
        row["input_identity"]["audit_library"] or {}
    ).get("sha256"):
        raise ValueError("Development episode audit library differs from benchmark")
    if archive_record["physical_binding"]["controller_sha256"] != row["input_identity"]["controller_sha256"]:
        raise ValueError("Development episode controller differs from benchmark")
    checkpoints = row["checkpoint_measurements"]
    if (
        len(checkpoints) != 3
        or checkpoints[-1].get("readback_verified") is not True
        or checkpoints[-1].get("full_rollout_transitions") != 4096
    ):
        raise ValueError("Three completed actual checkpoint measurements required")
    cp = checkpoints[-1]
    cp_path = Path(cp["path"])
    if cp_path.name != "current-size-probe.pt":
        raise ValueError("Actual current-size-probe.pt required")
    state = _checkpoint(
        cp_path, {key: cp[key] for key in ("path", "bytes", "sha256")}, row["workers"], physical_profile
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    archive = output / "development-episode.jsonl.gz"
    started = time.monotonic()
    archive_bytes = append_training_episode_receipt(archive, archive_record)
    archive_seconds = time.monotonic() - started
    count = 1048576 + 3 * max(candidates)
    # 128 bytes exceeds the generated case ID width; check every frozen declaration.
    width = 128
    for entry in binding.files["declaration"]:
        path = binding.root / entry["path"]
        if path.suffix == ".jsonl":
            with path.open() as stream:
                for line in stream:
                    value = json.loads(line)
                    if isinstance(value, dict) and "case_id" in value:
                        width = max(width, len(value["case_id"].encode()) + 32)
    payload = dict(
        schema="dual-training-shape-size-envelope-v1",
        scope="DECLARED_SHAPE_SIZE_PROBE",
        selectable=False,
        physical_resume_available=False,
        production_transitions=0,
        actual_development_checkpoint=state,
        full_rollout_transitions=4096,
        max_curve_rows=256,
        max_assignment_rows=count,
        history_padding=history_padding(assignment_rows=count, curve_rows=256, case_id_width=width),
        source_checkpoint={key: cp[key] for key in ("path", "bytes", "sha256")},
    )
    target = output / "checkpoint-envelope.pt"
    started = time.monotonic()
    fd, temporary = tempfile.mkstemp(dir=output, prefix="envelope-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    write_seconds = time.monotonic() - started
    read_started = time.monotonic()
    _assert_same_payload(payload, torch.load(target, map_location="cpu", weights_only=False))
    read_seconds = time.monotonic() - read_started
    envelope = _file_record(target)
    if envelope["bytes"] < max(item["bytes"] for item in checkpoints):
        raise ValueError("Size envelope is smaller than an actual checkpoint")
    envelope.update(
        path=target.name,
        scope="DECLARED_SHAPE_SIZE_PROBE",
        full_rollout_transitions=4096,
        max_curve_rows=256,
        max_assignment_rows=count,
        atomic_write_seconds=write_seconds,
        readback_wall_seconds=read_seconds,
        readback_verified=True,
    )
    retained_sources = sum(p.stat().st_size for p in episode.rglob("*") if p.is_file())
    sample = dict(
        schema="dual-benchmark-cost-sample-v1",
        kind="training_archive",
        measurement_cohort="development",
        actual_execution=True,
        measurement_scope="ACTUAL_ARCHIVE_IO_ONLY_NO_NEW_PHYSICS",
        units=units,
        wall_seconds=archive_seconds,
        retained_bytes=archive_bytes,
        peak_storage_bytes=retained_sources + archive_bytes + envelope["bytes"] + cp["bytes"],
        experiment_inputs=binding.files,
        machine_id=observed["machine_id"],
        execution_profile_sha256=profile_file["sha256"],
        checkpoint_envelope=envelope,
        cleanup_credit_bytes=0,
        original_episode_preserved=True,
        production_transitions=0,
        selectable=False,
        archive=_file_record(archive),
        evidence=dict(
            benchmark=_file_record(benchmark),
            episode=episode_evidence,
            execution_profile=profile_file,
            experiment_binding=_file_record(experiment_binding),
            source_checkpoint={key: cp[key] for key in ("path", "bytes", "sha256")},
        ),
    )
    binding.verify()
    after = machine_inventory(output)
    if after["machine_id"] != observed["machine_id"] or after["dependencies"] != observed["dependencies"]:
        raise ValueError("Machine changed during storage measurement")
    atomic_json(output / "training-archive-cost.json", sample)
    cost = _file_record(output / "training-archive-cost.json")
    cost["path"] = "training-archive-cost.json"
    atomic_json(output / "cost-inventory.json", dict(schema="dual-benchmark-costs-v1", samples=[cost]))
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "episode", "experiment-binding", "execution-profile", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--actions", type=Path)
    args = parser.parse_args()
    print(json.dumps(measure(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
