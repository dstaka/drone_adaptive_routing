"""Runnable DEVELOPMENT ONLY native rollout/PPO timing, never qualification.

Each measured configuration has one disposable warm-up update, then exactly
three repetitions of 4096 aggregate joint transitions plus five PPO epochs.
Worlds live in distinct spawn processes because native MuJoCo control callbacks
are process-global. Policy inference/update can use CPU or CUDA; physics remains
native CPU. Compact replay is retained; detailed internal raw is disabled. The
environment's online monitor remains responsible for transient internal audit.

Each runner result is dual-development-benchmark-v1. The separate ``reduce``
command emits dual-benchmark-v1 only when every quota-capped CPU/CUDA configuration,
native qualification, prospective cost samples and storage evidence agree.
Missing evidence emits dual-benchmark-diagnostic-v1 without selecting a backend.
GPU physics remains unqualified. Disposable counters never enter production seeds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .dual_coordination_policy import JointActorCritic, JointPPO, PPOBatch, generalized_advantage
from .dual_coordination_progress import RunJournal, atomic_json
from .util import sha256

PHASE_TIMINGS = (
    "controller_reference_supervisor",
    "physics_audit",
    "mask_observation",
    "record_io",
    "total_step",
)


def validate_phase_timings(phases, *, execution_seconds=None):
    if (
        not isinstance(phases, dict)
        or set(phases) != set(PHASE_TIMINGS)
        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in phases.values())
    ):
        raise ValueError("Complete finite nonnegative native phase timings required")
    total = phases["total_step"]
    if total <= 0 or not math.isclose(
        total, sum(phases[k] for k in PHASE_TIMINGS[:-1]), rel_tol=1e-8, abs_tol=1e-8
    ):
        raise ValueError("Native phase sum differs from total step")
    if execution_seconds is not None and (
        type(execution_seconds) not in (int, float)
        or not math.isfinite(execution_seconds)
        or execution_seconds + 1e-8 < total
    ):
        raise ValueError("Native phase total exceeds measured worker execution")


def validate_phase_coverage(receipt, *, workers, transitions=4096):
    expected = [transitions // workers + int(i < transitions % workers) for i in range(workers)]
    for key in ("joint_transitions", "phase_timed_transitions"):
        if type(receipt.get(key)) is not int or receipt[key] != transitions:
            raise ValueError("Every measured transition requires native phase timing coverage")
    for key in ("world_transition_counts", "world_phase_timed_counts"):
        counts = receipt.get(key)
        if not isinstance(counts, list) or any(type(n) is not int for n in counts) or counts != expected:
            raise ValueError("Per-worker native phase coverage differs from actual collection")
    validate_phase_timings(
        receipt.get("summed_worker_phase_seconds"),
        execution_seconds=receipt.get("summed_worker_execution_seconds", -1),
    )


def training_checkpoint_writes(*, workers, checkpoint_rounds=64):
    if type(workers) is not int or workers < 1 or checkpoint_rounds != 64:
        raise ValueError("Positive workers and declared 64-round recovery schedule required")
    # Three sequential seeds; three stage entries, before/after each update,
    # and each 64 collection rounds. Writes replace current-recovery.pt.
    return 3 * (3 + 2 * 256 + (256 * math.ceil(4096 / workers)) // checkpoint_rounds)


def worker_candidates(cpu_quota):
    if type(cpu_quota) not in (int, float) or not math.isfinite(cpu_quota) or cpu_quota < 1:
        raise ValueError("Measured finite CPU quota of at least one core required")
    cap = math.floor(cpu_quota)
    return tuple(n for n in (1, 2, 4, 7) if n <= cap)


@dataclass
class DisposablePilotBudget:
    seed: int = 10272799
    transitions: int = 0
    updates: int = 0

    @property
    def selectable(self):
        return False

    @property
    def production_transitions(self):
        return 0

    def check_update(self, transitions, *, backend):
        if transitions != 4096 or type(transitions) is not int or backend != "native-mujoco":
            raise ValueError("Pilot requires4096 aggregate native joint transitions per update")
        if self.updates >= 4:
            raise ValueError("Disposable pilot budget exhausted after warm-up plus three measured updates")

    def record_update(self, transitions, *, backend):
        self.check_update(transitions, backend=backend)
        self.transitions += transitions
        self.updates += 1


@dataclass(frozen=True)
class NativeDevelopmentFactory:
    original: str
    output: str
    cases: tuple
    design: dict
    online_auditor: object = None
    native_audit_library: str | None = None
    replay_stride_ticks: int = 20

    def __call__(self, worker, episode):
        from .dual_coordination_env import DualCoordinationEnv

        if not self.cases or any(case.get("split") != 15 for case in self.cases):
            raise ValueError("Declared split15 development cases required")
        # Same declared initial sequence per configuration; worker offset avoids
        # synchronized clones while all workers cycle the same fixed case cohort.
        case = self.cases[(worker + episode) % len(self.cases)]
        return DualCoordinationEnv(
            self.original,
            Path(self.output) / f"worker-{worker:02d}" / f"mission-{episode:06d}",
            case,
            self.design,
            record_raw=False,
            mode="development",
            online_auditor=self.online_auditor,
            native_audit_library=self.native_audit_library,
            replay_stride_ticks=self.replay_stride_ticks,
            collect_timings=True,
        )


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def _peak_rss_bytes():
    return int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    )


def capture_input_identity(factory):
    """Observe executable inputs; never infer identities from a caller's labels."""
    source = Path(__file__).resolve().parent
    sources = sorted(source.glob("*.py")) + sorted((source.parents[1] / "native").glob("*.cpp"))
    original = Path(factory.original).resolve(strict=True)
    assets = {original}
    for mesh in ET.parse(original).getroot().iter("mesh"):
        name = mesh.get("filename") or mesh.get("file")
        if name:
            assets.add((original.parent / name).resolve(strict=True))
    library = _file_record(factory.native_audit_library) if factory.native_audit_library else None
    build_receipt = None
    if library:
        build_receipt = _file_record(Path(factory.native_audit_library).with_suffix(".json"))
    return dict(
        source=[_file_record(p) for p in sources],
        assets=[_file_record(p) for p in sorted(assets)],
        controller_sha256=_json_hash(factory.design),
        cases_sha256=_json_hash(factory.cases),
        audit_backend="compiled-native" if library else "python-native",
        audit_library=library,
        audit_build_receipt=build_receipt,
        replay_stride_ticks=factory.replay_stride_ticks,
        physics_backend="native-mujoco",
        online_auditor_override=factory.online_auditor is not None,
        collect_timings=True,
    )


def machine_inventory(output):
    """Measure local resource limits and devices without assuming a server type."""
    logical = os.cpu_count() or 1
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else logical
    limits = []
    cgroup_paths = {Path("/sys/fs/cgroup")}
    group_file = Path("/proc/self/cgroup")
    if group_file.is_file():
        for line in group_file.read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                cgroup_paths.add(Path("/sys/fs/cgroup") / relative.lstrip("/"))
    for group in cgroup_paths:
        for parent in (group, *group.parents):
            if not str(parent).startswith("/sys/fs/cgroup"):
                break
            path = parent / "cpu.max"
            if path.is_file():
                quota, period = path.read_text().split()
                if quota != "max":
                    limits.append(int(quota) / int(period))
    v1 = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    if v1.is_file() and int(v1.read_text()) > 0:
        limits.append(int(v1.read_text()) / int(v1.with_name("cpu.cfs_period_us").read_text()))
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    memory = dict(
        physical_bytes=physical, available_bytes=None, cgroup_limit_bytes=None, cgroup_current_bytes=None
    )
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        rows = {r.split(":")[0]: r.split(":")[1].strip() for r in meminfo.read_text().splitlines()}
        memory["available_bytes"] = int(rows["MemAvailable"].split()[0]) * 1024
    for filename, key in (("memory.max", "cgroup_limit_bytes"), ("memory.current", "cgroup_current_bytes")):
        values = []
        for group in cgroup_paths:
            path = group / filename
            if path.is_file() and path.read_text().strip() != "max":
                values.append(int(path.read_text()))
        if values:
            memory[key] = min(values) if filename == "memory.max" else max(values)
    dependencies = {}
    for package in ("torch", "numpy", "mujoco", "mujoco-warp", "warp-lang", "scipy"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    dependencies.update(
        python=platform.python_version(), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()
    )
    gpu = []
    for index in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(index)
        gpu.append(
            dict(
                index=index,
                name=p.name,
                total_memory_bytes=p.total_memory,
                capability=list(torch.cuda.get_device_capability(index)),
                uuid=str(getattr(p, "uuid", "unavailable")),
            )
        )
    disk = shutil.disk_usage(output)
    cpu = dict(
        logical_cores=logical,
        affinity_cores=affinity,
        cgroup_quota_limits=sorted(set(limits)),
        effective_quota=min([logical, affinity, *limits]),
    )
    machine_id = _json_hash(
        dict(
            host=platform.node(),
            platform=platform.platform(),
            cpu=cpu,
            gpu=gpu,
            physical_memory=physical,
            filesystem_device=Path(output).stat().st_dev,
        )
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(
        observed_at=datetime.now(timezone.utc).isoformat(),
        machine_id=machine_id,
        cpu=cpu,
        memory=memory,
        gpu_devices=gpu,
        dependencies=dependencies,
        disk=dict(
            path=str(Path(output).resolve()),
            total_bytes=disk.total,
            used_bytes=disk.used,
            free_bytes=disk.free,
        ),
        process_peak_rss_bytes=int(usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024)),
    )


def _worker(connection, factory, worker):
    env = None
    episode = 0
    try:
        torch.set_num_threads(1)
        started = time.monotonic()
        env = factory(worker, episode)
        if env.terminated:
            raise ValueError("Development world failed before its first policy decision")
        connection.send(
            dict(
                ok=True,
                observation=env.observation(),
                mask=env.legal_mask(),
                initialization_wall_seconds=time.monotonic() - started,
            )
        )
        while True:
            command, payload = connection.recv()
            if command == "close":
                env.close()
                env = None
                connection.send(
                    dict(
                        ok=True, command="close", final_flush_complete=True, peak_rss_bytes=_peak_rss_bytes()
                    )
                )
                break
            if command == "checkpoint":
                started = time.monotonic()
                state = env.to_dict()  # Includes actual replay flush and full native state.
                connection.send(
                    dict(
                        ok=True,
                        command="checkpoint",
                        state=state,
                        episode=episode,
                        python_rng=random.getstate(),
                        numpy_rng=np.random.get_state(),
                        # Byte transport avoids torch shared-memory managers;
                        # full RNG state is retained without an extra process.
                        torch_rng=torch.get_rng_state().numpy().tobytes(),
                        wall_seconds=time.monotonic() - started,
                    )
                )
                continue
            if command != "step":
                raise ValueError("Unknown development worker command")
            started = time.monotonic()
            observation, reward, terminal, info = env.step(payload)
            step_seconds = time.monotonic() - started
            duration = info["elapsed_simulated_s"]
            phase_timings = info.get("timings")
            if phase_timings is not None:
                validate_phase_timings(phase_timings, execution_seconds=step_seconds)
            elif isinstance(factory, NativeDevelopmentFactory):
                raise ValueError("Native benchmark requires phase timings on every step")
            if not math.isfinite(reward) or not math.isfinite(duration) or not 0 <= duration <= 0.200000001:
                raise ValueError("Finite reward and actual interval0..0.20s required")
            if type(terminal) is not bool:
                raise ValueError("Environment must explicitly mark every terminal failure")
            reset_seconds = 0.0
            if terminal:
                started = time.monotonic()
                env.close()
                episode += 1
                env = factory(worker, episode)
                if env.terminated:
                    raise ValueError("Reset world failed before its first policy decision")
                observation = env.observation()
                reset_seconds = time.monotonic() - started
            connection.send(
                dict(
                    ok=True,
                    observation=observation,
                    mask=env.legal_mask(),
                    reward=float(reward),
                    terminal=terminal,
                    elapsed_simulated_s=duration,
                    terminal_reason=info.get("terminal_reason"),
                    step_wall_seconds=step_seconds,
                    reset_wall_seconds=reset_seconds,
                    phase_timings=phase_timings,
                )
            )
    except BaseException:
        try:
            connection.send(dict(ok=False, error=traceback.format_exc()))
        except (OSError, EOFError, BrokenPipeError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                try:
                    connection.send(
                        dict(ok=False, error="Final replay flush failed: " + traceback.format_exc())
                    )
                except (OSError, EOFError, BrokenPipeError):
                    pass
        connection.close()


class ProcessWorlds:
    """Persistent one-process-per-world development pool, with explicit resets."""

    def __init__(
        self, factory, *, workers, timeout_seconds=650.0, close_timeout_seconds=60.0, heartbeat=None
    ):
        if type(workers) is not int or workers < 1 or timeout_seconds <= 0:
            raise ValueError("Positive workers and bounded worker timeout required")
        if not math.isfinite(close_timeout_seconds) or close_timeout_seconds <= 0:
            raise ValueError("Positive finite close timeout required")
        self.close_timeout_seconds = close_timeout_seconds
        self.close_wall_seconds = None
        self.workers, self.timeout_seconds, self.heartbeat = workers, timeout_seconds, heartbeat
        self.connections, self.processes = [], []
        self.observations, self.masks, self.initialization_seconds = [], [], []
        self.worker_peak_rss_bytes = [None] * workers
        self.closed = False
        context = mp.get_context("spawn")
        try:
            for worker in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, factory, worker))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for worker in range(workers):
                result = self._receive(worker)
                self.observations.append(result["observation"])
                self.masks.append(result["mask"])
                self.initialization_seconds.append(result["initialization_wall_seconds"])
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"Worker cleanup also failed: {cleanup_error}")
            raise

    def _receive(self, worker):
        started = last_heartbeat = time.monotonic()
        while not self.connections[worker].poll(1.0):
            now = time.monotonic()
            if self.heartbeat is not None and now - last_heartbeat >= 30:
                self.heartbeat()
                last_heartbeat = now
            if now - started > self.timeout_seconds or not self.processes[worker].is_alive():
                raise RuntimeError(f"Development worker{worker} failed or exceeded IPC timeout")
        try:
            result = self.connections[worker].recv()
        except EOFError as error:
            raise RuntimeError(f"Development worker{worker} exited without result") from error
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Unknown development worker failure"))
        return result

    def step(self, workers, actions):
        if self.closed or len(workers) != len(actions):
            raise ValueError("Open worlds and one joint action per selected world required")
        for worker, action in zip(workers, actions):
            self.connections[worker].send(("step", int(action)))
        results = []
        for worker in workers:
            result = self._receive(worker)
            self.observations[worker], self.masks[worker] = result["observation"], result["mask"]
            results.append(result)
        return results

    def close(self):
        if self.closed:
            return
        started = time.monotonic()
        deadline = started + self.close_timeout_seconds
        errors = []
        pending = []
        for worker, (connection, process) in enumerate(zip(self.connections, self.processes)):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                    pending.append(worker)
                except (OSError, BrokenPipeError) as error:
                    errors.append(f"worker{worker} close dispatch failed: {error}")
            else:
                errors.append(f"worker{worker} exited before close acknowledgement (exit={process.exitcode})")
        for worker in pending:
            connection = self.connections[worker]
            acknowledged = False
            while time.monotonic() < deadline:
                if connection.poll(min(1.0, max(0.0, deadline - time.monotonic()))):
                    try:
                        result = connection.recv()
                    except EOFError:
                        errors.append(f"worker{worker} EOF before final flush acknowledgement")
                    else:
                        if result.get("ok") is not True:
                            errors.append(f"worker{worker} final flush failed: {result.get('error')}")
                        elif (
                            result.get("command") != "close" or result.get("final_flush_complete") is not True
                        ):
                            errors.append(f"worker{worker} missing explicit final flush acknowledgement")
                        else:
                            acknowledged = True
                            self.worker_peak_rss_bytes[worker] = result.get("peak_rss_bytes")
                    break
            if not acknowledged and not any(f"worker{worker}" in message for message in errors):
                errors.append(f"worker{worker} close/flush timeout")
        for worker, (connection, process) in enumerate(zip(self.connections, self.processes)):
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                errors.append(f"worker{worker} did not exit after bounded final flush")
                process.terminate()
                process.join(timeout=5.0)
            if process.exitcode != 0:
                errors.append(f"worker{worker} nonzero close exit={process.exitcode}")
            connection.close()
        self.close_wall_seconds = time.monotonic() - started
        self.closed = True
        if errors:
            raise RuntimeError("; ".join(errors))

    def checkpoint(self):
        if self.closed:
            raise ValueError("Open workers required for checkpoint measurement")
        for connection in self.connections:
            connection.send(("checkpoint", None))
        states = [self._receive(worker) for worker in range(self.workers)]
        if any(row.get("command") != "checkpoint" for row in states):
            raise RuntimeError("Worker checkpoint acknowledgement required")
        return states

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if value is None:
                raise
            value.add_note(f"Worker final flush also failed: {cleanup_error}")


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_rollout(
    policy, worlds, *, transitions=4096, heartbeat=None, require_phase_timings=False, recovery_rows=None
):
    """Collect exactly N joint decisions, preserving unequal final world lengths.

    Small N is for software smoke tests only. The benchmark runner fixes4096.
    GAE is computed independently for each world's chronological transitions;
    terminal failure/timeout cuts both bootstrap and cross-episode recurrence.
    """
    if type(transitions) is not int or transitions <= 0:
        raise ValueError("Positive aggregate joint transition count required")
    device = next(policy.parameters()).device
    policy.eval()
    observation_rows = {key: [] for key in worlds.observations[0]}
    masks, actions, logps, values, rewards, terminals = [], [], [], [], [], []
    histories = [[] for _ in range(worlds.workers)]
    simulated = inference_s = copies_s = world_wall_s = reset_worker_s = execution_worker_s = 0.0
    reasons = Counter()
    phases = {key: 0.0 for key in PHASE_TIMINGS}
    phase_timed_transitions = 0
    world_phase_timed_counts = [0] * worlds.workers
    started = last_heartbeat = time.monotonic()
    while len(actions) < transitions:
        ids = list(range(min(worlds.workers, transitions - len(actions))))
        tick = time.monotonic()
        current = {
            k: torch.as_tensor(np.stack([worlds.observations[i][k] for i in ids]), device=device)
            for k in observation_rows
        }
        current_masks = torch.as_tensor(
            np.stack([worlds.masks[i] for i in ids]), device=device, dtype=torch.bool
        )
        _synchronize(device)
        copies_s += time.monotonic() - tick
        tick = time.monotonic()
        with torch.no_grad():
            selected, old_logp, predicted = policy.act(current, current_masks)
        _synchronize(device)
        inference_s += time.monotonic() - tick
        tick = time.monotonic()
        selected_list = selected.cpu().tolist()
        copies_s += time.monotonic() - tick
        tick = time.monotonic()
        outcomes = worlds.step(ids, selected_list)
        world_wall_s += time.monotonic() - tick
        for local, (worker, outcome) in enumerate(zip(ids, outcomes)):
            if require_phase_timings and outcome.get("phase_timings") is None:
                raise ValueError("Native benchmark requires phase timings on every transition")
            if recovery_rows is not None:
                recovery_rows.append(
                    dict(
                        worker=worker,
                        observation={k: current[k][local].detach().cpu().numpy().copy() for k in current},
                        mask=current_masks[local].detach().cpu().numpy().copy(),
                        action=int(selected[local]),
                        log_prob=float(old_logp[local]),
                        value=float(predicted[local]),
                        reward=float(outcome["reward"]),
                        terminal=outcome["terminal"],
                        elapsed=outcome["elapsed_simulated_s"],
                    )
                )
            histories[worker].append(len(actions))
            for key in observation_rows:
                observation_rows[key].append(current[key][local])
            masks.append(current_masks[local])
            actions.append(selected[local])
            logps.append(old_logp[local])
            values.append(predicted[local])
            rewards.append(outcome["reward"])
            terminals.append(outcome["terminal"])
            simulated += outcome["elapsed_simulated_s"]
            reset_worker_s += outcome["reset_wall_seconds"]
            execution_worker_s += outcome["step_wall_seconds"]
            if outcome.get("phase_timings") is not None:
                validate_phase_timings(
                    outcome["phase_timings"], execution_seconds=outcome["step_wall_seconds"]
                )
                phase_timed_transitions += 1
                world_phase_timed_counts[worker] += 1
                for key in PHASE_TIMINGS:
                    phases[key] += outcome["phase_timings"][key]
            if outcome["terminal"]:
                reasons[str(outcome["terminal_reason"])] += 1
        if heartbeat is not None and time.monotonic() - last_heartbeat >= 30:
            heartbeat(len(actions), transitions)
            last_heartbeat = time.monotonic()
    with torch.no_grad():
        last_observation = {
            k: torch.as_tensor(np.stack([o[k] for o in worlds.observations]), device=device)
            for k in observation_rows
        }
        _, last_values = policy(last_observation)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    value_tensor = torch.stack(values)
    terminal_tensor = torch.tensor(terminals, dtype=torch.bool, device=device)
    advantages, returns = torch.empty_like(value_tensor), torch.empty_like(value_tensor)
    for worker, history in enumerate(histories):
        if not history:
            continue
        advantage, target = generalized_advantage(
            reward_tensor[history, None],
            value_tensor[history, None],
            terminal_tensor[history, None],
            last_values[worker : worker + 1],
        )
        advantages[history], returns[history] = advantage[:, 0], target[:, 0]
    batch = PPOBatch(
        {k: torch.stack(v) for k, v in observation_rows.items()},
        torch.stack(masks),
        torch.stack(actions),
        torch.stack(logps),
        advantages,
        returns,
    )
    _synchronize(device)
    receipt = dict(
        scope="DEVELOPMENT_ONLY",
        joint_transitions=transitions,
        world_transition_counts=[len(h) for h in histories],
        elapsed_simulated_seconds=simulated,
        completed_episodes=sum(reasons.values()),
        terminal_reasons=dict(reasons),
        rollout_wall_seconds=time.monotonic() - started,
        policy_inference_wall_seconds=inference_s,
        tensor_copy_wall_seconds=copies_s,
        world_roundtrip_wall_seconds=world_wall_s,
        summed_worker_execution_seconds=execution_worker_s,
        summed_worker_reset_seconds=reset_worker_s,
        worker_time_note="Worker sums may overlap; never add them to elapsed wall time",
        phase_timed_transitions=phase_timed_transitions,
        world_phase_timed_counts=world_phase_timed_counts,
        summed_worker_phase_seconds=phases,
    )
    if require_phase_timings:
        validate_phase_coverage(receipt, workers=worlds.workers, transitions=transitions)
    return batch, receipt


def benchmark_summary(repetitions, *, workers, device):
    if len(repetitions) != 3 or any(
        type(r.get("joint_transitions")) is not int
        or r.get("joint_transitions") != 4096
        or type(r.get("epochs")) is not int
        or r.get("epochs") != 5
        or not math.isfinite(r.get("total_wall_seconds", float("nan")))
        or r["total_wall_seconds"] <= 0
        for r in repetitions
    ):
        raise ValueError("Exactly three complete4096-transition/five-epoch measured repetitions required")
    return dict(
        schema="dual-development-benchmark-v1",
        scope="DEVELOPMENT_ONLY",
        formal_qualification=False,
        production_transitions=0,
        selectable=False,
        workers=workers,
        policy_device=str(device),
        physics_backend="native-mujoco",
        repetitions=3,
        transitions_per_repetition=4096,
        epochs_per_update=5,
        measured_transitions=12288,
        warmup_transitions=4096,
        total_pilot_transitions=16384,
        median_repetition_seconds=float(np.median([r["total_wall_seconds"] for r in repetitions])),
        measurements=repetitions,
        storage_feasible=None,
        whole_campaign_projection_seconds=None,
    )


def _load_experiment_binding(path):
    from .dual_coordination_campaign import BoundFiles, EXPERIMENT_ROLES

    binding = BoundFiles.from_dict(json.loads(Path(path).read_text()))
    if set(binding.files) != EXPERIMENT_ROLES:
        raise ValueError("Complete experiment roles required")
    return binding


def _require_identity_binding(identity, binding):
    for role in ("source", "assets"):
        measured = {(r["sha256"], r["bytes"]) for r in identity[role]}
        declared = {(r["sha256"], r["bytes"]) for r in binding.files[role]}
        if not measured or not measured <= declared:
            raise ValueError(f"Measured {role} differs from experiment binding")
    if identity["audit_library"]:
        r = identity["audit_library"]
        if (r["sha256"], r["bytes"]) not in {(a["sha256"], a["bytes"]) for a in binding.files["audit"]}:
            raise ValueError("Measured audit library differs from experiment binding")
    for role, key in (("controller", "controller_sha256"), ("declaration", "cases_sha256")):
        hashes = set()
        for r in binding.files[role]:
            path = binding.root / r["path"]
            try:
                data = (
                    json.loads(path.read_text())
                    if path.suffix != ".jsonl"
                    else [json.loads(line) for line in path.read_text().splitlines()]
                )
                hashes.add(_json_hash(data))
            except (ValueError, UnicodeError):
                continue
        if identity[key] not in hashes:
            raise ValueError(f"Measured {role} content differs from experiment binding")
    if identity.get("online_auditor_override"):
        raise ValueError("Arbitrary online auditor cannot be a benchmark admission input")


def _assert_same_payload(expected, actual):
    """Read back our own local checkpoint, including every array and RNG byte."""
    if isinstance(expected, torch.Tensor):
        equal = isinstance(actual, torch.Tensor) and torch.equal(expected.cpu(), actual.cpu())
    elif isinstance(expected, np.ndarray):
        equal = (
            isinstance(actual, np.ndarray)
            and expected.dtype == actual.dtype
            and np.array_equal(expected, actual)
        )
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError("Checkpoint readback keys differ")
        for key in expected:
            _assert_same_payload(expected[key], actual[key])
        return
    elif isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise ValueError("Checkpoint readback sequence differs")
        for left, right in zip(expected, actual):
            _assert_same_payload(left, right)
        return
    else:
        equal = expected == actual
    if not equal:
        raise ValueError("Checkpoint readback payload differs")


def measure_checkpoint(trainer, worlds, recovery_rows, path):
    """Measure actual native snapshots, full rollout, NN/Adam/RNG atomic I/O.

    This development size sample is deliberately neither selectable nor a
    production recovery checkpoint. The production sampler history and archives
    need their separate measured storage envelope before admission.
    """
    if len(recovery_rows) != 4096 or trainer.budget.selectable:
        raise ValueError("Disposable complete 4096-row checkpoint size sample required")
    path = Path(path)
    prior = path.stat().st_size if path.exists() else 0
    started = time.monotonic()
    environment = dict(
        schema="dual-development-checkpoint-size-v1",
        workers=worlds.checkpoint(),
        partial_rollout=dict(
            schema="dual-partial-rollout-v1", workers=worlds.workers, count=4096, rows=recovery_rows
        ),
    )
    trainer.save(path, environment_state=environment, backend_state=None)
    write_seconds = time.monotonic() - started
    read_started = time.monotonic()
    # Only a checkpoint written above in this process is unpickled.
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in dict(
        environment_state=environment,
        policy=trainer.policy.state_dict(),
        optimizer=trainer.optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    ).items():
        _assert_same_payload(expected, loaded[key])
    if loaded["selectable"] or loaded["physical_resume_available"]:
        raise ValueError("Development checkpoint must not claim production recovery")
    record = _file_record(path)
    return dict(
        **record,
        wall_seconds=write_seconds,
        readback_wall_seconds=time.monotonic() - read_started,
        readback_verified=True,
        full_rollout_transitions=4096,
        prior_bytes=prior,
        atomic_peak_bytes=prior + record["bytes"],
        retained=True,
        scope="DEVELOPMENT_ONLY_SIZE_AND_IO_SAMPLE",
    )


def run_configuration(factory, *, workers, device, output, journal_root, cpu_quota, experiment_binding=None):
    """Run one real development configuration; caller explicitly starts long work."""
    if not isinstance(factory, NativeDevelopmentFactory):
        raise ValueError("Native development factory required for a reported performance run")
    if workers not in worker_candidates(cpu_quota):
        raise ValueError("Worker count outside measured CPU quota/candidate set")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Available CPU or CUDA policy device required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    journal = RunJournal(journal_root)
    attempt = journal.start(
        f"development-benchmark-{workers}-{device.type}",
        "Disposable native rollout/update benchmark; warm-up plus three measured repetitions",
        category="benchmark",
        log_path=output / "benchmark.json",
    )
    try:
        inputs = capture_input_identity(factory)
        binding = _load_experiment_binding(experiment_binding) if experiment_binding else None
        if binding:
            _require_identity_binding(inputs, binding)
        before = machine_inventory(output)
        measured_quota = before["cpu"]["effective_quota"]
        if cpu_quota > measured_quota + 1e-6 or workers not in worker_candidates(measured_quota):
            raise ValueError("Declared CPU quota/workers exceed measured process quota")
        if output.resolve() not in Path(factory.output).resolve().parents:
            raise ValueError(
                "Worker output must be beneath configuration output for measured storage accounting"
            )
        atomic_json(
            output / "inputs-and-machine.json",
            dict(
                input_identity=inputs,
                machine=before,
                experiment_inputs=binding.files if binding else None,
                requested_cpu_quota=cpu_quota,
            ),
        )
        torch.set_num_threads(1)
        random.seed(10272799)
        np.random.seed(10272799)
        torch.manual_seed(10272799)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(10272799)
            torch.cuda.reset_peak_memory_stats(device)
        initialized = time.monotonic()
        policy = JointActorCritic().to(device)
        trainer = JointPPO(policy, seed=42)
        trainer.budget = DisposablePilotBudget()  # Distinct from production seed counters.
        repetitions, checkpoints = [], []
        with ProcessWorlds(
            factory,
            workers=workers,
            heartbeat=lambda: journal.heartbeat(attempt, reason="Native worker executing"),
        ) as worlds:
            initialization_s = time.monotonic() - initialized
            for repetition in range(4):
                start = time.monotonic()
                recovery_rows = []
                batch, rollout = collect_rollout(
                    policy,
                    worlds,
                    require_phase_timings=True,
                    recovery_rows=recovery_rows,
                    heartbeat=lambda n, total: journal.heartbeat(
                        attempt,
                        completed=n,
                        total=total,
                        reason=f"Disposable repetition{repetition}:0=warm-up;1..3=measured",
                    ),
                )
                update_started = time.monotonic()
                update = trainer.update(batch, backend="native-mujoco")
                _synchronize(device)
                rollout.update(
                    epochs=5,
                    update_wall_seconds=time.monotonic() - update_started,
                    total_wall_seconds=time.monotonic() - start,
                    update_metrics=update,
                )
                if repetition:
                    repetitions.append(rollout)
                    sample = measure_checkpoint(
                        trainer, worlds, recovery_rows, output / "current-size-probe.pt"
                    )
                    for previous in checkpoints:
                        previous["retained"] = False
                    checkpoints.append(sample)
                    atomic_json(output / "checkpoint-measurements.json", checkpoints)
                else:
                    atomic_json(output / "warmup.json", rollout)
                atomic_json(output / f"repetition-{repetition}.json", rollout)
                journal.heartbeat(
                    attempt,
                    completed=trainer.budget.transitions,
                    total=16384,
                    reason="Disposable pilot transitions only; production counters remain0",
                )
                del batch
            current = output / "current-size-probe.pt"
            terminal = output / "terminal-size-probe.pt"
            os.link(current, terminal)
            same_inode = (current.stat().st_dev, current.stat().st_ino) == (
                terminal.stat().st_dev,
                terminal.stat().st_ino,
            )
            if not same_inode:
                raise ValueError("Terminal checkpoint hardlink differs from current payload")
            receipt = benchmark_summary(repetitions, workers=workers, device=device)
            receipt.update(
                initialization_wall_seconds=initialization_s,
                worker_initialization_wall_seconds=worlds.initialization_seconds,
                torch_version=torch.__version__,
                numpy_version=np.__version__,
                cpu_quota=cpu_quota,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                checkpoint_saved=True,
                checkpoint_measurements=checkpoints,
                checkpoint_storage_probe=dict(
                    schema="dual-checkpoint-storage-probe-v1",
                    overwrite_verified=True,
                    terminal_hardlink_verified=True,
                    unique_current_payloads=1,
                    full_rollout_transitions=4096,
                    max_atomic_bytes=max(cp["atomic_peak_bytes"] for cp in checkpoints),
                    terminal=_file_record(terminal),
                    production_history_envelope_measured=False,
                ),
            )
        receipt.update(
            final_flush_wall_seconds=worlds.close_wall_seconds,
            total_configuration_wall_seconds=time.monotonic() - initialized,
            machine_before=before,
            machine_after=machine_inventory(output),
            input_identity=inputs,
            experiment_inputs=binding.files if binding else None,
            peak_storage_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
            completed=True,
            worker_peak_rss_bytes=worlds.worker_peak_rss_bytes,
            timing_scope="Complete measured env phases for every transition; worker sums are diagnostic and never added to elapsed wall time",
        )
        if capture_input_identity(factory) != inputs:
            raise ValueError("Executable benchmark inputs changed during measurement")
        if receipt["machine_after"]["dependencies"] != before["dependencies"]:
            raise ValueError("Dependencies changed during benchmark measurement")
        if binding:
            binding.verify()
        atomic_json(output / "benchmark.json", receipt)
        journal.finish(
            attempt,
            status="passed",
            reason="Three measured repetitions completed; no production admission implied",
        )
        return receipt
    except BaseException as error:
        atomic_json(output / "failure.json", dict(completed=False, error=repr(error)))
        journal.finish(attempt, status="failed", reason=repr(error))
        raise


def _positive(value, label, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"Finite {'nonnegative' if zero else 'positive'} {label} required")
    return value


def reduce_benchmarks(
    configuration_paths,
    *,
    output,
    experiment_binding=None,
    qualification_path=None,
    execution_profile_path=None,
    cost_measurements_path=None,
    backend_restriction_path=None,
):
    """Reduce retained measurements; incomplete evidence yields no admission schema.

    This reducer does not execute experiments or certify physics. Qualifying
    evidence must be supplied by the separate native qualification producer.
    Cost sample records bind actual measurements to inputs, machine and the
    predeclared execution/replay profile. No synthetic default timing is used.
    """
    result = dict(
        schema="dual-benchmark-diagnostic-v1",
        selected_backend=None,
        selected_workers=None,
        storage_feasible=None,
        whole_campaign_projection_seconds=None,
        missing_evidence=[],
        gpu_physics=dict(
            qualified=False,
            status="unqualified",
            backend="warp-cuda",
            reason="36 paired trials and a complete stage-audit/controller/mask adapter are not implemented",
        ),
        configurations=[],
        evidence=[],
    )

    def read(path):
        result["evidence"].append(_file_record(path))
        return json.loads(Path(path).read_text())

    try:
        if not configuration_paths or any(
            p is None
            for p in (experiment_binding, qualification_path, execution_profile_path, cost_measurements_path)
        ):
            raise ValueError(
                "Complete configuration runs, experiment binding, qualification, execution profile and cost measurements required"
            )
        binding = _load_experiment_binding(experiment_binding)
        restriction = None
        if backend_restriction_path is not None:
            restriction = read(backend_restriction_path)
            if (
                restriction.get("schema") != "dual-backend-restriction-v1"
                or restriction.get("decision") != "cpu-only"
                or restriction.get("source") != "user"
                or not isinstance(restriction.get("authorization"), str)
                or not restriction["authorization"].strip()
                or restriction.get("experiment_binding_sha256") != sha256(experiment_binding)
            ):
                raise ValueError("Explicit user CPU-only restriction bound to this experiment required")
            result["backend_restriction"] = restriction
            result["reducer_identity"] = _file_record(__file__)
        result["evidence"].append(_file_record(experiment_binding))
        qualification, profile = read(qualification_path), read(execution_profile_path)
        from .dual_coordination_campaign import _qualification

        _qualification(qualification)
        if qualification.get("experiment_inputs") != binding.files:
            raise ValueError("Qualification experiment inputs differ")
        if (
            profile.get("schema") != "dual-execution-profile-v1"
            or type(profile.get("replays_per_mission")) is not int
            or profile["replays_per_mission"] not in (0, 1)
            or qualification.get("execution_profile_sha256") != sha256(execution_profile_path)
            or qualification.get("compact_online_evidence_verified") is not True
        ):
            raise ValueError("Qualified predeclared compact evidence/replay execution profile required")
        counts = dict(
            training_updates=768,
            validation_missions=1500,
            validation_diagnostic_missions=600,
            heldout_missions=1500,
            physical_missions=127,
        )
        for key, count in counts.items():
            if type(profile.get(key)) is not int or profile[key] != count:
                raise ValueError(f"Declared {key} must equal {count}")
        for key in ("checkpoint_rounds", "media_items", "recovery_attempts"):
            if type(profile.get(key)) is not int or profile[key] <= 0:
                raise ValueError(f"Predeclared positive {key} required")
        if profile["checkpoint_rounds"] != 64:
            raise ValueError("Actual trainer 64-round recovery schedule required")
        configurations = []
        identity = machine_id = candidates = dependencies = None
        available_devices = None
        seen = set()
        for path in configuration_paths:
            row = read(path)
            if row.get("schema") != "dual-development-benchmark-v1" or row.get("completed") is not True:
                raise ValueError("Completed native development benchmark configuration required")
            device = str(row["policy_device"]).split(":")[0]
            if device not in ("cpu", "cuda") or row.get("physics_backend") != "native-mujoco":
                raise ValueError("Only measured native CPU physics with CPU/CUDA policy is implemented")
            summary = benchmark_summary(row["measurements"], workers=row["workers"], device=device)
            for measurement in row["measurements"]:
                validate_phase_coverage(measurement, workers=row["workers"])
            for key in (
                "repetitions",
                "transitions_per_repetition",
                "epochs_per_update",
                "measured_transitions",
                "warmup_transitions",
                "total_pilot_transitions",
                "median_repetition_seconds",
            ):
                if row.get(key) != summary[key]:
                    raise ValueError(f"Configuration {key} differs from complete measured repetitions")
            if row.get("experiment_inputs") != binding.files:
                raise ValueError("Configuration experiment inputs differ")
            _require_identity_binding(row["input_identity"], binding)
            if identity is None:
                identity = row["input_identity"]
                machine_id = row["machine_before"]["machine_id"]
                candidates = worker_candidates(row["machine_before"]["cpu"]["effective_quota"])
                dependencies = row["machine_before"]["dependencies"]
                available_devices = ("cpu", "cuda") if row["machine_before"]["gpu_devices"] else ("cpu",)
            if row["input_identity"] != identity:
                raise ValueError("Configuration source/assets/controller/audit identities differ")
            for phase in ("machine_before", "machine_after"):
                host = row[phase]
                if (
                    host["machine_id"] != machine_id
                    or worker_candidates(host["cpu"]["effective_quota"]) != candidates
                ):
                    raise ValueError("Configuration machine/quota changed")
                if (("cpu", "cuda") if host["gpu_devices"] else ("cpu",)) != available_devices:
                    raise ValueError("GPU inventory changed during configuration measurement")
                if host["dependencies"] != dependencies:
                    raise ValueError("Measured dependencies changed between configurations")
                _positive(host["disk"]["free_bytes"], "measured free disk")
                _positive(host["memory"]["physical_bytes"], "measured physical RAM")
            key = (row["workers"], device)
            if (
                key in seen
                or row["workers"] not in candidates
                or device not in (("cpu",) if restriction else available_devices)
            ):
                raise ValueError("Duplicate or undeclared worker/device configuration")
            seen.add(key)
            if len(row.get("checkpoint_measurements", [])) != 3:
                raise ValueError("Three actual complete checkpoint/flush size and I/O measurements required")
            probe = row.get("checkpoint_storage_probe", {})
            if (
                probe.get("schema") != "dual-checkpoint-storage-probe-v1"
                or probe.get("overwrite_verified") is not True
                or probe.get("terminal_hardlink_verified") is not True
                or probe.get("unique_current_payloads") != 1
                or probe.get("full_rollout_transitions") != 4096
            ):
                raise ValueError("Measured bounded checkpoint overwrite/hardlink probe required")
            checkpoints = row["checkpoint_measurements"]
            for index, cp in enumerate(checkpoints):
                _positive(cp["wall_seconds"], "checkpoint wall seconds")
                _positive(cp["bytes"], "checkpoint bytes")
                if (
                    cp.get("readback_verified") is not True
                    or cp.get("full_rollout_transitions") != 4096
                    or cp.get("retained") is not (index == 2)
                    or cp["path"] != checkpoints[-1]["path"]
                    or cp.get("prior_bytes") != (checkpoints[index - 1]["bytes"] if index else 0)
                    or cp.get("atomic_peak_bytes") != cp["bytes"] + cp["prior_bytes"]
                ):
                    raise ValueError("Checkpoint overwrite/full-rollout measurements differ")
                if index == 2:
                    actual = _file_record(cp["path"])
                    if actual["bytes"] != cp["bytes"] or actual["sha256"] != cp["sha256"]:
                        raise ValueError("Actual retained checkpoint sample differs")
                    result["evidence"].append(actual)
            terminal = _file_record(probe["terminal"]["path"])
            if terminal != probe["terminal"] or terminal["sha256"] != checkpoints[-1]["sha256"]:
                raise ValueError("Terminal checkpoint payload differs")
            a, b = Path(terminal["path"]).stat(), Path(checkpoints[-1]["path"]).stat()
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino) or probe["max_atomic_bytes"] != max(
                cp["atomic_peak_bytes"] for cp in checkpoints
            ):
                raise ValueError("Actual bounded checkpoint retention differs")
            result["evidence"].append(terminal)
            for key in (
                "final_flush_wall_seconds",
                "total_configuration_wall_seconds",
                "initialization_wall_seconds",
                "peak_storage_bytes",
            ):
                _positive(row[key], key)
            configurations.append(row)
        if seen != {
            (workers, device)
            for workers in candidates
            for device in (("cpu",) if restriction else available_devices)
        }:
            raise ValueError(
                "Full three-repetition CPU/CUDA results for every quota-capped worker candidate required"
            )
        if qualification.get("audit_backend", "python-native") != identity["audit_backend"]:
            raise ValueError("Qualification audit backend differs from timed backend")
        if (
            identity["audit_library"]
            and qualification.get("audit_library_sha256") != identity["audit_library"]["sha256"]
        ):
            raise ValueError("Qualification compiled library differs from timed library")
        costs = read(cost_measurements_path)
        if costs.get("schema") != "dual-benchmark-costs-v1":
            raise ValueError("Actual campaign cost sample inventory required")
        units = dict(
            training_archive=3 * 256 * 4096,
            validation_mission=profile["validation_missions"] + profile["validation_diagnostic_missions"],
            heldout_mission=1500,
            physical_mission=127,
            qualification=1,
            media=profile["media_items"],
            recovery=profile["recovery_attempts"],
        )
        if profile["replays_per_mission"]:
            units.update(
                validation_replay=units["validation_mission"], heldout_replay=1500, physical_replay=127
            )
        samples = {key: [] for key in units}
        sample_paths = set()
        from .dual_coordination_campaign import _path

        for record in costs.get("samples", []):
            path = _path(Path(cost_measurements_path).resolve().parent, record["path"])
            if (
                path in sample_paths
                or path.stat().st_size != record["bytes"]
                or sha256(path) != record["sha256"]
            ):
                raise ValueError("Duplicate, changed or unconfined cost sample")
            sample_paths.add(path)
            sample = read(path)
            if (
                sample.get("schema") != "dual-benchmark-cost-sample-v1"
                or sample.get("actual_execution") is not True
                or sample.get("measurement_cohort") != "development"
                or sample.get("experiment_inputs") != binding.files
                or sample.get("machine_id") != machine_id
                or sample.get("execution_profile_sha256") != sha256(execution_profile_path)
            ):
                raise ValueError("Cost sample execution/input/machine/profile binding differs")
            if sample.get("kind") not in samples:
                raise ValueError("Undeclared cost sample kind")
            for key in ("units", "wall_seconds", "peak_storage_bytes"):
                _positive(sample[key], f"cost sample {key}")
            _positive(sample["retained_bytes"], "retained bytes", zero=True)
            if sample["kind"] == "training_archive":
                envelope = sample.get("checkpoint_envelope", {})
                if (
                    envelope.get("scope") != "DECLARED_SHAPE_SIZE_PROBE"
                    or envelope.get("full_rollout_transitions") != 4096
                    or envelope.get("max_curve_rows") != 256
                    or type(envelope.get("max_assignment_rows")) is not int
                    or envelope["max_assignment_rows"] < 1048576 + 3 * max(candidates)
                ):
                    raise ValueError("Measured training checkpoint finite-history shape envelope required")
                envelope_path = _path(Path(cost_measurements_path).resolve().parent, envelope["path"])
                observed = _file_record(envelope_path)
                if observed["bytes"] != envelope.get("bytes") or observed["sha256"] != envelope.get("sha256"):
                    raise ValueError("Measured training checkpoint envelope artifact differs")
                _positive(envelope.get("atomic_write_seconds"), "checkpoint envelope atomic write seconds")
                result["evidence"].append(observed)
            samples[sample["kind"]].append(sample)
        if any(
            len(rows) < (3 if kind.endswith(("_mission", "_replay")) else 1) for kind, rows in samples.items()
        ):
            raise ValueError(
                "Three development mission/replay samples and measured training_archive/qualification/media/recovery costs required"
            )
        components = {
            kind: dict(
                units=units[kind],
                seconds_per_unit=float(np.median([r["wall_seconds"] / r["units"] for r in rows])),
                retained_bytes_per_unit=max(r["retained_bytes"] / r["units"] for r in rows),
                peak_bytes=max(r["peak_storage_bytes"] for r in rows),
            )
            for kind, rows in samples.items()
        }
        other_seconds = sum(v["units"] * v["seconds_per_unit"] for v in components.values())
        pilot_seconds = sum(r["total_configuration_wall_seconds"] for r in configurations)
        retained = sum(v["units"] * v["retained_bytes_per_unit"] for v in components.values())
        transient = max(v["peak_bytes"] for v in components.values())
        storage_pilots = sum(r["peak_storage_bytes"] for r in configurations)
        projected = []
        for row in configurations:
            cp_seconds = float(np.median([cp["wall_seconds"] for cp in row["checkpoint_measurements"]]))
            cp_bytes = max(cp["bytes"] for cp in row["checkpoint_measurements"])
            # A full production-history envelope must be measured separately;
            # pilot snapshots alone cannot establish its maximum retained size.
            envelopes = [sample["checkpoint_envelope"]["bytes"] for sample in samples["training_archive"]]
            if any(type(n) is not int or n < cp_bytes for n in envelopes):
                raise ValueError(
                    "Measured training archive and complete checkpoint history storage envelope required"
                )
            cp_bytes = max(envelopes)
            # Add the measured finite-history envelope I/O conservatively to
            # the native snapshot + pilot write time; never assume size-free I/O.
            cp_seconds += max(
                sample["checkpoint_envelope"]["atomic_write_seconds"]
                for sample in samples["training_archive"]
            )
            checkpoint_writes = training_checkpoint_writes(
                workers=row["workers"], checkpoint_rounds=profile["checkpoint_rounds"]
            )
            seconds = (
                768 * row["median_repetition_seconds"]
                + checkpoint_writes * cp_seconds
                + 3 * (row["initialization_wall_seconds"] + row["final_flush_wall_seconds"])
                + other_seconds
                + pilot_seconds
            )
            storage = (
                retained
                + transient
                + storage_pilots
                + row["peak_storage_bytes"]  # conservative active-world proxy: includes development replay
                + 4 * cp_bytes  # two completed sequential seeds plus current + atomic replacement
            )
            free = min(r["machine_after"]["disk"]["free_bytes"] for r in configurations)
            projected.append(
                dict(
                    backend="native-" + str(row["policy_device"]).split(":")[0],
                    workers=row["workers"],
                    whole_campaign_projection_seconds=seconds,
                    projected_storage_bytes=storage,
                    checkpoint_writes=checkpoint_writes,
                    peak_unique_checkpoint_payloads=4,
                    measured_free_bytes=free,
                    storage_headroom_factor=1.25,
                    storage_feasible=free >= 1.25 * storage,
                    learning_seconds=768 * row["median_repetition_seconds"],
                )
            )
        result["configurations"] = projected
        cpu = min(
            (r for r in projected if r["backend"] == "native-cpu" and r["storage_feasible"]),
            key=lambda x: x["whole_campaign_projection_seconds"],
            default=None,
        )
        if cpu is None:
            raise ValueError("Measured storage headroom insufficient for every CPU candidate")
        best = min(
            (r for r in projected if r["storage_feasible"]),
            key=lambda x: x["whole_campaign_projection_seconds"],
        )
        chosen = (
            best
            if best["whole_campaign_projection_seconds"] < 0.9 * cpu["whole_campaign_projection_seconds"]
            else cpu
        )
        result.update(
            schema="dual-benchmark-v1",
            repetitions=3,
            transitions_per_repetition=4096,
            epochs_per_update=5,
            selected_backend=chosen["backend"],
            selected_workers=chosen["workers"],
            storage_feasible=True,
            whole_campaign_projection_seconds=chosen["whole_campaign_projection_seconds"],
            whole_campaign_speedup=cpu["whole_campaign_projection_seconds"]
            / chosen["whole_campaign_projection_seconds"],
            learning_only_speedup=cpu["learning_seconds"] / chosen["learning_seconds"],
            experiment_inputs=binding.files,
            execution_profile=profile,
            cost_components=components,
            pilot_wall_seconds=pilot_seconds,
            production_transitions=0,
            projection_assumptions=dict(
                training_updates=768,
                validation_selection_missions=1500,
                validation_diagnostic_missions=600,
                seeds=3,
                stage_updates_all_seeds=[192, 528, 48],
                mission_costs="Actual development analogues; no validation/heldout access",
                training_storage="Measured compact training archives plus conservative replay-retaining development active-world proxy",
                checkpoint_storage="Three sequential seeds: peak four unique payloads, terminal/current hardlinks; write count affects time only",
                storage_headroom_factor=1.25,
                phase_timings=list(PHASE_TIMINGS),
            ),
            warp_fidelity36_passed=False,
            gpu_selection_note="GPU acceleration was not beneficial for the measured configuration"
            if chosen["backend"] == "native-cpu" and "cuda" in available_devices
            else "Native CPU physics with CUDA policy selected"
            if chosen["backend"] == "native-cuda"
            else "CUDA unavailable on measured machine",
        )
        if restriction is not None:
            result["gpu_selection_note"] = (
                "CPU-only continuation requested by user; GPU physics speedup unmeasured"
            )
        binding.verify()
        for record in result["evidence"]:
            if _file_record(record["path"]) != record:
                raise ValueError("Benchmark evidence changed during reduction")
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update(
            schema="dual-benchmark-diagnostic-v1",
            selected_backend=None,
            selected_workers=None,
            storage_feasible=None,
            whole_campaign_projection_seconds=None,
        )
        result["missing_evidence"].append(str(error))
    atomic_json(output, result)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "reduce":
        parser = argparse.ArgumentParser(
            description="Reduce measured configurations; incomplete inputs yield diagnostic only"
        )
        parser.add_argument("--configuration", action="append", default=[])
        parser.add_argument("--experiment-binding")
        parser.add_argument("--qualification")
        parser.add_argument("--execution-profile")
        parser.add_argument("--cost-measurements")
        parser.add_argument("--backend-restriction")
        parser.add_argument("--output", required=True)
        args = parser.parse_args(argv[1:])
        return reduce_benchmarks(
            args.configuration,
            output=args.output,
            experiment_binding=args.experiment_binding,
            qualification_path=args.qualification,
            execution_profile_path=args.execution_profile,
            cost_measurements_path=args.cost_measurements,
            backend_restriction_path=args.backend_restriction,
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="Original MuJoCo source XML path")
    parser.add_argument("--cases", required=True, help="JSON list of fixed declared split15 cases")
    parser.add_argument("--design", required=True, help="Existing controller design JSON")
    parser.add_argument("--output", required=True, help="New configuration output directory")
    parser.add_argument("--journal-root", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--cpu-quota", required=True, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--native-audit-library", help="Development-only compiled audit shared library")
    parser.add_argument("--replay-stride-ticks", type=int, choices=(1, 20), default=20)
    parser.add_argument("--experiment-binding", help="Existing BoundFiles JSON with all experiment roles")
    args = parser.parse_args(argv)
    factory = NativeDevelopmentFactory(
        str(Path(args.original).resolve()),
        str(Path(args.output).resolve() / "worlds"),
        tuple(json.loads(Path(args.cases).read_text())),
        json.loads(Path(args.design).read_text()),
        native_audit_library=str(Path(args.native_audit_library).resolve())
        if args.native_audit_library
        else None,
        replay_stride_ticks=args.replay_stride_ticks,
    )
    run_configuration(
        factory,
        workers=args.workers,
        device=args.device,
        output=args.output,
        journal_root=args.journal_root,
        cpu_quota=args.cpu_quota,
        experiment_binding=args.experiment_binding,
    )


if __name__ == "__main__":
    main()
