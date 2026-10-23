"""Independent curriculum accounting over already reconciled valid transitions.

This arithmetic alone cannot certify physical learning. Its caller must first
reconcile the training records with native getters and shared settling. In
particular stochastic training actions are not the final mean policy outputs.
"""

import numpy as np


class TrainingResetReplay:
    """Regenerate the declared independent load and pose reset streams."""

    def __init__(self, method, seed):
        from .payload_contract import PayloadSampler
        from .payload_solver_mujoco import FREE_MASS_KG

        if method not in {"randomized", "fixed"}:
            raise ValueError("Known training reset method required")
        self.sampler = PayloadSampler(FREE_MASS_KG, seed, "training", randomized=method == "randomized")
        self.pose = np.random.default_rng(np.random.SeedSequence([seed, 710, 101]))
        self.episode = 0

    def verify(self, reset):
        from .payload_cases import initial_state

        payload = self.sampler.reset([0])[0]
        expected = initial_state(self.pose, payload.side_m)
        self.episode += 1
        if (reset["episode"] != self.episode or reset["draw_index"] != self.sampler.draw_indices[0]
                or reset["payload"] != vars(payload) or set(reset["initial"]) != set(expected)
                or any(not np.allclose(reset["initial"][key], value, atol=1e-14, rtol=0)
                       for key, value in expected.items())):
            raise ValueError("Actual training reset differs from declared method/seed/draw stream")
        return True


def exposure_summary(records, *, destinations=None):
    """Count stage-local consecutive intervals and distinct within-episode goals."""
    expected = None if destinations is None else {tuple(map(float, p)) for p in destinations}
    if expected is not None and len(expected) != 10:
        raise ValueError("Ten distinct bound destinations required")
    counts, longest = [0] * 6, [0] * 6
    certificates, goals = [], set()
    previous_step = previous_episode = previous_local = previous_stage = 0
    previous_done = False
    run = 0
    for row in records:
        step, episode, local, stage = [row[k] for k in ["valid_step", "episode", "episode_step", "stage"]]
        if (
            any(type(v) is not int for v in [step, episode, local, stage])
            or step != previous_step + 1
            or not 0 <= stage < 6
            or (episode == previous_episode and local != previous_local + 1)
            or (episode != previous_episode and (episode != previous_episode + 1 or local != 1))
        ):
            raise ValueError("Exposure requires consecutive actual valid steps and episodes")
        if type(row.get("done")) is not bool or (
            previous_episode and ((episode != previous_episode) != previous_done)
        ):
            raise ValueError("Episode reset/continuation differs from actual terminal")
        if episode != previous_episode:
            goals = set()
        run = run + 1 if episode == previous_episode and stage == previous_stage else 1
        counts[stage] += 1
        longest[stage] = max(longest[stage], run)
        target = np.asarray(row["target"], dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or type(row["shared_complete"]) is not bool:
            raise ValueError("Measured command and independently verified completion required")
        if stage == 5 and row["shared_complete"]:
            goal = tuple(target.tolist())
            if expected is not None and goal not in expected:
                raise ValueError("Actual stage5 target differs from bound layout")
            goals.add(goal)
            if len(goals) == 10 and episode not in certificates:
                certificates.append(episode)
        previous_step, previous_episode, previous_local, previous_stage = step, episode, local, stage
        previous_done = row["done"]
    return dict(
        valid_transitions=previous_step,
        stage_valid_transitions=counts,
        stage_max_continuous_steps=longest,
        full_stage_ten_target_episodes=certificates,
        destinations_bound=expected is not None,
        physical_acceptance=False,
    )


def require_exposure(summary, *, min_transitions=10000, min_continuous=500):
    """Reject insufficient experience before freezing final training parameters."""
    if (
        not summary.get("destinations_bound")
        or len(summary["stage_valid_transitions"]) != 6
        or len(summary["stage_max_continuous_steps"]) != 6
        or any(x < min_transitions for x in summary["stage_valid_transitions"])
        or any(x < min_continuous for x in summary["stage_max_continuous_steps"])
        or not summary["full_stage_ten_target_episodes"]
    ):
        raise ValueError("Actual curriculum exposure is insufficient")
    return True


def reconcile_training_step(action, transition, native, previous, side, history):
    """Reconcile stochastic inputs and measured actor fields, without mean replay."""
    from .payload_retry_contract import actor_observation
    from .payload_solver_trace import observation_sample

    raw_action = np.asarray(action["raw_action"], dtype=float)
    if raw_action.shape != (4,) or not np.isfinite(raw_action).all() or native["kind"] != "STEP":
        raise ValueError("Finite stochastic action and native STEP required")
    if (
        type(native["step"]) is not int
        or native["step"] <= 0
        or not np.isclose(native["actual"]["time_s"], native["step"] * 0.01, atol=1e-7, rtol=0)
    ):
        raise ValueError("Actual native clock differs from full control intervals")
    target = transition["target"]
    if not np.array_equal(native["target"], target):
        raise ValueError("Native training target differs")
    obs = actor_observation(observation_sample(previous, side), target, history)
    for recorded, tolerance in [
        (action["actor_observation"], 1e-7),
        (native["actor_observation_before_step"], 1e-10),
    ]:
        if not np.allclose(recorded, obs, atol=tolerance, rtol=1e-7):
            raise ValueError("Training observation differs from native measured state/history")
    clipped = np.clip(raw_action, -1, 1)
    if not np.array_equal(native["actions"], clipped):
        raise ValueError("Native applied action differs from current stochastic input")
    if not np.allclose(
        native["actual"]["rpm"], (1 + 0.8 * clipped) * 14468.429183500699, atol=1e-7, rtol=1e-12
    ):
        raise ValueError("Actual native RPM differs from current action")
    for key, value in native["actual"].items():
        if key == "time_s":
            if not np.isclose(transition["actual"][key], value, atol=1e-7, rtol=0):
                raise ValueError("Actual native training clock differs")
        elif key not in transition["actual"] or not np.array_equal(transition["actual"][key], value):
            raise ValueError("Training telemetry differs from native getter: " + key)
    return clipped


def audit_training_prefix(root, steps):
    """Replay an immutable run prefix against native per-geometry step streams.

    Certifies action/observation/native/sample/exposure accounting. It does not
    reconstruct past stochastic actor weights or random-number generator states.
    """
    from contextlib import ExitStack
    import gzip
    import json
    from pathlib import Path
    from .payload_contract import SharedSettling, training_outside_envelope
    from .payload_solver_qualification import physical_errors
    from .payload_solver_validation import measured_predicates, verify_initial_sample
    from .payload_solver_trace import match_native_sample
    from .util import sha256

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    common = manifest["common"]
    if steps not in manifest["checkpoints"] or common["N"] != 1 or common["T"] != 100:
        raise ValueError("Actual checkpoint prefix must be predeclared with N1/T100")
    reset_replay = TrainingResetReplay(manifest["method"], common["seed"])
    completed = json.loads((root / "completed.json").read_text())
    if completed["status"] != "TRAINED_NOT_VALIDATED" or completed["counters"]["invalid"]:
        raise ValueError("Closed valid learning run required")
    sidecar_path = root / f"checkpoint_s{steps}.json"
    sidecar = json.loads(sidecar_path.read_text())
    checkpoint = root / f"checkpoint_s{steps}.pt"
    if sha256(checkpoint) != sidecar["checkpoint_sha256"]:
        raise ValueError("Actual prefix checkpoint hash differs")
    import torch

    info = torch.load(checkpoint, map_location="cpu", weights_only=False)["infos"]
    if (
        info["method"] != manifest["method"]
        or info["common_training_plan"] != common
        or info["valid_environment_transitions"] != steps
        or info["completed_rollout_updates"] != steps // 100
    ):
        raise ValueError("Embedded checkpoint prefix provenance differs")
    expected_updates = steps // 100
    if sidecar["optimizer"] != dict(
        rollout_updates=expected_updates,
        optimizer_gradient_steps=20 * expected_updates,
        consumed_transitions=steps,
    ):
        raise ValueError("Actual optimizer/sample prefix counts differ")
    files = [
        root / "environment" / name
        for name in ["training.jsonl.gz", "reset-properties.jsonl.gz", "terminal.jsonl.gz"]
    ]
    native_files = {
        side: root / "environment" / f"runtime-{side:.3f}" / "raw.jsonl.gz" for side in [0.03, 0.045, 0.06]
    }
    before_hashes = {
        str(p.relative_to(root)): sha256(p)
        for p in [*files, *native_files.values(), sidecar_path, root / "manifest.json"]
    }
    checked = dict(valid=0, generated=0, invalid=0, completed_training_stops=0, failed_episodes=0)
    experiences = []
    local_episodes = {side: 0 for side in native_files}
    with ExitStack() as stack:

        def lines(path):
            stream = stack.enter_context(gzip.open(path, "rt"))
            return (json.loads(line) for line in stream)

        records, resets, terminals = [lines(p) for p in files]
        natives = {side: lines(path) for side, path in native_files.items()}
        episode = 0
        previous = history = gate = runtime = None
        side = stage = local = command_steps = episode_stops = start_stage = command_count = 0

        def reconciled():
            nonlocal episode, previous, history, gate, side, stage, local, command_steps, episode_stops
            nonlocal start_stage, command_count
            nonlocal runtime
            for index in range(1, steps + 1):
                action, row = next(records), next(records)
                if (
                    action["kind"] != "ACTION_INPUT"
                    or row["kind"] != "VALID_TRANSITION"
                    or action["generated_step"] != index
                    or row["valid_step"] != index
                    or action["episode"] != row["episode"]
                ):
                    raise ValueError("Training input/valid sequence differs")
                if row["episode"] != episode:
                    if row["episode"] != episode + 1:
                        raise ValueError("Training logical episode skipped")
                    reset = next(resets)
                    reset_replay.verify(reset)
                    runtime = reset["runtime"]
                    if reset["episode"] != row["episode"]:
                        raise ValueError("Actual reset draw differs from logical episode")
                    side = reset["payload"]["side_m"]
                    native_reset = next(natives[side])
                    local_episodes[side] += 1
                    if (
                        native_reset["kind"] != "RESET"
                        or native_reset["step"] != 0
                        or native_reset["episode_id"] != local_episodes[side]
                        or native_reset["actual"] != reset["actual"]
                        or np.any(native_reset["actions"])
                        or np.any(native_reset["actual"]["rpm"])
                        or native_reset["actual"]["time_s"] != 0.0
                    ):
                        raise ValueError("Native reset state/action history differs")
                    if physical_errors(
                        reset["runtime"],
                        dict(side_m=side, mass_ratio=reset["payload"]["mass_ratio"], substeps=16),
                        fixed=False,
                    ):
                        raise ValueError("Actual training reset physical properties differ")
                    verify_initial_sample(reset["actual"], reset["initial"])
                    episode, local, command_steps, episode_stops = row["episode"], 0, 0, 0
                    command_count = 0
                    previous, history = native_reset["actual"], np.zeros(4)
                    stage = 0 if common["pilot"] else min(5, (index - 1) * 6 // common["curriculum_steps"])
                    start_stage = stage
                    gate = SharedSettling(0.01)
                    initial_sample = dict(
                        previous, contacts=bool(previous["contact"]), done=False, reset=False
                    )
                    match_native_sample(initial_sample, previous, runtime)
                    valid = measured_predicates(initial_sample, native_reset["target"], side)["valid"]
                    gate.update(0.0, valid, native_reset["target_id"])
                local += 1
                command_steps += 1
                native = next(natives[side])
                if (
                    native["episode_id"] != local_episodes[side]
                    or native["step"] != local
                    or row["episode_step"] != local
                    or row["stage"] != stage
                ):
                    raise ValueError("Native local steps/current-command stage differ")
                match_native_sample(row["actual"], native["actual"], runtime)
                history = reconcile_training_step(action, row, native, previous, side, history)
                measured = measured_predicates(row["actual"], row["target"], side)
                failed = bool(measured["unsafe"] or training_outside_envelope(row["actual"], row["target"]))
                complete = gate.update(
                    local * 0.01, bool(measured["valid"] and not failed), native["target_id"]
                )
                if row["shared_complete"] != complete or row["unsafe"] != failed:
                    raise ValueError("Training shared completion/failure differs from measured predicates")
                episode_stops += int(complete)
                done = bool(failed or local >= 12000 or (complete and episode_stops >= 10))
                if row["done"] != done:
                    raise ValueError("Actual terminal condition differs")
                checked["valid"] += 1
                checked["generated"] += 1
                checked["completed_training_stops"] += int(complete)
                previous = native["actual"]
                if done:
                    terminal = next(terminals)
                    for key, value in dict(
                        valid_step=index,
                        episode=episode,
                        episode_steps=local,
                        stage=stage,
                        command_count=command_count,
                        failed=failed,
                        timeout=local >= 12000,
                        sample=row["actual"],
                    ).items():
                        if terminal[key] != value:
                            raise ValueError("Terminal record differs from actual native transition")
                    checked["failed_episodes"] += int(failed)
                    experiences.append(
                        dict(
                            episode=episode,
                            start_stage=start_stage,
                            end_stage=stage,
                            continuous_steps=local,
                            completed_target_count=episode_stops,
                            attempted_target_count=command_count + 1,
                            failed=failed,
                        )
                    )
                yield row
                if not done and (complete or command_steps >= 800):
                    retarget = next(natives[side])
                    if (
                        retarget["kind"] != "RETARGET"
                        or retarget["actual"] != previous
                        or retarget["step"] != local
                        or not np.array_equal(retarget["actions"], history)
                    ):
                        raise ValueError("Retarget changed native bodies/time/actions")
                    command_steps = 0
                    command_count += 1
                    stage = 0 if common["pilot"] else min(5, index * 6 // common["curriculum_steps"])
                    gate = SharedSettling(0.01)
                    gate.update(local * 0.01, False, retarget["target_id"])

        import yaml

        layout = yaml.safe_load(Path("configs/demo_scenarios_dispersed/normal.yaml").read_text())
        summary = exposure_summary(reconciled(), destinations=layout["coords"][1:])
    counters = sidecar["counters"]
    for key, value in checked.items():
        if counters[key] != value:
            raise ValueError("Saved sample accounting differs from actual raw")
    if counters["stage_valid_transitions"] != summary["stage_valid_transitions"]:
        raise ValueError("Saved stage exposure differs from actual raw")
    if counters["stage_episode_experience"] != experiences:
        raise ValueError("Saved continuous episode experience differs from raw")
    if any(sha256(root / p) != h for p, h in before_hashes.items()):
        raise ValueError("Learning evidence changed during independent replay")
    summary.update(
        native_action_observation_reconciled=True,
        checkpoint_sha256=sha256(checkpoint),
        artifact_sha256=before_hashes,
        sample_counters=checked,
        past_stochastic_actor_rng_replayed=False,
    )
    return summary
