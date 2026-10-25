#!/usr/bin/env python
"""Matched frozen-policy noise experiment on the qualified physical backend.

Noise is an explicit development diagnostic input, never an acceptance policy.
No optimizer, reward change, hidden force, state filter or threshold change.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_cases import initial_state
from adaptive_drone_patrol.payload_contract import SharedSettling
from adaptive_drone_patrol.payload_solver_flight import SolverAdmission, SolverFlight
from adaptive_drone_patrol.payload_solver_validation import measured_predicates
from adaptive_drone_patrol.util import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["qualification", "locations", "original", "checkpoint", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--method", choices=["fixed", "randomized"], default="randomized")
    parser.add_argument("--scales", default="0,0.5,1")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    scales = [float(x) for x in args.scales.split(",")]
    if not scales or len(set(scales)) != len(scales) or any(x not in [0, 0.5, 1] for x in scales):
        parser.error("Only predeclared noise scales0/0.5/1 supported")
    if args.steps != 3000:
        parser.error("Matched30-second diagnostic required")
    if args.out.exists():
        raise FileExistsError("Preserve all noise experiments")
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Frozen actual checkpoint differs")
    admission = SolverAdmission.from_receipt(
        json.loads(args.qualification.read_text()), json.loads(args.locations.read_text())
    )
    import torch

    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = state["actor_state_dict"]
    key = "distribution.std_param"
    if key not in actor:
        raise ValueError("Actual inherited scalar Gaussian distribution required")
    std = actor[key].cpu().numpy().astype(float)
    if std.shape != (4,) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Actual frozen actor noise is invalid")
    case = dict(
        case_id="development-noise-83011",
        side_m=0.045,
        mass_ratio=0.2,
        initial=initial_state(np.random.default_rng(np.random.SeedSequence([83011, 720, 0])), 0.045),
    )
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "declaration.json",
        dict(
            scales=scales,
            steps=args.steps,
            frozen_checkpoint_sha256=args.checkpoint_sha256,
            actual_actor_std=std.tolist(),
            case=case,
            target=[0, 0, 1],
            noise_seed=83111,
            actual_binding=admission.binding,
            source_sha256=sha256(__file__),
            scope="Development frozen actor, no learning or acceptance",
        ),
    )
    results = []
    for scale in scales:
        root = args.out / f"scale-{scale:g}"
        flight = SolverFlight.from_checkpoint(
            admission=admission,
            checkpoint=args.checkpoint,
            expected_hash=args.checkpoint_sha256,
            original=args.original,
            case=case,
            output=root / "flight",
            allow_development=True,
            expected_method=args.method,
        )
        rng = np.random.default_rng(83111)
        mean_policy = flight.policy
        gate = SharedSettling(0.01)
        initial = measured_predicates(flight.sample(), [0, 0, 1], 0.045)
        gate.update(0.0, initial["valid"], "hover")
        aggregate, action_inputs, settled_steps, completed_at, error = [], [], 0, None, None
        started = time.monotonic()
        with (
            (root / "policy-inputs.jsonl").open("w") as inputs,
            (root / "samples.jsonl").open("w") as samples,
        ):

            def policy(obs):
                mean = mean_policy(obs).astype(float)
                noise = rng.standard_normal(4) * std * scale
                action = mean + noise
                inputs.write(
                    json.dumps(
                        dict(
                            observation=obs.tolist(),
                            mean=mean.tolist(),
                            noise=noise.tolist(),
                            raw_action=action.tolist(),
                        ),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                inputs.flush()
                action_inputs.append(action)
                return action

            flight.policy = policy
            try:
                for step in range(1, args.steps + 1):
                    row = flight.advance()
                    samples.write(json.dumps(row, separators=(",", ":")) + "\n")
                    samples.flush()
                    predicates = measured_predicates(row, [0, 0, 1], 0.045)
                    complete = gate.update(step * 0.01, predicates["valid"], "hover")
                    if complete and completed_at is None:
                        completed_at = step * 0.01
                    settled_steps += int(predicates["valid"])
                    aggregate.append(row["metrics"])
            except Exception as exc:
                error = repr(exc)
            finally:
                flight.close()
        metrics = [
            "drone_error_m",
            "drone_speed_m_s",
            "package_speed_m_s",
            "swing_angle_deg",
            "swing_rate_deg_s",
            "package_spin_deg_s",
        ]
        result = dict(
            scale=scale,
            steps=len(aggregate),
            error=error,
            first_shared_complete_s=completed_at,
            simultaneous_valid_samples=settled_steps,
            mean={k: float(np.mean([r[k] for r in aggregate])) for k in metrics} if aggregate else {},
            final500_mean={k: float(np.mean([r[k] for r in aggregate[-500:]])) for k in metrics}
            if aggregate
            else {},
            motor_saturation_fraction=float(np.mean(np.abs(action_inputs) >= 1)) if action_inputs else None,
            valid_transitions_per_second=len(aggregate) / (time.monotonic() - started),
            samples_sha256=sha256(root / "samples.jsonl"),
            native_raw_sha256=sha256(root / "flight/runtime/raw.jsonl.gz"),
            policy_inputs_sha256=sha256(root / "policy-inputs.jsonl"),
            physical_acceptance=False,
        )
        results.append(result)
        write_json(args.out / "results.json", results)
        print(json.dumps(result), flush=True)
    return 0 if all(r["error"] is None for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
