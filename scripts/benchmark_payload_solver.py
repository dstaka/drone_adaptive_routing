#!/usr/bin/env python
"""Measure serial independent native scenes; never admit an N>1 PPO batch."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_cases import initial_state
from adaptive_drone_patrol.payload_contract import PayloadSampler
from adaptive_drone_patrol.payload_solver_flight import SolverAdmission
from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG
from adaptive_drone_patrol.payload_solver_policy import mean_policy
from adaptive_drone_patrol.payload_solver_qualification import physical_errors
from adaptive_drone_patrol.payload_solver_runtime import SolverRuntime
from adaptive_drone_patrol.util import sha256


class Slot:
    """Independent geometry scenes and episode streams for one diagnostic slot."""

    def __init__(self, index, root, original):
        self.index, self.root, self.original = index, root, original
        self.sampler = PayloadSampler(FREE_MASS_KG, 83101, "training", randomized=True)
        self.rng = np.random.default_rng(np.random.SeedSequence([83101, index, 711]))
        self.cores = {}
        self.resets = 0
        self.active = True
        self.reset()

    def reset(self):
        payload = self.sampler.reset([self.index])[self.index]
        if payload.side_m not in self.cores:
            root = self.root / f"size-{payload.side_m:.3f}"
            fixture = MuJoCoFixture(self.original, root / "scene", payload, substeps=16)
            self.cores[payload.side_m] = SolverRuntime(fixture, root / "runtime")
        self.core = self.cores[payload.side_m]
        self.core.fixture.apply_payload(payload)
        self.core.payload = payload
        self.resets += 1
        self.core.reset_case(
            initial=initial_state(self.rng, payload.side_m),
            target=[0, 0, 1],
            target_id=f"{self.index}-{self.resets}",
        )
        actual = self.core.fixture.raw_state()
        if np.any(actual["rpm"]) or np.any(self.core.last_actions):
            raise ValueError("Actual selected reset retained input/history")
        properties = self.core.fixture.runtime()
        if physical_errors(
            properties, dict(side_m=payload.side_m, mass_ratio=payload.mass_ratio, substeps=16), fixed=False
        ):
            raise ValueError("Actual independent slot geometry/mass/inertia differs")
        write_json(
            self.root / f"reset-{self.resets}.json",
            dict(
                payload=vars(payload),
                runtime=properties,
                actual=actual,
                draw_index=self.sampler.draw_indices[self.index],
            ),
        )
        self.active = True

    def snapshot(self):
        c = self.core
        return dict(
            actual=c.fixture.raw_state(),
            actions=c.last_actions.tolist(),
            target=c.target.tolist(),
            target_id=c.target_id,
            dwell=copy.deepcopy(vars(c.dwell)),
            step=c.step_count,
            episode=c.episode_id,
            active=self.active,
        )

    def advance(self, policy):
        if not self.active:
            return 0
        row = self.core.step(policy(self.core.observation))
        if row["unsafe"]:
            self.active = False
            write_json(self.root / f"unsafe-{self.resets}.json", row)
            return 0
        return 1

    def close(self):
        for core in self.cores.values():
            core.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["qualification", "locations", "original", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--sizes", default="1,8,32,128")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--partial-resets", type=int, default=30)
    args = parser.parse_args()
    sizes = [int(v) for v in args.sizes.split(",")]
    if not sizes or len(set(sizes)) != len(sizes) or any(n not in [1, 8, 32, 128] for n in sizes):
        parser.error("Only declared N1/8/32/128 diagnostics are supported")
    if args.steps < 100 or args.partial_resets < 30:
        parser.error("At least100 steps and30 actual partial resets required")
    if args.out.exists():
        raise FileExistsError("Preserve benchmark attempts")
    if not args.checkpoint or not args.checkpoint_sha256 or sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Exact real learned checkpoint required for inference timing")
    admission = SolverAdmission.from_receipt(
        json.loads(args.qualification.read_text()), json.loads(args.locations.read_text())
    )
    import torch

    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["infos"]["actual_solver_binding"] != admission.binding:
        raise ValueError("Diagnostic checkpoint physics differs")
    policy = mean_policy(checkpoint["actor_state_dict"])
    args.out.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        sizes=sizes,
        steps_per_slot=args.steps,
        selective_resets=args.partial_resets,
        seed=83101,
        checkpoint_sha256=args.checkpoint_sha256,
        actual_binding=admission.binding,
        source_sha256=sha256(__file__),
        execution="Serial CPU native scenes; no optimizer, vector PPO or GPU scaling claim",
        admission="Only existing fully qualified N1 remains eligible for learning",
    )
    write_json(args.out / "declaration.json", declaration)
    results = []
    for n in sizes:
        slots = []
        try:
            for i in range(n):
                slots.append(Slot(i, args.out / f"n{n}" / f"slot-{i}", args.original))
            started = time.monotonic()
            valid, attempted = 0, 0
            for step in range(args.steps):
                for slot in slots:
                    attempted += int(slot.active)
                    valid += slot.advance(policy)
                if step == args.steps // 2:
                    for slot in slots:
                        before = slot.snapshot()
                        slot.core.set_target(
                            [0.15 * ((slot.index % 3) - 1), 0, 1], target_id=f"independent-{slot.index}"
                        )
                        after = slot.snapshot()
                        for key in ["actual", "actions", "step", "episode", "active"]:
                            if before[key] != after[key]:
                                raise ValueError("Retarget changed native state/history")
            isolation = 0
            for index in range(args.partial_resets):
                selected = index % n
                before = [slot.snapshot() for slot in slots]
                slots[selected].reset()
                for i, slot in enumerate(slots):
                    if i != selected and slot.snapshot() != before[i]:
                        raise ValueError("Partial reset changed an ongoing neighbor")
                reset_state = slots[selected].snapshot()
                for i, slot in enumerate(slots):
                    if i != selected:
                        attempted += int(slot.active)
                        valid += slot.advance(policy)
                if slots[selected].snapshot() != reset_state:
                    raise ValueError("Neighbor advance changed reset slot")
                isolation += 1
            elapsed = time.monotonic() - started
            result = dict(
                N=n,
                valid_transitions=valid,
                attempted_transitions=attempted,
                inactive_slots=sum(not s.active for s in slots),
                effective_valid_transitions_per_second=valid / elapsed,
                actual_selective_resets=isolation,
                neighboring_isolation=n > 1,
                measured_geometry_scenes=sum(len(s.cores) for s in slots),
                qualified_ppo_batch=False,
            )
            results.append(result)
            write_json(args.out / "results.json", results)
            print(json.dumps(result), flush=True)
        finally:
            for slot in slots:
                slot.close()
    write_json(
        args.out / "selection.json",
        dict(
            selected_N=1,
            reason="Serial independent scenes provide isolation diagnostics, not a verified vector PPO implementation; retain fully qualified N1",
            performance_results_sha256=sha256(args.out / "results.json"),
            qualified_batch_binding=admission.binding,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
