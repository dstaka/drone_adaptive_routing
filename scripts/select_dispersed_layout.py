#!/usr/bin/env python
"""Select and disclose one 3D demo fixture without reading held-out outcomes."""

import argparse
import copy
import json
import yaml
from adaptive_drone_patrol.dispersed_layout import layout_candidate, layout_checks, coordinate_hash, GENERATOR
from adaptive_drone_patrol.analytical import (
    demo_fixture,
    event_comparison,
    explain_decision,
    verify_explanation,
)
from adaptive_drone_patrol.objective import RoutingState
from adaptive_drone_patrol.routing_agent import RoutingAgent
from adaptive_drone_patrol.scene import scene_manifest
from adaptive_drone_patrol.util import load_config, unique_dir, json_write, sha256


def select(config, out):
    cfg = load_config(config)
    if cfg["generator"] != GENERATOR:
        raise ValueError("Generator/config mismatch")
    agent = RoutingAgent(cfg["routing_model"], cfg["routing_model_sha256"])
    out = unique_dir(out)
    attempts = []
    for attempt in range(cfg["max_attempts"]):
        coords = layout_candidate(cfg["layout_seed"], attempt)
        checks = layout_checks(coords)
        record = {
            "attempt": attempt,
            "coordinate_sha256": coordinate_hash(coords),
            "rejection_reasons": checks["rejection_reasons"].copy(),
        }
        attempts.append(record)
        if not checks["passed"]:
            continue
        normal = demo_fixture(cfg["priority_seed"], cfg["priority_profile"])
        normal.coords = coords
        normal.validate_physical()
        agent.validate_scenario(normal)
        st = RoutingState(normal)
        first = agent.predict(st)
        st.complete(first, st.departure(first))
        following = agent.predict(st)
        emergency, cost = copy.deepcopy(normal), copy.deepcopy(normal)
        emergency.scenario_id = "emergency"
        emergency.event = {"kind": "priority", "after": 1, "checkpoint": "C", "value": 100}
        cost.scenario_id = "cost_shift"
        edge = [normal.checkpoint_ids[first], normal.checkpoint_ids[following]]
        cost.event = {"kind": "cost", "after": 1, "edges": [edge], "value": 5}
        comparisons = []
        for s in [emergency, cost]:
            boundary = RoutingState(s)
            boundary.complete(first, boundary.departure(first))
            comparisons.append(event_comparison(boundary, agent.predict))
        b, c = comparisons
        target = normal.checkpoint_ids.index("C")
        b_ok = (
            b["applicable"]
            and b["no_event_first"] != b["post_event_first"]
            and b["improvement"] > 1e-6
            and target in b["post_event_order"]
            and b["post_event_order"].index(target) < b["no_event_order"].index(target)
        )
        c_ok = c["no_event_first"] != c["post_event_first"] and c["improvement"] > 1e-6
        record.update(first=first, proposed_cost_edge=edge, comparisons=comparisons)
        if not b_ok:
            record["rejection_reasons"].append("C emergency response gate")
        if not c_ok:
            record["rejection_reasons"].append("directed cost response gate")
        if record["rejection_reasons"]:
            continue
        try:
            scene = scene_manifest(normal)
        except ValueError as exc:
            record["rejection_reasons"].append(str(exc))
            continue
        explanations, replays = [], []
        for s in [normal, emergency, cost]:
            state, decisions = RoutingState(s), []
            run_id = f"analytical-dispersed-{attempt}-{s.scenario_id}"
            while not state.terminal:
                decision_id = f"decision-{int(state.visited.sum())}"
                e = explain_decision(state, agent.predict, agent.hash, run_id, decision_id)
                verify_explanation(e, agent.hash, run_id, decision_id)
                explanations.append(e)
                action = agent.predict(state)
                decisions.append(
                    {"decision_id": decision_id, "action": action, "boundary_hash": e["boundary_hash"]}
                )
                state.complete(action, state.departure(action))
                state.apply_event()
            replays.append(
                {
                    "run_id": run_id,
                    "evidence_kind": "analytical",
                    "scenario_id": s.scenario_id,
                    "policy_hash": agent.hash,
                    "decisions": decisions,
                    "cost": state.cost,
                    "nominal_distance_m": state.distance,
                    "legs": state.legs,
                }
            )
        if not any(e["nearer_lower_priority_gate"] for e in explanations):
            record["rejection_reasons"].append("exact L/H gap/distance/policy gate")
            continue
        for s in [normal, emergency, cost]:
            (out / f"{s.scenario_id}.yaml").write_text(yaml.safe_dump(s.to_dict(), sort_keys=False))
        manifest = {
            "schema_version": 1,
            "generator": GENERATOR,
            "seed": cfg["layout_seed"],
            "priority_seed": cfg["priority_seed"],
            "priorities": normal.priorities,
            "selected_attempt": attempt,
            "coords": coords,
            "checks": checks,
            "coordinate_sha256": coordinate_hash(coords),
            "attempts": attempts,
            "nominal_horizontal_area_ratio": 9,
            "nominal_bounds_m": [[-2.7, 2.7], [-1.8, 1.8], [1, 3]],
            "policy_sha256": agent.hash,
            "configuration_sha256": sha256(config),
            "scenario_hashes": {
                s.scenario_id: sha256(out / f"{s.scenario_id}.yaml") for s in [normal, emergency, cost]
            },
            "declared_cost_edge": edge,
            "cost_change": [1, 5],
            "operator_review": "PENDING",
            "acceptance_frozen": False,
            "selection_split": "demo; no held-out outcomes read",
        }
        json_write(out / "layout.json", manifest)
        json_write(out / "scene_manifest.json", scene)
        json_write(
            out / "analytical_replay.json",
            {"schema_version": 2, "evidence_kind": "analytical", "runs": replays},
        )
        json_write(
            out / "decision_explanations.json",
            {"schema_version": 2, "evidence_kind": "analytical", "decisions": explanations},
        )
        print(
            json.dumps({"attempt": attempt, "checks": checks, "cost_edge": edge, "coords": coords}),
            flush=True,
        )
        return manifest
    json_write(out / "failed_selection.json", {"attempts": attempts, "status": "INCOMPLETE"})
    raise RuntimeError("No qualifying fixture under the frozen criteria")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/dispersed_layout.yaml")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    select(a.config, a.out)
