#!/usr/bin/env python
"""Assess frozen integer models on disjoint expanded-domain cases."""

import argparse
import json
from pathlib import Path
import numpy as np
from adaptive_drone_patrol.dispersed_layout import expanded_scenario, GENERATOR
from adaptive_drone_patrol.benchmark import benchmark, evaluate_scenario
from adaptive_drone_patrol.routing_agent import RoutingAgent
from adaptive_drone_patrol.util import unique_dir, json_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = unique_dir(a.out)
    models = json.loads(Path(a.models).read_text())
    agents = [(m, RoutingAgent(m["path"], m["sha256"])) for m in models["models"]]
    # The selected model remains the adopted seed44; validation cannot reselect it.
    cases = [
        expanded_scenario(10000 + i, "validation", ["static", "priority", "cost"][i % 3]) for i in range(100)
    ]
    json_write(out / "validation-scenarios.json", [s.to_dict() for s in cases])
    results = []
    for item, agent in agents:
        details = [evaluate_scenario(s, agent.predict) for s in cases]
        results.append(
            {
                "seed": item["seed"],
                "mean_cost": float(np.mean([d["analytical_cost"] for d in details])),
                "episodes": 100,
                "details": details,
            }
        )
    json_write(
        out / "validation.json",
        {
            "models": results,
            "selection_changed": False,
            "geometry_generator": GENERATOR,
            "selected_seed": models["selected"]["seed"],
        },
    )
    for split, n in [("static", 100), ("dynamic", 50)]:
        benchmark(a.models, split, n, out / split, 10, "dispersed-constrained3d-v1")


if __name__ == "__main__":
    main()
