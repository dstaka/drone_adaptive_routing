#!/usr/bin/env python3
"""Capture3 final seed1 videos AFTER one reconstructed full formal acceptance."""

import argparse
import json
from pathlib import Path

from adaptive_drone_patrol.payload_hybrid_final import (
    FinalContext,
    FinalFreeze,
    FinalAcceptance,
    ROUTING_SHA256,
)
from adaptive_drone_patrol.payload_hybrid_flight import ReferenceAdmission
from adaptive_drone_patrol.payload_mujoco_final_media import capture_final_three, MediaAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-rebinding",
        "design",
        "original",
        "routing",
        "freeze",
        "formal-root",
        "font",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument(
        "--reader-rebinding",
        type=Path,
        help="Explicit validated historical/current reader binding receipt for reused development raw",
    )
    parser.add_argument(
        "--media-module", required=True, help="Project-owned overview renderer/layout/HUD auditing adapter"
    )
    args = parser.parse_args()
    from adaptive_drone_patrol.routing_agent import RoutingAgent

    media = MediaAdapter.from_module(args.media_module)
    reference = ReferenceAdmission.from_rebinding(json.loads(args.reference_rebinding.read_text()))
    context = FinalContext.from_reference(
        reference,
        design=args.design,
        original=args.original,
        routing=args.routing,
        reader_rebinding=args.reader_rebinding,
    )
    planner = RoutingAgent(args.routing, expected_hash=ROUTING_SHA256)
    freeze = FinalFreeze.load(args.freeze, context, planner)
    acceptance = FinalAcceptance.from_campaign(freeze, args.formal_root, planner)
    input_paths = {
        name: getattr(args, name)
        for name in ("reference_rebinding", "design", "original", "routing", "freeze")
    }
    if args.reader_rebinding is not None:
        input_paths["reader_rebinding"] = args.reader_rebinding
    accepted_report = args.formal_root / "final-independent-audit.json"
    if accepted_report.exists():
        if json.loads(accepted_report.read_text()) != acceptance.record:
            raise ValueError(
                "Saved formal acceptance proof differs from fresh complete independent reconstruction"
            )
        input_paths["formal_acceptance_proof"] = accepted_report
    result = capture_final_three(
        acceptance=acceptance,
        formal_root=args.formal_root,
        output=args.output,
        font=args.font,
        camera=args.camera,
        input_paths=input_paths,
        media=media,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
