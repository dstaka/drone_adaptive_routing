#!/usr/bin/env python3
"""Reconstruct every native/control/work interval of seven330s duration traces."""

import argparse
import json
from pathlib import Path

from adaptive_drone_patrol.payload_hybrid_duration_audit import audit_group
from adaptive_drone_patrol.payload_hybrid_duration import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    receipt = audit_group(args.output, source_root=args.source_root)
    destination = args.output / "independent-duration7.json"
    if destination.exists() and json.loads(destination.read_text()) != receipt:
        raise ValueError("Preserve prior independent receipt; evidence or result changed")
    write_json(destination, receipt)
    print(json.dumps(receipt), flush=True)
    if not receipt["passed"]:
        raise SystemExit("Actual independent duration gate failed")


if __name__ == "__main__":
    main()
