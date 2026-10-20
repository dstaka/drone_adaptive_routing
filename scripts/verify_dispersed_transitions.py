#!/usr/bin/env python
"""Independently verify every sample and declared target in expanded-domain probes."""

import argparse
import json
from pathlib import Path
from adaptive_drone_patrol.transition_check import verify, evidence_binding

__all__ = ["verify", "evidence_binding"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    result = verify(a.root)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
