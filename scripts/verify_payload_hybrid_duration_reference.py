#!/usr/bin/env python3
"""Hash actual2669 preserved reference raws on their storage host; no engine import."""

import argparse
import json
from pathlib import Path

from adaptive_drone_patrol.payload_hybrid_duration import verify_preserved_raw, write_json
from adaptive_drone_patrol.util import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-rebinding",
        "historical-receipt",
        "qualification-locations",
        "qualification-root",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing full actual raw verification receipt")
    locations = {
        name: str(path if (path := Path(value)).is_absolute() else args.qualification_root / path)
        for name, value in json.loads(args.qualification_locations.read_text()).items()
    }
    receipt = verify_preserved_raw(args.reference_rebinding, args.historical_receipt, locations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, receipt)
    print(
        json.dumps(
            dict(
                path=str(args.output.resolve()),
                receipt_sha256=sha256(args.output),
                raw_files=receipt["raw_files"],
                raw_bytes=receipt["raw_bytes"],
                raw_manifest_sha256=receipt["raw_manifest_sha256"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
