#!/usr/bin/env python
"""Screen one predeclared 120-second anchored native-ball raw recording."""
import argparse
import hashlib
import json
from pathlib import Path

from adaptive_drone_patrol.payload_passive_check import ROOT, verify_passive_probe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--source-root', default=str(ROOT))
    parser.add_argument('--out', required=True)
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():
        raise FileExistsError('Passive checker output already exists')
    receipt=verify_passive_probe(args.probe, source_root=args.source_root)
    receipt['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with out.open('x') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(receipt['status'])
    return 2 if receipt['status'] == 'QUALIFICATION_BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
