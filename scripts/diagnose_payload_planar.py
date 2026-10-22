#!/usr/bin/env python3
"""Write a unique CPU planar-replay receipt; imports no simulation engine."""
import argparse
import hashlib
import json
from pathlib import Path

from adaptive_drone_patrol.payload_recovery_planar import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--substeps', type=int, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique output required; existing receipt preserved')
    result = diagnose(args.probe, substeps=args.substeps)
    result['cli_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], local_replay=result['local_replay']), ensure_ascii=False))


if __name__ == '__main__':
    main()
