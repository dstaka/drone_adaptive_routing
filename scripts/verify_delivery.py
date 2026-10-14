#!/usr/bin/env python
"""Verify retained artifact hashes and sizes against the local manifest."""
import argparse
import json
from pathlib import Path
from adaptive_drone_patrol.evidence import verify_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--require-local-artifacts', action='store_true')
    a = p.parse_args()
    count = verify_manifest(json.loads(Path(a.manifest).read_text()), Path.cwd(), require_local_artifacts=a.require_local_artifacts)
    print(f'PASS: {count} local artifacts verified')


if __name__ == '__main__':
    main()
