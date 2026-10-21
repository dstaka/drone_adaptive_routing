#!/usr/bin/env python
"""Declare paired payload cases before evaluation; this is not acceptance."""

import argparse
from pathlib import Path
from adaptive_drone_patrol.payload_cases import write_case_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = write_case_manifest(path, Path.cwd())
    print(f'{path}: {digest} (127 inputs per policy, NOT_RUN)')


if __name__ == '__main__':
    main()
