#!/usr/bin/env python
"""Independently accept version-2 realtime media; legacy checks stay separate."""

import argparse
from pathlib import Path
from adaptive_drone_patrol.video_check import check_delivery
from adaptive_drone_patrol.video import ffmpeg_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--font", default="outputs/assets/NotoSansJP.ttf")
    args = parser.parse_args()
    if (Path(args.root) / "verification.json").exists():
        raise FileExistsError("Verification already exists")
    check_delivery(args.root, args.ffprobe, ffmpeg_executable(), args.font)


if __name__ == "__main__":
    main()
