#!/usr/bin/env python
"""Build a Japanese submission from complete paired Genesis patrol recordings."""
import argparse
from adaptive_drone_patrol.util import load_config
from adaptive_drone_patrol.video import compose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--input-root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    compose(a.input_root, a.out, cfg['font_path'], cfg.get('duration_s', 120), cfg.get('fps', 30))


if __name__ == '__main__':
    main()
