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
    p.add_argument('--mode', choices=['full', 'submission'], required=True)
    p.add_argument('--explanations', required=True)
    p.add_argument('--duration', type=float, help='Submission cap; never changes flight speed')
    a = p.parse_args()
    cfg = load_config(a.config)
    if cfg.get('playback_rate', 1.0) != 1.0:
        p.error('Realtime deliveries require playback_rate=1.0')
    duration = a.duration if a.duration is not None else (cfg.get('submission_max_s') if a.mode == 'submission' else None)
    compose(a.input_root, a.out, cfg['font_path'], duration, cfg.get('fps', 30),
            mode=a.mode, explanations=a.explanations)


if __name__ == '__main__':
    main()
