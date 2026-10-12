#!/usr/bin/env python
"""Validate physical event/frame references and export a deterministic replay index."""
import argparse
import json
from pathlib import Path
from adaptive_drone_patrol.evidence import validate_run
from adaptive_drone_patrol.util import unique_dir, json_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    summary = validate_run(a.run)
    out = unique_dir(a.out)
    events = [json.loads(line) for line in (Path(a.run) / 'events.jsonl').read_text().splitlines()]
    json_write(out / 'replay.json', {'summary': summary, 'events': events,
               'note': 'Telemetry replay only; no reconstruction of missing physics footage.'})
    print('PASS: same-run evidence validated')


if __name__ == '__main__':
    main()
