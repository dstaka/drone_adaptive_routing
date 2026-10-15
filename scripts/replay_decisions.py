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
    p.add_argument('--analytical', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.analytical:
        from adaptive_drone_patrol.analytical import verify_explanation
        directory = Path(a.run)
        replay = json.loads((directory / 'analytical_replay.json').read_text())
        explanations = json.loads((directory / 'decision_explanations.json').read_text())
        lookup = {(r['run_id'], d['decision_id']): (r, d) for r in replay['runs'] for d in r['decisions']}
        if len(lookup) != len(explanations['decisions']):
            raise ValueError('Replay/explanation decision counts mismatch')
        for value in explanations['decisions']:
            run, decision = lookup.pop((value['run_id'], value['decision_id']))
            verify_explanation(value, run['policy_hash'], run['run_id'], decision['decision_id'])
            if (value['boundary_hash'] != decision['boundary_hash'] or
                    value['policy_action'] != decision['action']):
                raise ValueError('Replay/explanation boundary or action mismatch')
        out = unique_dir(a.out)
        json_write(out / 'replay.json', {'status': 'PASS', 'evidence_kind': 'analytical',
                   'decisions_verified': len(explanations['decisions'])})
        print('PASS: analytical decision references and costs verified')
        return
    summary = validate_run(a.run)
    out = unique_dir(a.out)
    events = [json.loads(line) for line in (Path(a.run) / 'events.jsonl').read_text().splitlines()]
    json_write(out / 'replay.json', {'summary': summary, 'events': events,
               'note': 'Telemetry replay only; no reconstruction of missing physics footage.'})
    print('PASS: same-run evidence validated')


if __name__ == '__main__':
    main()
