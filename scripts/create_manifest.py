#!/usr/bin/env python
"""Inventory retained source and evidence with hashes and regeneration commands."""
import argparse
from pathlib import Path
from adaptive_drone_patrol.util import json_write, sha256


def regeneration(path):
    name = str(path)
    if name.startswith('outputs/genesis/') or name.startswith('outputs/videos/raw_'):
        return 'python scripts/run_demo_scenarios.py --config configs/execution.yaml --mode genesis --record --trials 3 --out outputs/genesis_new'
    if name.startswith('outputs/videos/'):
        return 'python scripts/build_video.py --config configs/video.yaml --input-root outputs/genesis --out outputs/videos/submission_new.mp4'
    if name.startswith('outputs/report/'):
        return 'python scripts/build_report.py --source docs/report_ja.md --out outputs/report/report_new.pdf'
    if name.startswith('outputs/evaluation/static'):
        return 'python scripts/eval_routing.py --models outputs/routing_boundary/models.json --split static --episodes 100 --out outputs/evaluation/static_new'
    if name.startswith('outputs/evaluation/dynamic'):
        return 'python scripts/eval_routing.py --models outputs/routing_boundary/models.json --split dynamic --episodes 50 --out outputs/evaluation/dynamic_new'
    if name.startswith('outputs/evaluation/flight'):
        return 'python scripts/eval_flight.py --config configs/flight_eval.yaml --execution configs/execution.yaml --mode sequential-targets --out outputs/evaluation/flight_new'
    if name.startswith('outputs/evaluation/figures'):
        return 'python scripts/plot_results.py --out outputs/evaluation/figures_new'
    if name.startswith('outputs/routing'):
        return 'python scripts/train_routing.py --config configs/routing_train.yaml --seeds 42 43 44 --out outputs/routing_new'
    return 'See README.md and docs/proven-baseline.md; preserve original operator inputs unchanged.'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='artifacts/manifest.json')
    a = p.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    roots = ['src', 'scripts', 'tests', 'configs', 'requirements', 'docs', 'third_party', 'outputs']
    files = [Path(x) for x in ['spec.md', 'README.md', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml', 'LICENSE', '.gitignore',
                                'RL_drone_hovering.ipynb'] if Path(x).is_file()]
    for root in roots:
        files.extend(x for x in Path(root).rglob('*') if x.is_file() and '__pycache__' not in str(x)
                     and x.suffix not in ['.pyc', '.lock']
                     and (not x.name.startswith('execution') or x.name == 'execution.example.yaml') and not x.name.startswith('delivery_receipt'))
    artifacts = []
    for path in sorted(set(files)):
        run_id = path.parts[2] if str(path).startswith('outputs/genesis/') and len(path.parts) > 3 else 'delivery-20261013'
        artifacts.append({'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size,
                          'run_id': run_id, 'regeneration_command': regeneration(path)})
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_write(output, {'schema_version': 1, 'artifacts': artifacts,
                       'note': 'Includes ignored local evidence; original inputs are preserved, not regenerated.'})
    print(f'Inventoried {len(artifacts)} retained files')


if __name__ == '__main__':
    main()
