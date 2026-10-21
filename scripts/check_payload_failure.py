#!/usr/bin/env python
"""Confirm recorded numerical divergence from actual simulator samples and hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_trace(directory):
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text())
    raw=directory/'raw-states.jsonl'
    if digest(raw)!=summary['raw_state_sha256']:
        raise ValueError('Actual raw state hash differs from diagnostic recording')
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    if len(rows)!=summary['samples'] or not rows or [r['index'] for r in rows]!=list(range(len(rows))):
        raise ValueError('Missing or incomplete diagnostic samples')
    if summary['status']!='FAILED' or summary['mission_success']:
        raise ValueError('A diagnostic failure cannot certify a successful mission')
    speeds=[]
    for r in rows:
        if not np.isfinite(r['actions']).all() or max(abs(x) for x in r['actions'])>1:
            raise ValueError('Motor command differs from retained bounded interface')
        speeds.append(float(np.linalg.norm(r['package_velocity'])))
    if not np.isfinite(speeds).all() or max(speeds)<=1e4:
        raise ValueError('Expected finite catastrophic numerical excursion was not recorded')
    return dict(samples=len(rows),max_package_speed_m_s=max(speeds),
                first_speed_above_1e4_index=next(r['index'] for r,v in zip(rows,speeds) if v>1e4),
                checkpoint_sha256=summary['checkpoint_sha256'],raw_state_sha256=digest(raw))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    root=Path(a.delivery)
    manifest=root/'delivery-files.json'
    files=json.loads(manifest.read_text())['files']
    for item in files:
        path=root/item['path']
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError('External delivery integrity mismatch')
    traces={name:verify_trace(root/'outputs'/name) for name in
            ('diagnostic_v1','diagnostic_substeps8_v1','diagnostic_substeps16_v1')}
    checkpoint=root/'outputs/train_randomized_v2/model_2000.pt'
    if any(r['checkpoint_sha256']!=digest(checkpoint) for r in traces.values()):
        raise ValueError('Actual failed actor differs between numerical diagnostics')
    out=Path(a.out)
    with out.open('x') as f:
        json.dump(dict(status='CONFIRMED_NUMERICAL_FAILURE',physical_training_admitted=False,
                       verified_delivery_files=len(files),delivery_manifest_sha256=digest(manifest),
                       checker_sha256=digest(__file__),diagnostics=traces),f,indent=2)
        f.write('\n')
    print('CONFIRMED_NUMERICAL_FAILURE; no acceptance trials or videos admitted')


if __name__=='__main__':
    main()
