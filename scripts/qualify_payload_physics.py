#!/usr/bin/env python
"""Bind measured pretraining diagnostics; this does not admit a flight policy."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import yaml
from verify_payload_probes import verify, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probes',required=True)
    p.add_argument('--bank',required=True)
    p.add_argument('--out',required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    thresholds = yaml.safe_load(Path('configs/payload_preflight.yaml').read_text())
    result = verify(a.probes,thresholds)
    failures = []
    if thresholds.get('operating_domain_admission') != 'admitted':
        failures.append('Operating-domain admission blocked: policy-driven numerical divergence requires new validation')
    for row in result['checks']:
        summary = json.loads((Path(a.probes)/row['probe']/'summary.json').read_text())
        if summary['source_sha256']!=digest('scripts/probe_payload_physics.py'):
            failures.append('Probe source mismatch')
    for side in (.03,.045,.06):
        summary = json.loads((Path(a.probes)/f'anchor_{side:.3f}'/'summary.json').read_text())
        reference = summary['reference_mass_kg']
        actual = next(r['measured_period_s'] for r in result['checks'] if r['probe']==f'anchor_{side:.3f}')
        for e,ratio in enumerate((.1,.2,.3)):
            m,d,L = reference*ratio,side/2,.3
            M = np.array([[.0001*L*L/3+(m+.000002)*L*L,m*L*d],[m*L*d,m*d*d+m*side*side/6]])
            K = np.diag([9.81*(.0001*L/2+(m+.000002)*L),9.81*m*d])
            period = 2*math.pi/math.sqrt(min(np.linalg.eigvals(np.linalg.solve(M,K))))
            if actual[e] is None or abs(actual[e]-period)/period>.1:
                failures.append('Pendulum period mismatch')
    for i,signs in enumerate(((-1,-1,-1),(-1,1,1),(1,1,-1),(1,-1,1))):
        rows = [json.loads(s) for s in (Path(a.probes)/f'rotor_{i}'/'samples.jsonl').read_text().splitlines()]
        if np.sign(rows[0]['drone_angular_velocity'][1]).tolist()!=list(signs):
            failures.append('Rotor impulse sign mismatch')
    bank = Path(a.bank)
    summary = json.loads((bank/'summary.json').read_text())
    rows = json.loads((bank/'reset-samples.json').read_text())
    if summary['sample_sha256']!=digest(bank/'reset-samples.json'):
        failures.append('Stale reset receipt')
    if summary['bank_source_sha256']!=digest('src/adaptive_drone_patrol/payload_bank.py'):
        failures.append('Bank execution source mismatch')
    if summary['source_sha256']!=digest('scripts/probe_payload_bank.py'):
        failures.append('Bank probe source mismatch')
    if len(rows)!=30 or {r['side_m'] for r in rows}!={.03,.045,.06}:
        failures.append('Incomplete reset samples')
    for r in rows:
        if (not r['passed'] or not r['inactive_scenes_unchanged'] or r['initial_center_error_m']>1e-6 or
            abs(r['mass'][0]/r['original_runtime_mass_kg']-r['mass_ratio'])>1e-6 or
            not np.allclose(r['collision_extents'],r['side_m'],atol=1e-5)):
            failures.append('Effective reset property/state mismatch')
    if result['status']=='FAIL':
        failures.append('Numerical diagnostic failed')
    result.update(status='PASS' if not failures else 'FAIL', failures=failures,
                  physical_training_admitted=not failures, policy_admitted=False,
                  bank_binding={name:digest(bank/name) for name in ('summary.json','reset-samples.json')},
                  source_binding={name:digest(name) for name in ('src/adaptive_drone_patrol/payload_bank.py',
                    'src/adaptive_drone_patrol/payload_asset.py','src/adaptive_drone_patrol/payload_contract.py',
                    'scripts/probe_payload_bank.py','scripts/probe_payload_physics.py','scripts/verify_payload_probes.py')},
                  thresholds_sha256=digest('configs/payload_preflight.yaml'),
                  approximation='Compression-capable rigid link, not tension-only rope',
                  internal_reaction_wrench='Not exposed by the audited public API; coupled motion and rotor impulses measured',
                  remaining=['Loaded policy training/validation/254 missions/recording/operator review'])
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],failures)
    raise SystemExit(0 if not failures else 2)


if __name__=='__main__':
    main()
