#!/usr/bin/env python
"""Verify reset properties and isolated episode geometry before training."""

import argparse
from pathlib import Path
import numpy as np
from adaptive_drone_patrol.payload_bank import PayloadSceneBank
from adaptive_drone_patrol.payload_contract import PayloadSampler, payload_metrics
from adaptive_drone_patrol.payload_cases import initial_state
from adaptive_drone_patrol.util import json_write, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    import genesis as gs
    gs.init(backend=gs.gpu,precision='32',logging_level='warning',seed=7)
    bank = PayloadSceneBank(out/'scenes')
    sampler = PayloadSampler(bank.reference_mass,7,'training')
    rng = np.random.default_rng(np.random.SeedSequence([7,710,99]))
    rows = []
    for i in range(30):
        payload = sampler.reset([0])[0]
        initial = initial_state(rng,payload.side_m)
        before = {size: drone.get_qpos().cpu().tolist() for size,(_,drone,_,_) in bank.banks.items()}
        sample = bank.reset(payload,initial)
        mass = bank.package.get_mass().cpu().tolist()
        all_masses = bank.drone.get_links_mass().cpu().numpy().reshape(-1)
        original_mass = sum(float(v) for link,v in zip(bank.drone.links,all_masses)
                            if link.name in ('base_link','prop0_link','prop1_link','prop2_link','prop3_link'))
        inertia = bank.drone.get_links_inertia([bank.package.idx-bank.drone.link_start]).cpu().tolist()
        bounds = bank.package.geoms[0].get_AABB().cpu().numpy()
        extents = bounds[...,1,:]-bounds[...,0,:]
        unchanged = all(before[size] == drone.get_qpos().cpu().tolist()
                        for size,(_,drone,_,_) in bank.banks.items() if size != payload.side_m)
        initial_error = max(np.linalg.norm(np.array(sample[k])-v) for k,v in initial.items()
                            if k in sample and k.endswith('xyz'))
        row = dict(index=i,side_m=payload.side_m,mass_ratio=payload.mass_ratio,
                   mass=mass,inertia=inertia,collision_extents=extents.tolist(),
                   original_runtime_mass_kg=original_mass,all_link_masses_kg=all_masses.tolist(),
                   inactive_scenes_unchanged=unchanged,initial_center_error_m=initial_error,
                   measured=payload_metrics(sample,[0,0,1],payload))
        row['passed'] = (np.allclose(mass,payload.mass,rtol=1e-5) and
                         np.allclose(np.array(inertia).reshape(3,3),np.eye(3)*payload.inertia[0],rtol=1e-5,atol=1e-12) and
                         np.allclose(extents,payload.side_m,atol=1e-5) and unchanged and initial_error<1e-6)
        row['passed'] = bool(row['passed'] and abs(original_mass-bank.reference_mass)<1e-7)
        rows.append(row)
        json_write(out/'reset-samples.json',rows)
        bank.step([0,0,0,0])
    json_write(out/'reset-samples.json',rows)
    status='PASS' if all(r['passed'] for r in rows) else 'FAIL'
    json_write(out/'summary.json',dict(status=status,resets=len(rows),source_sha256=sha256(__file__),
                                      bank_source_sha256=sha256('src/adaptive_drone_patrol/payload_bank.py'),
                                      sample_sha256=sha256(out/'reset-samples.json'),
                                      sizes_seen=sorted({r['side_m'] for r in rows}),mission_success=False))
    print(status)
    raise SystemExit(0 if status=='PASS' else 2)


if __name__=='__main__':
    main()
