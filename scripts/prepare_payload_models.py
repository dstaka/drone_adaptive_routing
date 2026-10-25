#!/usr/bin/env python
"""Freeze matched final checkpoints before held-out evaluation, preserving failures."""

import argparse
import json
from pathlib import Path
import time
from adaptive_drone_patrol.util import sha256,json_write,load_config
from adaptive_drone_patrol.payload_contract import require_physical_admission
from adaptive_drone_patrol.payload_checkpoint import actor_signature, validate_matched_manifests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--randomized',required=True)
    p.add_argument('--fixed',required=True)
    p.add_argument('--preflight',required=True)
    p.add_argument('--cases',required=True)
    p.add_argument('--validation')
    p.add_argument('--validation-models')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    out=Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    preflight=json.loads(Path(a.preflight).read_text())
    require_physical_admission(preflight,load_config('configs/payload_preflight.yaml'),
                               sha256('configs/payload_preflight.yaml'))
    policies={}
    manifests=[]
    steps=[]
    initializations=[]
    import torch
    for method,directory in [('randomized',a.randomized),('fixed',a.fixed)]:
        path=Path(directory)
        manifest=json.loads((path/'training_manifest.json').read_text())
        completed=json.loads((path/'completed.json').read_text())
        manifests.append(manifest)
        for source_key,actual_path in [('training_source_sha256','src/adaptive_drone_patrol/payload_training.py'),
                                       ('bank_source_sha256','src/adaptive_drone_patrol/payload_bank.py'),
                                       ('training_script_sha256','scripts/train_payload_flight.py'),
                                       ('parent_config_sha256','inputs/cfgs.pkl'),('parent_sha256','inputs/model_4296.pt')]:
            if manifest.get(source_key)!=sha256(actual_path):
                raise ValueError('Training evidence differs from actual source or parent artifact')
        if manifest['method']!=method or manifest['preflight_sha256']!=sha256(a.preflight):
            raise ValueError('Training method/preflight does not match')
        checkpoint=path/'final.pt'
        if sha256(checkpoint)!=completed['checkpoint_sha256']:
            raise ValueError('Actual final checkpoint differs from completed training')
        if completed['actual_environment_steps']!=manifest['environment_steps']:
            raise ValueError('Training step budget incomplete')
        steps.append(completed['actual_environment_steps'])
        initializations.append(torch.load(path/'initialization.pt',map_location='cpu',weights_only=False))
        policies[method]=dict(actor_sha256=actor_signature(torch.load(checkpoint,map_location='cpu',weights_only=False)['actor_state_dict']),
                              checkpoint=str(checkpoint),sha256=sha256(checkpoint),bytes=checkpoint.stat().st_size,
                              training_manifest=str(path/'training_manifest.json'),
                              training_manifest_sha256=sha256(path/'training_manifest.json'),
                              training_completed=str(path/'completed.json'),
                              training_completed_sha256=sha256(path/'completed.json'))
    validate_matched_manifests(*manifests)
    if steps[0]!=steps[1]:
        raise ValueError('Comparison environment-step budgets differ')
    for kind in ('actor_state_dict','critic_state_dict'):
        for key,value in initializations[0][kind].items():
            if not torch.equal(value,initializations[1][kind][key]):
                raise ValueError('Comparison initialization parameters differ')
    validation_binding=None
    validation_receipt=None
    if a.validation:
        validation=Path(a.validation)
        completed=json.loads((validation/'completed.json').read_text())
        if completed['trials']!=12 or completed['results_sha256']!=sha256(validation/'results.json'):
            raise ValueError('Incomplete paired validation')
        if not a.validation_models:
            raise ValueError('The actual validation candidate manifest is required')
        from check_payload_trials import verify_validation
        validation_receipt=verify_validation(validation,policies,sha256('inputs/cfgs.pkl'),a.validation_models)
        validation_binding={str(p):sha256(p) for p in sorted(validation.rglob('*')) if p.is_file()}
        validation_binding[a.validation_models]=sha256(a.validation_models)
    sources={str(p):sha256(p) for p in sorted(Path('src/adaptive_drone_patrol').glob('payload*.py'))}
    for name in ('scripts/evaluate_payload_flight.py','configs/payload_validation.yaml','configs/payload_preflight.yaml'):
        sources[name]=sha256(name)
    json_write(out,dict(schema='payload-model-freeze-v1',created_epoch=time.time(),policies=policies,
               matched_environment_steps=steps[0],initialization_parameters_equal=True,
               selection_rule='Final matched-budget checkpoints, declared before validation; no held-out selection',
               selection_frozen_before_heldout=bool(a.validation),validation_binding=validation_binding,
               validation_root=a.validation,validation_models=a.validation_models,validation_receipt=validation_receipt,
               preflight=a.preflight,preflight_sha256=sha256(a.preflight),
               preflight_probes='outputs/probes_v4',preflight_bank='outputs/bank_v4',
               parent_cfg='inputs/cfgs.pkl',parent_cfg_sha256=sha256('inputs/cfgs.pkl'),
               routing_model='inputs/seed_44.zip',routing_sha256=sha256('inputs/seed_44.zip'),
               routing_sidecar_sha256=sha256('inputs/seed_44.schema.json'),
               planner_python='.venv-cpu312/bin/python',cases_sha256=sha256(a.cases),source_binding=sources,
               limits=dict(control_dt_s=.01,shared_dwell_s=1.,leg_s=30.,mission_s=120.,wall_s=600.,
                           drone_error_m=.2,drone_speed_m_s=.3,package_error_m=.2,package_speed_m_s=.1,
                           swing_deg=5.,swing_rate_deg_s=10.,package_spin_deg_s=10.,floor_m=.1,roll_pitch_deg=60.),
               recording_cases=['basic-normal-1','basic-emergency-1','basic-cost_shift-1'],
               media_status='Pending; no loaded camera/operator approval asserted'))
    print(out,sha256(out))


if __name__=='__main__':
    main()
