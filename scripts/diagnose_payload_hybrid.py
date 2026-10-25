#!/usr/bin/env python3
"""Frozen controller-only development. No optimizer or acceptance consumer."""

import argparse
import json
from pathlib import Path

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_hybrid_runtime import run_case
from adaptive_drone_patrol.payload_hybrid_manifest import validate_cases, validate_gate_result
from adaptive_drone_patrol.payload_solver_qualification import current_solver_binding
from adaptive_drone_patrol.util import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-rebinding',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--design',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',choices=('C0','C1','C2'),default='C0')
    p.add_argument('--feedback-off',action='store_true')
    p.add_argument('--substeps',type=int,default=16)
    args = p.parse_args()
    reference = json.loads(args.reference_rebinding.read_text())
    if (reference['schema']!='hybrid-unchanged-reference-rebinding-v1' or
            not reference['preserved_raw_bytes_identical'] or reference['learning_admitted'] or
            current_solver_binding()!=reference['actual_binding'] or
            sha256(args.original)!=reference['actual_binding']['original_asset_sha256']):
        raise ValueError('Identical preserved reference qualification required')
    manifest = json.loads(args.manifest.read_text())
    validate_cases(manifest['cases'])
    design = json.loads(args.design.read_text())
    if design['recipe'] not in {h['id'] for h in manifest['hypotheses']}:
        raise ValueError('Undeclared controller hypothesis')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'inputs.json',dict(
        manifest_sha256=sha256(args.manifest),design_sha256=sha256(args.design),
        reference_sha256=sha256(args.reference_rebinding),gate=args.gate,
        source_sha256=sha256(__file__)))
    outcomes=[]
    for case in manifest['cases']:
        if case['gate']!=args.gate:
            continue
        result = run_case(original=args.original,output=args.output/case['case_id'],case=case,
            design=design,rebound_binding=reference['actual_binding'],
            vibration_feedback=not args.feedback_off,substeps=args.substeps)
        outcomes.append(result)
        print(case['case_id'],result['success'],result['failure'],result['simulated_s'],flush=True)
        write_json(args.output/'progress.json',dict(outcomes=outcomes,gate_complete=False))
    result=dict(gate=args.gate,complete=True,passed=validate_gate_result(args.gate,outcomes),
        outcomes=outcomes,independent_raw_audit='PENDING',learning_admitted=False)
    write_json(args.output/'gate-result.json',result)
    print('GATE_RECORDED',result['passed'],'RAW_AUDIT_PENDING',flush=True)


if __name__=='__main__':
    main()
