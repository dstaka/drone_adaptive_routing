#!/usr/bin/env python3
"""Fresh independent all127 raw/command/native/RK reconstruction."""
import argparse,json,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from adaptive_drone_patrol.elevated_campaign import JointFreeze,formal_cases,audit_prerequisite,audit_source_hashes,validate_audit_receipt
from adaptive_drone_patrol.elevated_artifacts import inventory
from adaptive_drone_patrol.elevated_formal import bind_formal_statistics
from adaptive_drone_patrol.util import json_write,sha256


def worker(params):
    trial,case,paths=params
    before=inventory(trial);sources=audit_source_hashes()
    try:
        result=audit_prerequisite(trial,case,paths)
        row=dict(case_id=case['case_id'],success=result['passed'],proof=result)
    except BaseException as exc:row=dict(case_id=case['case_id'],success=False,error=repr(exc),traceback=traceback.format_exc())
    row.update(artifacts=before,checked_sources=sources)
    validate_audit_receipt(trial,case,row)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--formal',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    f=JointFreeze.load(a.freeze);cases=formal_cases(f);declaration=json.loads((a.formal/'declaration.json').read_text())
    if declaration['cases']!=cases or declaration['freeze_sha256']!=sha256(a.freeze):raise ValueError('Formal case/source freeze changed')
    out=a.formal/'independent-audit';out.mkdir(exist_ok=False);results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(worker,[(a.formal/c['case_id'],c,f.record['paths']) for c in cases]):
            results.append(result);json_write(out/f"{result['case_id']}.json",result);print('AUDIT',result['case_id'],result['success'],flush=True)
    stats=bind_formal_statistics(a.formal,cases,results,sha256(a.freeze),sha256(__file__))
    json_write(a.formal/'independent-statistics.json',stats);f.require_current()

if __name__=='__main__':main()
