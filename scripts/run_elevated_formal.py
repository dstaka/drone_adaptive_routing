#!/usr/bin/env python3
"""Open the fixed127 cohort after joint freeze and routing held-out gates."""
import argparse,json,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from adaptive_drone_patrol.elevated_campaign import JointFreeze,formal_cases
from adaptive_drone_patrol.elevated_flight import run_development
from adaptive_drone_patrol.elevated_formal import formal_statistics,claim_physical_campaign
from adaptive_drone_patrol.analytical import content_hash
from adaptive_drone_patrol.util import json_write,sha256


def worker(params):
    case,root,freeze=params;freeze.require_current();logical=Path(root)/case['case_id']
    # Fixed storage allocation from seed labels, before examining any loads:
    # 27 basic/endpoints + j0/1 in each of the nine heldout cells =45 RAM cases.
    ram=case['suite']!='heldout' or case['execution_seed']%100<2
    output=(Path('/dev/shm')/Path(freeze.record['run']).name/'formal-memory-v1'/case['case_id']) if ram else logical
    if case not in formal_cases(freeze):raise ValueError('Formal case differs from frozen allocation')
    if output.exists() or output.is_symlink() or logical.exists() or logical.is_symlink():raise FileExistsError('Existing formal trial must never be mutated: '+str(output))
    try:
        result=run_development(case,output=output,**{k:v for k,v in freeze.record['paths'].items()})
        row=dict(case_id=case['case_id'],success=result['success'],numeric=result['numeric'],steps=result['mission']['steps'])
    except FileExistsError:
        raise
    except BaseException as exc:
        output.mkdir(parents=True,exist_ok=True)
        row=dict(case_id=case['case_id'],success=False,error=repr(exc),traceback=traceback.format_exc())
        json_write(output/'formal-interruption.json',row)
    if output!=logical:
        logical.symlink_to(output.resolve(),target_is_directory=True)
    json_write(output/'formal-attempt.json',dict(attempt=1,freeze_digest=content_hash(freeze.record),case_digest=content_hash(case),result=row))
    freeze.require_current();return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    f=JointFreeze.load(a.freeze);run=Path(f.record['run'])
    routing=json.loads((run/'routing-heldout-result.json').read_text())
    if not routing['absolute_heldout_gate'] or routing['selected']!=f.record['selected'] or routing.get('reselection') is not False:raise ValueError('Selected frozen routing held-out gate unmet')
    claim_physical_campaign(run,content_hash(f.record),a.output)
    cases=formal_cases(f);a.output.mkdir(parents=True,exist_ok=False)
    json_write(a.output/'declaration.json',dict(schema='elevated-physical127-declaration-v1',cases=cases,freeze_sha256=sha256(a.freeze),freeze_digest=content_hash(f.record),attempts_per_case=1,storage_plan='45 cases in existing /dev/shm: all27 basic/endpoints and heldout j0/1 in each of nine cells; remaining82 heldout on disk; logical case paths linked; recover with symlink dereference and size/SHA verification'))
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(worker,[(c,a.output,f) for c in cases]):
            results.append(result);json_write(a.output/'progress.json',results);print(result['case_id'],result['success'],flush=True)
    json_write(a.output/'runner-statistics.json',formal_statistics(cases,results))

if __name__=='__main__':main()
