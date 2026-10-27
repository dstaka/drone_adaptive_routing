"""Fixed physical denominators and uncertainty, separate from routing cases."""
from collections import Counter
from .payload_solver_acceptance import wilson


def claim_physical_campaign(run,freeze_digest,output):
    """Durably claim the one physical exposure, independent of output choice."""
    import json,os
    from pathlib import Path
    receipt=Path(run)/'physical-exposure.json'
    with receipt.open('x') as stream:
        json.dump(dict(schema='elevated-physical-exposure-v1',freeze_digest=freeze_digest,
            output=str(Path(output).resolve()),status='CLAIMED_BEFORE_CASE_GENERATION'),stream)
        stream.flush();os.fsync(stream.fileno())
    return receipt


def bind_formal_statistics(formal,cases,results,freeze_sha256,source_sha256):
    from pathlib import Path
    from .elevated_artifacts import file_record
    root=Path(formal);stats=formal_statistics(cases,results)
    stats.update(freeze_sha256=freeze_sha256,source_sha256=source_sha256,
        declaration=file_record(root/'declaration.json'),
        case_audits={c['case_id']:file_record(root/'independent-audit'/f"{c['case_id']}.json") for c in cases})
    return stats


def verify_formal_acceptance(formal,cases,freeze_sha256,freeze_digest):
    """Recompute admission from all127 current evidence sets, never a PASS flag."""
    import json
    from pathlib import Path
    from .analytical import content_hash
    from .elevated_artifacts import file_record
    from .elevated_campaign import validate_audit_receipt
    from .util import sha256
    root=Path(formal);stats=json.loads((root/'independent-statistics.json').read_text())
    declaration=json.loads((root/'declaration.json').read_text())
    if (declaration['cases']!=cases or declaration['freeze_sha256']!=freeze_sha256
            or declaration['freeze_digest']!=freeze_digest or declaration['attempts_per_case']!=1
            or stats['freeze_sha256']!=freeze_sha256 or stats['declaration']!=file_record(root/'declaration.json')
            or set(stats['case_audits'])!={c['case_id'] for c in cases}
            or stats['source_sha256']!=sha256('scripts/audit_elevated_formal.py')):
        raise ValueError('Formal allocation/freeze/audit coverage differs')
    rows=[]
    for case in cases:
        path=root/'independent-audit'/f"{case['case_id']}.json"
        if file_record(path)!=stats['case_audits'][case['case_id']]:raise ValueError('Formal audit receipt changed')
        row=json.loads(path.read_text());validate_audit_receipt(root/case['case_id'],case,row)
        attempt=json.loads((root/case['case_id']/'formal-attempt.json').read_text())
        if attempt['attempt']!=1 or attempt['freeze_digest']!=freeze_digest or attempt['case_digest']!=content_hash(case):
            raise ValueError('Formal execution attempt differs')
        rows.append(row)
    current=bind_formal_statistics(root,cases,rows,freeze_sha256,stats['source_sha256'])
    if current!=stats or not current['passed']:raise ValueError('Independent formal admission unmet or stale')
    return current


def formal_statistics(cases,results):
    allocation=Counter(c['suite'] for c in cases)
    if len(cases)!=127 or allocation!=dict(basic=9,endpoints=18,heldout=100):
        raise ValueError('Exact127-case denominator required')
    ids=[c['case_id'] for c in cases];actual=[r['case_id'] for r in results]
    if len(actual)!=127 or len(set(actual))!=127 or len(set(ids))!=127 or set(ids)!=set(actual):
        raise ValueError('One actual result per declared case required')
    by_id={r['case_id']:r for r in results}
    def rate(subset):
        k=sum(by_id[c['case_id']]['success'] is True for c in subset)
        return dict(successes=k,total=len(subset),wilson95=wilson(k,len(subset)))
    groups={suite:rate([c for c in cases if c['suite']==suite]) for suite in allocation}
    scenario_rates={s:rate([c for c in cases if c['scenario']==s]) for s in ('normal','emergency','cost_shift')}
    return dict(schema='elevated-physical127-statistics-v1',groups=groups,per_scenario=scenario_rates,total=127,
        successes=sum(r['success'] is True for r in results),failures=[r for r in results if r['success'] is not True],
        passed=groups['basic']['successes']==9 and groups['endpoints']['successes']==18 and groups['heldout']['successes']>=95,
        routing_cases_in_denominator=0)
