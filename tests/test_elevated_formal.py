import pytest
from adaptive_drone_patrol.elevated_formal import formal_statistics


def test_campaign_claim_rejects_second_output_before_exposure(tmp_path):
    from adaptive_drone_patrol.elevated_formal import claim_physical_campaign
    claim_physical_campaign(tmp_path,'frozen-digest',tmp_path/'first')
    with pytest.raises(FileExistsError):
        claim_physical_campaign(tmp_path,'frozen-digest',tmp_path/'different-output')


def test_formal_acceptance_rehashes_every_case_before_admission(tmp_path):
    import json
    from adaptive_drone_patrol.elevated_artifacts import inventory,file_record
    from adaptive_drone_patrol.elevated_campaign import audit_source_hashes
    from adaptive_drone_patrol.elevated_formal import bind_formal_statistics,verify_formal_acceptance
    from adaptive_drone_patrol.analytical import content_hash
    from adaptive_drone_patrol.util import sha256
    cases=[dict(case_id=f'{suite}-{i}',suite=suite,scenario=['normal','emergency','cost_shift'][i%3]) for suite,n in [('basic',9),('endpoints',18),('heldout',100)] for i in range(n)]
    (tmp_path/'independent-audit').mkdir()
    (tmp_path/'declaration.json').write_text(json.dumps(dict(cases=cases,freeze_sha256='freeze',freeze_digest='digest',attempts_per_case=1)))
    rows=[]
    for case in cases:
        root=tmp_path/case['case_id'];root.mkdir();(root/'raw').write_bytes(b'raw')
        (root/'formal-attempt.json').write_text(json.dumps(dict(attempt=1,freeze_digest='digest',case_digest=content_hash(case))))
        row=dict(case_id=case['case_id'],success=True,artifacts=inventory(root),checked_sources=audit_source_hashes())
        rows.append(row);(tmp_path/'independent-audit'/f"{case['case_id']}.json").write_text(json.dumps(row))
    stats=bind_formal_statistics(tmp_path,cases,rows,'freeze',sha256('scripts/audit_elevated_formal.py'))
    (tmp_path/'independent-statistics.json').write_text(json.dumps(stats))
    assert verify_formal_acceptance(tmp_path,cases,'freeze','digest')['passed']
    # A non-recording case must invalidate aggregate admission too.
    (tmp_path/cases[-1]['case_id']/'raw').write_bytes(b'bad')
    with pytest.raises(ValueError):verify_formal_acceptance(tmp_path,cases,'freeze','digest')


def test_fixed127_denominator_preserves_failures_and_excludes_routing():
    cases=[dict(case_id=f'{suite}-{i}',suite=suite,scenario=['normal','emergency','cost_shift'][i%3]) for suite,n in [('basic',9),('endpoints',18),('heldout',100)] for i in range(n)]
    results=[dict(case_id=c['case_id'],success=not (c['suite']=='heldout' and int(c['case_id'].split('-')[-1])<5)) for c in cases]
    s=formal_statistics(cases,results)
    assert s['passed'] and s['successes']==122 and s['groups']['heldout']['successes']==95
    assert s['routing_cases_in_denominator']==0
    results[-1]['success']=False
    assert not formal_statistics(cases,results)['passed']
    with pytest.raises(ValueError):formal_statistics(cases,results[:-1])
    results[-1]=results[0]
    with pytest.raises(ValueError):formal_statistics(cases,results)
