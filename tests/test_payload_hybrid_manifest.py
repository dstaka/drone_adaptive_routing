import numpy as np
import pytest

from adaptive_drone_patrol.payload_hybrid_manifest import development_cases, validate_gate_result, validate_cases


def test_manifest_has_exact_counts_independent_starts_and_nonoverlapping_targets():
    cases = development_cases()
    assert [sum(c['gate']==g for c in cases) for g in ('C0','C1','C2')] == [10,10,30]
    assert len({c['seed'] for c in cases}) == 50
    assert [c['initial']['swing_angle_deg'] for c in cases[:10]] == [5.]*5+[10.]*5
    for c in cases[10:]:
        assert len(c['targets']) == 3
        assert np.linalg.norm(np.subtract(c['targets'][1], c['targets'][0])) == pytest.approx(.5)
        assert c['initial']['package_quaternion_wxyz'] == [1,0,0,0]
    assert development_cases() == cases


def test_gate_needs_all_declared_attempts_and_actual_motion():
    cases = [c for c in development_cases() if c['gate']=='C1']
    good = [dict(case_id=c['case_id'], success=True, safe=True, completed_targets=3,
                 max_displacement_m=.5) for c in cases]
    assert validate_gate_result('C1', good)
    bad = [dict(r) for r in good]
    bad[0]['safe'] = False
    assert not validate_gate_result('C1', bad)
    assert not validate_gate_result('C1', good[:9])
    assert not validate_gate_result('C1', [dict(r, max_displacement_m=0.) for r in good])


def test_platform_roundoff_in_initial_geometry_keeps_frozen_manifest_authoritative():
    cases = development_cases()
    cases[0]['initial']['package_xyz'][0] += 1e-17
    assert validate_cases(cases)
    cases[0]['targets'][0][0] = 1e-17
    with pytest.raises(ValueError):
        validate_cases(cases)
