"""Predeclared paired payload cases; no synthetic case is a flight result."""

from collections import Counter
from pathlib import Path
import math
import numpy as np
import pytest
from adaptive_drone_patrol.payload_cases import build_cases, write_case_manifest, verify_case_manifest
from adaptive_drone_patrol.payload_contract import LINK_M, DRONE_ATTACHMENT, rotation


def test_all_127_cases_are_paired_balanced_and_reproducible():
    cases = build_cases()
    assert cases == build_cases()
    assert len(cases) == len({c['case_id'] for c in cases}) == 127
    assert Counter(c['suite'] for c in cases) == {'basic': 9, 'endpoints': 18, 'heldout': 100}
    held = [c for c in cases if c['suite'] == 'heldout']
    counts = Counter((c['scenario'], c['side_m']) for c in held)
    assert counts.pop(('normal', .03)) == 12
    assert set(counts.values()) == {11}
    assert len({c['mass_ratio'] for c in held}) == 100
    assert all(.1 <= c['mass_ratio'] <= .3 for c in cases)
    for seed in [1, 2, 3]:
        initial = [c['initial'] for c in cases if c['suite'] == 'basic' and c['execution_seed'] == seed]
        assert initial[0] == initial[1] == initial[2]
    assert len({str(c['initial']) for c in cases if c['suite'] == 'basic'}) == 3


def test_initial_conditions_are_consistent_with_fixed_link_and_bounds():
    for c in build_cases():
        state = c['initial']
        d, p = np.array(state['drone_xyz']), np.array(state['package_xyz'])
        anchor = d + rotation(state['drone_quaternion_wxyz']) @ DRONE_ATTACHMENT
        end = p + np.array([0, 0, c['side_m']/2])
        assert np.linalg.norm(end-anchor) == pytest.approx(LINK_M)
        assert abs(d[0]) <= .02 and abs(d[1]) <= .02 and d[2] == 1
        assert all(v == 0 for k, v in state.items() if k.endswith('velocity') for v in v)
        assert state['swing_angle_deg'] <= 10
        if c['suite'] == 'endpoints':
            assert state['swing_angle_deg'] == 10
        assert np.linalg.norm(rotation(state['package_quaternion_wxyz']) - np.eye(3)) < 1e-12
        assert np.linalg.norm(end-anchor) * math.cos(math.radians(10)) > .29


def test_manifest_integrity_and_inherited_contract(tmp_path):
    root = Path(__file__).parents[1]
    path = tmp_path/'cases.json'
    digest = write_case_manifest(path, root)
    m = verify_case_manifest(path, digest, root)
    assert m['scenario_contract']['priority_seed'] == 40029
    assert m['results'] == 'NOT_RUN'
    assert m['recording_cases'] == ['basic-normal-1', 'basic-emergency-1', 'basic-cost_shift-1']
    # Re-hashing a swapped case must not make it a predeclared case.
    import json
    from adaptive_drone_patrol.util import sha256
    m['cases'][0]['side_m'] = .06
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        verify_case_manifest(path, sha256(path), root)
    with pytest.raises(ValueError):
        verify_case_manifest(path, digest, root)
