"""A failure receipt must be supported by the bound raw simulator samples."""

import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('failure_checker',Path('scripts/check_payload_failure.py'))
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def fixture(tmp_path,speed):
    raw=tmp_path/'raw-states.jsonl'
    raw.write_text(json.dumps(dict(index=0,actions=[0,0,0,0],package_velocity=[speed,0,0]))+'\n')
    (tmp_path/'summary.json').write_text(json.dumps(dict(raw_state_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                samples=1,status='FAILED',mission_success=False,checkpoint_sha256='actor')))
    return tmp_path


def test_ordinary_velocity_cannot_be_reported_as_catastrophic_failure(tmp_path):
    with pytest.raises(ValueError,match='excursion'):
        checker.verify_trace(fixture(tmp_path,.1))


def test_stale_samples_rejected(tmp_path):
    path=fixture(tmp_path,1e8)
    (path/'raw-states.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='hash'):
        checker.verify_trace(path)


def test_actual_bound_excursion_is_failure_evidence(tmp_path):
    result=checker.verify_trace(fixture(tmp_path,1e8))
    assert result['max_package_speed_m_s']==1e8
