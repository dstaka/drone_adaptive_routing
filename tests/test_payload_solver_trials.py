"""Held-out attempts must never be selectively rerun or silently omitted."""

import json
import pytest


@pytest.fixture(autouse=True)
def isolated_claim_registry(tmp_path, monkeypatch):
    from adaptive_drone_patrol import payload_solver_heldout

    monkeypatch.setattr(payload_solver_heldout, 'heldout_registry', lambda: tmp_path / 'claims')


def test_one_freeze_cannot_start_a_second_attempt_at_another_output(tmp_path, monkeypatch):
    from adaptive_drone_patrol.payload_solver_trials import claim_heldout_once

    monkeypatch.chdir(tmp_path)
    freeze = tmp_path / 'freeze.json'
    freeze.write_text(json.dumps(dict(schema='test-identity-only')))
    marker = claim_heldout_once(freeze, tmp_path / 'first')
    assert marker['freeze_sha256'] and marker['required_trials'] == 254
    with pytest.raises(FileExistsError):
        claim_heldout_once(freeze, tmp_path / 'second')
    assert not (tmp_path / 'second').exists()


def test_verification_error_remains_a_failed_original_trial_with_all_artifacts(tmp_path):
    from adaptive_drone_patrol.payload_solver_trials import failed_trial
    from adaptive_drone_patrol.util import sha256

    (tmp_path / 'raw.jsonl').write_text('original failure\n')
    row = failed_trial('fixed', dict(case_id='heldout-normal-0-0'), tmp_path, ValueError('raw mismatch'))
    assert row['case_id'] == 'heldout-normal-0-0' and row['method'] == 'fixed'
    assert not row['success'] and not row['physical_acceptance']
    assert row['artifact_sha256']['raw.jsonl'] == sha256(tmp_path / 'raw.jsonl')
    assert 'raw mismatch' in row['error']


def test_same_freeze_bytes_at_another_path_cannot_reopen_the_suite(tmp_path, monkeypatch):
    from adaptive_drone_patrol.payload_solver_trials import claim_heldout_once

    monkeypatch.chdir(tmp_path)
    a, b = tmp_path / 'a.json', tmp_path / 'copy.json'
    a.write_text('{"same":"freeze"}')
    b.write_bytes(a.read_bytes())
    claim_heldout_once(a, tmp_path / 'first')
    with pytest.raises(FileExistsError):
        claim_heldout_once(b, tmp_path / 'second')


def test_changed_freeze_metadata_cannot_reopen_same_canonical_stream(tmp_path):
    from adaptive_drone_patrol.payload_solver_trials import claim_heldout_once
    from adaptive_drone_patrol.payload_solver_heldout import require_unopened_stream

    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text('{"models":"first","comment":"original"}')
    b.write_text('{"models":"changed","comment":"new freeze"}')
    assert require_unopened_stream()
    claim_heldout_once(a, tmp_path / 'first')
    with pytest.raises(FileExistsError, match='Canonical'):
        require_unopened_stream()
    with pytest.raises(FileExistsError):
        claim_heldout_once(b, tmp_path / 'second')
    assert not (tmp_path / 'second').exists()


def test_retained_opening_can_be_verified_on_host_without_original_temp_registry(tmp_path):
    import shutil
    from adaptive_drone_patrol.payload_solver_trials import claim_heldout_once
    from adaptive_drone_patrol.payload_solver_heldout import claim_path, verify_opening

    freeze = tmp_path / 'freeze.json'
    freeze.write_text('{"same":"original freeze"}')
    original = tmp_path / 'original'
    claim_heldout_once(freeze, original)
    delivered = tmp_path / 'delivered'
    shutil.copytree(original,delivered)
    claim_path().unlink()
    assert verify_opening(freeze,delivered)
