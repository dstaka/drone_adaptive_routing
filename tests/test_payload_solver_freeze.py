"""Acceptance identity binds exact portable artifacts, not serialized PASS labels."""

import hashlib
import pytest


def test_changed_frozen_source_is_rejected_even_when_status_is_unchanged(tmp_path, monkeypatch):
    from adaptive_drone_patrol.payload_solver_freeze import verify_file_snapshot

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "source.py"
    path.write_text("x=1\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_file_snapshot({"source.py": digest})
    path.write_text("x=2\n")
    with pytest.raises(ValueError, match="changed"):
        verify_file_snapshot({"source.py": digest})


@pytest.mark.parametrize("path", ["../outside.py", "/private/source.py"])
def test_freeze_requires_portable_explicit_artifact_labels(path):
    from adaptive_drone_patrol.payload_solver_freeze import verify_file_snapshot

    with pytest.raises(ValueError, match="relative"):
        verify_file_snapshot({path: "a" * 64})


def test_status_only_freeze_cannot_open_heldout_cases():
    from adaptive_drone_patrol.payload_solver_freeze import require_acceptance_freeze

    with pytest.raises(ValueError):
        require_acceptance_freeze(dict(status="FROZEN_BEFORE_HELDOUT"), None)


def test_changed_routing_reference_cannot_escape_unchanged_artifact_snapshot():
    from adaptive_drone_patrol.payload_solver_freeze import validate_freeze_references

    with pytest.raises(ValueError, match='snapshot'):
        validate_freeze_references(dict(routing_model='other/router.zip', camera_assets=[],
            artifact_sha256={'inputs/router.zip':'a'*64}))
