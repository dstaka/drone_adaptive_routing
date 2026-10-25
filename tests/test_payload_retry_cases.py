"""Retry migration preserves predeclared cases without granting admission."""

import copy
import importlib.util
import hashlib
import numpy as np
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from adaptive_drone_patrol.payload_cases import write_case_manifest

REPO = Path(__file__).parents[1]
SCRIPT = REPO / 'scripts/prepare_payload_retry_cases.py'
ORIGINAL_SHA256 = 'f98e0632d96a4d7b40feda667bff6e9ca831fc7f67e1734f2438e3c527ecf01c'


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / 'repository'
    for relative in [
        *[f'configs/demo_scenarios_dispersed/{name}.yaml'
          for name in ('normal', 'emergency', 'cost_shift')],
        'configs/payload_retry_probe_manifest.json',
        'src/adaptive_drone_patrol/payload_ball_asset.py',
    ]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    source = root / 'outputs/suspended_20261021_v1/cases-v1.json'
    source.parent.mkdir(parents=True)
    original = REPO / 'outputs/suspended_20261021_v1/cases-v1.json'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == ORIGINAL_SHA256
    shutil.copyfile(original, source)
    return root, source, root / 'outputs/retry/cases.json'


def invoke(root, out, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), '--root', str(root), '--out', str(out), *extra],
        cwd=REPO, capture_output=True, text=True, timeout=20,
    )


def test_migration_keeps_all_original_physics_and_provisional_status(inputs):
    # Dropping/replacing a draw, recording choice or old status must fail this test.
    root, source, out = inputs
    original_bytes = source.read_bytes()
    result = invoke(root, out)
    assert result.returncode == 0, result.stderr
    wrapper = json.loads(out.read_text())
    old = json.loads(original_bytes)
    assert wrapper['inherited_manifest'] == old
    assert len(wrapper['inherited_manifest']['cases']) == 127
    assert wrapper['inherited_manifest']['recording_cases'] == [
        'basic-normal-1', 'basic-emergency-1', 'basic-cost_shift-1',
    ]
    assert wrapper['inherited_manifest']['attachment_status'].startswith('PROVISIONAL:')
    assert wrapper['schema'] == 'payload-retry-cases-wrapper-v1'
    assert wrapper['profile'] == 'cf2x-two-native-ball-root-aligned-v3'
    assert wrapper['status'] == 'EXPERIMENTAL_DIAGNOSTICS_ONLY'
    assert wrapper['acceptance_freeze'] == 'NOT_FROZEN_FOR_ACCEPTANCE'
    assert wrapper['qualified_for_training'] is False
    assert wrapper['results'] == 'NOT_RUN'
    assert wrapper['original_manifest_sha256'] == ORIGINAL_SHA256
    assert source.read_bytes() == original_bytes
    assert str(root) not in out.read_text()


@pytest.mark.parametrize('mutation', ['draw', 'derived_ulp', 'old_status', 'recording', 'whitespace'])
def test_changed_original_bytes_cannot_become_retry_cases(inputs, mutation):
    # Even metadata-only edits cannot bypass the immutable old manifest hash.
    root, source, out = inputs
    old = json.loads(source.read_text())
    if mutation == 'draw':
        old['cases'][0]['initial']['drone_xyz'][0] += .001
    elif mutation == 'derived_ulp':
        old['cases'][0]['initial']['package_xyz'][0] = float(np.nextafter(old['cases'][0]['initial']['package_xyz'][0], 1.))
    elif mutation == 'old_status':
        old['results'] = 'PASS'
    elif mutation == 'recording':
        old['recording_cases'][0] = 'basic-normal-2'
    source.write_text(json.dumps(old, indent=2) + ('\n\n' if mutation == 'whitespace' else '\n'))
    result = invoke(root, out)
    assert result.returncode != 0
    assert 'hash' in result.stderr.lower()
    assert not out.exists()


def test_changed_scenario_bytes_fail_even_if_yaml_meaning_matches(inputs):
    root, _, out = inputs
    scenario = root / 'configs/demo_scenarios_dispersed/emergency.yaml'
    scenario.write_bytes(scenario.read_bytes() + b'\n# changed bytes\n')
    result = invoke(root, out)
    assert result.returncode != 0
    assert 'scenario' in result.stderr.lower()
    assert not out.exists()


def test_source_and_manifest_bindings_use_actual_current_contents(inputs):
    # A source or declaration change must produce a different binding, never stale hashes.
    root, _, out = inputs
    assert invoke(root, out).returncode == 0
    first = json.loads(out.read_text())
    asset = root / 'src/adaptive_drone_patrol/payload_ball_asset.py'
    asset.write_bytes(asset.read_bytes() + b'\n# next experimental revision\n')
    manifest = root / 'configs/payload_retry_probe_manifest.json'
    declaration = json.loads(manifest.read_text())
    declaration['notes'] = 'New experimental declaration; no admission'
    manifest.write_text(json.dumps(declaration))
    next_out = out.with_name('next-cases.json')
    result = invoke(root, next_out)
    assert result.returncode == 0, result.stderr
    second = json.loads(next_out.read_text())
    binding = second['source_binding']
    assert binding['probe_manifest']['sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    asset_record = binding['sources']['src/adaptive_drone_patrol/payload_ball_asset.py']
    assert asset_record == hashlib.sha256(asset.read_bytes()).hexdigest()
    assert second['source_binding_sha256'] != first['source_binding_sha256']
    assert second['inherited_manifest'] == first['inherited_manifest']
    assert second['acceptance_freeze'] == 'NOT_FROZEN_FOR_ACCEPTANCE'


@pytest.mark.parametrize('field,value', [
    ('profile', 'payload-articulation-v1-isolated-scenes'),
    ('status', 'QUALIFIED_FOR_TRAINING'),
])
def test_old_profile_or_claimed_admission_is_rejected(inputs, field, value):
    root, _, out = inputs
    manifest = root / 'configs/payload_retry_probe_manifest.json'
    declaration = json.loads(manifest.read_text())
    declaration[field] = value
    manifest.write_text(json.dumps(declaration))
    result = invoke(root, out)
    assert result.returncode != 0
    assert 'experimental declaration' in result.stderr.lower()
    assert not out.exists()


def test_reconstruction_matches_original_and_never_rerolls(inputs):
    root, source, out = inputs
    expected = json.loads(source.read_text())
    source.unlink()
    check = root / 'recalculated.json'
    regenerated_hash = write_case_manifest(check, root)
    result = invoke(root, out, '--reconstruct')
    if regenerated_hash != ORIGINAL_SHA256:
        assert result.returncode != 0 and not out.exists()
        assert 'Reconstructed' in result.stderr
        assert not source.exists()
        return
    assert result.returncode == 0, result.stderr
    wrapper = json.loads(out.read_text())
    assert wrapper['inherited_manifest'] == expected
    assert wrapper['original_manifest_sha256'] == ORIGINAL_SHA256
    assert wrapper['migration_method'] == 'INDEPENDENT_EXACT_RECONSTRUCTION'
    assert not source.exists()


def test_existing_output_is_preserved(inputs):
    root, _, out = inputs
    out.parent.mkdir(parents=True)
    out.write_text('retain this earlier attempt\n')
    result = invoke(root, out)
    assert result.returncode != 0
    assert 'exists' in result.stderr.lower()
    assert out.read_text() == 'retain this earlier attempt\n'


def test_original_bytes_survive_generator_package_position_roundoff(inputs, monkeypatch):
    root, source, _ = inputs
    spec=importlib.util.spec_from_file_location('retry_cases', SCRIPT)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original=json.loads(source.read_text())
    regenerated=copy.deepcopy(original['cases'])
    regenerated[0]['initial']['package_xyz'][0]=float(np.nextafter(regenerated[0]['initial']['package_xyz'][0],1.))
    monkeypatch.setattr(module.payload_cases,'build_cases',lambda:regenerated)
    assert module.verify_original_cases(source,root)==original
    regenerated[0]['mass_ratio']=float(np.nextafter(regenerated[0]['mass_ratio'],1.))
    with pytest.raises(ValueError):
        module.verify_original_cases(source,root)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==ORIGINAL_SHA256
