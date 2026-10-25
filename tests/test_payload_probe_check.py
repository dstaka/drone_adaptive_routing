"""Independent measured-probe calculations with explicit synthetic fixtures."""

import importlib.util
from pathlib import Path
import pytest


def checker():
    spec = importlib.util.spec_from_file_location('payload_checker', Path(__file__).parents[1]/'scripts/verify_payload_probes.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_independent_geometry_uses_rotated_cube_and_exact_attachments():
    m = checker()
    sample = dict(drone_xyz=[[0,0,1]], drone_quaternion_wxyz=[[1,0,0,0]],
                  package_xyz=[[0,0,.6575]], package_quaternion_wxyz=[[1,0,0,0]])
    result = m.geometry(sample, .045, 0)
    assert result['residual_m'] == pytest.approx(0, abs=1e-12)
    assert result['floor_m'] == pytest.approx(.635)
    sample['package_xyz'][0][0] = .1
    assert m.geometry(sample, .045, 0)['residual_m'] > .01


def test_changed_sample_cannot_reuse_receipt(tmp_path):
    m = checker()
    (tmp_path/'samples.jsonl').write_text('{}\n')
    (tmp_path/'summary.json').write_text('{}')
    before = m.binding(tmp_path)
    (tmp_path/'samples.jsonl').write_text('{"step": 1}\n')
    assert m.binding(tmp_path) != before


def test_missing_probe_set_is_rejected(tmp_path):
    m = checker()
    with pytest.raises(ValueError):
        m.verify(tmp_path, {})
