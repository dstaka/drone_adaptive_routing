import copy
import numpy as np
import pytest
from adaptive_drone_patrol.elevated_layout import transform_coordinates, elevated_checks, build_profiles
from adaptive_drone_patrol.util import load_config


def test_transform_preserves_horizontal_base_and_non_geometry(tmp_path):
    old = load_config('configs/demo_scenarios_dispersed/normal.yaml')
    coords = transform_coordinates(old['coords'])
    assert coords[0] == [0., 0., 1.]
    assert np.array_equal(np.array(coords)[:, :2], np.array(old['coords'])[:, :2])
    assert coords[1][2] == pytest.approx(2.1525485966228524, abs=1e-14)
    assert coords[7][2] == 4.867951028521382
    assert elevated_checks(coords)['passed']
    manifest = build_profiles(tmp_path)
    assert manifest['profile'] == 'payload-elevated-z-v1'
    for name in ('normal', 'emergency', 'cost_shift'):
        before = load_config(f'configs/demo_scenarios_dispersed/{name}.yaml')
        after = load_config(tmp_path / f'{name}.yaml')
        assert after.pop('coords') == coords
        before.pop('coords')
        assert before == after
    assert np.ptp(np.array(coords)[1:, 2]) == pytest.approx(2.7413168908485783)


@pytest.mark.parametrize('mutation', ['nan', 'old', 'base', 'shape'])
def test_rejects_invalid_elevated_coordinates(mutation):
    old = load_config('configs/demo_scenarios_dispersed/normal.yaml')['coords']
    coords = transform_coordinates(old)
    if mutation == 'nan': coords[2][2] = float('nan')
    if mutation == 'old': coords = old
    if mutation == 'base': coords[0][2] = 2.
    if mutation == 'shape': coords.pop()
    assert not elevated_checks(coords)['passed']
    damaged = copy.deepcopy(old)
    damaged[1][2] = 9.
    with pytest.raises(ValueError): transform_coordinates(damaged)
