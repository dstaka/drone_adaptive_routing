"""Opt-in elevated geometry; historical dispersed inputs/readers stay immutable."""
import copy
from pathlib import Path
import numpy as np
import yaml
from .dispersed_layout import layout_checks, coordinate_hash
from .scenario import Scenario
from .util import load_config, sha256, json_write

PROFILE = 'payload-elevated-z-v1'


def transform_coordinates(coords):
    checks = layout_checks(coords)
    if not checks['passed']:
        raise ValueError('Baseline dispersed geometry rejected: ' + str(checks['rejection_reasons']))
    result = np.array(coords, dtype=float, copy=True)
    result[1:, 2] = 2. + 1.5 * (result[1:, 2] - 1.)
    return result.tolist()


def elevated_checks(coords):
    try:
        a = np.asarray(coords, dtype=float)
        if a.shape != (11, 3) or not np.isfinite(a).all():
            raise ValueError('shape/nonfinite')
        old = a.copy()
        old[1:, 2] = 1. + (old[1:, 2] - 2.) / 1.5
        result = layout_checks(old.tolist())
        # Separation is also independently checked on the executed coordinates.
        Scenario('elevated-check', 'demo', 0, a.tolist(), [1]*10,
                 np.ones((11,11)).tolist(), [.5]*10, beta=.05,
                 schema_version=3, checkpoint_ids=list('ABCDEFGHIJ')).validate_physical()
        result.update(spans_m=np.ptp(a[1:],axis=0).tolist(),height_min_max_m=[float(a[1:,2].min()),float(a[1:,2].max())],
                      min_separation_m=min(float(np.linalg.norm(x-y)) for i,x in enumerate(a[1:]) for y in a[1:][i+1:]),
                      min_base_distance_m=float(np.linalg.norm(a[1:]-a[0],axis=1).min()),
                      height_band_counts=np.bincount(np.searchsorted([3.,4.],a[1:,2],side='right'),minlength=3).tolist())
        return result
    except (TypeError, ValueError):
        return {'passed': False, 'rejection_reasons': ['invalid-elevated-geometry']}


def build_profiles(directory, baseline='configs/demo_scenarios_dispersed'):
    directory, baseline = Path(directory), Path(baseline)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='payload-elevated-layout-v1', profile=PROFILE, units='metres',
                    transform='z_new=2.0+1.5*(z_old-1.0); A-J only', scenarios={})
    shared = None
    for name in ('normal', 'emergency', 'cost_shift'):
        source = baseline / f'{name}.yaml'
        data = copy.deepcopy(load_config(source))
        data['coords'] = transform_coordinates(data['coords'])
        if shared is not None and data['coords'] != shared:
            raise ValueError('Mixed scenario geometry')
        shared = data['coords']
        dest = directory / f'{name}.yaml'
        if dest.exists():
            raise FileExistsError(dest)
        dest.write_text(yaml.safe_dump(data, sort_keys=False))
        manifest['scenarios'][name] = dict(baseline_sha256=sha256(source), sha256=sha256(dest))
    manifest.update(coords=shared, coordinate_sha256=coordinate_hash(shared), checks=elevated_checks(shared))
    json_write(directory / 'layout.json', manifest)
    return manifest
