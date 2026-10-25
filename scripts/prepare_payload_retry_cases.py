#!/usr/bin/env python
"""Wrap unchanged historical cases for diagnostics; never freeze or admit them.

The source binding identifies current generator/declaration bytes, not a tested
physical asset or qualified backend. Acceptance requires a separate complete
qualification, validation and source/model/settings freeze.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

from adaptive_drone_patrol import payload_cases, payload_contract, payload_qualification
from adaptive_drone_patrol.payload_cases import write_case_manifest
from adaptive_drone_patrol.payload_qualification import validate_declaration
from adaptive_drone_patrol.util import sha256

ORIGINAL_SHA256 = 'f98e0632d96a4d7b40feda667bff6e9ca831fc7f67e1734f2438e3c527ecf01c'
PROFILE = 'cf2x-two-native-ball-root-aligned-v3'
ORIGINAL_PATH = 'outputs/suspended_20261021_v1/cases-v1.json'
ASSET_SOURCE = 'src/adaptive_drone_patrol/payload_ball_asset.py'
PROBE_MANIFEST = 'configs/payload_retry_probe_manifest.json'


def verify_original_cases(source, root):
    """Retain exact original bytes across float64 matrix arithmetic backends.

    Only the independent re-computation of derived package center coordinates
    may differ within eight float64 rounding units at unit scale. Input bytes,
    every random draw, pose quaternion, ID and all other fields remain exact.
    Even a one-ULP edit of the original file fails the mandatory original hash.
    """
    if sha256(source) != ORIGINAL_SHA256:
        raise ValueError('Case manifest hash mismatch')
    inherited=json.loads(Path(source).read_text())
    expected=payload_cases.build_cases()
    if (inherited['schema']!='payload-cases-v1' or len(inherited['cases'])!=len(expected) or
            inherited['scenario_contract']!=payload_cases._scenario_contract(Path(root))):
        raise ValueError('Actual scenario or case contract differs')
    for actual, regenerated in zip(inherited['cases'],expected):
        left,right=copy.deepcopy(actual),copy.deepcopy(regenerated)
        a,b=[np.asarray(c['initial'].pop('package_xyz'),dtype=float) for c in (left,right)]
        rounding=8*np.finfo(float).eps*np.maximum(1.,np.maximum(np.abs(a),np.abs(b)))
        if (left!=right or a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a).all() or
                not np.isfinite(b).all() or np.any(np.abs(a-b)>rounding)):
            raise ValueError('Cases differ from predeclared draws or derived geometry')
    return inherited


def prepare_retry_cases(source, output, root, *, reconstruct=False,
                        asset_source=ASSET_SOURCE, probe_manifest=PROBE_MANIFEST):
    """Validate immutable cases and write a new, explicitly unfrozen wrapper."""
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Output already exists; use a new attempt path')
    if reconstruct:
        with tempfile.TemporaryDirectory(prefix='payload-retry-cases-') as temporary:
            reconstructed = Path(temporary) / 'cases.json'
            write_case_manifest(reconstructed, root)
            if sha256(reconstructed) != ORIGINAL_SHA256:
                raise ValueError('Reconstructed case/scenario hash differs from the original declaration')
            inherited = verify_original_cases(reconstructed, root)
    else:
        source = Path(source)
        if not source.is_absolute():
            source = root / source
        inherited = verify_original_cases(source, root)

    declaration_path = (root / probe_manifest).resolve()
    asset_path = (root / asset_source).resolve()
    # Portable labels also prevent silently binding a source outside the checkout.
    declaration_label = declaration_path.relative_to(root).as_posix()
    asset_label = asset_path.relative_to(root).as_posix()
    declaration = json.loads(declaration_path.read_text())
    if (declaration.get('profile') != PROFILE or
            declaration.get('status') != 'EXPERIMENTAL_DIAGNOSTICS_ONLY'):
        raise ValueError('The retry experimental declaration is required; no old profile or admission claim')
    validate_declaration(declaration)
    sources = {
        asset_label: sha256(asset_path),
        'scripts/prepare_payload_retry_cases.py': sha256(__file__),
        'src/adaptive_drone_patrol/payload_cases.py': sha256(payload_cases.__file__),
        'src/adaptive_drone_patrol/payload_contract.py': sha256(payload_contract.__file__),
        'src/adaptive_drone_patrol/payload_qualification.py': sha256(payload_qualification.__file__),
    }
    binding = {
        'schema': 'payload-retry-case-source-binding-v1',
        'profile': PROFILE,
        'probe_manifest': {'path': declaration_label, 'sha256': sha256(declaration_path)},
        'sources': sources,
        'scope': 'Current source/declaration identity only; generated asset and runtime physics NOT_QUALIFIED',
    }
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    wrapper = {
        'schema': 'payload-retry-cases-wrapper-v1',
        'profile': PROFILE,
        'status': 'EXPERIMENTAL_DIAGNOSTICS_ONLY',
        'acceptance_freeze': 'NOT_FROZEN_FOR_ACCEPTANCE',
        'qualified_for_training': False,
        'results': 'NOT_RUN',
        'migration_method': 'INDEPENDENT_EXACT_RECONSTRUCTION' if reconstruct else 'VERIFIED_ORIGINAL_BYTES',
        'migration_scope': 'Metadata wrapper only; all historical cases, draws, scenario and recording choices retained',
        'original_manifest_sha256': ORIGINAL_SHA256,
        'inherited_manifest': inherited,
        'source_binding': binding,
        'source_binding_sha256': digest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(wrapper, indent=2, allow_nan=False) + '\n')
    return sha256(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--source', type=Path, default=Path(ORIGINAL_PATH))
    inputs.add_argument('--reconstruct', action='store_true')
    parser.add_argument('--asset-source', type=Path, default=Path(ASSET_SOURCE))
    parser.add_argument('--probe-manifest', type=Path, default=Path(PROBE_MANIFEST))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    digest = prepare_retry_cases(args.source, args.out, args.root, reconstruct=args.reconstruct,
                                 asset_source=args.asset_source, probe_manifest=args.probe_manifest)
    print(f'{args.out}: {digest} (127 unchanged inputs; EXPERIMENTAL_DIAGNOSTICS_ONLY; NOT_FROZEN_FOR_ACCEPTANCE)')


if __name__ == '__main__':
    main()
