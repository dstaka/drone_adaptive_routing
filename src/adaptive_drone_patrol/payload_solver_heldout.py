"""Host-local one-opening guard for the unchanged canonical held-out stream."""

from pathlib import Path
import json
import tempfile

from .util import sha256

ORIGINAL_CASES_SHA = 'f98e0632d96a4d7b40feda667bff6e9ca831fc7f67e1734f2438e3c527ecf01c'


def heldout_registry():
    """Keep the claim independent of workspace, output, model and freeze names."""
    return Path(tempfile.gettempdir()) / 'payload-solver-canonical-heldout-claims'


def claim_path():
    return heldout_registry() / (ORIGINAL_CASES_SHA + '.json')


def require_unopened_stream():
    if claim_path().exists():
        raise FileExistsError('Canonical heldout stream was already opened; preserve the original attempt')
    return True


def verify_opening(freeze_path, output):
    """Read portable original opening evidence without claiming another trial."""
    marker = json.loads((Path(output) / 'opened.json').read_text())
    if (marker.get('schema') != 'payload-solver-heldout-opened-v2'
            or marker.get('canonical_cases_sha256') != ORIGINAL_CASES_SHA
            or marker.get('freeze_sha256') != sha256(freeze_path)
            or marker.get('required_trials') != 254):
        raise ValueError('Retained canonical heldout opening differs')
    # Delivery to another host does not transfer its temporary host-local guard.
    # Verification cannot start physics; new execution always atomically claims.
    if claim_path().exists() and json.loads(claim_path().read_text()) != marker:
        raise ValueError('Host canonical opening differs from retained original attempt')
    return True
