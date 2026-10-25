"""Native field omissions and contact aliases cannot weaken safety replay."""

import copy
import json
from pathlib import Path

import pytest


@pytest.fixture
def native():
    return json.loads((Path(__file__).parent/'data/payload_solver/native-reset.json').read_text())


def test_actual_native_reset_has_complete_schema(native):
    from adaptive_drone_patrol.payload_solver_trace import match_native_sample

    actual=native['actual']
    assert match_native_sample(dict(actual,contacts=False),actual,native['properties'])


@pytest.mark.parametrize('mutation',['qpos','qacc','rod_xyz','warning_counts','qvel_nan','contact_type','force_shape'])
def test_incomplete_or_warned_native_getters_are_rejected(native,mutation):
    from adaptive_drone_patrol.payload_solver_trace import match_native_sample

    actual=copy.deepcopy(native['actual'])
    if mutation in ['qpos','qacc','rod_xyz']:
        actual.pop(mutation)
    elif mutation=='warning_counts':
        actual['warning_counts'][0]=1
    elif mutation=='qvel_nan':
        actual['qvel'][0]=float('nan')
    elif mutation=='contact_type':
        actual['contact']='false'
    else:
        actual['actual_xfrc_applied'].pop()
    with pytest.raises(ValueError):
        match_native_sample(dict(actual,contacts=False),actual,native['properties'])


def test_actual_contact_true_cannot_be_hidden_in_plural_alias(native):
    from adaptive_drone_patrol.payload_solver_trace import match_native_sample

    actual=copy.deepcopy(native['actual'])
    actual['contact']=True
    with pytest.raises(ValueError,match='contact'):
        match_native_sample(dict(actual,contacts=False),actual,native['properties'])
