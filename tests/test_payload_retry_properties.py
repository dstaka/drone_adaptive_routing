"""Every episode draw needs measured solver properties, not setter arguments."""
import importlib
import numpy as np
import pytest
from adaptive_drone_patrol.payload_contract import Payload


def measured(payload):
    side=payload.side_m
    return dict(original_mass_kg=payload.reference_mass_kg, package_mass_kg=payload.mass,
        package_inertia_kg_m2=(np.eye(3)*payload.inertia[0]).tolist(),
        package_xyz=[0,0,1-.02-.3-side/2], package_quaternion_wxyz=[1,0,0,0],
        drone_xyz=[0,0,1],drone_quaternion_wxyz=[1,0,0,0],
        package_aabb=[[ -side/2,-side/2,1-.02-.3-side],[side/2,side/2,1-.02-.3]],
        velocity=[0]*12, qpos=[0,0,1,1,0,0,0,1,0,0,0,1,0,0,0],
        asset_sha256='a'*64,asset_bytes_verified=True)


def test_measured_reset_properties_are_required_for_every_draw():
    module=importlib.import_module('adaptive_drone_patrol.payload_retry_properties')
    p=Payload(.02700047753751278,.2,.045)
    record=measured(p)
    assert module.validate_reset_properties(record,p)==[]
    for key in ('package_mass_kg','original_mass_kg'):
        bad=measured(p)
        bad[key]*=1.001
        assert module.validate_reset_properties(bad,p)
    bad=measured(p)
    bad['package_inertia_kg_m2'][0][0]*=2
    assert module.validate_reset_properties(bad,p)
    bad=measured(p)
    bad['package_aabb'][1][0]+=.001
    assert module.validate_reset_properties(bad,p)
    bad=measured(p)
    bad['velocity'][3]=.01
    assert module.validate_reset_properties(bad,p)
    bad=measured(p)
    bad['asset_bytes_verified']=False
    assert module.validate_reset_properties(bad,p)


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_reset_properties_reject_nonfinite_measured_mass(value):
    module=importlib.import_module('adaptive_drone_patrol.payload_retry_properties')
    p=Payload(.02700047753751278,.2,.045)
    record=measured(p)
    record['package_mass_kg']=value
    assert module.validate_reset_properties(record,p)
