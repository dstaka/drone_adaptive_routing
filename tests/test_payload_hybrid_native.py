import os

import numpy as np
import pytest

pytest.importorskip('mujoco')

from adaptive_drone_patrol.payload_hybrid_native import verify_native_kinematics


def test_native_kinematics_reconstructs_velocity_and_rejects_forged_body_pose(tmp_path):
    from adaptive_drone_patrol.payload_solver_mujoco import MuJoCoFixture,FREE_MASS_KG
    from adaptive_drone_patrol.payload_contract import Payload
    original=os.environ.get('PAYLOAD_SOLVER_CF2X')
    if not original:
        pytest.skip('Actual source asset required')
    f=MuJoCoFixture(original,tmp_path/'fixture',Payload(FREE_MASS_KG,.2,.045),substeps=16)
    f.data.qvel[:]=np.linspace(-.01,.01,12)
    f.mj.mj_forward(f.model,f.data)
    actual=f.raw_state()
    assert verify_native_kinematics(f.model,f.data,actual)<1e-10
    actual['package_xyz'][0]+=.001
    with pytest.raises(ValueError,match='kinematic'):
        verify_native_kinematics(f.model,f.data,actual)
