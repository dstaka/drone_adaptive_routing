"""CPU native-ball reset kinematics; no mocked dynamics are certified."""
import importlib
import math
from pathlib import Path

import numpy as np
import pytest


def api():
    path=Path(__file__).resolve().parents[1] / 'src/adaptive_drone_patrol/payload_ball_bank.py'
    assert path.is_file(), 'shared native-ball backend must exist'
    return importlib.import_module('adaptive_drone_patrol.payload_ball_bank')


def apply(q, v):
    # Quaternion-vector algebra, independent of the implementation's matrices.
    w, vector=q[0], np.asarray(q[1:])
    v=np.asarray(v)
    return v+2*np.cross(vector, np.cross(vector, v)+w*v)


def multiply(a, b):
    return np.r_[a[0]*b[0]-np.dot(a[1:], b[1:]),
                 a[0]*np.asarray(b[1:])+b[0]*np.asarray(a[1:])+np.cross(a[1:], b[1:])]


def axis_angle(axis, degrees):
    angle=math.radians(degrees)/2
    axis=np.asarray(axis, dtype=float)
    return np.r_[math.cos(angle), axis/np.linalg.norm(axis)*math.sin(angle)]


def initial(direction, drone_q=(1.,0,0,0), package_q=(1.,0,0,0), side=.045):
    drone=np.array([.1,-.2,1.3])
    anchor=drone+apply(drone_q, [0,0,-.02])
    package=anchor+.3*np.asarray(direction)-apply(package_q,[0,0,side/2])
    return dict(drone_xyz=drone.tolist(),drone_quaternion_wxyz=list(drone_q),package_xyz=package.tolist(),
                package_quaternion_wxyz=list(package_q),drone_velocity=[0,0,0],drone_angular_velocity=[0,0,0],
                package_velocity=[0,0,0],package_angular_velocity=[0,0,0])


@pytest.mark.parametrize('degrees', [0, 5, 89, 90, 91, 170, 179.999999, 180])
@pytest.mark.parametrize('side', [.03,.045,.06])
def test_native_ball_forward_kinematics_reconstructs_attachment_and_package(degrees, side):
    angle=math.radians(degrees)
    direction=[math.sin(angle),0,-math.cos(angle)]
    drone_q=axis_angle([1,2,3],37)
    package_q=axis_angle([3,-2,1],123)
    state=initial(direction,drone_q,package_q,side)
    upper,lower=api().ball_joint_configuration(state,side)
    rod_q=multiply(drone_q,upper)
    actual_package_q=multiply(rod_q,lower)
    assert np.linalg.norm(upper) == pytest.approx(1,abs=1e-12)
    assert np.linalg.norm(lower) == pytest.approx(1,abs=1e-12)
    assert apply(rod_q,[0,0,-1]) == pytest.approx(direction,abs=1e-10)
    for basis in np.eye(3):
        assert apply(actual_package_q,basis) == pytest.approx(apply(package_q,basis),abs=1e-10)
    anchor=np.asarray(state['drone_xyz'])+apply(drone_q,[0,0,-.02])
    reconstructed=anchor+apply(rod_q,[0,0,-.3])-apply(actual_package_q,[0,0,side/2])
    assert reconstructed == pytest.approx(state['package_xyz'],abs=1e-10)


def test_identity_initial_state_gives_identity_native_ball_coordinates():
    upper,lower=api().ball_joint_configuration(initial([0,0,-1]),.045)
    assert upper == pytest.approx([1,0,0,0],abs=1e-12)
    assert lower == pytest.approx([1,0,0,0],abs=1e-12)


@pytest.mark.parametrize('mode', ['length','nonfinite','quaternion','linear_velocity','angular_velocity','side'])
def test_reset_rejects_invalid_geometry_or_nonzero_initial_motion(mode):
    state=initial([0,0,-1])
    side=.045
    if mode == 'length':
        state['package_xyz'][2]+=.001
    elif mode == 'nonfinite':
        state['package_xyz'][0]=float('nan')
    elif mode == 'quaternion':
        state['drone_quaternion_wxyz']=[2,0,0,0]
    elif mode == 'linear_velocity':
        state['package_velocity']=[.1,0,0]
    elif mode == 'angular_velocity':
        state['drone_angular_velocity']=[0,.1,0]
    else:
        side=.04
    with pytest.raises(ValueError):
        api().ball_joint_configuration(state,side)


@pytest.mark.parametrize(('dt','substeps','fixture'),[(0,2,'loaded'),(.01,0,'loaded'),(.01,1.5,'loaded'),
                                                     (float('nan'),2,'loaded'),(.01,2,'decorative')])
def test_builder_rejects_bad_physics_before_importing_simulator(tmp_path,dt,substeps,fixture):
    from adaptive_drone_patrol.payload_contract import Payload
    with pytest.raises(ValueError):
        api().build_ball_fixture(tmp_path,Payload(.02700047753751278,.2,.045),
                                 fixture=fixture,dt=dt,substeps=substeps)


def test_backend_import_and_pure_kinematics_do_not_import_genesis():
    import subprocess
    import sys
    import os
    root=Path(__file__).resolve().parents[1]
    code="import sys; import adaptive_drone_patrol.payload_ball_bank; assert 'genesis' not in sys.modules"
    completed=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(root/'src')),
                             capture_output=True,text=True)
    assert completed.returncode == 0,completed.stderr


def test_all_inherited_initial_cases_keep_exact_centers_under_native_ball_fk():
    from adaptive_drone_patrol.payload_cases import build_cases
    cases=build_cases()
    assert len(cases) == 127
    for case in cases:
        state,side=case['initial'],case['side_m']
        upper,lower=api().ball_joint_configuration(state,side)
        rod=multiply(state['drone_quaternion_wxyz'],upper)
        cube=multiply(rod,lower)
        anchor=np.asarray(state['drone_xyz'])+apply(state['drone_quaternion_wxyz'],[0,0,-.02])
        center=anchor+apply(rod,[0,0,-.3])-apply(cube,[0,0,side/2])
        assert center == pytest.approx(state['package_xyz'],abs=1e-10)


def test_unselected_bank_never_imports_simulator_for_sample_or_step():
    bank=api().BallSceneBank.__new__(api().BallSceneBank)
    bank.active=None
    with pytest.raises(ValueError,match='Select'):
        bank.sample()
    with pytest.raises(ValueError,match='Select'):
        bank.step([0,0,0,0])
    with pytest.raises(ValueError,match='finite'):
        bank.step([float('nan'),0,0,0])


@pytest.mark.parametrize('align_root', [1, 'true', None])
def test_alignment_switch_is_explicit_boolean_before_gpu_construction(tmp_path,align_root):
    from adaptive_drone_patrol.payload_contract import Payload
    with pytest.raises(ValueError):
        api().build_ball_fixture(tmp_path,Payload(.02700047753751278,.2,.045),align_root=align_root)
    with pytest.raises(ValueError):
        api().BallSceneBank(tmp_path / 'bank',align_root=align_root)
