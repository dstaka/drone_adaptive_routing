"""Compiled integration is an explicit unqualified development candidate."""
import pytest


def test_compiled_report_never_turns_missing_intervals_into_safe():
    from adaptive_drone_patrol.dual_coordination_compiled_audit import compiled_report
    audit=dict(control_steps=1,rk_points=64,native_intervals=16,body_intervals=16,
               point_evaluations=81,certified_intervals=15,python_fallback_intervals=1,
               safety_flags=0,minimum_clearance_m=1.,safe=False)
    metrics=dict(floor_minima_m=[.6,.6],max_swing_deg=0.,max_spin_deg_s=0.,
                 max_roll_pitch_deg=0.,constraint_upper=0.,tilt_upper=0.,sweep_lower=1.,
                 self_lower=None,violations=[],reconstruction_failures=[{'status':'uncertain'}])
    report=compiled_report(audit,metrics,self_pairs=[],library_hash='a'*64)
    assert report['safe'] is False and report['reconstructed_path_safe'] is False
    assert report['full_physical_acceptance_qualified'] is False
    assert report['rk_states']==64 and report['native_poses']==17
    assert report['body_reconstructed_intervals']==16
    report=compiled_report(audit,metrics,self_pairs=[(1,2)],library_hash='a'*64)
    assert report['self_collision_pairs']==[[1,2]]


def test_native_compiled_environment_matches_python_and_resumes_exactly(tmp_path):
    import os
    from pathlib import Path
    import numpy as np
    if not os.environ.get('DUAL_NATIVE_AUDIT_LIBRARY'):
        pytest.skip('Explicit development native compiled library required')
    pytest.importorskip('mujoco')
    import json
    from adaptive_drone_patrol.dual_coordination_env import DualCoordinationEnv
    from adaptive_drone_patrol.dual_coordination_cases import generate_case
    from adaptive_drone_patrol.dual_coordination_schema import encode_action,WAIT
    original=Path(os.environ['DUAL_CF2X'])
    design=json.loads(Path(os.environ['DUAL_CONTROLLER']).read_text())
    case=generate_case(15,0,'normal')
    case['starts']=[[-3.5,-3.5,1.],[3.5,3.5,1.]]
    case['initials']=[dict(position=p,quaternion_wxyz=[1.,0.,0.,0.],swing_deg=0.,direction_rad=0.) for p in case['starts']]
    case['goals'][0]=[.5,-3.5,1.]
    common=dict(case=case,design=design,replay_stride_ticks=20)
    python=DualCoordinationEnv(original,tmp_path/'python',**common)
    compiled=DualCoordinationEnv(original,tmp_path/'compiled',native_audit_library=os.environ['DUAL_NATIVE_AUDIT_LIBRARY'],**common)
    resumed=None
    try:
        action=encode_action(0,WAIT,1)
        for _ in range(3):
            left=python.step(action)
            right=compiled.step(action)
            assert not left[2] and not right[2],(left[3],right[3])
            np.testing.assert_allclose(python.fixture.data.qpos,compiled.fixture.data.qpos,rtol=0,atol=1e-12)
            np.testing.assert_allclose(python.fixture.data.qvel,compiled.fixture.data.qvel,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(python.fixture.data.ctrl,compiled.fixture.data.ctrl)
            np.testing.assert_array_equal(python.fixture.rpm,compiled.fixture.rpm)
        snapshot=compiled.to_dict()
        resumed=DualCoordinationEnv(original,tmp_path/'resumed',native_audit_library=os.environ['DUAL_NATIVE_AUDIT_LIBRARY'],**common)
        resumed.restore(snapshot)
        for env in (compiled,resumed):
            assert not env.step(action)[2]
        np.testing.assert_array_equal(compiled.fixture.data.qpos,resumed.fixture.data.qpos)
        np.testing.assert_array_equal(compiled.fixture.data.qvel,resumed.fixture.data.qvel)
        report=compiled.online_auditor.report()
        assert report['control_steps']==80 and report['rk_states']==5120
        assert report['reconstructed_intervals']==1280 and report['safe']
    finally:
        for env in (python,compiled,resumed):
            if env is not None:
                env.close()
