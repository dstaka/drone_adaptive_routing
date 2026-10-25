import pytest

from adaptive_drone_patrol.payload_hybrid_media import require_development_capture


def test_early_capture_requires_complete_independent_proof_and_is_not_final_acceptance():
    summary=dict(status='SUCCEEDED',completed_count=10,steps=15000,sim_time_s=150.,wall_residence_s=300.)
    proof=dict(success=True,physical_acceptance=True,completed_count=10,steps=15000)
    assert require_development_capture(summary,proof)['adoption_status']=='DEVELOPMENT_NOT_FINAL'
    for changed in (dict(proof,success=False),dict(proof,physical_acceptance=False),dict(proof,steps=14999)):
        with pytest.raises(ValueError):
            require_development_capture(summary,changed)
    with pytest.raises(ValueError):
        require_development_capture(dict(summary,wall_residence_s=600.),proof)


def test_software_recomposition_requires_every_raw_and_caption_pixel_exact():
    import numpy as np
    from adaptive_drone_patrol.payload_hybrid_media import compare_recomposition
    raw=np.zeros((8,8,3),np.uint8)
    hud=np.zeros((12,8,3),np.uint8)
    rerender=raw.copy()
    rerender[1,1,0]=1
    expected=np.concatenate([rerender,hud[8:]])
    assert compare_recomposition(raw,hud,raw,hud,raw_height=8)['raster_changed_pixels']==0
    with pytest.raises(ValueError):
        compare_recomposition(raw,hud,rerender,expected,raw_height=8)
    expected[-1,0,0]=1
    with pytest.raises(ValueError):
        compare_recomposition(raw,hud,raw,expected,raw_height=8)


def test_cached_hud_exactly_matches_full_recomposition_across_native_times():
    import copy
    from pathlib import Path
    import numpy as np
    import matplotlib
    from adaptive_drone_patrol.payload_hybrid_media import compose_development_hud,compose_development_hud_cached
    from adaptive_drone_patrol.payload_solver_media import load_camera
    import os
    font=Path(os.environ.get('HYBRID_FONT','outputs/assets/NotoSansJP.ttf'))
    if not font.is_file():
        font=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    config=load_camera()
    state=dict(scenario_id='normal',selected='A',visited=[],priorities=[81,42,43,64,23,84,6,98,22,56],
               event_text=None,target=[0,0,1])
    native=dict(time_s=0.,drone_xyz=[0,0,1],package_xyz=[0,0,.6575],
        drone_quaternion_wxyz=[1,0,0,0],package_quaternion_wxyz=[1,0,0,0],
        drone_velocity=[0,0,0],package_velocity=[0,0,0],
        drone_angular_velocity=[0,0,0],package_angular_velocity=[0,0,0])
    raw=np.zeros((720,1792,3),np.uint8)
    for i in range(3):
        s=copy.deepcopy(state)
        native['time_s']=i*.01
        native['package_angular_velocity']=[.1*i,0,0]
        raw[:]=i*20
        if i==2:
            s.update(visited=['A'],selected='B',event_text='test')
        args=(raw,s,native,dict(side_m=.045,mass_ratio=.2),config,font,'hash')
        assert np.array_equal(compose_development_hud(*args),compose_development_hud_cached(*args))
