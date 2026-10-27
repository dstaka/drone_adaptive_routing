import copy
import pytest
from adaptive_drone_patrol.elevated_binding import development_binding, verify_binding_scenario
from adaptive_drone_patrol.payload_solver_acceptance import verify_case_scenario
from adaptive_drone_patrol.scenario import Scenario
from adaptive_drone_patrol.util import load_config

B='outputs/suspended_mujoco_final_20261025_r3a/inputs/routing/seed_44.zip'


def test_explicit_elevated_binding_accepts_only_exact_scenario_and_checkpoint():
    binding=development_binding(B,'configs/payload_mujoco_elevated_camera.json',provisional=True)
    scenario=Scenario.from_dict(load_config('configs/demo_scenarios_payload_elevated/normal.yaml'))
    assert verify_binding_scenario(scenario,{'scenario':'normal'},binding)
    assert verify_case_scenario(scenario,{'scenario':'normal'},elevated_binding=binding)
    with pytest.raises(ValueError): verify_case_scenario(scenario,{'scenario':'normal'})
    old=Scenario.from_dict(load_config('configs/demo_scenarios_dispersed/normal.yaml'))
    with pytest.raises(ValueError): verify_binding_scenario(old,{'scenario':'normal'},binding)
    bad=copy.deepcopy(binding); bad['routing_sha256']='f'*64
    with pytest.raises(ValueError): verify_binding_scenario(scenario,{'scenario':'normal'},bad)
    scenario.coords[1][2]+=.01
    with pytest.raises(ValueError): verify_binding_scenario(scenario,{'scenario':'normal'},binding)


def test_no_selection_receipt_cannot_admit_required_development():
    with pytest.raises(ValueError): development_binding(B,'configs/payload_mujoco_elevated_camera.json')
