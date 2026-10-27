import pytest
from adaptive_drone_patrol.elevated_campaign import JointFreeze,formal_cases
from adaptive_drone_patrol.elevated_routing_data import generate_case


def test_formal_inputs_reject_forged_or_absent_freeze_without_drawing():
    with pytest.raises(ValueError):formal_cases(None)
    with pytest.raises(ValueError):JointFreeze({},object())
    with pytest.raises(ValueError):generate_case('held-out',-1,freeze={})


def test_controller_rebinding_compares_values_and_preserves_format_hashes(tmp_path):
    from adaptive_drone_patrol.elevated_campaign import require_same_controller
    original=tmp_path/'original';trace=tmp_path/'trace'
    original.write_text('{"gain":[1,2],"recipe":"C-local-lqr-v1"}')
    trace.write_text('{\n "recipe": "C-local-lqr-v1",\n "gain": [1, 2]\n}')
    require_same_controller(trace,original)
    trace.write_text('{"gain":[1,3],"recipe":"C-local-lqr-v1"}')
    with pytest.raises(ValueError):require_same_controller(trace,original)
