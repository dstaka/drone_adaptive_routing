"""Fine-tuning settings preserve controller contracts and safety bounds."""
import copy


def test_finetune_config_is_new_and_preserves_original_bounds():
    from adaptive_drone_patrol.flight_training import finetune_configs
    original = [{'num_actions':4,'episode_length_s':15.,'visualize_target':True,
                 'visualize_camera':True,'termination_if_roll_greater_than':180,
                 'termination_if_pitch_greater_than':180},
                {'obs_scales':{'rel_pos':1/3}}, {'reward_scales':{'target':10}},
                {'pos_x_range':[-1,1],'pos_y_range':[-1,1],'pos_z_range':[1,1]},
                {'algorithm':{'learning_rate':.0003},'save_interval':100}]
    saved = copy.deepcopy(original)
    result = finetune_configs(original)
    assert original == saved and result[1] == original[1] and result[3] == original[3]
    assert result[0]['termination_if_roll_greater_than'] == 60
    assert result[0]['termination_if_pitch_greater_than'] == 60
    assert result[4]['save_interval'] == 100
