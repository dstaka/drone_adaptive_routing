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


def test_dispersed_training_changes_domain_but_keeps_observation_semantics():
    from adaptive_drone_patrol import flight_training
    original = [{'num_actions':4}, {'obs_scales':{'rel_pos':1/3}},
                {'reward_scales':{'target':10}},
                {'pos_x_range':[-1,1],'pos_y_range':[-1,1],'pos_z_range':[1,1]},
                {'algorithm':{'learning_rate':.0001}}]
    saved = copy.deepcopy(original)
    result = flight_training.dispersed_configs(original)
    assert original == saved
    assert result[1] == original[1]
    assert result[3]['pos_x_range'] == [-2.7,2.7]
    assert result[3]['pos_y_range'] == [-1.8,1.8]
    assert result[3]['pos_z_range'] == [1.,3.]
    assert result[0]['termination_if_x_greater_than'] >= 5.4
    assert result[0]['termination_if_y_greater_than'] >= 3.6
    assert result[0]['termination_if_roll_greater_than'] == 60


def test_safe_curriculum_preserves_safety_and_does_not_restore_failed_model():
    from adaptive_drone_patrol import flight_training
    original=[{}, {'obs_scales':{'rel_pos':1/3}}, {'reward_scales':{'target':10}},
              {}, {'algorithm':{'learning_rate':.0001,'schedule':'adaptive','entropy_coef':.004}}]
    configure=getattr(flight_training,'dispersed_curriculum_configs',None)
    assert configure is not None
    result=configure(original)
    assert result[1]==original[1]
    assert result[2]['reward_scales']['crash']==-200
    assert result[4]['algorithm']['schedule']=='fixed'
    assert result[4]['algorithm']['learning_rate']==.00003
    assert result[0]['termination_if_roll_greater_than']==60
    assert result[3]['pos_z_range']==[1.,3.]
