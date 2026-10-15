"""Bounded fine-tuning configuration without changing inference motor dynamics."""
import copy


def finetune_configs(original):
    result = copy.deepcopy(original)
    env, _, rewards, _, train = result
    env.update(episode_length_s=20., visualize_target=False, visualize_camera=False,
               termination_if_roll_greater_than=60., termination_if_pitch_greater_than=60.)
    rewards['reward_scales'].update(hover=.03, velocity=-.001, angular=-.001, crash=-10.)
    train['algorithm'].update(learning_rate=.0001)
    train['save_interval'] = 100
    return result
