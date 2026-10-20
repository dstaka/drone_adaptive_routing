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


def dispersed_configs(original):
    """Train the wider target domain with unchanged clipped relative-position units."""
    result = finetune_configs(original)
    env, _, _, command, _ = result
    command.update(pos_x_range=[-2.7,2.7], pos_y_range=[-1.8,1.8], pos_z_range=[1.,3.])
    # Upstream termination is target-relative, not a world-coordinate fence.
    # Keep roll/pitch/floor gates; allow the new domain's full diagonal retarget.
    env.update(termination_if_x_greater_than=6., termination_if_y_greater_than=4.,
               termination_if_z_greater_than=3., episode_length_s=30.)
    return result


def dispersed_curriculum_configs(original):
    """Protect the stable parent during gradual expansion, with stronger crash cost."""
    result = dispersed_configs(original)
    result[2]['reward_scales'].update(target=2., crash=-200., hover=.5, velocity=-.03,
                                      angular=-.01, yaw=.1, altitude=-.5, upright=-1.)
    result[4]['algorithm'].update(learning_rate=.00003, schedule='fixed', entropy_coef=0., max_grad_norm=.3)
    return result
