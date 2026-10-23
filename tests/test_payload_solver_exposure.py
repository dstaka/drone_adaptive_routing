"""Exposure must describe continuous actual experience, not attempted commands."""

import pytest


def rows(episodes):
    step = 0
    for episode, stages in enumerate(episodes, 1):
        for local, (stage, target, complete) in enumerate(stages, 1):
            step += 1
            yield dict(
                valid_step=step,
                episode=episode,
                episode_step=local,
                stage=stage,
                target=target,
                shared_complete=complete,
                done=local == len(stages),
            )


def test_short_episodes_cannot_be_summed_into_continuous_experience():
    from adaptive_drone_patrol.payload_solver_exposure import exposure_summary

    result = exposure_summary(rows([[(0, [0, 0, 1], False)] * 300] * 2))
    assert result["stage_valid_transitions"][0] == 600
    assert result["stage_max_continuous_steps"][0] == 300


def test_stage_boundary_does_not_give_both_stages_a_whole_episode():
    from adaptive_drone_patrol.payload_solver_exposure import exposure_summary

    result = exposure_summary(rows([[(0, [0, 0, 1], False)] * 300 + [(1, [0, 0, 1], False)] * 300]))
    assert result["stage_max_continuous_steps"][:2] == [300, 300]


def test_repeated_success_is_not_ten_distinct_destinations():
    from adaptive_drone_patrol.payload_solver_exposure import exposure_summary, require_exposure

    result = exposure_summary(rows([[(5, [0, 0, 1], True)] * 10]))
    assert result["full_stage_ten_target_episodes"] == []
    with pytest.raises(ValueError, match="exposure"):
        require_exposure(result, min_transitions=1, min_continuous=1)


def test_full_stage_certificate_belongs_to_one_episode():
    from adaptive_drone_patrol.payload_solver_exposure import exposure_summary

    goals = [[i / 10, 0, 1] for i in range(10)]
    stages = [(5, target, True) for target in goals]
    assert exposure_summary(rows([stages]))["full_stage_ten_target_episodes"] == [1]
    assert exposure_summary(rows([stages[:5], stages[5:]]))["full_stage_ten_target_episodes"] == []


def test_duplicate_or_gapped_valid_steps_cannot_inflate_exposure():
    from adaptive_drone_patrol.payload_solver_exposure import exposure_summary

    records = list(rows([[(0, [0, 0, 1], False)] * 3]))
    records[1]["valid_step"] = 3
    with pytest.raises(ValueError, match="consecutive"):
        exposure_summary(records)


@pytest.mark.parametrize("change", ["raw_action", "history", "rpm", "actual", "clock"])
def test_training_raw_cannot_hide_input_or_native_state_corruption(change):
    import copy
    from adaptive_drone_patrol.payload_solver_exposure import reconcile_training_step

    before = dict(
        drone_xyz=[0, 0, 1],
        package_xyz=[0, 0, 0.6575],
        drone_quaternion_wxyz=[1, 0, 0, 0],
        package_quaternion_wxyz=[1, 0, 0, 0],
        drone_velocity=[0, 0, 0],
        package_velocity=[0, 0, 0],
        drone_angular_velocity=[0, 0, 0],
        package_angular_velocity=[0, 0, 0],
    )
    obs = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0]
    actual = dict(before, time_s=0.01, rpm=[14468.429183500699] * 4)
    native = dict(
        kind="STEP",
        step=1,
        target=[0, 0, 1],
        actions=[0] * 4,
        actor_observation_before_step=obs.copy(),
        actual=copy.deepcopy(actual),
    )
    transition = dict(actual=copy.deepcopy(actual), target=[0, 0, 1])
    action = dict(raw_action=[0] * 4, actor_observation=obs.copy())
    reconcile_training_step(action, transition, native, before, 0.045, [0] * 4)
    if change == "raw_action":
        action["raw_action"][0] = 0.5
    elif change == "history":
        action["actor_observation"][13] = 1
    elif change == "rpm":
        native["actual"]["rpm"][0] += 100
        transition["actual"]["rpm"][0] += 100
    elif change == "clock":
        native["actual"]["time_s"] = transition["actual"]["time_s"] = 0.02
    else:
        transition["actual"]["drone_xyz"][0] += 0.1
    with pytest.raises(ValueError):
        reconcile_training_step(action, transition, native, before, 0.045, [0] * 4)


@pytest.mark.parametrize("change", ["early_reset", "continued_terminal"])
def test_exposure_episode_boundary_requires_an_actual_terminal(change):
    from adaptive_drone_patrol.payload_solver_exposure import exposure_summary

    if change == "early_reset":
        records = list(rows([[(0, [0, 0, 1], False)], [(0, [0, 0, 1], False)]]))
        records[0]["done"] = False
    else:
        records = list(rows([[(0, [0, 0, 1], False)] * 2]))
        records[0]["done"] = True
    with pytest.raises(ValueError, match="terminal"):
        exposure_summary(records)


def test_training_audit_rejects_unplanned_checkpoint_prefix(tmp_path):
    import json
    from adaptive_drone_patrol.payload_solver_exposure import audit_training_prefix

    (tmp_path / "manifest.json").write_text(json.dumps(dict(checkpoints=[300000], common=dict(N=1, T=100))))
    with pytest.raises(ValueError, match="predeclared"):
        audit_training_prefix(tmp_path, 1000000)
