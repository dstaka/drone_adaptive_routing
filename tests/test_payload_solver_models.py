"""Policy labels and common final training provenance cannot be substituted."""

import copy
import pytest


def test_final_pair_rejects_same_policy_under_two_method_labels():
    from adaptive_drone_patrol.payload_solver_models import verify_pair_metadata

    common = dict(seed=7, N=1, T=100, S=300000, K=3000, pilot=False)
    info = dict(
        method="randomized",
        training_split="final",
        common_training_plan=common,
        valid_environment_transitions=300000,
        initial_actor_tensor_sha256="a" * 64,
        initial_critic_tensor_sha256="b" * 64,
    )
    plan = dict(common=common, initial_actor_tensor_sha256="a" * 64, initial_critic_tensor_sha256="b" * 64)
    with pytest.raises(ValueError, match="method"):
        verify_pair_metadata({"randomized": info, "fixed": copy.deepcopy(info)}, plan)
    fixed = dict(info, method="fixed")
    assert verify_pair_metadata({"randomized": info, "fixed": fixed}, plan)
    fixed["common_training_plan"] = dict(common, S=1000000, K=10000)
    with pytest.raises(ValueError, match="plan"):
        verify_pair_metadata({"randomized": info, "fixed": fixed}, plan)


def test_final_model_metadata_cannot_replace_missing_actual_learning_run(tmp_path):
    from adaptive_drone_patrol.payload_solver_models import audit_final_training

    item = dict(checkpoint=str(tmp_path / 'checkpoint_s300000.pt'), sha256='a' * 64)
    with pytest.raises(ValueError, match='actual final training'):
        audit_final_training('randomized', item, dict(common=dict(S=300000)))
