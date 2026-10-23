"""Common sample selection cannot favor one method or omit earlier rungs."""

import copy
from pathlib import Path
import pytest


def exposure():
    return dict(
        stage_valid_transitions=[10000] * 6,
        stage_max_continuous_steps=[500] * 6,
        full_stage_ten_target_episodes=[20],
        destinations_bound=True,
    )


def candidate(s, successes):
    return dict(
        S=s,
        conditions={"randomized": 6, "fixed": 6},
        successes={"randomized": successes, "fixed": 0},
        exposure={"randomized": exposure(), "fixed": exposure()},
    )


def test_smallest_common_rung_requires_randomized_six_real_sequences():
    from adaptive_drone_patrol.payload_solver_training_plan import select_common_rung

    assert select_common_rung([candidate(300000, 5), candidate(1000000, 6)]) == 1000000


def test_both_methods_need_actual_exposure_even_when_only_randomized_has_success_gate():
    from adaptive_drone_patrol.payload_solver_training_plan import select_common_rung

    row = candidate(300000, 6)
    row["exposure"]["fixed"]["stage_max_continuous_steps"][5] = 499
    with pytest.raises(ValueError, match="qualifying"):
        select_common_rung([row])


@pytest.mark.parametrize("change", ["missing_prior", "fixed_denominator", "duplicate"])
def test_missing_or_duplicate_evidence_cannot_choose_a_rung(change):
    from adaptive_drone_patrol.payload_solver_training_plan import select_common_rung

    rows = [candidate(300000, 5), candidate(1000000, 6)]
    if change == "missing_prior":
        rows = rows[1:]
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    else:
        rows[0]["conditions"]["fixed"] = 5
    with pytest.raises(ValueError):
        select_common_rung(rows)


def test_status_only_final_plan_cannot_authorize_final_learning():
    from adaptive_drone_patrol.payload_solver_training_plan import require_final_plan

    with pytest.raises(ValueError, match='frozen'):
        require_final_plan(dict(schema='payload-solver-final-training-plan-v1',
            status='FINAL_PLAN_FROZEN_BEFORE_BOTH_RUNS', heldout_opened=False))


def test_fabricated_success_and_exposure_with_unrelated_file_cannot_admit_final(tmp_path, monkeypatch):
    from adaptive_drone_patrol.payload_solver_training_plan import require_final_plan
    from adaptive_drone_patrol.util import sha256

    monkeypatch.chdir(tmp_path)
    Path('unrelated.txt').write_text('This is not actual development evidence.\n')
    forged = dict(schema='payload-solver-final-training-plan-v1',
        status='FINAL_PLAN_FROZEN_BEFORE_BOTH_RUNS', heldout_opened=False,
        common=dict(S=300000, N=1, T=100, K=3000, seed=7, pilot=False,
                    development_evaluation=dict(ladder=[300000,1000000,3000000])),
        artifact_sha256={'unrelated.txt':sha256('unrelated.txt')},
        development_candidates=[candidate(300000,6)])
    with pytest.raises(ValueError, match='replay'):
        require_final_plan(forged)
