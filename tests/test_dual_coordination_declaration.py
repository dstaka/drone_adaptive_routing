"""Prospective declarations must never open fresh held-out inputs."""

import json
import pytest


def test_declaration_writes_named_development_cases_and_no_heldout(tmp_path):
    from adaptive_drone_patrol.dual_coordination_declaration import declare_campaign

    result = declare_campaign(
        tmp_path / "declared",
        layout_path="configs/demo_scenarios_payload_elevated/layout.json",
        training_count=8,
    )
    assert result["heldout_generated"] is False
    assert result["production_ready"] is False  # partial test declaration is not a real campaign
    assert result["training_cases"] == 8
    assert result["validation_cases"] == 300
    assert len(result["video_case_ids"]) == 3
    assert not (tmp_path / "declared" / "routing-heldout.jsonl").exists()
    cases = [
        json.loads(line) for line in (tmp_path / "declared" / "validation.jsonl").read_text().splitlines()
    ]
    assert len({c["case_id"] for c in cases}) == 300
    assert {c["split"] for c in cases} == {20}
    with pytest.raises(FileExistsError):
        declare_campaign(
            tmp_path / "declared", layout_path="configs/demo_scenarios_payload_elevated/layout.json"
        )
