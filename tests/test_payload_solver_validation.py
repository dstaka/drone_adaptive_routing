"""Independent sequence replay must reject fabricated shared completions."""

import json
import pytest
from test_payload_recovery_mission import CpuFlight

TARGETS = [[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [0.0, 0.0, 1.0]]


def run(tmp_path):
    from adaptive_drone_patrol.payload_solver_validation import record_sequence

    class ContinuousFlight(CpuFlight):
        def set_target(self, target, target_id=None):
            super().set_target(target, target_id)
            self.current["effective_target"] = self.target.tolist()

    flight = ContinuousFlight(cruise=0.01)
    out = tmp_path / "sequence"
    record_sequence(flight, TARGETS, out, {"evidence_kind": "cpu-fixture"})
    return out


def test_sequence_requires_real_motion_and_100_full_intervals(tmp_path):
    from adaptive_drone_patrol.payload_solver_validation import verify_sequence

    proof = verify_sequence(run(tmp_path), TARGETS)
    assert proof["passed"] and proof["completed_targets"] == 3
    assert proof["measured_distance_m"] > 0.9
    assert not proof["physical_acceptance"]


@pytest.mark.parametrize("change", ["stationary", "short", "retarget_reset", "swapped_target"])
def test_stale_or_relabelled_samples_never_establish_sequence_acceptance(tmp_path, change):
    from adaptive_drone_patrol.payload_solver_validation import verify_sequence
    from adaptive_drone_patrol.util import sha256

    out = run(tmp_path)
    rows = [json.loads(x) for x in (out / "samples.jsonl").read_text().splitlines()]
    if change == "short":
        rows = rows[:100]
    elif change == "stationary":
        initial = rows[0]["sample"]
        for row in rows[1:]:
            for key in initial:
                if key.startswith(("drone_", "package_", "attachment_")):
                    row["sample"][key] = initial[key]
    elif change == "retarget_reset":
        row = next(r for r in rows if r.get("kind") == "RETARGET" and r["target_index"] == 1)
        row["sample"]["drone_xyz"][0] += 0.3
    else:
        for row in rows:
            if row.get("target_index") == 1:
                row["target"] = [0.0, 0.0, 1.0]
                row["sample"]["effective_target"] = [0.0, 0.0, 1.0]
    (out / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = json.loads((out / "summary.json").read_text())
    summary["samples_sha256"] = sha256(out / "samples.jsonl")
    (out / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        verify_sequence(out, TARGETS)


@pytest.mark.parametrize("wall", [float("nan"), -1.0])
def test_invalid_wall_clock_never_passes_saved_replay(tmp_path, wall):
    from adaptive_drone_patrol.payload_solver_validation import verify_sequence

    out = run(tmp_path)
    summary = json.loads((out / "summary.json").read_text())
    summary["wall_s"] = wall
    (out / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="limits"):
        verify_sequence(out, TARGETS)


def test_physical_label_without_native_raw_cannot_admit_validation(tmp_path):
    from adaptive_drone_patrol.payload_solver_validation import verify_sequence
    from adaptive_drone_patrol.util import sha256

    out = run(tmp_path)
    declaration = json.loads((out / "declaration.json").read_text())
    declaration["metadata"]["evidence_kind"] = "physical"
    (out / "declaration.json").write_text(json.dumps(declaration))
    summary = json.loads((out / "summary.json").read_text())
    summary["declaration_sha256"] = sha256(out / "declaration.json")
    (out / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="native raw"):
        verify_sequence(out, TARGETS)


def test_real_timeout_prefix_can_be_replayed_without_claiming_completion(tmp_path):
    from adaptive_drone_patrol.payload_solver_validation import record_sequence, verify_sequence

    flight = CpuFlight(cruise=.001, pause=5000)
    out = tmp_path / 'timeout'
    record_sequence(flight, TARGETS, out, dict(evidence_kind='cpu-fixture'))
    proof = verify_sequence(out, TARGETS, allow_incomplete=True)
    assert proof['passed'] is False and proof['completed_targets'] == 0
    assert proof['failure_verified'] and proof['steps'] == 3000
    with pytest.raises(ValueError, match='Incomplete'):
        verify_sequence(out, TARGETS)


def test_incomplete_status_cannot_justify_arbitrary_shortened_evidence(tmp_path):
    from adaptive_drone_patrol.payload_solver_validation import verify_sequence
    from adaptive_drone_patrol.util import sha256

    out = run(tmp_path)
    rows = [json.loads(x) for x in (out / 'samples.jsonl').read_text().splitlines()][:50]
    (out / 'samples.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    summary = json.loads((out / 'summary.json').read_text())
    summary.update(status='FAILED', completed=[], steps=48,
                   samples_sha256=sha256(out / 'samples.jsonl'))
    (out / 'summary.json').write_text(json.dumps(summary))
    (out / 'failure-sample.json').write_text(json.dumps(dict(step=48, error='claimed failure')))
    with pytest.raises(ValueError):
        verify_sequence(out, TARGETS, allow_incomplete=True)


def test_supplied_physical_binding_cannot_be_bypassed_with_cpu_fixture_label(tmp_path):
    from adaptive_drone_patrol.payload_solver_validation import verify_sequence

    with pytest.raises(ValueError, match='physical'):
        verify_sequence(run(tmp_path), TARGETS, native={})
