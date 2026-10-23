"""Independent audit of a real closed learning run and corrupted private copies."""

import gzip
import json
import os
from pathlib import Path
import shutil

import pytest


@pytest.mark.parametrize("change", ["none", "terminal", "native_step", "summary", "seed", "method", "contact_alias", "native_schema"])
def test_actual_closed_pilot_raw_reconciles_and_corruption_is_rejected(tmp_path, change):
    source = os.environ.get("PAYLOAD_SOLVER_CLOSED_PILOT")
    if not source:
        pytest.skip("Actual closed20k solver pilot required")
    from adaptive_drone_patrol.payload_solver_exposure import audit_training_prefix

    root = tmp_path / "pilot"
    shutil.copytree(Path(source), root)
    if change in ["seed", "method"]:
        path = root / "manifest.json"
        value = json.loads(path.read_text())
        if change == "seed":
            value["common"]["seed"] += 1
        else:
            value["method"] = "fixed"
        path.write_text(json.dumps(value))
        # Keep metadata self-consistent; only actual reset regeneration exposes it.
        import torch
        from adaptive_drone_patrol.util import sha256
        checkpoint = root / "checkpoint_s20000.pt"
        state = torch.load(checkpoint,map_location="cpu",weights_only=False)
        state["infos"]["common_training_plan"] = value["common"]
        state["infos"]["method"] = value["method"]
        torch.save(state,checkpoint)
        sidecar = root / "checkpoint_s20000.json"
        info = json.loads(sidecar.read_text())
        info["checkpoint_sha256"] = sha256(checkpoint)
        sidecar.write_text(json.dumps(info))
    elif change == "summary":
        path = root / "checkpoint_s20000.json"
        value = json.loads(path.read_text())
        value["counters"]["stage_valid_transitions"][0] += 100
        path.write_text(json.dumps(value))
    elif change in ["terminal", "native_step", "contact_alias", "native_schema"]:
        if change == "terminal":
            path = root / "environment/terminal.jsonl.gz"
        else:
            path = next(
                p
                for p in (root / "environment").glob("runtime-*/raw.jsonl.gz")
                if len(gzip.open(p, "rb").read()) > 100
            )
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        if change == "terminal":
            rows[0]["valid_step"] += 1
        elif change == "native_step":
            row = next(r for r in rows if r["kind"] == "STEP")
            row["step"] += 1
        else:
            native = next(r for r in rows if r["kind"] == "STEP")
            qpos = native["actual"]["qpos"]
            if change == "native_schema":
                native["actual"].pop("qacc")
            else:
                native["actual"]["contact"] = True
            # Change both descriptions consistently. The previous sparse-field
            # equality checker accepted this omitted getter/false safety alias.
            training = root / "environment/training.jsonl.gz"
            rewritten = training.with_name("corrupt-training.jsonl.gz")
            found = False
            with gzip.open(training,"rt") as source_stream, gzip.open(rewritten,"wt") as target_stream:
                for line in source_stream:
                    value = json.loads(line)
                    if value["kind"] == "VALID_TRANSITION" and value["actual"]["qpos"] == qpos:
                        if change == "native_schema":
                            value["actual"].pop("qacc")
                        else:
                            value["actual"]["contact"] = True
                        found = True
                    target_stream.write(json.dumps(value)+"\n")
            assert found
            rewritten.replace(training)
        with gzip.open(path, "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    if change == "none":
        proof = audit_training_prefix(root, 20000)
        assert proof["valid_transitions"] == 20000
        assert proof["native_action_observation_reconciled"]
        assert proof["stage_max_continuous_steps"] == [12000, 0, 0, 0, 0, 0]
        assert not proof["physical_acceptance"]
        assert not proof["past_stochastic_actor_rng_replayed"]
    else:
        with pytest.raises(ValueError):
            audit_training_prefix(root, 20000)
