"""Batch diagnostics cannot masquerade as qualified PPO batches."""

import subprocess
import sys
from pathlib import Path


def test_unsupported_batch_is_rejected_before_physics_or_outputs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_payload_solver.py"
    out = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--qualification",
            "missing.json",
            "--locations",
            "missing.json",
            "--original",
            "missing.urdf",
            "--sizes",
            "2",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "N1/8/32/128" in result.stderr
    assert not out.exists()
