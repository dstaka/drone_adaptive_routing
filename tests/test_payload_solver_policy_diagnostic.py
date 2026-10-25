"""Diagnostic noise settings are explicit and cannot change acceptance."""

import subprocess
import sys
from pathlib import Path


def test_unbounded_noise_probe_rejected_before_engine_and_outputs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/diagnose_payload_policy.py"
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
            "--checkpoint",
            "missing.pt",
            "--checkpoint-sha256",
            "0" * 64,
            "--out",
            str(tmp_path / "out"),
            "--scales",
            "0,3",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "0/0.5/1" in result.stderr
    assert not (tmp_path / "out").exists()
