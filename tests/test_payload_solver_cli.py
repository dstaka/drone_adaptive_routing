"""Public solver qualification entry rejects unsupported or incomplete stages."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_powered_cli_rejects_missing_passive_raw_before_creating_case(tmp_path):
    output = tmp_path / "powered"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/qualify_payload_solver.py"),
            "--stage",
            "powered",
            "--out",
            str(output),
            "--original",
            str(tmp_path / "missing.urdf"),
            "--passive-probes",
            str(tmp_path / "missing-passive"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not output.exists()
    assert "passive" in result.stderr.lower()
