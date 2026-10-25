"""Actual versioned solver libraries must be bound alongside extension modules."""

from pathlib import Path
from types import SimpleNamespace
import pytest


def test_engine_inventory_includes_actual_versioned_solver_binary(tmp_path):
    from adaptive_drone_patrol.payload_solver_engine import engine_hashes

    names = [
        "mujoco/libmujoco.so.3.15.0",
        "mujoco/_functions.cpython-312.so",
        "mujoco/include/mujoco.h",
        "mujoco/__init__.py",
    ]
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode())
    dist = SimpleNamespace(files=list(map(Path, names)), locate_file=lambda p: tmp_path / p)
    assert set(engine_hashes(dist)) == set(names)
    (tmp_path / names[0]).unlink()
    with pytest.raises(ValueError, match="solver binary"):
        engine_hashes(dist)


def test_changed_library_cannot_match_pre_capture_inventory(tmp_path):
    from adaptive_drone_patrol.payload_solver_engine import installed_inventory, verify_installed_inventory

    p = tmp_path / "mujoco/libmujoco.so.3.15.0"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"captured actual solver")
    dist = SimpleNamespace(files=[Path("mujoco/libmujoco.so.3.15.0")], locate_file=lambda p: tmp_path / p)
    captured = installed_inventory(dist)
    verify_installed_inventory(captured, dist)
    p.write_bytes(b"different solver")
    with pytest.raises(ValueError, match="inventory differs"):
        verify_installed_inventory(captured, dist)
