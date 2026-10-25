"""Byte-preserving storage guards are not physical or learning admission."""

import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from adaptive_drone_patrol.util import sha256


@pytest.fixture
def closed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("run")
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(dict(schema="payload-solver-learning-v1", common=dict(S=100)))
    )
    (root / "completed.json").write_text(
        json.dumps(dict(status="TRAINED_NOT_VALIDATED", counters=dict(valid=100, invalid=0)))
    )
    with gzip.open(root / "raw.jsonl.gz", "wb") as stream:
        stream.write(b"actual fixture bytes\n" * 1000)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    job = dict(
        schema="payload-solver-job-v1",
        status="SUCCEEDED",
        exit_code=0,
        pid=child.pid,
        command=["python", "scripts/train_payload_recovery.py", "--out", "run"],
    )
    Path("job.json").write_text(json.dumps(job))
    shutil.copytree(root, "persistent-gold")
    files = {
        str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha256(p))
        for p in root.rglob("*")
        if p.is_file()
    }
    receipt = dict(schema="payload-solver-verified-delivery-v1", root="run", verified=True, files=files)
    return root, job, receipt


def test_only_closed_learning_bytes_matching_external_verified_gold_can_move(closed):
    from adaptive_drone_patrol.payload_solver_storage import closed_inventory

    root, job, receipt = closed
    plan = closed_inventory(root, job, receipt)
    assert plan["files"] == receipt["files"]
    assert plan["physical_acceptance"] is False


@pytest.mark.parametrize("change", ["active", "alive", "missing", "changed", "wrong_root", "not_learning"])
def test_unsafe_storage_handoff_preserves_all_original_files(closed, change):
    from adaptive_drone_patrol.payload_solver_storage import closed_inventory

    root, job, receipt = closed
    if change == "active":
        job["status"] = "RUNNING"
    elif change == "alive":
        job["pid"] = os.getpid()
    elif change == "missing":
        receipt["files"].pop("raw.jsonl.gz")
    elif change == "changed":
        receipt["files"]["raw.jsonl.gz"]["sha256"] = "0" * 64
    elif change == "wrong_root":
        receipt["root"] = "other"
    else:
        (root / "manifest.json").write_text('{"schema":"qualification"}')
    before = {p.name: sha256(p) for p in root.iterdir()}
    with pytest.raises(ValueError):
        closed_inventory(root, job, receipt)
    assert before == {p.name: sha256(p) for p in root.iterdir()}
    assert all(not p.is_symlink() for p in root.iterdir())


@pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Actual Linux sealed descriptors required")
def test_real_sealed_shadow_has_independent_reads_and_restores_identical_regular_file(closed):
    from adaptive_drone_patrol.payload_solver_storage import closed_inventory, hold_closed_files

    root, job, receipt = closed
    plan = closed_inventory(root, job, receipt)
    held = hold_closed_files(root, plan, available_bytes=1000000, reserve_bytes=1000, min_file_bytes=1)
    try:
        path = root / "raw.jsonl.gz"
        assert path.is_symlink() and path.is_file()
        assert sha256(path) == receipt["files"]["raw.jsonl.gz"]["sha256"]
        assert gzip.open(path, "rb").read() == b"actual fixture bytes\n" * 1000
        with path.open("rb") as a, path.open("rb") as b:
            a.read(5)
            assert b.tell() == 0
        with pytest.raises(OSError):
            with path.open("r+b") as stream:
                stream.write(b"x")
    finally:
        held.restore()
    assert all(not p.is_symlink() for p in root.iterdir())
    assert {p.name: sha256(p) for p in root.iterdir()} == {
        name: row["sha256"] for name, row in receipt["files"].items()
    }


@pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Actual Linux sealed descriptors required")
def test_insufficient_memory_cannot_change_any_original_path(closed):
    from adaptive_drone_patrol.payload_solver_storage import closed_inventory, hold_closed_files

    root, job, receipt = closed
    with pytest.raises(ValueError, match="capacity"):
        hold_closed_files(
            root,
            closed_inventory(root, job, receipt),
            available_bytes=1,
            reserve_bytes=1000,
            min_file_bytes=1,
        )
    assert all(not p.is_symlink() for p in root.iterdir())


@pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Actual Linux sealed descriptors required")
def test_interrupt_after_atomic_replace_retains_every_installed_descriptor(closed, monkeypatch):
    from adaptive_drone_patrol.payload_solver_storage import (
        closed_inventory,
        hold_closed_files,
        ShadowInstallError,
    )

    root, job, receipt = closed
    real_replace = os.replace
    interrupted = False

    def replace_then_interrupt(source, target):
        nonlocal interrupted
        real_replace(source, target)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("Injected after successful atomic replacement")

    monkeypatch.setattr(os, "replace", replace_then_interrupt)
    with pytest.raises(ShadowInstallError) as error:
        hold_closed_files(
            root,
            closed_inventory(root, job, receipt),
            available_bytes=1000000,
            reserve_bytes=1000,
            min_file_bytes=1,
        )
    held = error.value.hold
    try:
        assert held.installed
        assert all(sha256(root / name) == row["sha256"] for name, row in receipt["files"].items())
    finally:
        held.restore()
    assert all(not p.is_symlink() for p in root.iterdir())


@pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Actual Linux sealed descriptors required")
@pytest.mark.parametrize("report_failure", ["missing_ready_parent", "broken_stdout"])
def test_reporting_failure_keeps_caretaker_alive_until_external_exact_restore(closed, report_failure):
    import time
    from adaptive_drone_patrol.payload_solver_storage import measured_memory_available

    if measured_memory_available() < 11 * 1024**3:
        pytest.skip("Actual capacity guard requires reserve")
    root, job, receipt = closed
    payload = os.urandom(2 * 1024**2)
    with gzip.open(root / "raw.jsonl.gz", "wb") as stream:
        stream.write(payload)
    shutil.copyfile(root / "raw.jsonl.gz", "persistent-gold/raw.jsonl.gz")
    receipt["files"]["raw.jsonl.gz"] = dict(
        bytes=(root / "raw.jsonl.gz").stat().st_size, sha256=sha256(root / "raw.jsonl.gz")
    )
    Path("delivery.json").write_text(json.dumps(receipt))
    repo = Path(__file__).resolve().parents[1]
    ready = "missing/ready.json" if report_failure == "missing_ready_parent" else "ready.json"
    with Path("caretaker.log").open("wb") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                str(repo / "scripts/hold_payload_solver_raw.py"),
                "--root",
                "run",
                "--job-status",
                "job.json",
                "--delivery",
                "delivery.json",
                "--ready",
                ready,
            ],
            env=dict(os.environ, PYTHONPATH=str(repo / "src")),
            stdout=subprocess.PIPE,
            stderr=log,
        )
        if report_failure == "broken_stdout":
            child.stdout.close()
        try:
            deadline = time.monotonic() + 10
            while not (root / "raw.jsonl.gz").is_symlink() and time.monotonic() < deadline:
                assert child.poll() is None
                time.sleep(0.02)
            time.sleep(0.1)
            assert child.poll() is None
            assert gzip.open(root / "raw.jsonl.gz", "rb").read() == payload
        finally:
            for path in root.iterdir():
                if path.is_symlink():
                    temporary = path.with_name(path.name + ".external-restore")
                    shutil.copyfile(Path("persistent-gold") / path.name, temporary)
                    os.replace(temporary, path)
            child.terminate()
            child.wait(timeout=10)
            if child.stdout and not child.stdout.closed:
                child.stdout.close()
        assert child.returncode == 0
        assert all(not p.is_symlink() for p in root.iterdir())


@pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Actual Linux sealed descriptors required")
def test_restore_read_error_cannot_release_other_live_shadow_descriptors(closed, monkeypatch):
    from adaptive_drone_patrol import payload_solver_storage as storage

    root, job, receipt = closed
    held = storage.hold_closed_files(
        root,
        storage.closed_inventory(root, job, receipt),
        available_bytes=1000000,
        reserve_bytes=1000,
        min_file_bytes=1,
    )
    try:
        name = next(iter(held.files))
        temporary = root / "first-external-restore"
        shutil.copyfile(Path("persistent-gold") / name, temporary)
        os.replace(temporary, root / name)
        with monkeypatch.context() as patch:
            calls = []

            def read_error(path):
                calls.append(path)
                raise OSError("Injected transient restore read failure")

            patch.setattr(storage, "sha256", read_error)
            assert held.release_if_restored() is False
            assert calls == [root / name]
        assert held.descriptors
        assert all(sha256(root / name) == row["sha256"] for name, row in receipt["files"].items())
    finally:
        held.restore()
