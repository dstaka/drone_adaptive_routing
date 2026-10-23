"""Complete installed engine identity, including versioned Linux solver binaries."""

import json
from pathlib import Path
import re

from .util import sha256

HISTORICAL_PROVENANCE_SHA256 = "f9daa00f36cdd85f9cccb9fccda15971c00bdbeb6a251e250f0633429a1db519"
SOLVER_LIBRARY = "mujoco/libmujoco.so.3.15.0"


def is_engine_file(path):
    label = str(path)
    return label.endswith((".py", ".h")) or bool(re.search(r"\.so(?:\.\d+)*$", label))


def engine_hashes(distribution):
    result = {
        str(p): sha256(distribution.locate_file(p))
        for p in distribution.files
        if is_engine_file(p) and distribution.locate_file(p).is_file()
    }
    if SOLVER_LIBRARY not in result:
        raise ValueError("Actual versioned solver binary is missing")
    return result


def installed_inventory(distribution):
    return {
        str(p): sha256(distribution.locate_file(p))
        for p in distribution.files
        if not str(p).endswith(".pyc") and distribution.locate_file(p).is_file()
    }


def verify_installed_inventory(expected, distribution):
    if installed_inventory(distribution) != expected:
        raise ValueError("Full pre-capture installed engine inventory differs")
    return engine_hashes(distribution)


def verify_engine_provenance(output):
    """Reconcile the retained pre-capture full inventory without editing old plans.

    The original capture plans omitted versioned shared libraries from their
    projection. A separate all-files snapshot was already captured before these
    experiments. Its exact retained bytes, package RECORD, complete inventory
    and upstream archives close that omission; a new PASS label cannot do so.
    """
    import importlib.metadata as metadata

    root = Path(output)
    p = root / "provenance.json"
    if sha256(p) != HISTORICAL_PROVENANCE_SHA256:
        raise ValueError("Exact independently retained pre-capture inventory required")
    record = json.loads(p.read_text())
    if record["version"] != "3.15.0":
        raise ValueError("Historical actual engine version differs")
    engine = verify_installed_inventory(record["installed_files"], metadata.distribution("mujoco"))
    for label in ["tag.json", "source.tar.gz", "pypi.json"]:
        f = root / label
        if f.stat().st_size != record[label]["bytes"] or sha256(f) != record[label]["sha256"]:
            raise ValueError("Retained upstream engine artifact differs")
    if sha256(root / "LICENSE") != record["license_sha256"]:
        raise ValueError("Retained upstream engine license differs")
    return dict(
        provenance_sha256=sha256(p),
        complete_engine=engine,
        installed_inventory=record["installed_files"],
        package_record_sha256=record["installed_files"]["mujoco-3.15.0.dist-info/RECORD"],
        actual_solver_library_sha256=engine[SOLVER_LIBRARY],
        upstream_commit=json.loads((root / "tag.json").read_text())["object"]["sha"],
        scope="Pre-capture all-files snapshot reconciled exactly with current actual distribution; old plans/raw preserved",
    )
