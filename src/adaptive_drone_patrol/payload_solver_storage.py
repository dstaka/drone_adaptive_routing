"""Keep closed, externally backed-up learning bytes readable in sealed RAM.

The caller must actually verify delivery on the persistent destination first.
This module matches current bytes to that receipt; it cannot prove another
host's persistence and never grants physical, optimizer or acceptance admission.
Qualification directories and active gzip writers cannot use this mechanism.
"""

import json
import os
from pathlib import Path
import shutil
import uuid

from .util import sha256


def closed_inventory(root, job, delivery):
    """Require reaped successful learning and exact externally delivered bytes."""
    root = Path(root)
    if root.is_absolute() or ".." in root.parts:
        raise ValueError("Portable owned learning root required")
    if (
        job.get("schema") != "payload-solver-job-v1"
        or job.get("status") != "SUCCEEDED"
        or job.get("exit_code") != 0
    ):
        raise ValueError("Closed successful writer required")
    args = job.get("command", [])
    if "--out" not in args or args[args.index("--out") + 1] != str(root):
        raise ValueError("Closed writer output differs")
    pid = job.get("pid")
    if type(pid) is not int or pid <= 1:
        raise ValueError("Original writer process identity required")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise ValueError("Cannot establish original writer exit") from exc
    else:
        raise ValueError("Original writer is still alive or its PID was reused")
    manifest = json.loads((root / "manifest.json").read_text())
    completed = json.loads((root / "completed.json").read_text())
    if (
        manifest.get("schema") != "payload-solver-learning-v1"
        or completed.get("status") != "TRAINED_NOT_VALIDATED"
        or completed["counters"]["valid"] != manifest["common"]["S"]
        or completed["counters"]["invalid"] != 0
    ):
        raise ValueError("Only closed complete learning files can be shadowed")
    if (
        delivery.get("schema") != "payload-solver-verified-delivery-v1"
        or delivery.get("root") != str(root)
        or delivery.get("verified") is not True
    ):
        raise ValueError("Exact previously verified external delivery required")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Original closed files must be regular before shadowing")
        if path.is_file():
            files[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    if not files or files != delivery.get("files"):
        raise ValueError("Current closed bytes differ from verified persistent gold")
    return dict(
        root=str(root),
        files=files,
        physical_acceptance=False,
        persistence="External verified gold required; RAM is only a readable temporary copy",
    )


class ShadowHold:
    """Own sealed descriptors until every original path is restored or delivered."""

    def __init__(self, root, files):
        self.root, self.files = Path(root), files
        self.descriptors, self.installed = {}, []

    def close_uninstalled(self):
        if self.installed:
            raise ValueError("Restore original regular files before releasing their descriptors")
        for fd in self.descriptors.values():
            os.close(fd)
        self.descriptors.clear()

    def release_if_restored(self):
        """Release only after an external restore left every exact regular file."""
        try:
            restored = all(
                not (self.root / name).is_symlink()
                and (self.root / name).is_file()
                and sha256(self.root / name) == item["sha256"]
                for name, item in self.files.items()
            )
        except (OSError, ValueError):
            return False
        if not restored:
            return False
        self.installed.clear()
        self.close_uninstalled()
        return True

    def restore(self):
        """Restore identical regular bytes, checking each before atomic replacement."""
        for name in list(self.installed):
            path = self.root / name
            temporary = path.with_name(path.name + ".restore-" + uuid.uuid4().hex)
            with path.open("rb") as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if sha256(temporary) != self.files[name]["sha256"]:
                raise ValueError("Restored persistent bytes differ; retain all live descriptors")
            os.replace(temporary, path)
            self.installed.remove(name)
        self.close_uninstalled()

    def proof(self):
        if any(sha256(self.root / name) != item["sha256"] for name, item in self.files.items()):
            raise ValueError("Readable closed bytes changed during storage handoff")
        return dict(
            root=str(self.root),
            pid=os.getpid(),
            files=self.files,
            sealed_read_only_files=list(self.installed),
            physical_acceptance=False,
            status="RAM_SHADOWS_REQUIRE_EXTERNAL_PERSISTENT_GOLD",
        )


class ShadowInstallError(RuntimeError):
    """Retain ownership on partial install so a caretaker can keep bytes alive."""

    def __init__(self, hold):
        super().__init__("Partial storage install: retain descriptors and restore from verified gold")
        self.hold = hold


def hold_closed_files(
    root, plan, *, available_bytes, reserve_bytes=10 * 1024**3, min_file_bytes=1024**2, owner=None
):
    """Seal exact copies, then atomically substitute unchanged leaf file paths."""
    import fcntl

    if not hasattr(os, "memfd_create"):
        raise ValueError("Actual Linux sealed descriptors required")
    root = Path(root)
    if str(root) != plan["root"]:
        raise ValueError("Owned closed root differs")
    selected = {n: v for n, v in plan["files"].items() if v["bytes"] >= min_file_bytes}
    if available_bytes < 2 * sum(v["bytes"] for v in selected.values()) + reserve_bytes:
        raise ValueError("Insufficient measured memory capacity for copies and reserve")
    held = owner if owner is not None else ShadowHold(root, plan["files"])
    if held.root != root or held.files != plan["files"] or held.installed or held.descriptors:
        raise ValueError("Fresh matching descriptor owner required")
    temporary_links = []
    try:
        for index, (name, item) in enumerate(selected.items()):
            path = root / name
            if path.is_symlink() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError("Closed original changed before sealing")
            fd = os.memfd_create("payload-closed-" + str(index), os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
            try:
                with path.open("rb") as source, os.fdopen(os.dup(fd), "wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                fcntl.fcntl(
                    fd,
                    fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE,
                )
                os.fchmod(fd, 0o400)
                read_only = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
            finally:
                os.close(fd)
            held.descriptors[name] = read_only
            if sha256(f"/proc/self/fd/{read_only}") != item["sha256"]:
                raise ValueError("Sealed copy hash differs")
        # Rehash the complete closed set before any original inode is replaced.
        held.proof()
        for name, fd in held.descriptors.items():
            path = root / name
            temporary = path.with_name(path.name + ".shadow-" + uuid.uuid4().hex)
            temporary.symlink_to(f"/proc/{os.getpid()}/fd/{fd}")
            temporary_links.append((temporary, path, name))
        for temporary, path, name in temporary_links:
            # Conservative ownership must precede the atomic operation: even
            # an interrupt immediately after replacement cannot lose this FD.
            held.installed.append(name)
            os.replace(temporary, path)
        held.proof()
        return held
    except BaseException as exc:
        for temporary, _, _ in temporary_links:
            if temporary.is_symlink():
                temporary.unlink()
        if held.installed:
            raise ShadowInstallError(held) from exc
        held.close_uninstalled()
        raise


def measured_memory_available():
    """Read the effective container limit, never the host's larger free memory."""
    for base, limit_name, usage_name in [
        (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes"),
        (Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
    ]:
        if (base / limit_name).is_file() and (base / usage_name).is_file():
            limit_text = (base / limit_name).read_text().strip()
            if limit_text == "max":
                continue
            limit, usage = int(limit_text), int((base / usage_name).read_text())
            if 0 < limit < 2**60 and 0 <= usage < limit:
                return limit - usage
    raise ValueError("Finite effective cgroup memory capacity required")
