"""Build and measure an UNQUALIFIED compiled native audit shadow.

The native context owns copies of the model (including runtime damping) and
data. RK diagnostics use scratch forward dynamics and also preserve the live
integration acceleration finite check. Detached stepping never writes the
source model/data/forces. Only explicit copy_to_fixture publishes full data
after matching the complete model. Boundary poses support the independent
Python sweep audit. Qualification remains the integrating caller's obligation.
The process must serialize MuJoCo callback users; existing callbacks are refused.
"""

from __future__ import annotations

import ctypes
import copy
import hashlib
import json
import platform
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .dual_coordination_progress import atomic_json
from .util import sha256

QUALIFICATION = "UNQUALIFIED_SHADOW_ONLY"
_METRIC_WIDTH = 23
_NATIVE_KINDS = {2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}


def _record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def _pip_dependencies():
    import mujoco

    if mujoco.__version__ != "3.15.0":
        raise ValueError("MuJoCo3.15.0 required")
    root = Path(mujoco.__file__).resolve().parent
    libraries = list(root.glob("libmujoco.so.3.15.0")) + list(root.glob("libmujoco.3.15.0.dylib"))
    if len(libraries) != 1:
        raise ValueError("Explicit MuJoCo header/library paths required for this installation")
    return root / "include" / "mujoco", libraries[0]


def build_native_audit(output, *, headers=None, library=None, compiler=None):
    """Compile only this candidate and retain source/header/library/tool hashes.

    headers is the directory containing mujoco.h; library is the exact3.15
    shared library. No dependency download or package installation is performed.
    """
    if (headers is None) != (library is None):
        raise ValueError("Supply both headers and library, or use the installed Python package")
    if headers is None:
        headers, library = _pip_dependencies()
    headers, library = Path(headers).resolve(), Path(library).resolve()
    source = Path(__file__).resolve().parents[2] / "native" / "dual_native_audit.cpp"
    if not (headers / "mujoco.h").is_file() or not library.is_file():
        raise ValueError("Existing MuJoCo SDK headers and shared library required")
    executable = shutil.which(compiler or "c++")
    if executable is None:
        raise ValueError("C++ compiler unavailable")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / ("dual_native_audit.dylib" if sys.platform == "darwin" else "dual_native_audit.so")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("Use a fresh build output directory to retain attempts")
    source_origin = _record(source)
    shutil.copyfile(source, root / "dual_native_audit.cpp")
    source = root / "dual_native_audit.cpp"
    include = root / "include"
    include.mkdir()
    (include / "mujoco").symlink_to(headers, target_is_directory=True)
    rpaths = {library.parent}
    for parent in library.parents:
        if parent.suffix == ".framework":
            rpaths.add(parent.parent)
    argv = [
        executable,
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I",
        str(include),
        "-MD",
        "-MF",
        str(root / "dependencies.d"),
        str(source),
        str(library),
        *[f"-Wl,-rpath,{p}" for p in sorted(rpaths)],
        "-o",
        str(destination),
    ]
    receipt = dict(
        schema="dual-native-audit-build-v1",
        qualification=QUALIFICATION,
        mujoco_version="3.15.0",
        source=_record(source),
        source_origin=source_origin,
        headers=[_record(p) for p in sorted(headers.rglob("*.h"))],
        library=_record(library),
        compiler={
            **_record(executable),
            "version": subprocess.check_output([executable, "--version"], text=True),
        },
        argv=argv,
        platform=platform.platform(),
        python=sys.version,
        numpy=np.__version__,
    )
    start = time.perf_counter()
    process = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (root / "compile.log").write_text(process.stdout)
    receipt.update(
        elapsed_wall_seconds=time.perf_counter() - start,
        returncode=process.returncode,
        compile_log=_record(root / "compile.log"),
    )
    if process.returncode == 0:
        receipt["artifact"] = _record(destination)
        dependency_text = (root / "dependencies.d").read_text().replace("\\\n", "")
        dependencies = shlex.split(dependency_text.split(":", 1)[1])
        receipt["dependencies"] = [_record(p) for p in sorted(set(dependencies))]
    atomic_json(destination.with_suffix(".json"), receipt)
    if process.returncode:
        raise RuntimeError(f"Native candidate compile failed; see {root / 'compile.log'}")
    return destination


def _array(value, shape):
    value = np.ascontiguousarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Finite native array shape{shape} required")
    return value


def _ptr(value):
    return value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class CompiledAuditShadow:
    """Independent native clone; always unqualified and never a production gate."""

    def __init__(self, library):
        self.library = Path(library).resolve()
        self.receipt = json.loads(self.library.with_suffix(".json").read_text())
        if self.receipt.get("qualification") != QUALIFICATION or self.receipt.get("returncode") != 0:
            raise ValueError("Successful shadow-only build receipt required")
        for record in [
            self.receipt["source"],
            self.receipt["library"],
            self.receipt["artifact"],
            *self.receipt["headers"],
        ]:
            if _record(record["path"]) != record:
                raise ValueError("Frozen native build input/artifact changed")
        if Path(self.receipt["artifact"]["path"]) != self.library:
            raise ValueError("Exact compiled artifact required")
        # Keep the GIL while native global callbacks are temporarily installed.
        self.lib = ctypes.PyDLL(str(self.library))
        lib = self.lib
        pointer, doubles = ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)
        signatures = dict(
            da_version=([], ctypes.c_int),
            da_error=([], ctypes.c_char_p),
            da_create=([pointer, pointer], pointer),
            da_create_xml=([ctypes.c_char_p], pointer),
            da_destroy=([pointer], None),
            da_ngeom=([pointer], ctypes.c_int),
            da_state_size=([pointer], ctypes.c_int),
            da_model_size=([pointer], ctypes.c_int),
            da_model_bytes=([pointer, ctypes.POINTER(ctypes.c_ubyte)], ctypes.c_int),
            da_get_state=([pointer, doubles, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            da_set_state=([pointer, doubles, ctypes.POINTER(ctypes.c_int), doubles], ctypes.c_int),
            da_copy_to=([pointer, pointer, pointer], ctypes.c_int),
            da_self_count=([pointer], ctypes.c_int),
            da_self_pairs=([pointer, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            da_package_offsets=([pointer, doubles], ctypes.c_int),
            da_body_geometry=([pointer, doubles], ctypes.c_int),
            da_distance=([pointer, doubles, doubles, ctypes.c_int, ctypes.c_int, doubles], ctypes.c_int),
            da_fast_interval=([pointer, doubles, doubles, doubles, doubles, doubles], ctypes.c_int),
            da_shapes=([pointer, doubles], ctypes.c_int),
            da_inspect=([pointer, doubles, doubles], ctypes.c_int),
            da_reset=([pointer, doubles, doubles, ctypes.c_double], ctypes.c_int),
            da_step=([pointer, doubles, doubles, doubles, doubles], ctypes.c_int),
        )
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result
        if lib.da_version() != 3015000:
            raise ValueError("Native library is not MuJoCo3.15.0")
        self.handle = None

    @classmethod
    def from_fixture(cls, fixture, library):
        if fixture.mj.__version__ != "3.15.0":
            raise ValueError("MuJoCo3.15.0 fixture required")
        obj = cls(library)
        # _address is MuJoCo's native object pointer; both ABI and version above
        # are fixed. C++ copies immediately, retaining no Python-owned pointers.
        obj.handle = obj.lib.da_create(fixture.model._address, fixture.data._address)
        obj._initialize()
        obj._last_rpm = np.array(fixture.rpm, dtype=float, copy=True)
        if [
            [i for i, s in enumerate(obj.shapes) if s["owner"] == owner] for owner in (0, 1)
        ] != fixture.audit_geom_ids:
            obj.close()
            raise ValueError("Compiled surface ownership differs from fixture metadata")
        return obj

    @classmethod
    def from_xml(cls, path, library):
        """SDK candidate: reconstruct the reference auditor's angular damping.

        This starts at the XML initial state. For a current live fixture state,
        use from_fixture, which copies every runtime model/data field instead.
        """
        obj = cls(library)
        obj.handle = obj.lib.da_create_xml(str(Path(path).resolve()).encode())
        obj._initialize()
        return obj

    def _initialize(self):
        if not self.handle:
            raise ValueError(self.lib.da_error().decode())
        self.ngeom = self.lib.da_ngeom(self.handle)
        shapes = np.zeros((self.ngeom, 7))
        self._check(self.lib.da_shapes(self.handle, _ptr(shapes)))
        self.shapes = [
            dict(
                owner=None if s[0] < 0 else int(s[0]),
                rotor=bool(s[1]),
                kind=_NATIVE_KINDS.get(int(s[2])),
                size=s[3:6].tolist(),
                body_id=int(s[6]),
            )
            for s in shapes
        ]
        pairs = np.zeros((self.lib.da_self_count(self.handle), 2), dtype=np.int32)
        self._check(self.lib.da_self_pairs(self.handle, pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))))
        self.self_pairs = [tuple(map(int, p)) for p in pairs]
        offsets = np.zeros((2, 3))
        self._check(self.lib.da_package_offsets(self.handle, _ptr(offsets)))
        self.package_offsets = offsets.tolist()
        model_bytes = np.zeros(self.lib.da_model_size(self.handle), dtype=np.uint8)
        self._check(
            self.lib.da_model_bytes(self.handle, model_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        )
        self.model_sha256 = hashlib.sha256(model_bytes.tobytes()).hexdigest()
        self._reset_audit()

    def _reset_audit(self):
        point = self.inspect()
        self._native_steps = 0
        self._last_rpm = np.zeros((2, 4))
        self._audit = dict(
            origin_time=point["world_time"],
            control_steps=0,
            rk_points=0,
            native_intervals=0,
            body_intervals=0,
            point_evaluations=1,
            certified_intervals=0,
            python_fallback_intervals=0,
            safety_flags=point["flags"],
            minimum_clearance_m=point["min_clearance_m"],
            safe=point["flags"] == 0,
        )

    def _check(self, code):
        if code:
            raise ValueError(self.lib.da_error().decode())

    def _open(self):
        if not self.handle:
            raise ValueError("Native shadow is closed")

    @staticmethod
    def _point(row, state):
        return dict(
            world_time=float(row[0]),
            min_clearance_m=float(row[1]),
            floor_minima_m=row[2:4].tolist(),
            constraint_max_m=float(row[4]),
            max_roll_pitch_deg=float(row[5]),
            max_swing_deg=float(row[6]),
            max_spin_deg_s=float(row[7]),
            finite=bool(row[8]),
            flags=int(row[9]),
            closest_pair=row[10:12].astype(int).tolist(),
            contacts=int(row[12]),
            wash_incursions=int(row[13]),
            exact_evaluations=int(row[14]),
            warning_counts=row[15:22].astype(int).tolist(),
            live_acceleration_finite=bool(row[22]),
            qpos=state[:30].tolist(),
            qvel=state[30:].tolist(),
        )

    def inspect(self):
        self._open()
        metrics, state = np.zeros(_METRIC_WIDTH), np.zeros(54)
        self._check(self.lib.da_inspect(self.handle, _ptr(metrics), _ptr(state)))
        return self._point(metrics, state)

    def reset_state(self, qpos, qvel, *, world_time=0.0):
        self._open()
        qpos, qvel = _array(qpos, (30,)), _array(qvel, (24,))
        self._check(self.lib.da_reset(self.handle, _ptr(qpos), _ptr(qvel), world_time))
        self._reset_audit()

    def step_rpm(self, rpm):
        self._open()
        rpm = _array(rpm, (2, 4))
        metrics, states, geometry = (
            np.zeros((81, _METRIC_WIDTH)),
            np.zeros((81, 54)),
            np.zeros((17, self.ngeom, 12)),
        )
        start = time.perf_counter()
        self._check(self.lib.da_step(self.handle, _ptr(rpm), _ptr(metrics), _ptr(states), _ptr(geometry)))
        elapsed = time.perf_counter() - start
        self._native_steps += 1
        self._last_rpm = rpm.copy()
        points = [self._point(m, s) for m, s in zip(metrics, states)]
        bodies = np.zeros((17, 2, 2, 12))
        self._check(self.lib.da_body_geometry(self.handle, _ptr(bodies)))
        for row, pose, body in zip(points[64:], geometry, bodies):
            row.update(geom_xpos=pose[:, :3].tolist(), geom_xmat=pose[:, 3:].tolist())
            row.update(
                assembly_xpos=body[:, :, :3].tolist(),
                assembly_xmat=body[:, :, 3:].reshape(2, 2, 3, 3).tolist(),
            )
        return dict(
            schema="dual-native-audit-shadow-v1",
            qualification=QUALIFICATION,
            production_qualified=False,
            scratch_forward_recomputes_acceleration=True,
            native_elapsed_seconds=elapsed,
            rk_points=points[:64],
            boundaries=points[64:],
        )

    def step_audited(self, rpm, *, fast_path=True):
        """Candidate atomic control boundary: all points and all intervals audited.

        This remains a detached clone. copy_to_fixture is a separate explicit
        operation; this API does not confer production qualification.
        """
        if self._native_steps != self._audit["control_steps"] or not self._audit["safe"]:
            raise ValueError("Complete safe prior candidate audit required before continuation")
        result = self.step_rpm(rpm)
        interval = self.audit_intervals(result, fast_path=fast_path)
        points = result["rk_points"] + result["boundaries"]
        self._audit["control_steps"] += 1
        self._audit["rk_points"] += 64
        self._audit["native_intervals"] += 16
        self._audit["body_intervals"] += 16
        self._audit["point_evaluations"] += 80
        self._audit["certified_intervals"] += interval["certified_intervals"]
        self._audit["python_fallback_intervals"] += interval["python_fallback_intervals"]
        for point in points:
            self._audit["safety_flags"] |= point["flags"]
            self._audit["minimum_clearance_m"] = min(
                self._audit["minimum_clearance_m"], point["min_clearance_m"]
            )
        self._audit["safe"] = (
            self._audit["safety_flags"] == 0
            and self._audit["certified_intervals"] == self._audit["native_intervals"]
        )
        result["interval_audit"] = interval
        result["cumulative_audit"] = copy.deepcopy(self._audit)
        return result

    def snapshot(self):
        """Exact integration state plus complete audit counters at a control boundary."""
        self._open()
        if self._native_steps != self._audit["control_steps"]:
            raise ValueError("Incomplete candidate audit cannot be a resumable boundary")
        state = np.zeros(self.lib.da_state_size(self.handle))
        warnings = np.zeros((7, 2), dtype=np.int32)
        self._check(
            self.lib.da_get_state(
                self.handle, _ptr(state), warnings.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            )
        )
        return dict(
            schema="dual-compiled-candidate-checkpoint-v1",
            production_qualified=False,
            model_sha256=self.model_sha256,
            library_sha256=self.receipt["artifact"]["sha256"],
            state_spec="mjSTATE_INTEGRATION",
            state=state.tolist(),
            warning_state=warnings.tolist(),
            last_rpm=self._last_rpm.tolist(),
            audit=copy.deepcopy(self._audit),
        )

    def restore(self, snapshot):
        self._open()
        saved = copy.deepcopy(snapshot)
        if (
            saved.get("schema") != "dual-compiled-candidate-checkpoint-v1"
            or saved.get("production_qualified") is not False
            or saved.get("model_sha256") != self.model_sha256
            or saved.get("library_sha256") != self.receipt["artifact"]["sha256"]
            or saved.get("state_spec") != "mjSTATE_INTEGRATION"
        ):
            raise ValueError("Checkpoint native model/build/state contract differs")
        state = _array(saved["state"], (self.lib.da_state_size(self.handle),))
        warnings = np.asarray(saved["warning_state"])
        if (
            warnings.shape != (7, 2)
            or warnings.dtype.kind not in "iu"
            or np.any(warnings < 0)
            or np.any(warnings > np.iinfo(np.int32).max)
        ):
            raise ValueError("Exact native warning counters required")
        warnings = np.ascontiguousarray(warnings, dtype=np.int32)
        rpm = _array(saved["last_rpm"], (2, 4))
        from .payload_solver_mujoco import HOVER_RPM

        if np.any(rpm < 0) or np.any(rpm > 1.8 * HOVER_RPM):
            raise ValueError("Original checkpoint RPM authority required")
        audit = saved["audit"]
        if not isinstance(audit, dict) or set(audit) != set(self._audit):
            raise ValueError("Complete candidate audit state required")
        steps = audit["control_steps"]
        counts = (
            "control_steps",
            "rk_points",
            "native_intervals",
            "body_intervals",
            "point_evaluations",
            "certified_intervals",
            "python_fallback_intervals",
            "safety_flags",
        )
        if (
            any(type(audit[k]) is not int or audit[k] < 0 for k in counts)
            or audit["rk_points"] != steps * 64
            or audit["native_intervals"] != steps * 16
            or audit["body_intervals"] != steps * 16
            or audit["point_evaluations"] != 1 + 80 * steps
            or audit["certified_intervals"] > 16 * steps
            or audit["python_fallback_intervals"] > 16 * steps
            or type(audit["safe"]) is not bool
            or audit["safe"] != (audit["safety_flags"] == 0 and audit["certified_intervals"] == 16 * steps)
            or not np.isfinite(audit["origin_time"])
            or not np.isfinite(audit["minimum_clearance_m"])
            or not np.isclose(state[0], audit["origin_time"] + 0.01 * steps, atol=1e-10, rtol=0)
        ):
            raise ValueError("Inconsistent candidate audit checkpoint counters/time")
        self._check(
            self.lib.da_set_state(
                self.handle, _ptr(state), warnings.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), _ptr(rpm)
            )
        )
        self._native_steps = steps
        self._last_rpm = rpm
        self._audit = audit

    def copy_to_fixture(self, fixture):
        """Explicit experimental state publication; never called automatically.

        C++ compares complete serialized runtime models before copying every
        mjData field, including warmstart, controls, applied forces and warnings.
        Python RPM metadata is updated; transient Python raw capture lists are
        cleared because this candidate exports compact evidence instead.
        """
        self.snapshot()  # enforce an audited boundary, including unsafe terminal states
        self._check(self.lib.da_copy_to(self.handle, fixture.model._address, fixture.data._address))
        fixture.rpm = self._last_rpm.copy()
        fixture.stage_rows = []
        fixture.substep_rows = []

    def pair_distance(self, positions, rotations, left, right):
        self._open()
        positions = _array(positions, (self.ngeom, 3))
        rotations = _array(np.asarray(rotations).reshape(self.ngeom, 9), (self.ngeom, 9))
        result = np.zeros(1)
        self._check(
            self.lib.da_distance(self.handle, _ptr(positions), _ptr(rotations), left, right, _ptr(result))
        )
        return float(result[0])

    def audit_intervals(self, result, *, fast_path=False):
        """Use unchanged Python interval contracts on SDK-returned native poses."""
        from .dual_coordination_audit import swept_reconstruction, body_interval_certificate, _radius

        self._open()
        rows = result["boundaries"]
        if len(rows) != 17:
            raise ValueError("Exactly17 native boundary poses required")
        cross = [
            (a, b)
            for a, x in enumerate(self.shapes)
            for b, y in enumerate(self.shapes)
            if x["owner"] == 0 and y["owner"] == 1
        ]
        pairs = cross + self.self_pairs
        left, right = np.asarray(pairs, dtype=int).T
        radii = np.array([_radius(self.shapes[a]) + _radius(self.shapes[b]) for a, b in pairs])
        index = {pair: i for i, pair in enumerate(pairs)}
        last = None
        lower = None

        def distance(positions, rotations, a, b):
            nonlocal last, lower
            if positions is not last:
                lower = np.linalg.norm(positions[left] - positions[right], axis=1) - radii - 1e-12
                last = positions
            value = lower[index[a, b]]
            margin = 0.2 if self.shapes[a]["owner"] != self.shapes[b]["owner"] else 0.0
            return float(value) if value >= margin else self.pair_distance(positions, rotations, a, b)

        start = time.perf_counter()
        reports = []
        fallbacks = 0
        for a, b in zip(rows, rows[1:]):
            if not np.isclose(b["world_time"] - a["world_time"], 0.01 / 16, rtol=0, atol=1e-12):
                raise ValueError("Consecutive native substep times required")
            quick = self.fast_interval(a, b) if fast_path else None
            if quick is not None and quick["certified"]:
                reports.append(dict(sweep=quick["sweep"], body=quick["body"]))
            else:
                fallbacks += 1
                sweep = swept_reconstruction(
                    a, b, shapes=self.shapes, pair_distance=distance, self_pairs=self.self_pairs
                )
                body = body_interval_certificate(a, b, package_offsets=self.package_offsets)
                reports.append(dict(sweep=sweep, body=body))
        return dict(
            schema="dual-candidate-interval-audit-v1",
            production_qualified=False,
            native_intervals=16,
            body_intervals=16,
            python_fallback_intervals=fallbacks,
            certified_intervals=sum(
                x["sweep"]["status"] == x["body"]["status"] == "certified" for x in reports
            ),
            elapsed_seconds=time.perf_counter() - start,
            reports=reports,
        )

    def fast_interval(self, start, end):
        """Return a sufficient root-cell certificate, otherwise request fallback.

        It preserves all original margins and numerical guards. Arcs larger
        than.05rad deliberately fall back, avoiding ambiguous SO(3) midpoints.
        It never classifies a near or unresolved interval as safe.
        """
        self._open()

        def arrays(row):
            geometry = np.concatenate(
                (
                    np.asarray(row["geom_xpos"]).reshape(self.ngeom, 3),
                    np.asarray(row["geom_xmat"]).reshape(self.ngeom, 9),
                ),
                axis=1,
            )
            bodies = np.concatenate(
                (
                    np.asarray(row["assembly_xpos"]).reshape(4, 3),
                    np.asarray(row["assembly_xmat"]).reshape(4, 9),
                ),
                axis=1,
            )
            return _array(geometry, (self.ngeom, 12)), _array(bodies, (4, 12))

        ga, ba = arrays(start)
        gb, bb = arrays(end)
        out = np.zeros(8)
        self._check(self.lib.da_fast_interval(self.handle, _ptr(ga), _ptr(gb), _ptr(ba), _ptr(bb), _ptr(out)))
        if out[0] != 1:
            return dict(certified=False, reason="root bound not sufficient; Python subdivision required")
        sweep = dict(
            status="certified",
            contract="linear-centers-shortest-rotation-v1",
            lower_clearance_m=float(out[1]),
            self_lower_clearance_m=float(out[2]) if np.isfinite(out[2]) else None,
            self_pairs_checked=len(self.self_pairs),
            minimum_sampled_clearance_m=float(out[7]),
            witness=None,
            witness_fraction=None,
            sampled_poses=2,
            unresolved_leaves=0,
        )
        body = dict(
            status="certified",
            contract="linear-body-centers-shortest-rotation-attachment-chord-v1",
            witness=None,
            checked_subintervals=int(out[5]),
            unresolved_leaves=0,
            constraint_upper_bound_m=float(out[3]),
            tilt_upper_bound_deg=float(out[4]),
        )
        return dict(certified=True, sweep=sweep, body=body, exact_distance_evaluations=int(out[6]))

    def close(self):
        if getattr(self, "handle", None):
            self.lib.da_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def compare_python_shadow(fixture, shadow, rpm):
    """One measured interval against original stepping and independent diagnostics.

    The caller advances both branches together. This diagnostic never promotes
    the candidate, and runs Python sweeps on its17 returned native poses.
    """
    from .dual_coordination_audit import NativePointAuditor, PointAuditLedger

    auditor = NativePointAuditor(fixture.model_path, fixture.runtime())
    reference_sweeper = NativePointAuditor(fixture.model_path, fixture.runtime())
    before = fixture.data.qpos.copy(), fixture.data.qvel.copy(), float(fixture.data.time)
    compiled = shadow.step_rpm(rpm)
    if not (
        np.array_equal(before[0], fixture.data.qpos)
        and np.array_equal(before[1], fixture.data.qvel)
        and before[2] == fixture.data.time
    ):
        raise ValueError("Shadow modified original integration data")
    fixture.step_rpm(rpm, observe_stages=True)
    start = time.perf_counter()
    maximum = 0.0
    checks = [
        ("rk", compiled["rk_points"], fixture.stage_rows),
        ("boundary", compiled["boundaries"], fixture.substep_rows),
    ]
    for label, left, right in checks:
        if len(left) != len(right):
            raise ValueError("Native callback/boundary count mismatch")
        for candidate, reference in zip(left, right):
            fields = ("qpos", "qvel", "world_time") + (
                ("geom_xpos", "geom_xmat") if label == "boundary" else ()
            )
            for field in fields:
                residual = float(np.max(np.abs(np.asarray(candidate[field]) - np.asarray(reference[field]))))
                maximum = max(maximum, residual)
                if residual > 1e-9:
                    raise ValueError(f"Compiled shadow {label}/{field} differs")
            mj, data = auditor.mj, auditor.data
            mj.set_mjcb_control(None)
            mj.mj_resetData(auditor.model, data)
            data.qpos[:], data.qvel[:], data.ctrl[:], data.time = (
                reference["qpos"],
                reference["qvel"],
                fixture.data.ctrl,
                reference["world_time"],
            )
            mj.mj_forward(auditor.model, data)
            point = auditor._diagnostics(
                reference.get("warning_counts", [int(w.number) for w in data.warning])
            )
            for field in (
                "min_clearance_m",
                "floor_minima_m",
                "constraint_max_m",
                "max_roll_pitch_deg",
                "max_swing_deg",
                "max_spin_deg_s",
            ):
                residual = float(np.max(np.abs(np.asarray(candidate[field]) - np.asarray(point[field]))))
                maximum = max(maximum, residual)
                if residual > 1e-8:
                    raise ValueError(f"Compiled shadow diagnostic {field} differs")
            if (
                candidate["contacts"] != len(point["contacts"])
                or candidate["wash_incursions"] != len(point["wash_incursions"])
                or candidate["finite"] != point["finite"]
                or candidate["warning_counts"] != point["warning_counts"]
                or candidate["closest_pair"] != point["closest_pair"]
            ):
                raise ValueError("Compiled shadow safety classifications differ")
            ledger = PointAuditLedger()
            ledger.observe(point, time=reference["world_time"], location=label)
            bits = {
                "nonfinite native state": 1,
                "assembly margin": 2,
                "floor clearance": 4,
                "forbidden contact": 8,
                "rotor-wash exclusion": 16,
                "constraint residual": 32,
                "tilt": 64,
                "native numerical warning": 128,
            }
            expected_flags = 0
            for violation in ledger.violations:
                expected_flags |= bits[violation["reason"]]
            if candidate["flags"] != expected_flags:
                raise ValueError("Compiled safety threshold decision differs from independent ledger")
    sweep_start = time.perf_counter()
    for left, right in zip(compiled["boundaries"], compiled["boundaries"][1:]):
        auditor._sweep(left, right, "compiled-shadow")
    candidate_sweep_seconds = time.perf_counter() - sweep_start
    for left, right in zip(fixture.substep_rows, fixture.substep_rows[1:]):
        reference_sweeper._sweep(left, right, "compiled-shadow")
    if (
        auditor.sweeps != 16
        or reference_sweeper.sweeps != 16
        or auditor.sweep_failures != reference_sweeper.sweep_failures
        or not np.isclose(auditor.sweep_lower, reference_sweeper.sweep_lower, atol=1e-9, rtol=0)
    ):
        raise ValueError("Compiled boundary sweep differs from original boundary sweep")
    # The current independent auditor reconstructs assembly poses from qpos
    # when live=False, including self-collision, tilt and suspension intervals.
    interval_fields = (
        "body_intervals",
        "body_subintervals",
        "body_constraint_upper",
        "body_tilt_upper",
        "self_sweep_lower",
    )
    for name in interval_fields:
        if not np.isclose(getattr(auditor, name), getattr(reference_sweeper, name), atol=1e-9, rtol=0):
            raise ValueError(f"Compiled boundary {name} differs from original independent audit")
    if auditor.body_intervals != 16 or auditor.self_pairs != reference_sweeper.self_pairs:
        raise ValueError("Incomplete independent body/self-collision interval coverage")
    return dict(
        schema="dual-native-audit-shadow-comparison-v1",
        matched=True,
        production_qualified=False,
        qualification=QUALIFICATION,
        rk_points=64,
        native_intervals=16,
        maximum_residual=maximum,
        compiled_native_seconds=compiled["native_elapsed_seconds"],
        python_point_and_sweep_seconds=time.perf_counter() - start,
        python_sweep_seconds=candidate_sweep_seconds,
        sweep_comparison_passed=True,
        body_reconstructed_intervals=auditor.body_intervals,
        body_checked_subintervals=auditor.body_subintervals,
        constraint_upper_bound_m=auditor.body_constraint_upper,
        tilt_upper_bound_deg=auditor.body_tilt_upper,
        self_collision_pairs=auditor.self_pairs,
        self_lower_clearance_m=None
        if not np.isfinite(auditor.self_sweep_lower)
        else auditor.self_sweep_lower,
        minimum_clearance_m=min(p["min_clearance_m"] for p in compiled["rk_points"] + compiled["boundaries"]),
        python_sweep_failures=auditor.sweep_failures,
        python_sweep_lower_clearance_m=None if not np.isfinite(auditor.sweep_lower) else auditor.sweep_lower,
        limitations=auditor.proof_gaps,
        build_sha256=sha256(shadow.library),
    )
