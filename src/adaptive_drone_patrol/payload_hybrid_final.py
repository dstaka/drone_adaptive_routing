"""MuJoCo controller-only final campaign, separate from development video admission.

Formal draws are deliberately imported/called only after a typed, re-audited
freeze. No optimizer, held-out selection, or alternative controller is used.
"""

import ast
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

from .jobs import write_json
from .payload_cases import initial_state, SCENARIOS
from .payload_hybrid_contract import VIDEO_PROFILE, STRICT_PROFILE
from .util import sha256, load_config

ROOT = Path(__file__).resolve().parents[2]
ROUTING_SHA256 = "a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef"
_TOKEN = object()
PINNED_V1_READER_SOURCES = {
    "src/adaptive_drone_patrol/payload_hybrid_final.py": "a59211f6b96e7f8efc4f46bc192eb36bb0250bc2f312c33a4054979061c10ae5",
    "src/adaptive_drone_patrol/payload_hybrid_final_audit.py": "1f9f5e6a8061876dec2c76ce5683b2eb8aacfcb3e98a877279ab52222e14585f",
    "scripts/run_payload_hybrid_final.py": "febf66bd91ad9cdbabdf5b29c28d0059b98bcf0963a67389ec09c73a2cbc6e73",
}
READER_REBIND_SCOPE = "READ_ONLY_DEVELOPMENT_AUDIT_C0_PAIR_ALLOCATION_CORRECTION"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def robustness_cases():
    """15 new full missions; mass, pose and endpoint azimuth use separate streams."""
    result = []
    allocation = [(s, 0.045, 0.2, "nominal") for s in SCENARIOS]
    allocation += [
        (s, (0.03, 0.045, 0.06)[(2 * i + j) % 3], None, "interior")
        for i, s in enumerate(SCENARIOS)
        for j in range(2)
    ]
    allocation += [
        (s, (0.03, 0.045, 0.06)[i], m, "endpoint") for i, s in enumerate(SCENARIOS) for m in (0.1, 0.3)
    ]
    for i, (scenario, side, mass, kind) in enumerate(allocation):
        seed = 826251000 + i
        rng = np.random.default_rng(np.random.SeedSequence([seed, 826251, 1]))
        if mass is None:
            mass = float(np.random.default_rng(np.random.SeedSequence([seed, 826251, 2])).uniform(0.1, 0.3))
        direction = None
        if kind == "endpoint":
            direction = float(
                np.random.default_rng(np.random.SeedSequence([seed, 826251, 3])).uniform(0, 2 * np.pi)
            )
        result.append(
            dict(
                case_id=f"final-development-{i:02d}",
                suite="final-development",
                allocation=kind,
                scenario=scenario,
                execution_seed=seed,
                side_m=side,
                mass_ratio=mass,
                initial=initial_state(rng, side, direction),
            )
        )
    return result


def validation_cases():
    """Inherited three loads/ten commands, explicit seeds11–14; no formal input."""
    cfg = load_config(ROOT / "configs/payload_validation.yaml")
    loads = [
        dict(side_m=0.03, mass_ratio=0.13),
        dict(side_m=0.045, mass_ratio=0.2),
        dict(side_m=0.06, mass_ratio=0.27),
    ]
    targets = [
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0],
        [-2.7, -1.8, 1.0],
        [2.7, 1.8, 1.0],
        [2.7, 1.8, 3.0],
        [2.7, 1.8, 1.0],
        [-2.7, -1.8, 3.0],
        [2.7, 1.8, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    if cfg["loads"] != loads or cfg["targets"] != targets:
        raise ValueError("Inherited validation loads/targets changed")
    return [
        dict(
            case_id=f"final-validation-{seed}-{i}",
            suite="final-validation",
            execution_seed=seed,
            **load,
            targets=copy.deepcopy(targets),
            initial=initial_state(
                np.random.default_rng(np.random.SeedSequence([seed, 720, i])), load["side_m"]
            ),
        )
        for seed in (11, 12, 13, 14)
        for i, load in enumerate(loads)
    ]


def _source_closure():
    """Freeze recursive package imports, including imports inside reader functions."""
    package = Path(__file__).parent
    pending = {
        "payload_hybrid_final",
        "payload_hybrid_final_audit",
        "payload_hybrid_flight",
        "payload_hybrid_duration",
        "payload_hybrid_duration_audit",
        "payload_hybrid_audit",
        "payload_hybrid_manifest",
        "payload_recovery_mission",
        "payload_solver_acceptance",
        "routing_agent",
    }
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen or not (package / f"{name}.py").is_file():
            continue
        seen.add(name)
        tree = ast.parse((package / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and node.module:
                    pending.add(node.module.split(".")[0])
                elif node.module and node.module.startswith("adaptive_drone_patrol."):
                    pending.add(node.module.split(".")[1])
    paths = [package / f"{name}.py" for name in sorted(seen)]
    paths += [ROOT / "scripts/run_payload_hybrid_final.py"]
    paths += [ROOT / f"configs/demo_scenarios_dispersed/{s}.yaml" for s in SCENARIOS]
    paths += [
        ROOT / "configs/payload_validation.yaml",
        ROOT / "configs/payload_retry_probe_manifest.json",
        ROOT / "requirements/payload-reference-linux.lock.txt",
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def current_final_binding(*, design, original, routing):
    from .payload_solver_qualification import current_solver_binding

    if sha256(routing) != ROUTING_SHA256:
        raise ValueError("Original routing checkpoint44 required")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "mujoco", "torch", "gymnasium", "stable-baselines3", "sb3-contrib", "PyYAML")
    }
    actual = current_solver_binding()
    if sha256(original) != actual["original_asset_sha256"]:
        raise ValueError("Qualified original asset differs")
    return dict(
        schema="payload-hybrid-final-binding-v1",
        branch="controller",
        actual_reference=actual,
        source_sha256=_source_closure(),
        dependencies=versions,
        python=sys.version,
        design_sha256=sha256(design),
        design_value_sha256=digest(json.loads(Path(design).read_text())),
        original_asset_sha256=sha256(original),
        routing_sha256=sha256(routing),
        routing_schema_sha256=sha256(Path(routing).with_suffix(".schema.json")),
        video_profile_sha256=VIDEO_PROFILE.sha256,
        strict_profile_sha256=STRICT_PROFILE.sha256,
        controller="C-local-lqr-v1",
        motor_inputs="original four motor actions only",
        aggregate_simulated_deadline_s=None,
        leg_s=30.0,
        reset_through_flush_wall_s=600.0,
    )


def validate_reader_binding_delta(old, current):
    """Only three pinned v1 reader/guard files can differ; no physics exception."""
    if not isinstance(old, dict) or not isinstance(current, dict) or set(old) != set(current):
        raise ValueError("Complete identical final binding field inventories required")
    if digest({k: v for k, v in old.items() if k != "source_sha256"}) != digest(
        {k: v for k, v in current.items() if k != "source_sha256"}
    ):
        raise ValueError("Controller/reference/physics/profile/routing/design/dependencies changed")
    a, b = old["source_sha256"], current["source_sha256"]
    if set(a) != set(b) or any(a.get(p) != h for p, h in PINNED_V1_READER_SOURCES.items()):
        raise ValueError("Exact pinned v1 reader sources and identical source inventories required")
    changed = {p: dict(old_sha256=a[p], current_sha256=b[p]) for p in a if a[p] != b[p]}
    if not changed or not set(changed) <= set(PINNED_V1_READER_SOURCES):
        raise ValueError("Only the exact reviewed final reader/compatibility guards may change")
    return changed


def reader_source_package_files(path):
    """Hash original archive member bytes without extracting or executing them."""
    import tarfile
    from pathlib import PurePosixPath

    result = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Original reader source package contains an unsafe member path")
            if member.isdir():
                continue
            if not member.isfile() or str(name) in result:
                raise ValueError("Unique regular original source-package members required")
            with archive.extractfile(member) as stream:
                data = stream.read()
            result[str(name)] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    if any(result.get(p, {}).get("sha256") != h for p, h in PINNED_V1_READER_SOURCES.items()):
        raise ValueError("Original package does not contain every exact pinned v1 reader source")
    return result


def validate_reader_rebinding(record, current_binding):
    if (
        record.get("schema") != "payload-hybrid-final-reader-rebinding-v1"
        or record.get("scope") != READER_REBIND_SCOPE
        or record.get("formal_compatibility") is not False
        or digest(record["current_binding"]) != digest(current_binding)
        or record["reader_source_changes"]
        != validate_reader_binding_delta(record["old_binding"], current_binding)
        or record["pinned_v1_reader_sources"] != PINNED_V1_READER_SOURCES
        or sha256(record["old_source_package"]) != record["old_source_package_sha256"]
        or reader_source_package_files(record["old_source_package"]) != record["old_source_package_files"]
        or sha256(record["old_development_declaration"]) != record["old_development_declaration_sha256"]
    ):
        raise ValueError("Independent exact reader-only source derivation required")
    declaration = json.loads(Path(record["old_development_declaration"]).read_text())
    if digest(declaration["binding"]) != digest(record["old_binding"]):
        raise ValueError("Preserved development declaration differs from reader derivation")
    return copy.deepcopy(record)


def write_reader_rebinding(output, *, context, old_development, old_source_package):
    if not isinstance(context, FinalContext) or context.reader_rebinding is not None:
        raise ValueError("Fresh typed current context required for explicit reader derivation")
    context.require_current()
    if Path(output).exists():
        raise FileExistsError(output)
    declaration_path = (Path(old_development) / "declaration.json").resolve()
    old = json.loads(declaration_path.read_text())["binding"]
    changes = validate_reader_binding_delta(old, context.binding)
    package = Path(old_source_package).resolve()
    record = dict(
        schema="payload-hybrid-final-reader-rebinding-v1",
        scope=READER_REBIND_SCOPE,
        old_binding=old,
        current_binding=context.binding,
        reader_source_changes=changes,
        pinned_v1_reader_sources=copy.deepcopy(PINNED_V1_READER_SOURCES),
        old_source_package=str(package),
        old_source_package_sha256=sha256(package),
        old_source_package_files=reader_source_package_files(package),
        old_development_declaration=str(declaration_path),
        old_development_declaration_sha256=sha256(declaration_path),
        formal_compatibility=False,
        physical_execution="UNCHANGED_ALL_NONREADER_BOUND_BYTES_IDENTICAL",
        correction="Preserved20 feedback-off/refined diagnostics are C0, not C1; independently re-audit all old raw",
    )
    validate_reader_rebinding(record, context.binding)
    write_json(output, record)
    return record


class FinalContext:
    """Current reference capability plus exact final source/controller inputs."""

    def __init__(self, reference, paths, binding, token):
        if token is not _TOKEN:
            raise ValueError("Rebound physical reference required")
        self.reference = reference
        self.paths = copy.deepcopy(paths)
        self._binding = copy.deepcopy(binding)
        self._reader_rebinding = None
        self._reader_receipt_path = None
        self._reader_receipt_sha256 = None

    @classmethod
    def from_reference(cls, reference, *, design, original, routing, reader_rebinding=None):
        from .payload_hybrid_flight import ReferenceAdmission

        if not isinstance(reference, ReferenceAdmission):
            raise ValueError("Typed reference required")
        reference.require_current()
        paths = {
            k: str(Path(v).resolve())
            for k, v in dict(design=design, original=original, routing=routing).items()
        }
        binding = current_final_binding(**paths)
        if binding["actual_reference"] != reference.binding:
            raise ValueError("Reference binding differs")
        result = cls(reference, paths, binding, _TOKEN)
        if reader_rebinding is not None:
            result._reader_receipt_path = str(Path(reader_rebinding).resolve())
            result._reader_receipt_sha256 = sha256(reader_rebinding)
            result._reader_rebinding = validate_reader_rebinding(
                json.loads(Path(reader_rebinding).read_text()), binding
            )
        return result

    @property
    def binding(self):
        return copy.deepcopy(self._binding)

    @property
    def reader_rebinding(self):
        return copy.deepcopy(self._reader_rebinding)

    def require_current(self):
        self.reference.require_current()
        if current_final_binding(**self.paths) != self._binding:
            raise ValueError("Frozen final controller/reference/source/dependencies changed")
        if self._reader_rebinding is not None:
            if sha256(self._reader_receipt_path) != self._reader_receipt_sha256:
                raise ValueError("Explicit reader derivation receipt changed")
            validate_reader_rebinding(self._reader_rebinding, self._binding)

    def require_recorded_binding(self, recorded, stage, *, declaration_path=None):
        self.require_current()
        if digest(recorded) == digest(self._binding):
            return copy.deepcopy(recorded)
        if (
            stage not in ("development-declaration", "final-development", "final-validation")
            or self._reader_rebinding is None
            or digest(recorded) != digest(self._reader_rebinding["old_binding"])
        ):
            raise ValueError("Exact current binding required; reader derivation is development-audit only")
        validate_reader_rebinding(self._reader_rebinding, self._binding)
        if (
            declaration_path is not None
            and str(Path(declaration_path).resolve()) != self._reader_rebinding["old_development_declaration"]
        ):
            raise ValueError("Reader derivation does not authorize another development campaign")
        return copy.deepcopy(recorded)


def prepare_development(output, context):
    if not isinstance(context, FinalContext):
        raise ValueError("Typed final context required")
    context.require_current()
    if context.reader_rebinding is not None:
        raise ValueError("Reader-rebound context cannot prepare additional development physics")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="payload-hybrid-final-development-v1",
        binding=context.binding,
        robustness_cases=robustness_cases(),
        validation_cases=validation_cases(),
        formal_inputs="UNOPENED",
        selection="C only; no development tuning or retries inside this campaign",
    )
    write_json(root / "declaration.json", record)
    return record


def _development_declaration(root, context):
    if not isinstance(context, FinalContext):
        raise ValueError("Typed final context required")
    context.require_current()
    record = json.loads((Path(root) / "declaration.json").read_text())
    recorded = context.require_recorded_binding(
        record["binding"], "development-declaration", declaration_path=Path(root) / "declaration.json"
    )
    expected = dict(
        schema="payload-hybrid-final-development-v1",
        binding=recorded,
        robustness_cases=robustness_cases(),
        validation_cases=validation_cases(),
        formal_inputs="UNOPENED",
        selection="C only; no development tuning or retries inside this campaign",
    )
    if record != expected:
        raise ValueError("Predeclared final development campaign changed")
    return record


def require_development_attempt(output, context, case):
    root = Path(output)
    declaration = _development_declaration(root.parent.parent, context)
    stage = "validation" if case["suite"] == "final-validation" else "robustness"
    key = "validation_cases" if stage == "validation" else "robustness_cases"
    if root.parent.name != stage or root.name != case["case_id"] or case not in declaration[key]:
        raise ValueError("Actual development attempt differs from prior declaration")
    return digest(declaration)


def _record_validation(flight, case, root, *, clock=time.monotonic):
    from .payload_contract import SharedSettling
    from .payload_solver_validation import measured_predicates

    root.mkdir()
    completed = []
    failure = None
    with (root / "samples.jsonl").open("x") as stream:

        def emit(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()

        emit(dict(kind="INITIAL", step=0, sample=flight.sample()))
        try:
            for i, target in enumerate(case["targets"]):
                flight.set_target(target, target_id=str(i))
                begin = flight.runtime.step_count
                emit(
                    dict(
                        kind="RETARGET",
                        step=begin,
                        target_index=str(i),
                        target=target,
                        sample=flight.sample(),
                    )
                )
                gate = SharedSettling(0.01)
                gate.update(begin * 0.01, False, str(i))
                while True:
                    if clock() - flight.reset_wall_s >= 600:
                        raise TimeoutError("WALL_TIMEOUT")
                    if flight.runtime.step_count - begin >= 3000:
                        raise TimeoutError("LEG_TIMEOUT")
                    flight.advance()
                    sample = flight.sample()
                    step = flight.runtime.step_count
                    elapsed = clock() - flight.reset_wall_s
                    emit(
                        dict(
                            kind="STEP",
                            step=step,
                            target_index=str(i),
                            target=target,
                            sample=sample,
                            reset_elapsed_wall_s=elapsed,
                        )
                    )
                    m = measured_predicates(sample, target, case["side_m"], revision2=True)
                    if m["unsafe"]:
                        raise ValueError("SAFETY_FAILURE")
                    if elapsed >= 600:
                        raise TimeoutError("WALL_TIMEOUT")
                    if gate.update(step * 0.01, m["valid"], str(i)):
                        completed.append(i)
                        break
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
    write_json(
        root / "execution.json", dict(completed=completed, failure=failure, steps=flight.runtime.step_count)
    )


def run_final_case(context, case, output, planner=None, *, freeze=None):
    """Exactly one attempt, exclusive output; always preserve failure artifacts."""
    from .payload_hybrid_flight import HybridFlight

    if not isinstance(context, FinalContext):
        raise ValueError("Typed final context required")
    context.require_current()
    suite = case.get("suite")
    if context.reader_rebinding is not None and suite in ("final-development", "final-validation"):
        raise ValueError("Reader-only compatibility cannot execute additional development physics")
    if suite not in ("final-development", "final-validation") and (
        not isinstance(freeze, FinalFreeze) or freeze.context.binding != context.binding
    ):
        raise ValueError("Same typed final freeze/context required for formal execution")
    allowed = (
        robustness_cases()
        if suite == "final-development"
        else validation_cases()
        if suite == "final-validation"
        else formal_cases(freeze)
    )
    if case not in allowed:
        raise ValueError("Case differs from frozen allocation")
    root = Path(output)
    if suite not in ("final-development", "final-validation"):
        declaration = require_formal_declaration(root.parent, freeze)
        if case not in declaration["cases"]:
            raise ValueError("Actual formal case differs from predeclared frozen manifest")
        declaration_hash = digest(declaration)
    else:
        declaration_hash = require_development_attempt(root, context, case)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "case.json", case)
    write_json(root / "binding.json", context.binding)
    write_json(
        root / "attempt.json",
        dict(
            attempt=1,
            stage=suite,
            declaration_sha256=declaration_hash,
            freeze_sha256=digest(freeze.record) if freeze else None,
        ),
    )
    flight = None
    error = None
    flush = None
    try:
        flight = HybridFlight.create(
            admission=context.reference,
            original=context.paths["original"],
            output=root / "flight",
            case=case,
            design=json.loads(Path(context.paths["design"]).read_text()),
        )
        if suite == "final-validation":
            _record_validation(flight, case, root / "sequence")
        else:
            if planner is None or planner.hash != ROUTING_SHA256:
                raise ValueError("Original routing planner required")
            from .scenario import Scenario
            from .payload_recovery_mission import execute_payload_mission

            scenario = Scenario.from_dict(
                load_config(ROOT / f"configs/demo_scenarios_dispersed/{case['scenario']}.yaml")
            )
            execute_payload_mission(
                flight,
                planner,
                scenario,
                root / "mission",
                dict(
                    evidence_kind="physical",
                    case_id=case["case_id"],
                    checkpoint_sha256=sha256(root / "flight/design.json"),
                    controller="C-local-lqr-v1",
                    backend="MuJoCo3.15.0/float64/RK4/substeps16/N1",
                    actual_binding=context.reference.binding,
                    routing_model_hash=planner.hash,
                    flight_nn="NOT_USED",
                    scenario=case["scenario"],
                    final_stage=suite,
                    final_binding_sha256=digest(context.binding),
                ),
                physical_evidence=dict(hybrid_admission=context.reference),
                revision2=True,
            )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if flight is not None:
            try:
                flight.close()
                flush = time.monotonic() - flight.reset_wall_s
            except BaseException as exc:
                error = f"{error or ''}; close: {type(exc).__name__}: {exc}"
        write_json(
            root / "final-flush.json",
            dict(
                schema="payload-hybrid-final-flush-v1",
                flushed=flush is not None,
                elapsed_wall_s=flush,
                error=error,
                raw_sha256=sha256(root / "flight/runtime/raw.jsonl.gz")
                if (root / "flight/runtime/raw.jsonl.gz").is_file()
                else None,
                control_sha256=sha256(root / "flight/control.jsonl.gz")
                if (root / "flight/control.jsonl.gz").is_file()
                else None,
            ),
        )
    from .payload_hybrid_final_audit import audit_final_case

    result = audit_final_case(root, context=context, expected_case=case, planner=planner)
    write_json(root / "independent-check.json", result)
    return result


class FinalFreeze:
    """Minted only after current raw C gates,15 missions and12/120 validation."""

    def __init__(self, record, token, context=None):
        if token is not _TOKEN:
            raise ValueError("Independent final prerequisite audit required")
        self._record = copy.deepcopy(record)
        self.context = context
        self._formal_cases = None
        from .payload_hybrid_final_audit import VerifiedArtifactSet

        self._artifacts = VerifiedArtifactSet(self._record["prerequisite_artifacts"])

    @property
    def record(self):
        return copy.deepcopy(self._record)

    def require_current(self):
        self.context.require_current()
        self._artifacts.require_current()

    @classmethod
    def create(cls, context, *, development, diagnostic_roots, duration_root, planner):
        from .payload_hybrid_final_audit import audit_prerequisites

        if not isinstance(context, FinalContext):
            raise ValueError("Typed final context required")
        declaration = _development_declaration(development, context)
        proof = audit_prerequisites(context, development, diagnostic_roots, duration_root, planner)
        record = dict(
            schema="payload-hybrid-final-freeze-v1",
            binding=context.binding,
            branch="controller",
            development_declaration_sha256=digest(declaration),
            development=str(Path(development).resolve()),
            diagnostic_roots=[str(Path(p).resolve()) for p in diagnostic_roots],
            duration_root=str(Path(duration_root).resolve()),
            prerequisites=proof["summary"],
            reader_rebinding=context.reader_rebinding,
            prerequisite_artifacts=proof["artifacts"],
            formal_denominators=dict(basic=9, endpoints=18, heldout=100),
            formal_attempts_per_case=1,
            final_validation=dict(conditions=12, transitions=120),
            formal_inputs="UNOPENED_AT_FREEZE",
        )
        return cls(record, _TOKEN, context)

    @classmethod
    def load(cls, path, context, planner):
        saved = json.loads(Path(path).read_text())
        # Reconstruct every prerequisite from raw; a saved PASS never grants a capability.
        current = cls.create(
            context,
            development=saved["development"],
            diagnostic_roots=saved["diagnostic_roots"],
            duration_root=saved["duration_root"],
            planner=planner,
        )
        if current.record != saved:
            raise ValueError("Saved final freeze differs from independent reconstruction")
        return current


def formal_cases(freeze):
    if not isinstance(freeze, FinalFreeze):
        raise ValueError("Typed final freeze required before formal draw")
    freeze.require_current()
    if freeze._formal_cases is None:
        from .payload_cases import build_cases

        freeze._formal_cases = build_cases()
    return copy.deepcopy(freeze._formal_cases)


def prepare_formal(output, freeze):
    cases = formal_cases(freeze)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="payload-hybrid-final-formal-v1",
        freeze_sha256=digest(freeze.record),
        binding=freeze.context.binding,
        cases=cases,
        attempts_per_case=1,
        method="C",
    )
    write_json(root / "declaration.json", record)
    return record


def require_formal_declaration(output, freeze):
    cases = formal_cases(freeze)
    record = json.loads((Path(output) / "declaration.json").read_text())
    expected = dict(
        schema="payload-hybrid-final-formal-v1",
        freeze_sha256=digest(freeze.record),
        binding=freeze.context.binding,
        cases=cases,
        attempts_per_case=1,
        method="C",
    )
    if record != expected:
        raise ValueError("Formal declaration differs from freeze/original cases")
    return record


class FinalAcceptance:
    """Final delivery capability; development recordings cannot construct it."""

    def __init__(self, record, freeze, token):
        if token is not _TOKEN:
            raise ValueError("Complete independent formal campaign audit required")
        self._record = copy.deepcopy(record)
        self.freeze = freeze
        from .payload_hybrid_final_audit import VerifiedArtifactSet

        self._artifacts = VerifiedArtifactSet(self._record["artifacts"])

    @property
    def record(self):
        return copy.deepcopy(self._record)

    @classmethod
    def from_campaign(cls, freeze, output, planner):
        from .payload_hybrid_final_audit import audit_final_campaign

        proof = audit_final_campaign(freeze, output, planner)
        if not proof["accepted"]:
            raise ValueError("Formal9/9,18/18,95/100 acceptance not achieved")
        return cls(proof, freeze, _TOKEN)

    def require_current(self):
        self.freeze.require_current()
        self._artifacts.require_current()
