"""File-bound campaign admission and prospective held-out access.

These guards authenticate retained files and explicit qualification receipts;
no boolean flag admits training or opens held-out cohorts. Qualification itself
must come from physical experiments, never from this administrative module.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import json
import hashlib
import math
from collections import Counter
from pathlib import Path
from types import MappingProxyType

from .dual_coordination_progress import atomic_json
from .dual_coordination_schema import validate_case
from .util import sha256

LEARNING_ROLES = frozenset(
    (
        "source",
        "assets",
        "controller",
        "configuration",
        "declaration",
        "baseline",
        "audit",
        "qualification",
        "benchmark",
        "dependencies",
        "layout",
    )
)
JOINT_ROLES = LEARNING_ROLES | frozenset(
    ("camera", "selection", "terminal42", "terminal43", "terminal44", "learning_lock", "validation_evidence")
)
EXPERIMENT_ROLES = LEARNING_ROLES - {"qualification", "benchmark"}
NATIVE_CHECKS = frozenset(
    (
        "address_actuation",
        "reset_isolation",
        "passive_powered",
        "collision_detection",
        "stopping_extremes",
        "nominal330",
        "endpoints6",
        "mixed36",
        "streaming_audit",
        "deterministic_replay",
    )
)
_TOKEN = object()


def _path(root, relative):
    p = Path(relative)
    if not isinstance(relative, str) or p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("A confined relative file path is required")
    full = root / p
    for parent in (full, *full.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Symlinks are not frozen artifact identities")
    if not full.is_file() or root not in full.resolve().parents:
        raise ValueError(f"Missing or unconfined artifact: {relative}")
    return full


@dataclass(frozen=True)
class BoundFiles:
    root: Path
    files: dict

    @classmethod
    def capture(cls, root, roles):
        root = Path(root).resolve()
        if not isinstance(roles, dict) or not roles:
            raise ValueError("Nonempty named artifact roles required")
        files = {}
        for role, paths in roles.items():
            if not isinstance(role, str) or not isinstance(paths, list) or not paths:
                raise ValueError("Each role requires a nonempty file list")
            files[role] = []
            for relative in paths:
                p = _path(root, relative)
                files[role].append(dict(path=relative, bytes=p.stat().st_size, sha256=sha256(p)))
        return cls(root, files)

    def verify(self):
        for entries in self.files.values():
            if not isinstance(entries, list) or not entries:
                raise ValueError("Empty artifact role")
            for record in entries:
                if set(record) != {"path", "bytes", "sha256"}:
                    raise ValueError("Malformed frozen artifact")
                p = _path(self.root, record["path"])
                if p.stat().st_size != record["bytes"] or sha256(p) != record["sha256"]:
                    raise ValueError(f"Frozen file changed: {record['path']}")
        return self

    def to_dict(self):
        return dict(root=str(self.root), files=copy.deepcopy(self.files))

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"root", "files"} or not data["files"]:
            raise ValueError("Complete artifact binding required")
        return cls(Path(data["root"]).resolve(), copy.deepcopy(data["files"])).verify()


def _qualification(value):
    if not isinstance(value, dict) or value.get("schema") != "dual-qualification-v1":
        raise ValueError("Physical qualification receipt required")
    checks = value.get("checks", {})
    if set(checks) != NATIVE_CHECKS or any(v is not True for v in checks.values()):
        raise ValueError("All two-assembly native qualification checks must pass")
    if value.get("passed") is not True or value.get("safety_violations") != 0:
        raise ValueError("Qualified native safety is required")
    if value.get("profile") != "dual-payload-v1" or value.get("version") != "3.15.0":
        raise ValueError("Wrong physical profile")


def _benchmark(value):
    if not isinstance(value, dict) or value.get("schema") != "dual-benchmark-v1":
        raise ValueError("Measured end-to-end benchmark required")
    if value.get("repetitions") != 3 or value.get("transitions_per_repetition") != 4096:
        raise ValueError("Exactly three complete 4096-transition repetitions required")
    if value.get("epochs_per_update") != 5 or value.get("storage_feasible") is not True:
        raise ValueError("Complete update and measured storage feasibility required")
    if value.get("selected_backend") not in ("native-cpu", "native-cuda", "warp-cuda"):
        raise ValueError("A declared execution backend must be selected")
    if value.get("selected_backend") == "warp-cuda" and value.get("warp_fidelity36_passed") is not True:
        raise ValueError("GPU physics needs all36 paired fidelity trials")
    projection = value.get("whole_campaign_projection_seconds")
    if type(projection) not in (int, float) or not math.isfinite(projection) or projection <= 0:
        raise ValueError("Measured whole campaign projection required")


def _bound_receipt(binding, role, value):
    # Prevent accepting an unrelated in-memory receipt while binding other bytes.
    entries = binding.files[role]
    if len(entries) != 1 or json.loads(_path(binding.root, entries[0]["path"]).read_text()) != value:
        raise ValueError(f"{role} must equal the actual frozen receipt")
    if role in ("qualification", "benchmark"):
        expected = {key: binding.files[key] for key in EXPERIMENT_ROLES}
        if value.get("experiment_inputs") != expected:
            raise ValueError(f"{role} experiment inputs differ from frozen campaign inputs")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _declared_case_file(binding, filename):
    entries = binding.files["declaration"]
    declarations = [r for r in entries if Path(r["path"]).name == "declaration.json"]
    if len(declarations) != 1:
        raise ValueError("Exactly one prospective declaration required")
    record = declarations[0]
    raw = _path(binding.root, record["path"]).read_bytes()
    if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Frozen declaration changed")
    declaration = json.loads(raw)
    if (
        declaration.get("schema") != "dual-campaign-declaration-v1"
        or declaration.get("training_cases") != 10000
        or declaration.get("validation_cases") != 300
    ):
        raise ValueError("Full prospective training/validation300 declaration required")
    relative = str(Path(record["path"]).parent / filename)
    rows = [r for r in entries if r["path"] == relative]
    if len(rows) != 1 or declaration.get("files", {}).get(filename) != {
        k: rows[0][k] for k in ("bytes", "sha256")
    }:
        raise ValueError(f"Declared {filename} file must be frozen with its exact hash")
    return _path(binding.root, relative), rows[0]


def _validation_cases(binding):
    path, record = _declared_case_file(binding, "validation.jsonl")
    raw = path.read_bytes()
    if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Frozen validation300 manifest changed")
    cases = [validate_case(json.loads(line)) for line in raw.splitlines()]
    if (
        len(cases) != 300
        or len({c["case_id"] for c in cases}) != 300
        or {c["index"] for c in cases} != set(range(300))
        or any(c["split"] != 20 for c in cases)
        or Counter(c["scenario"] for c in cases)
        != dict.fromkeys(("normal", "priority", "availability-delay"), 100)
    ):
        raise ValueError("Actual validation300 requires unique split20 cases and100 per scenario")
    return {case["case_id"]: case for case in cases}


def _training_case_hashes(binding):
    """Parse the exact bound training bytes once, retaining only immutable hashes."""
    path, record = _declared_case_file(binding, "training.jsonl")
    scenarios = {"normal", "priority", "availability-delay"}
    digests, digest, size, count = {}, hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        for index, raw in enumerate(stream):
            if index >= 10000:
                raise ValueError("Exactly10000 training rows required")
            digest.update(raw)
            size += len(raw)
            row = json.loads(raw)
            if not isinstance(row, dict) or set(row) != scenarios:
                raise ValueError("Every training index requires all three scenarios")
            for scenario, case in row.items():
                validate_case(case)
                if (
                    case["index"] != index
                    or case["split"] != 10
                    or case["scenario"] != scenario
                    or not isinstance(case["case_id"], str)
                    or not case["case_id"]
                    or case["case_id"] in digests
                ):
                    raise ValueError("Unique split10 training cases with indices0..9999 required")
                digests[case["case_id"]] = _digest(case)
            count += 1
    if count != 10000:
        raise ValueError("Exactly10000 training rows required")
    if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError("Frozen training manifest changed")
    return MappingProxyType(digests)


def _validate_terminal_checkpoint(path, seed, *, expected_binding=None):
    """Validate trusted local checkpoint state without changing caller RNGs.

    This checks resumable learning state, not that physical training occurred.
    Physical state restoration is separately qualified by the environment adapter.
    """
    import random
    import numpy as np
    import torch
    from dataclasses import asdict
    from .dual_coordination_policy import JointActorCritic, PolicyConfig, PPOConfig, TrainingBudget

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = PPOConfig()
        budget = TrainingBudget(**state["budget"])
        if (
            state["schema"] != "dual-policy-v1"
            or budget.seed != seed
            or not budget.selectable
            or state["selectable"] is not True
            or state["policy_config"] != asdict(PolicyConfig())
            or state["ppo_config"] != asdict(config)
        ):
            raise ValueError("Wrong terminal policy/configuration/counters")
        backend = state["backend_state"]
        if expected_binding is not None:
            recovery=state.get('environment_state',{})
            if (recovery.get('schema')!='dual-training-recovery-v1'
                    or recovery.get('binding')!=expected_binding
                    or backend.get('learning_lock_sha256')!=expected_binding['learning_lock_sha256']
                    or backend.get('selected_backend')!=expected_binding['backend']):
                raise ValueError('Terminal checkpoint learning binding differs')
        if (
            backend["stage_transition_counts"] != [262144, 720896, 65536]
            or backend["stage3_backend"] != "native-mujoco"
            or state["physical_resume_available"] is not True
            or not isinstance(state["environment_state"], dict)
            or not state["environment_state"]
        ):
            raise ValueError("Exact physical stage accounting and recovery state required")
        with torch.random.fork_rng(devices=[]):
            model = JointActorCritic()
        expected = model.state_dict()
        weights = state["policy"]
        if set(weights) != set(expected):
            raise ValueError("Incomplete model weights and normalization")
        for name, tensor in weights.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype
                or not torch.isfinite(tensor).all()
                or (name.endswith("_scale") and not (tensor > 0).all())
            ):
                raise ValueError("Invalid model weights or normalization")
        model.load_state_dict(weights, strict=True)
        parameters = list(model.parameters())
        if state["parameter_count"] != sum(p.numel() for p in parameters):
            raise ValueError("Parameter count mismatch")
        optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        saved = state["optimizer"]
        groups = saved["param_groups"]
        expected_group = optimizer.state_dict()["param_groups"][0]
        if len(groups) != 1 or groups[0] != expected_group:
            raise ValueError("Frozen Adam configuration or parameter ordering differs")
        ids = groups[0]["params"]
        if set(saved["state"]) != set(ids):
            raise ValueError("Every parameter requires complete Adam moments")
        for key, parameter in zip(ids, parameters):
            moments = saved["state"][key]
            if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Incomplete Adam state")
            step = moments["step"]
            if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() != 10240:
                raise ValueError("Adam step disagrees with256 complete PPO updates")
            for name in ("exp_avg", "exp_avg_sq"):
                tensor = moments[name]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype
                    or not torch.isfinite(tensor).all()
                    or (name == "exp_avg_sq" and (tensor < 0).any())
                ):
                    raise ValueError("Invalid Adam moments")
        optimizer.load_state_dict(saved)
        torch.Generator().set_state(state["torch_rng"])
        np.random.RandomState().set_state(state["numpy_rng"])
        random.Random().setstate(state["python_rng"])
        cuda = state["cuda_rng"]
        if cuda is not None and (
            not isinstance(cuda, list)
            or not cuda
            or any(
                not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or t.ndim != 1 or t.numel() < 16
                for t in cuda
            )
        ):
            raise ValueError("Invalid CUDA RNG state")
        if any(
            not isinstance(state["versions"].get(k), str) or not state["versions"][k]
            for k in ("torch", "numpy")
        ):
            raise ValueError("Checkpoint library versions required")
    except (KeyError, TypeError, RuntimeError, AttributeError, IndexError) as exc:
        raise ValueError("Incomplete or unloadable terminal checkpoint") from exc
    return dict(
        sha256=sha256(path),
        transitions=budget.transitions,
        updates=budget.updates,
        stage3_transitions=backend["stage_transition_counts"][2],
    )


def _verify_retained_validation_bundle(bound, mission, development_reference):
    """Recompute the v2 receipt from one bounded archived capture, without physics.

    Only named, flat bundle members are copied into a temporary directory. No
    archived path is followed or restored into the campaign workspace. Original
    capture paths may have been retired after their bundle was verified.
    """
    import shutil
    import tempfile
    import zipfile
    from types import SimpleNamespace
    from .dual_coordination_evaluate import build_single_capture_receipts

    inventory = mission['replay']['inventory']
    record = inventory['compact_replay']
    if any(value != record for value in inventory.values()):
        raise ValueError('Single-capture v2 requires one complete retained bundle')
    try:
        with zipfile.ZipFile(_path(bound.root,record['path'])) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or '0/capture.json' not in names:
                raise ValueError('Duplicate or missing retained capture member')
            capture = json.loads(archive.read('0/capture.json'))
            if capture.get('schema') != 'dual-validation-capture-v1' or len(capture['replay_segments']) != 1:
                raise ValueError('One uninterrupted retained capture required')
            segment = capture['replay_segments'][0]
            filename = Path(segment['path']).name
            expected = {'0/'+name for name in ('capture.json','reset-inputs.json','proposed-actions.jsonl',filename)}
            if (not filename.endswith('.draw') or len(expected) != 4 or set(names) != expected
                    or any(member.is_dir() or member.flag_bits & 1 for member in archive.infolist())):
                raise ValueError('Exact flat reset/actions/capture/compact bundle required')
            with tempfile.TemporaryDirectory(prefix='dual-validation-stream-') as temporary:
                directory = Path(temporary)
                for name in expected - {'0/capture.json'}:
                    with archive.open(name) as source, (directory/Path(name).name).open('xb') as target:
                        shutil.copyfileobj(source,target,1024*1024)
                # Keep the declared bytes/hash; only relocate the retired source.
                segment['path'] = str(directory/filename)
                regenerated = build_single_capture_receipts(
                    SimpleNamespace(binding=bound),mission['result']['identity'],capture,
                    {'record':record},development_replay_reference=development_reference)
                if regenerated != mission or regenerated['replay']['retained_inputs_verified'] is not True:
                    raise ValueError('Retained stream differs from result/audit/replay receipts')
    except (OSError,zipfile.BadZipFile,KeyError,TypeError,AttributeError,IndexError) as error:
        raise ValueError('Incomplete or invalid retained validation bundle') from error


def _verify_validation_evidence(bound, selection, cases, manifests):
    """Authenticate the evaluator contract, separately from numeric selection.

    selection.validation_evidence names frozen dual-validation-evidence-v1 JSON
    documents. Their missions contain result, online audit, and replay receipts
    for each of five methods on each of the declared300 cases. Canonical JSON
    SHA-256 links the receipts; the replay inventory names retained reset inputs,
    actions and compact replay bytes. A single retained bundle may serve several
    inventory roles. Permanent RK/substep raw files are not required.

    This verifies provenance and completeness; it does not run a simulator or
    turn an administrative receipt into physical qualification. Producers must
    perform the online audit and may not leave proof gaps. v1 retains actual
    independent replay; v2 verifies one retained stream against an authenticated
    measured development replay reference without claiming another flight. Existing
    numeric-only reports and development captures do not satisfy this contract.
    """
    from .dual_coordination_selection import _outcomes

    bound.verify()
    if len(cases) != 300:
        raise ValueError("Declared validation300 required")
    paths = selection.get("validation_evidence")
    if not isinstance(paths, list) or not paths or len(set(paths)) != len(paths):
        raise ValueError("Bound validation evidence documents required")
    frozen = {r["path"]: r for r in bound.files["validation_evidence"]}
    if len(frozen) != len(bound.files["validation_evidence"]) or any(p not in frozen for p in paths):
        raise ValueError("Validation evidence paths must be uniquely frozen")
    expected = {}
    methods = {f"N{seed}": selection["candidates"][str(seed)] for seed in (42, 43, 44)}
    methods.update({"B-static": selection["static"], "B-dynamic": selection["dynamic"]})
    experiment_hash = _digest({role: bound.files[role] for role in EXPERIMENT_ROLES})
    for method, rows in methods.items():
        rows = _outcomes(rows)
        if {r["case_id"] for r in rows} != set(cases):
            raise ValueError("Selection cases differ from declared validation300")
        for outcome in rows:
            case = cases[outcome["case_id"]]
            if outcome["scenario"] != case["scenario"]:
                raise ValueError("Validation scenario differs from declared case")
            expected[method, case["case_id"]] = outcome
    seen, used = set(), set(paths)
    development_reference = None
    for path in paths:
        document = json.loads(_path(bound.root, path).read_text())
        if document.get("schema") != "dual-validation-evidence-v1" or not isinstance(
            document.get("missions"), list
        ):
            raise ValueError("Validation evaluator evidence schema required")
        for mission in document["missions"]:
            try:
                result, audit, replay = (mission[k] for k in ("result", "audit", "replay"))
                identity = result["identity"]
                key = identity["method"], identity["case_id"]
                if key not in expected or key in seen:
                    raise ValueError("Duplicate or undeclared validation mission")
                seen.add(key)
                method, case_id = key
                model_hash = (
                    manifests[int(method[1:])]["sha256"]
                    if method.startswith("N")
                    else _digest(bound.files["baseline"])
                )
                required_identity = dict(
                    method=method,
                    case_id=case_id,
                    case_sha256=_digest(cases[case_id]),
                    model_sha256=model_hash,
                    experiment_inputs_sha256=experiment_hash,
                )
                if any(r["identity"] != required_identity for r in (result, audit, replay)):
                    raise ValueError("Validation case/model/experiment identity mismatch")
                outcome = expected[key]
                tick = result["terminal_tick"]
                if (
                    result.get("schema") != "dual-validation-result-v1"
                    or result["outcome"] != outcome
                    or type(tick) is not int
                    or not 0 <= tick <= 30000
                    or (
                        outcome["success"]
                        and not math.isclose(outcome["team_time"], tick * 0.01, rel_tol=0, abs_tol=1e-9)
                    )
                ):
                    raise ValueError("Selection outcome differs from retained terminal result")
                coverage = dict(
                    control_steps=tick,
                    rk_states=tick * 64,
                    native_substep_intervals=tick * 16,
                    swept_intervals=tick * 16,
                )
                if (
                    audit.get("schema") != "dual-validation-audit-v1"
                    or audit["result_sha256"] != _digest(result)
                    or audit["auditor_sha256"] != _digest(bound.files["audit"])
                    or audit["outcome"] != outcome
                    or audit["complete"] is not True
                    or audit["errors"] != []
                    or audit["proof_gaps"] != []
                    or audit["coverage"] != coverage
                    or any(type(x) is not int for x in audit["coverage"].values())
                ):
                    raise ValueError("Complete bound online audit coverage required")
                inventory = replay["inventory"]
                if not isinstance(inventory, dict) or set(inventory) != {
                    "reset",
                    "actions",
                    "compact_replay",
                }:
                    raise ValueError("Reset/actions/compact replay inventory required")
                for record in inventory.values():
                    if (
                        not isinstance(record, dict)
                        or record.get("path") in paths
                        or frozen.get(record.get("path")) != record
                    ):
                        raise ValueError("Replay inventory differs from frozen retained bytes")
                    used.add(record["path"])
                if (
                    replay.get("schema") not in ("dual-validation-replay-v1", "dual-validation-replay-v2")
                    or replay["result_sha256"] != _digest(result)
                    or replay["audit_sha256"] != _digest(audit)
                    or replay["inventory_sha256"] != _digest(inventory)
                    or replay["configuration_sha256"] != _digest(bound.files["configuration"])
                    or type(replay["control_steps"]) is not int
                    or replay["control_steps"] != tick
                    or type(replay["reset_records"]) is not int
                    or replay["reset_records"] != 1
                    or replay["outcome"] != outcome
                ):
                    raise ValueError(
                        "Replay evidence must match result/audit and retained inventory"
                    )
                match_fields = ('passed','decisions_match','terminal_match','metrics_within_tolerance')
                if replay['schema'] == 'dual-validation-replay-v1':
                    if any(replay.get(k) is not True for k in match_fields):
                        raise ValueError('Actual independent deterministic replay must match')
                else:
                    if development_reference is None:
                        qualifications = bound.files['qualification']
                        if len(qualifications) != 1:
                            raise ValueError('One frozen development replay qualification required')
                        qualification = json.loads(_path(bound.root,qualifications[0]['path']).read_text())
                        development_reference = verified_development_replay_reference(bound,qualification)
                    if (replay.get('execution_mode') != 'retained-input-stream-validation'
                            or replay.get('native_reexecution_performed') is not False
                            or replay.get('development_replay_reference') != development_reference
                            or replay.get('retained_inputs_verified') is not True
                            or replay.get('compact_boundary_consistency') is not True
                            or replay.get('errors') != []
                            or type(replay.get('action_records')) is not int
                            or replay['action_records'] != outcome['decision_intervals']
                            or type(replay.get('retained_state_records')) is not int
                            or replay['retained_state_records'] < 1
                            or replay.get('reset_case_sha256') != required_identity['case_sha256']
                            or any(k in replay for k in match_fields)):
                        raise ValueError('Retained-input evidence or qualified development replay reference mismatch')
                    _verify_retained_validation_bundle(bound,mission,development_reference)
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError("Incomplete validation result/audit/replay receipt") from exc
    if seen != set(expected) or used != set(frozen):
        raise ValueError("Exactly1500 validation missions and their complete retained inventory required")


def create_learning_lock(path, *, root, roles, qualification, benchmark):
    if set(roles) != LEARNING_ROLES:
        raise ValueError("All and only declared learning artifact roles required")
    _qualification(qualification)
    _benchmark(benchmark)
    binding = BoundFiles.capture(root, roles)
    _bound_receipt(binding, "qualification", qualification)
    _bound_receipt(binding, "benchmark", benchmark)
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    _validation_cases(binding)
    receipt = dict(
        schema="dual-learning-lock-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        binding=binding.to_dict(),
        qualification=qualification,
        benchmark=benchmark,
        seeds=[42, 43, 44],
        stage_transitions=[262144, 720896, 65536],
        heldout_exposed=False,
    )
    atomic_json(target, receipt)
    return receipt


def _verify_learning_receipt(value, *, return_cases=False):
    if (
        not isinstance(value, dict)
        or value.get("schema") != "dual-learning-lock-v1"
        or value.get("seeds") != [42, 43, 44]
        or value.get("stage_transitions") != [262144, 720896, 65536]
        or value.get("heldout_exposed") is not False
    ):
        raise ValueError("Complete untouched learning lock required")
    binding = BoundFiles.from_dict(value["binding"])
    if set(binding.files) != LEARNING_ROLES:
        raise ValueError("Exact learning artifact roles required")
    _qualification(value.get("qualification"))
    _benchmark(value.get("benchmark"))
    _bound_receipt(binding, "qualification", value["qualification"])
    _bound_receipt(binding, "benchmark", value["benchmark"])
    cases = _validation_cases(binding)
    return (binding, cases) if return_cases else binding


class VerifiedLearningLock:
    """File-bound admission capability for split10/20 training and validation.

    Load only receipts whose qualification and benchmark were produced by the
    actual experiment runners. This class authenticates their frozen inputs;
    it does not execute or certify the physical experiments itself.
    """

    __slots__ = ("receipt", "path", "digest", "binding", "_case_digests")

    def __setattr__(self, name, value):
        if name == "_case_digests" and hasattr(self, name):
            raise AttributeError("Case admission cache is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name == "_case_digests":
            raise AttributeError("Case admission cache is immutable")
        object.__delattr__(self, name)

    def __init__(self, receipt, *, path=None, digest=None, token=None):
        if token is not _TOKEN:
            raise ValueError("Load a verified learning lock; direct construction is forbidden")
        self.receipt = copy.deepcopy(receipt)
        self.path, self.digest = Path(path), digest
        self.binding, cases = _verify_learning_receipt(receipt, return_cases=True)
        self._case_digests = MappingProxyType(
            {"validation": MappingProxyType({key: _digest(case) for key, case in cases.items()})}
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        value = json.loads(path.read_text())
        result = cls(value, path=path, digest=sha256(path), token=_TOKEN)
        return result.require_current()

    @property
    def selected_backend(self):
        return self.receipt["benchmark"]["selected_backend"]

    def verify_case(self, case, mode):
        """Return an isolated exact declared case for training or validation.

        Call require_current at construction/resume/phase boundaries. This hot
        path checks the immutable manifest identity loaded by that capability;
        it intentionally does not reread all campaign files on every reset.
        Training's10000 three-scenario rows are parsed lazily once per worker.
        Load a capability in each worker rather than sharing mutable case data.
        """
        if mode not in ("training", "validation"):
            raise ValueError("Case admission mode must be training or validation")
        validate_case(case)
        if case["split"] != (10 if mode == "training" else 20):
            raise ValueError("Case split differs from admission mode")
        if mode not in self._case_digests:
            # Recheck the capability before using its public binding to build
            # the lazy cache; normal reset calls never repeat this file scan.
            self.require_current()
            digests = _training_case_hashes(self.binding)
            object.__setattr__(
                self, "_case_digests", MappingProxyType({**self._case_digests, "training": digests})
            )
        if self._case_digests[mode].get(case["case_id"]) != _digest(case):
            raise ValueError("Case differs from the exact prospective manifest")
        return copy.deepcopy(case)

    def require_current(self):
        if sha256(self.path) != self.digest:
            raise ValueError("Learning lock receipt changed")
        value = json.loads(self.path.read_text())
        if value != self.receipt or value["binding"] != self.binding.to_dict():
            raise ValueError("Learning lock capability was mutated")
        self.binding.verify()
        if self.receipt['qualification'].get('compact_online_evidence_verified') is True:
            verified_development_replay_reference(self.binding,self.receipt['qualification'])
        return self


class VerifiedJointFreeze:
    def __init__(self, receipt, *, path=None, digest=None, token=None):
        if token is not _TOKEN:
            raise ValueError("Load a verified joint freeze receipt; direct construction is forbidden")
        self.receipt, self.path, self.digest = copy.deepcopy(receipt), Path(path), digest
        self.binding = BoundFiles.from_dict(receipt["binding"])

    @classmethod
    def load(cls, path):
        path = Path(path)
        value = json.loads(path.read_text())
        if (
            not isinstance(value, dict)
            or value.get("schema") != "dual-freeze-v1"
            or set(value.get("binding", {}).get("files", {})) != JOINT_ROLES
        ):
            raise ValueError("Complete joint freeze required")
        if value.get("selected_seed") not in (42, 43, 44) or value.get("selection_verified") is not True:
            raise ValueError("An eligible terminal seed is required")
        _qualification(value.get("qualification"))
        _benchmark(value.get("benchmark"))
        obj = cls(value, path=path, digest=sha256(path), token=_TOKEN)
        lock, result, manifests = _verify_joint_chain(obj.binding)
        lock_record = obj.binding.files["learning_lock"][0]
        if (
            value.get("learning_lock_sha256") != lock_record["sha256"]
            or value.get("qualification") != lock["qualification"]
            or value.get("benchmark") != lock["benchmark"]
            or value.get("selected_seed") != result["selected"]
            or value.get("terminal_manifests") != {str(k): v for k, v in manifests.items()}
        ):
            raise ValueError("Joint receipt differs from revalidated selection/checkpoint chain")
        obj.require_current()
        return obj

    def require_current(self):
        if sha256(self.path) != self.digest:
            raise ValueError("Joint freeze receipt changed")
        value = json.loads(self.path.read_text())
        if value != self.receipt or value["binding"] != self.binding.to_dict():
            raise ValueError("Joint freeze capability was mutated")
        self.binding.verify()
        if self.receipt['qualification'].get('compact_online_evidence_verified') is True:
            verified_development_replay_reference(self.binding,self.receipt['qualification'])
        return self


def verified_development_replay_reference(bound, qualification):
    """Authenticate a measured development replay for single-execution captures.

    The reference is linked by the frozen qualification JSON rather than placed
    in its own experiment-input hash, avoiding a circular content identity.
    This checks integrity/identity; the qualification producer must have reduced
    the actual paired native runs before emitting the qualification receipt.
    """
    from .dual_coordination_env import OBSERVATION_SCALES

    if (qualification.get('compact_online_evidence_verified') is not True
            or qualification.get('checks',{}).get('deterministic_replay') is not True):
        raise ValueError('Measured development replay qualification required')
    record=qualification.get('development_replay_reference',{})
    if not isinstance(record,dict) or set(record)!={'path','bytes','sha256'}:
        raise ValueError('Exact measured development replay reference required')
    path=_path(bound.root,record['path'])
    if record!=dict(path=record['path'],bytes=path.stat().st_size,sha256=sha256(path)):
        raise ValueError('Measured development replay reference changed')
    value=json.loads(path.read_text())
    names=('dual_coordination_env.py','dual_coordination_mission.py','dual_coordination_audit.py',
           'dual_coordination_reference.py','dual_coordination_safety.py','dual_payload_mujoco.py','payload_hybrid_controller.py')
    hashes={}
    for name in names:
        rows=[r for r in bound.files['source'] if Path(r['path']).name==name]
        if len(rows)!=1:
            raise ValueError('Exact frozen development replay executable required: '+name)
        hashes[name]=rows[0]['sha256']
    evidence=[r for r in bound.files['configuration'] if Path(r['path']).name=='dual_coordination_evidence.json']
    if len(evidence)!=1:
        raise ValueError('One prospective single-capture evidence profile required')
    profile=json.loads(_path(bound.root,evidence[0]['path']).read_text())
    if (profile.get('schema')!='dual-evidence-profile-v1' or profile.get('replays_per_mission')!=0
            or type(profile.get('replays_per_mission')) is not int or profile.get('replay_stride_ticks')!=20
            or profile.get('online_internal_audit') is not True):
        raise ValueError('Wrong single-capture evidence profile')
    expected=dict(experiment_inputs_sha256=_digest({r:bound.files[r] for r in EXPERIMENT_ROLES}),
                  source_sha256=_digest(bound.files['source']),audit_sha256=_digest(bound.files['audit']),
                  configuration_sha256=_digest(bound.files['configuration']),
                  profile_sha256=_digest(dict(scales=OBSERVATION_SCALES,source_sha256=hashes,profile='dual-payload-v1')),
                  audit_library_sha256=qualification.get('audit_library_sha256'),
                  evidence_profile_sha256=evidence[0]['sha256'])
    if (value.get('schema')!='dual-development-replay-qualification-v1'
            or any(value.get(k) is not True for k in ('passed','native_reexecution_performed','decisions_match','terminal_match','metrics_within_tolerance'))
            or type(value.get('decision_count')) is not int or value['decision_count']<1
            or type(value.get('checked_states')) is not int or value['checked_states']<1
            or type(value.get('maximum_state_absolute_error')) not in (int,float)
            or not math.isfinite(value['maximum_state_absolute_error'])
            or not 0<=value['maximum_state_absolute_error']<=1e-12):
        raise ValueError('Complete measured paired native replay required')
    if any(value.get(k)!=v for k,v in expected.items()) or qualification.get('evidence_profile_sha256')!=expected['evidence_profile_sha256']:
        raise ValueError('Measured development replay differs from frozen source/profile/library/configuration')
    return copy.deepcopy(record)


def expected_terminal_binding(bound, learning_lock_sha, benchmark, seed):
    """Identity shared by production, validation and the final joint freeze."""
    from dataclasses import asdict
    from .dual_coordination_policy import PolicyConfig

    workers=benchmark.get('selected_workers')
    if type(workers) is not int or not 1<=workers<=8:
        raise ValueError('Measured selected worker count required')
    if seed not in (42,43,44) or benchmark.get('selected_backend') not in ('native-cpu','native-cuda','warp-cuda'):
        raise ValueError('Finite seed and selected backend required')
    records={
        'original_sha256':[r for r in bound.files['assets'] if Path(r['path']).name=='cf2x.urdf'],
        'controller_sha256':bound.files['controller'],
        'manifest_sha256':[r for r in bound.files['declaration'] if Path(r['path']).name=='training.jsonl']}
    if any(len(rows)!=1 for rows in records.values()):
        raise ValueError('Exactly one frozen original/controller/training manifest required')
    return dict(learning_lock_sha256=learning_lock_sha,workers=workers,backend=benchmark['selected_backend'],
                seed=seed,policy_config=asdict(PolicyConfig()),
                **{name:rows[0]['sha256'] for name,rows in records.items()})


def _verify_joint_chain(bound):
    from .dual_coordination_selection import select_candidate, validate_terminal_checkpoints

    def single_json(role):
        entries = bound.files[role]
        if len(entries) != 1:
            raise ValueError(f"Exactly one {role} receipt required")
        return json.loads(_path(bound.root, entries[0]["path"]).read_text())

    lock = single_json("learning_lock")
    old = _verify_learning_receipt(lock)
    if any(bound.files[role] != old.files[role] for role in LEARNING_ROLES):
        raise ValueError("Learning configuration changed before joint freeze")
    _qualification(lock.get("qualification"))
    _benchmark(lock.get("benchmark"))
    _bound_receipt(bound, "qualification", lock["qualification"])
    _bound_receipt(bound, "benchmark", lock["benchmark"])
    cases = _validation_cases(bound)
    selection = single_json("selection")
    _bound_receipt(bound, "selection", selection)
    if selection.get("schema") != "dual-selection-bound-v1":
        raise ValueError("Selection must retain authenticated complete validation inputs")
    result = select_candidate(
        {int(k): v for k, v in selection["candidates"].items()}, selection["static"], selection["dynamic"]
    )
    if result["selected"] is None or selection.get("selected_seed") != result["selected"]:
        raise ValueError("No eligible model or selection mismatch")
    manifests = {}
    for seed in (42, 43, 44):
        entries = bound.files[f"terminal{seed}"]
        if len(entries) != 1:
            raise ValueError("Exactly one terminal checkpoint per seed required")
        p = _path(bound.root, entries[0]["path"])
        manifests[seed] = _validate_terminal_checkpoint(p, seed, expected_binding=expected_terminal_binding(
            old, bound.files["learning_lock"][0]["sha256"], lock["benchmark"], seed))
        if selection.get("checkpoint_sha256", {}).get(str(seed)) != sha256(p):
            raise ValueError("Validation model hash differs from terminal checkpoint")
    validate_terminal_checkpoints(manifests)
    _verify_validation_evidence(bound, selection, cases, manifests)
    return lock, result, manifests


def create_joint_freeze(path, *, learning_lock, root, roles, selection):
    """Selection receipt must contain all authenticated validation inputs.

    Recompute numerical selection here; mere selected_seed metadata is invalid.
    Terminal checkpoint bytes are opened only from trusted local campaign files.
    """
    if set(roles) != JOINT_ROLES:
        raise ValueError("Every joint role required")
    bound = BoundFiles.capture(root, roles)
    entries = bound.files["learning_lock"]
    if len(entries) != 1 or _path(bound.root, entries[0]["path"]).resolve() != Path(learning_lock).resolve():
        raise ValueError("Exact learning-lock file must be bound")
    _bound_receipt(bound, "selection", selection)
    lock, result, manifests = _verify_joint_chain(bound)
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    receipt = dict(
        schema="dual-freeze-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        binding=bound.to_dict(),
        qualification=lock["qualification"],
        benchmark=lock["benchmark"],
        selected_seed=result["selected"],
        selection_verified=True,
        terminal_manifests=manifests,
        learning_lock_sha256=sha256(learning_lock),
        heldout_exposed=False,
    )
    atomic_json(target, receipt)
    return VerifiedJointFreeze.load(target)


def _heldout_case_lists(freeze, *, layout_path):
    from .dual_coordination_cases import generate_case, canonical_case, keyed_rng

    if type(freeze) is not VerifiedJointFreeze:
        raise ValueError("Verified joint freeze required")
    freeze.require_current()
    layout_entries = freeze.binding.files["layout"]
    if (
        len(layout_entries) != 1
        or _path(freeze.binding.root, layout_entries[0]["path"]).resolve() != Path(layout_path).resolve()
    ):
        raise ValueError("Canonical layout must be the exact frozen input")
    scenarios = ("normal", "priority", "availability-delay")
    routing = [generate_case(30, i, scenarios[i // 100], freeze=freeze) for i in range(300)]
    physical = []
    for scenario in scenarios:
        for side in (0.03, 0.045, 0.06):
            for group, ratios in (("basic", (0.2,)), ("endpoints", (0.1, 0.3))):
                for ratio in ratios:
                    case = canonical_case(layout_path, scenario, index=len(physical))
                    case.update(case_id=f"physical-{group}-{scenario}-{side}-{ratio}", split=40)
                    case["loads"] = [dict(side_m=side, mass_ratio=ratio) for _ in range(2)]
                    if group == "endpoints":
                        for d, initial in enumerate(case["initials"]):
                            initial["swing_deg"] = 10.0
                            initial["direction_rad"] = float(
                                keyed_rng(40, case["index"], f"initial{d + 1}").uniform(0, 6.283185307179586)
                            )
                    physical.append(dict(group=group, case=validate_case(case)))
    pairs = [(a, b) for a in (0.03, 0.045, 0.06) for b in (0.03, 0.045, 0.06)] * 11 + [(0.03, 0.03)]
    for i, pair in enumerate(pairs):
        scenario = scenarios[0 if i < 34 else 1 if i < 67 else 2]
        case = generate_case(40, i, scenario, freeze=freeze)
        for load, side in zip(case["loads"], pair):
            load["side_m"] = side
        physical.append(dict(group="mixed", case=validate_case(case)))
    return routing, physical


def generate_heldout(output, freeze, *, layout_path):
    routing, physical = _heldout_case_lists(freeze, layout_path=layout_path)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "routing300.json", routing)
    atomic_json(root / "physical127.json", physical)
    atomic_json(
        root / "manifest.json",
        dict(
            schema="dual-heldout-manifest-v1",
            freeze_sha256=freeze.digest,
            opened_at=datetime.now(timezone.utc).isoformat(),
            routing_cases=300,
            physical_cases=127,
            files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(root.iterdir())},
        ),
    )
    return root


class VerifiedHeldout:
    """Runtime access to exact post-freeze cases, never a model-selection input.

    On load, regenerate the declared deterministic cohort only after validating
    the complete joint freeze. Authenticate case bytes against both that recipe
    and the original generation manifest, then cache immutable per-case digests.
    """

    __slots__ = ("freeze", "path", "digest", "receipt", "binding", "_case_digests")

    def __setattr__(self, name, value):
        if name == "_case_digests" and hasattr(self, name):
            raise AttributeError("Case admission cache is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name == "_case_digests":
            raise AttributeError("Case admission cache is immutable")
        object.__delattr__(self, name)

    def __init__(self, root, *, freeze=None, token=None):
        if token is not _TOKEN:
            raise ValueError("Load a verified held-out runtime capability")
        if type(freeze) is not VerifiedJointFreeze:
            raise ValueError("Verified joint freeze required")
        freeze.require_current()
        root = Path(root).resolve()
        self.freeze, self.path = freeze, _path(root, "manifest.json")
        self.digest, self.receipt = sha256(self.path), json.loads(self.path.read_text())
        self.binding = freeze.binding
        layout = self.binding.files["layout"]
        if len(layout) != 1:
            raise ValueError("Exact frozen layout required")
        routing, physical = _heldout_case_lists(freeze, layout_path=_path(self.binding.root, layout[0]["path"]))
        expected = {"routing300.json": routing, "physical127.json": physical}
        receipt = self.receipt
        if (receipt.get("schema") != "dual-heldout-manifest-v1"
                or receipt.get("freeze_sha256") != freeze.digest
                or receipt.get("routing_cases") != 300 or receipt.get("physical_cases") != 127
                or set(receipt.get("files", {})) != set(expected)):
            raise ValueError("Complete post-freeze held-out generation manifest required")
        opened = datetime.fromisoformat(receipt["opened_at"])
        frozen = datetime.fromisoformat(freeze.receipt["created_at"])
        if opened.tzinfo is None or frozen.tzinfo is None or opened < frozen:
            raise ValueError("Held-out generation must follow joint freeze")
        for name, cases in expected.items():
            path = _path(root, name)
            if json.loads(path.read_text()) != cases:
                raise ValueError("Held-out cases differ from frozen deterministic recipe")
        maps = {"routing-heldout": {c["case_id"]: _digest(c) for c in routing},
                "physical-heldout": {r["case"]["case_id"]: _digest(r["case"]) for r in physical}}
        if len(maps["routing-heldout"]) != 300 or len(maps["physical-heldout"]) != 127:
            raise ValueError("Held-out IDs must be unique within each cohort")
        self._case_digests = MappingProxyType({mode: MappingProxyType(rows) for mode, rows in maps.items()})
        self.require_current()

    @classmethod
    def load(cls, root, *, freeze):
        if type(freeze) is not VerifiedJointFreeze:
            raise ValueError("Verified joint freeze required")
        return cls(root, freeze=freeze, token=_TOKEN)

    @property
    def selected_backend(self):
        return self.freeze.receipt["benchmark"]["selected_backend"]

    def require_current(self):
        self.freeze.require_current()
        if sha256(self.path) != self.digest or json.loads(self.path.read_text()) != self.receipt:
            raise ValueError("Held-out manifest changed")
        for name, record in self.receipt["files"].items():
            path = _path(self.path.parent, name)
            if record != dict(bytes=path.stat().st_size, sha256=sha256(path)):
                raise ValueError("Held-out cohort changed")
        return self

    def verify_case(self, case, mode):
        validate_case(case)
        if mode not in self._case_digests or case["split"] != (30 if mode == "routing-heldout" else 40):
            raise ValueError("Held-out mode and split must match")
        if self._case_digests[mode].get(case["case_id"]) != _digest(case):
            raise ValueError("Case differs from exact post-freeze held-out manifest")
        return copy.deepcopy(case)
