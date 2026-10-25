"""Qualified inference shares the actual training runtime and motor semantics."""

import copy
from pathlib import Path

from .payload_checkpoint import actor_signature
from .payload_contract import Payload
from .payload_recovery_runtime import RecoveryPayloadFlight
from .payload_retry_contract import OBSERVATION_CONTRACT
from .payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG
from .payload_solver_policy import mean_policy
from .payload_solver_qualification import (
    current_solver_binding,
    require_qualification,
    physical_errors,
    QUALIFICATION_SCHEMA,
)
from .payload_solver_runtime import SolverRuntime
from .util import sha256
from .jobs import write_json

_ADMISSION_TOKEN = object()


class SolverAdmission:
    """Process-local capability created only by complete raw reconstruction.

    A stage consumes qualification once, then checks actual source/build and
    native properties on every case. Serialized PASS labels cannot create it.
    """

    def __init__(self, proof, token):
        if token is not _ADMISSION_TOKEN:
            raise ValueError("Complete raw admission capability required")
        self.proof = copy.deepcopy(proof)
        self.binding = copy.deepcopy(proof["actual_binding"])

    @classmethod
    def from_receipt(cls, receipt, locations):
        if receipt.get("schema") != QUALIFICATION_SCHEMA:
            raise ValueError("Complete actual raw qualification required")
        binding = current_solver_binding()
        proof = require_qualification(receipt, binding, locations)
        return cls(proof, _ADMISSION_TOKEN)

    def require_current(self):
        if current_solver_binding() != self.binding:
            raise ValueError("Actual consumer source/build changed after raw admission")


class SolverFlight(RecoveryPayloadFlight):
    """Same mission surface as recovery, backed by the qualified RK4 runtime."""

    @classmethod
    def from_checkpoint(
        cls,
        *,
        admission,
        checkpoint,
        expected_hash,
        original,
        case,
        output,
        allow_development=False,
        expected_method=None,
    ):
        if not isinstance(admission, SolverAdmission):
            raise ValueError("Raw-reconstructed process admission required")
        admission.require_current()
        checkpoint = Path(checkpoint)
        if sha256(checkpoint) != expected_hash:
            raise ValueError("Actual checkpoint bytes differ")
        if sha256(original) != admission.binding["original_asset_sha256"]:
            raise ValueError("Actual original asset differs")
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        info = state.get("infos", {})
        if (
            expected_method not in {"randomized", "fixed"}
            or info.get("method") != expected_method
            or info.get("observation_contract") != OBSERVATION_CONTRACT
            or info.get("actual_solver_binding") != admission.binding
            or info.get("training_split")
            not in (["final", "development"] if allow_development else ["final"])
        ):
            raise ValueError("Checkpoint physics/observation/split contract differs")
        root = Path(output)
        root.mkdir(parents=True, exist_ok=False)
        payload = Payload(FREE_MASS_KG, case["mass_ratio"], case["side_m"])
        fixture = MuJoCoFixture(original, root / "fixture", payload, substeps=16)
        errors = physical_errors(
            fixture.runtime(),
            dict(side_m=payload.side_m, mass_ratio=payload.mass_ratio, substeps=16),
            fixed=False,
        )
        if errors:
            raise ValueError("Actual inference native properties differ: " + ",".join(errors))
        write_json(root / "runtime-properties.json", fixture.runtime())
        core = SolverRuntime(fixture, root / "runtime")
        core.reset_case(initial=case["initial"], target=[0, 0, 1])
        flight = cls(core, mean_policy(state["actor_state_dict"]))
        flight.actor_sha256 = actor_signature(state["actor_state_dict"])
        flight.checkpoint_sha256 = expected_hash
        flight.config_sha256 = info["common_training_plan"]["parent_config_sha256"]
        flight.actual_execution_binding = copy.deepcopy(admission.binding)
        flight.admission = admission
        return flight

    def close(self):
        self.runtime.close()
