"""Frozen diagnostics for supported settings; no optimizer admission."""

from .payload_recovery_qualification import (
    make_manifest as recovery_manifest,
    SOURCE_FILES as RECOVERY_SOURCES,
)

SCHEMA = "payload-solver-domain-v1"
SOURCE_FILES = (
    *RECOVERY_SOURCES,
    "src/adaptive_drone_patrol/payload_solver_candidates.py",
    "src/adaptive_drone_patrol/payload_solver_reference.py",
)


def make_manifest(
    legacy, *, base_substeps=16, precision="64", integrator="implicitfast", execution_backend="cpu"
):
    """Keep all inherited conditions; numerical settings are explicit candidates."""
    if execution_backend not in ("cpu", "gpu"):
        raise ValueError("Unsupported execution backend")
    if precision not in ("32", "64") or integrator not in ("implicitfast", "approximate_implicitfast"):
        raise ValueError("Unsupported precision or integrator")
    result = recovery_manifest(legacy, base_substeps=base_substeps)
    result.update(schema=SCHEMA, profile=f"cf2x-solver-{integrator}-f{precision}-s{base_substeps}-v1")
    result["settings"].update(precision=precision, integrator=integrator, execution_backend=execution_backend)
    result["limits"]["absolute_unexplained_passive_energy_fraction"] = 0.05
    result["limits"]["unexplained_passive_energy_loss_fraction"] = 0.05
    return result


def validate_manifest(manifest):
    """Reject changed physical conditions and weakened conservative limits."""
    import json
    from .payload_recovery_qualification import ROOT, LEGACY

    try:
        expected = make_manifest(
            json.loads((ROOT / LEGACY).read_text()),
            base_substeps=manifest["base_substeps"],
            precision=manifest["settings"]["precision"],
            integrator=manifest["settings"]["integrator"],
            execution_backend=manifest["settings"]["execution_backend"],
        )
        if manifest != expected:
            raise ValueError("Candidate differs from its complete frozen template")
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid solver candidate") from exc
    return True
