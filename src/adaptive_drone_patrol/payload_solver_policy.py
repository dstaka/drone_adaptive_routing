"""Exact deterministic 128/128 Tanh actor; inference never creates an optimizer."""

import numpy as np
from .payload_retry_checkpoint import _validate_tensors


def mean_policy(actor_state):
    """Load the existing actor MLP weights without changing motor semantics."""
    import torch

    _validate_tensors(actor_state, "actor", 29)
    network = (
        torch.nn.Sequential(
            torch.nn.Linear(29, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 4),
        )
        .cpu()
        .eval()
    )
    state = {
        key.removeprefix("mlp."): value.cpu() for key, value in actor_state.items() if key.startswith("mlp.")
    }
    network.load_state_dict(state, strict=True)

    def infer(observation):
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (29,) or not np.isfinite(values).all():
            raise ValueError("Finite exact29 measured actor fields required")
        with torch.inference_mode():
            result = network(torch.as_tensor(values)).numpy()
        if result.shape != (4,) or not np.isfinite(result).all():
            raise ValueError("Invalid four-motor actor output")
        return result.copy()

    return infer
