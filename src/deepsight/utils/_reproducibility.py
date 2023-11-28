##
##
##

import random
from typing import Any

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict[str, Any]:
    """Get the random number generator state."""
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": {
            "cuda": torch.cuda.get_rng_state_all(),
            "cpu": torch.get_rng_state(),
        },
    }


def set_rng_state(state: dict[str, Any]) -> None:
    """Set the random number generator state."""
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.cuda.set_rng_state_all(state["torch"]["cuda"])
    torch.set_rng_state(state["torch"]["cpu"])
