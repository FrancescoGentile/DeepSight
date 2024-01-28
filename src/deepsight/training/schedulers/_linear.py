# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from torch.optim import Optimizer

from deepsight.training import EpochPhaseTimestamp
from deepsight.typing import Configurable, StateDict, Stateful
from deepsight.typing._types import Configs

from ._scheduler import LRScheduler


class LinearLR(LRScheduler, Configurable, Stateful):
    """Linearly decays the learning rate over a fixed number of steps.

    The learning rate is decreased linearly from the initial learning rate (taken
    from the optimizer) to 0 over the given number of steps. The learning rate is
    not decayed after the given number of steps (i.e. it remains at 0).
    """

    def __init__(self, optimizer: Optimizer, steps: int) -> None:
        """Initialize the scheduler.

        Args:
            optimizer : The wrapped optimizer.
            steps : The number of steps to linearly decay the learning rate.
        """
        if steps <= 0:
            raise ValueError("Steps must be positive.")

        super().__init__(optimizer)
        self._steps = steps
        self._start_step = -steps
        self._start_lrs = tuple(
            param_group["lr"] for param_group in self._optimizer.param_groups
        )

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def compute_lrs(self, timestamp: EpochPhaseTimestamp) -> tuple[float, ...]:
        step = timestamp.num_batches
        if step - self._start_step >= self._steps:
            self._start_step = step
            self._start_lrs = tuple(
                param_group["lr"] for param_group in self._optimizer.param_groups
            )

        factor = 1 - (step - self._start_step) / self._steps
        return tuple(lr * factor for lr in self._start_lrs)

    def get_configs(self, recursive: bool) -> Configs:
        return {"steps": self._steps}

    def state_dict(self) -> StateDict:
        return {
            "start_step": self._start_step,
            "start_lrs": self._start_lrs,
        }

    def load_state_dict(self, state_dict: StateDict) -> Any:
        self._start_step = state_dict["start_step"]
        self._start_lrs = state_dict["start_lrs"]

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(steps={self._steps})"

    def __repr__(self) -> str:
        return str(self)
