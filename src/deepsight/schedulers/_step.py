# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from torch.optim import Optimizer

from deepsight.time import Instant, PhaseTimestamp
from deepsight.typing import Configurable

from ._base import LRScheduler


class StepLR(LRScheduler, Configurable):
    """A learning rate scheduler with a stepwise decay.

    The learning rate is decayed by the given gamma factor at regular intervals.
    """

    def __init__(self, optimizer: Optimizer, interval: Instant, gamma: float) -> None:
        """Initializes the scheduler.

        Args:
            optimizer: The wrapped optimizer.
            interval: The interval at which the learning rate is decayed.
            gamma: The factor by which the learning rate is decayed.
        """
        super().__init__(optimizer)

        self._interval = interval
        self._gamma = gamma

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def compute_lrs(self, timestamp: PhaseTimestamp) -> tuple[float, ...]:
        lrs = tuple(group["lr"] for group in self._optimizer.param_groups)
        if timestamp % self._interval == 0:
            lrs = tuple(lr * self._gamma for lr in lrs)

        return lrs

    def get_config(self, recursive: bool) -> dict[str, Any]:
        return {
            "interval": repr(self._interval),
            "gamma": self._gamma,
        }

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(interval={self._interval}, gamma={self._gamma})"
        )
