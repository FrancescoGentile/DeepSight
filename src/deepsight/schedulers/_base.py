# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import abc

from torch.optim import Optimizer

from deepsight.time import PhaseTimestamp


class LRScheduler(abc.ABC):
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer) -> None:
        super().__init__()

        self._optimizer = optimizer

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def step(self, timestamp: PhaseTimestamp) -> None:
        """Updates the learning rate of the optimizer.

        Differently from the PyTorch convention of calling the method `step` after the
        optimizer's `step` method, this method is called before the optimizer's `step`
        method. That is, this method should set the learning rate of the optimizer for
        the current step.

        Args:
            timestamp: The timestamp of the current phase.
        """
        lrs = self.compute_lrs(timestamp=timestamp)

        for param_group, lr in zip(self._optimizer.param_groups, lrs, strict=True):
            param_group["lr"] = lr

    @abc.abstractmethod
    def compute_lrs(self, timestamp: PhaseTimestamp) -> tuple[float, ...]:
        """Computes the learning rates for the current step.

        Args:
            timestamp: The timestamp of the current phase.
        """
        ...
