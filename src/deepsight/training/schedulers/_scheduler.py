##
##
##

import abc

from torch.optim import Optimizer

from deepsight.training import EpochPhaseTimestamp


class LRScheduler(abc.ABC):
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer) -> None:
        super().__init__()

        self._optimizer = optimizer

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    @abc.abstractmethod
    def compute_lrs(self, timestamp: EpochPhaseTimestamp) -> tuple[float, ...]:
        """Compute the learning rates for the current step.

        Args:
            timestamp: The timestamp of the current phase.
        """
        ...

    def step(self, timestamp: EpochPhaseTimestamp) -> None:
        """Update the learning rate of the optimizer.

        Differently from the PyTorch convention of calling the method `step` after the
        optimizer's `step` method, this method is called before the optimizer's `step`
        method. That is, this method should set the learning rate of the optimizer for
        the current step.

        Args:
            timestamp: The timestamp of the current phase.
        """
        lrs = self.compute_lrs(timestamp)

        for param_group, lr in zip(self._optimizer.param_groups, lrs, strict=True):
            param_group["lr"] = lr
