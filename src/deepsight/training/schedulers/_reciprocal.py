##
##
##

from collections.abc import Sequence

from torch.optim import Optimizer

from deepsight.training.structs import EpochPhaseTimestamp
from deepsight.typing import Configurable
from deepsight.typing._types import Configs

from ._scheduler import LRScheduler


class ReciprocalLR(LRScheduler, Configurable):
    r"""A learning rate scheduler that uses a reciprocal schedule.

    The learning rate is increased linearly from 0 to the maximum learning rate
    during the warmup period, if any.

    During the main training period, the learning rate is decreased proportionally
    to the inverse of the step number. That is, the learning rate at step t is
    equal to:

    $$
        lr_t = max_lr * \\sqrt{\\frac{w}{t}}
    $$

    where `max_lr` is the maximum learning rate, `w` is the number of warmup steps,
    and `t` is the current step number. If there are no warmup steps, `w` is set to 1
    so that the learning rate is equal to `max_lr` at the first step.

    !!! note

        This scheduler does not require the number of training steps, so it can be used
        to train the model for an arbitrary number of steps.
    """

    def __init__(
        self, optimizer: Optimizer, max_lr: float | Sequence[float], warmup_steps: int
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer : The wrapped optimizer.
            max_lr : The maximum learning rate. If a list is provided, it must have the
                same length as the number of parameter groups in the optimizer.
            warmup_steps: The number of warmup steps. If 0, no warmup is performed.
        """
        if isinstance(max_lr, (float, int)):
            max_lr = [float(max_lr)] * len(optimizer.param_groups)
        elif len(max_lr) != len(optimizer.param_groups):
            raise ValueError(
                "Expected max_lr to be a float or a sequence of floats "
                f"with length {len(optimizer.param_groups)}."
            )

        if warmup_steps < 0:
            raise ValueError("Expected warmup_duration to be non-negative.")

        super().__init__(optimizer)
        self._max_lr = tuple(max_lr)
        self._warmup_steps = warmup_steps + 1
        self._warmup_deltas = tuple(lr / warmup_steps for lr in self._max_lr)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def compute_lrs(self, timestamp: EpochPhaseTimestamp) -> tuple[float, ...]:
        step = timestamp.num_batches
        if step < self._warmup_steps:
            # we are in the warmup period
            return tuple(delta * (step + 1) for delta in self._warmup_deltas)

        # we are in the main training period
        factor = (self._warmup_steps / (step + 1)) ** 0.5
        return tuple(lr * factor for lr in self._max_lr)

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "max_lr": self._max_lr,
            "warmup_steps": self._warmup_steps,
        }

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(max_lr={self._max_lr}, "
            f"warmup_duration={self._warmup_steps})"
        )

    def __repr__(self) -> str:
        return str(self)
