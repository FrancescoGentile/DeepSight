# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence

from torch.optim import Optimizer

from deepsight.time import PhaseTimestamp

from ._base import LRScheduler


class LambdaLR(LRScheduler):
    """A learning rate scheduler that uses a custom schedule.

    The learning rate at each step is computed as the product of the initial learning
    rate and the output of a custom function that takes the current timestamp as input.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lambdas: Callable[[PhaseTimestamp], float]
        | Sequence[Callable[[PhaseTimestamp], float]],
    ) -> None:
        """Initializes the scheduler.

        Args:
            optimizer: The wrapped optimizer.
            lambdas: A function or a list of functions that compute the learning rate
                multiplier at each step. If a single function is provided, it is used
                for all parameter groups. If a list is provided, it must have the same
                length as the number of parameter groups in the optimizer.

        Raises:
            ValueError: If the number of lambdas does not match the number of parameter
                groups in the optimizer.
        """
        super().__init__(optimizer)

        if callable(lambdas):
            lambdas = [lambdas] * len(self._optimizer.param_groups)

        if len(lambdas) != len(self._optimizer.param_groups):
            msg = (
                f"Expected {len(self._optimizer.param_groups)} lambdas, "
                f"but got {len(lambdas)}."
            )
            raise ValueError(msg)

        self._lambdas = lambdas
        self._start_lrs = tuple(group["lr"] for group in self._optimizer.param_groups)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def compute_lrs(self, timestamp: PhaseTimestamp) -> tuple[float, ...]:
        return tuple(
            lr * lambda_(timestamp)
            for lr, lambda_ in zip(self._start_lrs, self._lambdas, strict=True)
        )

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
