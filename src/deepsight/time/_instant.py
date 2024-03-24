# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochInstant:
    """An epoch during training.

    !!! note

        An instant is an exact point in time, not a range. Thus, this class represents
        the start of an epoch, not the entire epoch.

    Attributes:
        epoch: The epoch number.
    """

    epoch: int

    def __post_init__(self) -> None:
        if self.epoch < 0:
            msg = "The epoch number must be non-negative."
            raise ValueError(msg)


@dataclass(frozen=True)
class StepInstant:
    """A step during training.

    Attributes:
        step: The step number.
        from_current_epoch: Whether the step should be counted from the start of the
            current epoch or from the start of training.
    """

    step: int
    from_current_epoch: bool = False

    def __post_init__(self) -> None:
        if self.step < 0:
            msg = "The step number must be non-negative."
            raise ValueError(msg)


@dataclass(frozen=True)
class SampleInstant:
    """A sample during training.

    Attributes:
        sample: The sample number.
        from_current_epoch: Whether the sample should be counted from the start of the
            current epoch or from the start of training.
    """

    sample: int
    from_current_epoch: bool = False

    def __post_init__(self) -> None:
        if self.sample < 0:
            msg = "The sample number must be non-negative."
            raise ValueError(msg)


type Instant = EpochInstant | StepInstant | SampleInstant
"""An instant of time during training."""
