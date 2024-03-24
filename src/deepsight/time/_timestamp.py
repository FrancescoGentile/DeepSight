# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/mosaicml/composer/blob/dev/composer/core/time.py
# --------------------------------------------------------------------------- #

from typing import Any

from deepsight.typing import Stateful

from ._instant import EpochInstant, Instant, SampleInstant, StepInstant


class PhaseTimestamp(Stateful):
    """The timestamp of a phase."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(self, steps_per_epoch: int, samples_per_epoch: int) -> None:
        """Initializes the timestamp."""
        self._epoch = 0
        self._step_in_epoch = 0
        self._sample_in_epoch = 0
        self._steps_per_epoch = steps_per_epoch
        self._samples_per_epoch = samples_per_epoch

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_epochs(self) -> int:
        """The number of epochs in which the phase was fully run.

        !!! warning

            Since a phase does not necessarily run every epoch, this may not
            correspond to the number of epochs fully run by the training engine
            (i.e. the index of the current epoch).
        """
        return self._epoch

    @property
    def num_steps(self) -> int:
        """The number of steps performed since the beginning of training."""
        num_steps = self._epoch * self._steps_per_epoch
        num_steps += self._step_in_epoch

        return num_steps

    @property
    def num_samples(self) -> int:
        """The number of samples fully processed since the beginning of training."""
        num_samples = self._epoch * self._samples_per_epoch
        num_samples += self._sample_in_epoch

        return num_samples

    @property
    def num_steps_in_epoch(self) -> int:
        """The number of steps performed in the current epoch."""
        return self._step_in_epoch

    @property
    def num_samples_in_epoch(self) -> int:
        """The number of samples fully processed in the current epoch."""
        return self._sample_in_epoch

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def next_step(self, batch_size: int) -> None:
        """Increments the step counter."""
        self._step_in_epoch += 1
        self._sample_in_epoch += batch_size

        if self._step_in_epoch > self._steps_per_epoch:
            msg = "Trying to increment the step counter beyond the number of steps per epoch."  # noqa: E501
            raise RuntimeError(msg)

        if self._sample_in_epoch > self._samples_per_epoch:
            msg = "Trying to increment the sample counter beyond the number of samples per epoch."  # noqa: E501
            raise RuntimeError(msg)

    def terminate_epoch(self) -> None:
        """Terminates the current epoch."""
        if self._step_in_epoch != self._steps_per_epoch:
            msg = (
                f"The number of steps processed ({self._step_in_epoch}) in the current "
                "epoch does not match the expected number of steps per epoch"
                f"({self._steps_per_epoch})."
            )
            raise RuntimeError(msg)

        if self._sample_in_epoch != self._samples_per_epoch:
            msg = (
                f"The number of samples processed ({self._sample_in_epoch}) in the "
                "current epoch does not match the expected number of samples per epoch"
                f"({self._samples_per_epoch})."
            )
            raise RuntimeError(msg)

        self._epoch += 1
        self._step_in_epoch = 0
        self._sample_in_epoch = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_epochs": self._epoch,
            "batch_in_epoch": self._step_in_epoch,
            "sample_in_epoch": self._sample_in_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        self._epoch = state_dict["num_epochs"]
        self._step_in_epoch = state_dict["batch_in_epoch"]
        self._sample_in_epoch = state_dict["sample_in_epoch"]

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __eq__(self, other: object) -> bool:
        match other:
            case EpochInstant(epoch):
                # An epoch instant is matched only at the beginning of the epoch.
                res = epoch == self.num_epochs and self.num_samples_in_epoch == 0
            case StepInstant(step, from_current_epoch):
                if from_current_epoch:
                    res = step == self.num_steps_in_epoch
                else:
                    res = step == self.num_steps
            case SampleInstant(sample, from_current_epoch):
                if from_current_epoch:
                    res = sample == self.num_samples_in_epoch
                else:
                    res = sample == self.num_samples
            case _:
                msg = (
                    f"Cannot compare a {self.__class__.__name__} with an object "
                    f"of type {type(other).__name__}."
                )
                raise TypeError(msg)

        return res

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __lt__(self, other: Instant) -> bool:
        match other:
            case EpochInstant(epoch):
                res = self.num_epochs < epoch
            case StepInstant(step, from_current_epoch):
                if from_current_epoch:
                    res = self.num_steps < step
                else:
                    res = self.num_steps < step
            case SampleInstant(sample, from_current_epoch):
                if from_current_epoch:
                    res = self.num_samples < sample
                else:
                    res = self.num_samples < sample

        return res

    def __le__(self, other: Instant) -> bool:
        return self < other or self == other

    def __gt__(self, other: Instant) -> bool:
        return not self <= other

    def __ge__(self, other: Instant) -> bool:
        return not self < other

    def __mod__(self, other: Instant) -> int:
        match other:
            case EpochInstant(epoch):
                res = (self.num_epochs % epoch) * self._sample_in_epoch
                res += self.num_samples_in_epoch
            case StepInstant(step, from_current_epoch):
                if from_current_epoch:
                    res = self.num_steps_in_epoch % step
                else:
                    res = self.num_steps % step
            case SampleInstant(sample, from_current_epoch):
                if from_current_epoch:
                    res = self.num_samples_in_epoch % sample
                else:
                    res = self.num_samples % sample

        return res
