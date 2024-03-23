# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/mosaicml/composer/blob/dev/composer/core/time.py
# --------------------------------------------------------------------------- #

from collections.abc import Iterable
from typing import Any, Self

from deepsight.typing import Stateful

from ._phase import EpochPhase
from ._time import Instant, TimeUnit

#  --------------------------------------------------------------------------- #
# Epoch Phase Timestamp
# --------------------------------------------------------------------------- #


class EpochPhaseTimestamp(Stateful):
    """The timestamp of an epoch phase."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        label: str,
        num_epochs: int,
        batch_in_epoch: int,
        sample_in_epoch: int,
        batches_per_epoch: int,
        samples_per_epoch: int,
        started: bool,
        ended: bool,
    ) -> None:
        """Initialize the timestamp."""
        self._label = label
        self._num_epochs = num_epochs
        self._batch_in_epoch = batch_in_epoch
        self._sample_in_epoch = sample_in_epoch
        self._batches_per_epoch = batches_per_epoch
        self._samples_per_epoch = samples_per_epoch
        self._started = started
        self._ended = ended

    @classmethod
    def new[S, O, A, P](cls, phase: EpochPhase[S, O, A, P]) -> Self:  # type: ignore
        """Creates a new timestamp corresponding to the start of an epoch phase."""
        return cls(
            label=phase.label,
            num_epochs=0,
            batch_in_epoch=0,
            sample_in_epoch=0,
            batches_per_epoch=phase.dataloader.num_batches,
            samples_per_epoch=phase.dataloader.num_samples,
            started=False,
            ended=True,
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_epochs(self) -> int:
        """The number of epochs in which the phase was fully run."""
        return self._num_epochs

    @property
    def num_batches(self) -> int:
        """The number of batches fully processed since the beginning of training."""
        num_batches = self._num_epochs * self._batches_per_epoch
        num_batches += self._batch_in_epoch

        return num_batches

    @property
    def num_samples(self) -> int:
        """The number of samples fully processed since the beginning of training."""
        num_samples = self._num_epochs * self._samples_per_epoch
        num_samples += self._sample_in_epoch

        return num_samples

    @property
    def batch_in_epoch(self) -> int:
        """The number of batches fully processed in the current epoch.

        This can also be interpreted as the index of the batch currently being
        processed. For example, if `batch_in_epoch` is `5`, then this means that
        5 batches have been fully processed in the current epoch, and the 6th batch
        (i.e. the batch with index 5) is currently being processed.
        """
        return self._batch_in_epoch

    @property
    def sample_in_epoch(self) -> int:
        """The number of samples fully processed in the current epoch.

        This can also be interpreted as the index of the sample currently being
        processed. For example, if `sample_in_epoch` is `5`, then this means that
        5 samples have been fully processed in the current epoch, and the 6th sample
        (i.e. the sample with index 5) is currently being processed. Note that if the
        batch size is greater than 1, then this is equal to the index of the first
        sample in the current batch.
        """
        return self._sample_in_epoch

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def is_last_batch(self) -> bool:
        """Whether the current batch is the last one in the epoch."""
        return self._batch_in_epoch == self._batches_per_epoch - 1

    def has_started(self) -> bool:
        """Whether the phase has started."""
        return self._started

    def has_ended(self) -> bool:
        """Whether the phase has ended."""
        return self._ended

    def start(self) -> None:
        """Start a new epoch."""
        self._started = True
        self._ended = False

    def next_batch(self, batch_size: int) -> None:
        """Move to the next batch."""
        self._batch_in_epoch += 1
        self._sample_in_epoch += batch_size

    def end(self) -> None:
        """End the current epoch."""
        self._num_epochs += 1
        self._batch_in_epoch = 0
        self._sample_in_epoch = 0
        self._started = False
        self._ended = True

    def to_instant(self, unit: TimeUnit, from_epoch_begin: bool = False) -> Instant:
        """Convert the timestamp to an instant."""
        match unit:
            case TimeUnit.EPOCH:
                if from_epoch_begin:
                    raise ValueError("Cannot convert to epoch from start epoch.")

                return Instant.from_epoch(self.num_epochs, self._label)
            case TimeUnit.BATCH:
                if not from_epoch_begin:
                    return Instant.from_batch(self.num_batches, self._label, False)

                return Instant.from_batch(self.batch_in_epoch, self._label, True)
            case TimeUnit.SAMPLE:
                if not from_epoch_begin:
                    return Instant.from_sample(self.num_samples, self._label, False)

                return Instant.from_sample(self.sample_in_epoch, self._label, True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_epochs": self._num_epochs,
            "batch_in_epoch": self._batch_in_epoch,
            "sample_in_epoch": self._sample_in_epoch,
            "started": self._started,
            "ended": self._ended,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        self._num_epochs = state_dict["num_epochs"]
        self._batch_in_epoch = state_dict["batch_in_epoch"]
        self._sample_in_epoch = state_dict["sample_in_epoch"]
        self._started = state_dict["started"]
        self._ended = state_dict["ended"]


#  --------------------------------------------------------------------------- #
# Timestamp
# --------------------------------------------------------------------------- #


class Timestamp(Stateful):
    """The timestamp of the training process."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        num_epochs: int,
        phases: tuple[EpochPhaseTimestamp, ...],
    ) -> None:
        """Initialize the timestamp."""
        self._num_epochs = num_epochs
        self._phases = phases

    @classmethod
    def new[S, O, A, P](cls, phases: Iterable[EpochPhase[S, O, A, P]]) -> Self:  # type: ignore
        """Creates a new timestamp corresponding to the start of training."""
        return cls(
            num_epochs=0,
            phases=tuple(EpochPhaseTimestamp.new(phase) for phase in phases),
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_epochs(self) -> int:
        """The number of completed epochs.

        This can also be interpreted as the index of the current epoch.
        """
        return self._num_epochs

    @property
    def phases(self) -> tuple[EpochPhaseTimestamp, ...]:
        """The timestamps of the phases."""
        return self._phases

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def next_epoch(self) -> None:
        """Move to the next epoch."""
        if any(not phase.has_ended() for phase in self._phases):
            raise RuntimeError("Cannot move to next epoch while phase is running.")

        self._num_epochs += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_epochs": self._num_epochs,
            "phases": [phase.state_dict() for phase in self._phases],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        self._num_epochs = state_dict["num_epochs"]
        self._phases = tuple(
            EpochPhaseTimestamp(**phase_state_dict)
            for phase_state_dict in state_dict["phases"]
        )
