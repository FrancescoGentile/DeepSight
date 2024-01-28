# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/mosaicml/composer/blob/dev/composer/core/time.py
# --------------------------------------------------------------------------- #

import enum
from dataclasses import dataclass
from typing import Self

from deepsight.typing import str_enum


@str_enum
class TimeUnit(enum.Enum):
    """The unit of time."""

    EPOCH = "epoch"
    BATCH = "batch"
    SAMPLE = "sample"


class Instant:
    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        value: int,
        unit: TimeUnit | str,
        phase: str,
        from_epoch_begin: bool = False,
    ) -> None:
        """Initialize the instant."""
        self._value = value
        self._unit = TimeUnit(unit)
        self._phase = phase
        self._from_epoch_begin = from_epoch_begin

    @classmethod
    def from_epoch(cls, value: int, phase: str) -> Self:
        """Create an instant from an epoch."""
        return cls(value, TimeUnit.EPOCH, phase)

    @classmethod
    def from_batch(cls, value: int, phase: str, from_epoch_begin: bool = False) -> Self:
        """Create an instant from a batch."""
        return cls(value, TimeUnit.BATCH, phase, from_epoch_begin)

    @classmethod
    def from_sample(
        cls, value: int, phase: str, from_epoch_begin: bool = False
    ) -> Self:
        """Create an instant from a sample."""
        return cls(value, TimeUnit.SAMPLE, phase, from_epoch_begin)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def value(self) -> int:
        """The value of the instant."""
        return self._value

    @property
    def unit(self) -> TimeUnit:
        """The unit of the instant."""
        return self._unit

    @property
    def phase(self) -> str:
        """The phase of the instant."""
        return self._phase

    @property
    def from_epoch_begin(self) -> bool:
        """Whether the instant is from the beginning of the epoch."""
        return self._from_epoch_begin

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __eq__(self, other: object) -> bool:
        other_instant = self._parse(other)
        self._check(other_instant, "compare")
        return self.value == other_instant.value

    def __ne__(self, other: object) -> bool:
        other_instant = self._parse(other)
        self._check(other_instant, "compare")
        return self.value != other_instant.value

    def __lt__(self, other: Self | int) -> bool:
        other_instant = self._parse(other)
        self._check(other_instant, "compare")
        return self.value < other_instant.value

    def __le__(self, other: Self | int) -> bool:
        other_instant = self._parse(other)
        self._check(other_instant, "compare")
        return self.value <= other_instant.value

    def __gt__(self, other: Self | int) -> bool:
        other_instant = self._parse(other)
        self._check(other_instant, "compare")
        return self.value > other_instant.value

    def __ge__(self, other: Self | int) -> bool:
        other_instant = self._parse(other)
        self._check(other_instant, "compare")
        return self.value >= other_instant.value

    def __add__(self, other: int | Self) -> Self:
        other_instant = self._parse(other)
        self._check(other_instant, "add")
        return self.__class__(
            self.value + other_instant.value,
            self.unit,
            self.phase,
            self.from_epoch_begin,
        )

    def __sub__(self, other: int | Self) -> Self:
        other_instant = self._parse(other)
        self._check(other_instant, "subtract")
        return self.__class__(
            self.value - other_instant.value,
            self.unit,
            self.phase,
            self.from_epoch_begin,
        )

    def __mul__(self, other: int) -> Self:
        return self.__class__(
            self.value * other, self.unit, self.phase, self.from_epoch_begin
        )

    def __rmul__(self, other: int) -> Self:
        return self * other

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _check(self, other: Self, verb: str) -> None:
        if other.unit != self.unit:
            raise ValueError(f"Cannot {verb} instants of different units.")
        if other.phase != self.phase:
            raise ValueError(f"Cannot {verb} instants from different phases.")
        if other.from_epoch_begin != self.from_epoch_begin:
            raise ValueError(f"Cannot {verb} instants from different epochs.")

    def _parse(self, other: object) -> Self:
        if isinstance(other, int):
            return self.__class__(other, self.unit, self.phase, self.from_epoch_begin)
        if isinstance(other, self.__class__):
            return other

        raise TypeError(
            f"Expected an int or an {self.__class__.__name__} instance, "
            f"got {type(other)}."
        )


@dataclass(frozen=True)
class Interval:
    value: int
    unit: TimeUnit
    phase: str
    from_epoch_begin: bool = False

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #
