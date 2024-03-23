# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass
from typing import Self

from deepsight.typing import str_enum


@str_enum
class ColorSpace(enum.Enum):
    """Color space of an image."""

    GRAYSCALE = "grayscale"
    RGB = "rgb"

    @classmethod
    def from_pil_mode(cls, mode: str) -> Self:
        """Convert a PIL mode to an image mode.

        Args:
            mode: The PIL mode.

        Returns:
            The image mode.
        """
        match mode:
            case "L":
                return cls.GRAYSCALE  # type: ignore
            case "RGB":
                return cls.RGB  # type: ignore
            case _:
                raise ValueError(f"Unsupported PIL mode: {mode}")

    def to_pil_mode(self) -> str:
        """Convert the image mode to a PIL mode.

        Returns:
            The PIL mode.
        """
        match self:
            case self.GRAYSCALE:
                return "L"
            case self.RGB:
                return "RGB"

    def num_channels(self) -> int:
        """The number of channels."""
        match self:
            case self.GRAYSCALE:
                return 1
            case self.RGB:
                return 3


@str_enum
class InterpolationMode(enum.Enum):
    """Interpolation mode used to resize an image."""

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest_exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


@dataclass(frozen=True)
class ConstantPadding:
    """Padding with a constant value."""

    value: float


@dataclass(frozen=True)
class ReplicatePadding:
    """Replicate the edge values."""


@dataclass(frozen=True)
class ReflectPadding:
    """Reflect the values at the edge."""

    include_edge: bool


type PaddingMode = ConstantPadding | ReplicatePadding | ReflectPadding
"""The padding mode used to pad an image."""
