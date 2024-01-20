##
##
##

import enum
from typing import Self

from deepsight.typing import str_enum


@str_enum
class ImageMode(enum.Enum):
    """The image mode."""

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
