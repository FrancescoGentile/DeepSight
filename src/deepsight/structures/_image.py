##
##
##

from pathlib import Path
from typing import Annotated

import torch
from torch import Tensor
from torchvision import io
from typing_extensions import Self

from deepsight.typing import Moveable


class Image(Moveable):
    """A wrapper around a tensor representing an image."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(self, data: Annotated[Tensor, "C H W", bytes]) -> None:
        self._data = data

    @classmethod
    def open(cls, path: Path | str) -> Self:
        return cls(io.read_image(str(path), mode=io.ImageReadMode.RGB))

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def size(self) -> tuple[int, int]:
        """The size of the image as (height, width)."""
        return self.height, self.width

    @property
    def height(self) -> int:
        """The height of the image."""
        return self._data.shape[1]

    @property
    def width(self) -> int:
        """The width of the image."""
        return self._data.shape[2]

    @property
    def device(self) -> torch.device:
        return self._data.device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def to_tensor(self) -> Annotated[Tensor, "C H W", bytes]:
        """Return the image as a tensor.

        The tensor is in the format `(C, H, W)` and has dtype `torch.uint8`.
        """
        return self._data

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
        if self.device == device:
            return self

        return self.__class__(self._data.to(device, non_blocking=non_blocking))

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

    def __str__(self) -> str:
        return repr(self)

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_data",)
