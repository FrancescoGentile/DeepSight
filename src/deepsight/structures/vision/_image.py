##
##
##

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from PIL import Image as PILImage
from typing_extensions import Self

from deepsight.typing import Moveable, Number, PathLike, Tensor


class Image(Moveable):
    """A wrapper around a tensor representing an image."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(self, data: Tensor[Literal["3 H W"], Number]) -> None:
        self._data = data

    @classmethod
    def open(cls, path: PathLike) -> Self:
        """Opens an image from a file.

        Args:
            path: The path to the image file.

        Returns:
            The image.
        """
        image = PILImage.open(str(path)).convert("RGB")
        image = np.array(image)  # if np.asarray is used, the array is not writable
        image = torch.from_numpy(image).permute(2, 0, 1)

        return cls(image)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def data(self) -> Tensor[Literal["3 H W"], Number]:
        """The underlying tensor."""
        return self._data

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
    def dtype(self) -> torch.dtype:
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        return self._data.device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        if self.device == torch.device(device):
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
