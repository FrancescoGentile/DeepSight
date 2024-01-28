# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image as PILImage

from deepsight.typing import Moveable, Number, PathLike, Tensor

from . import _utils as utils
from ._enums import ImageMode, InterpolationMode


class Image(Moveable):
    """A wrapper around a tensor representing an image."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        data: Tensor[Literal["_ H W"], Number],
        mode: ImageMode | str,
    ) -> None:
        if data.ndim != 3:
            raise ValueError(
                f"Expected the image to be 3-dimensional. Got {data.ndim}-dimensional "
                f"image."
            )

        mode = ImageMode(mode)
        if data.shape[0] != mode.num_channels():
            raise ValueError(
                f"Expected the image to have {mode.num_channels()} channels. Got "
                f"{data.shape[0]} channels."
            )

        self._data = data
        self._mode = mode

    @classmethod
    def open(cls, path: PathLike, mode: ImageMode | str | None = None) -> Self:
        """Opens an image from a file.

        Args:
            path: The path to the image file.
            mode: The image mode to optionally convert the image to. If `None`, the
                image is not converted.

        Returns:
            The image.
        """
        pil_image = PILImage.open(str(path))
        if mode is not None:
            mode = ImageMode(mode)
            pil_image = pil_image.convert(mode=mode.to_pil_mode())
        else:
            mode = ImageMode.from_pil_mode(pil_image.mode)

        data = torch.from_numpy(np.array(pil_image))
        match mode:
            case ImageMode.GRAYSCALE:
                if data.ndim == 2:
                    data = data.unsqueeze_(0)
            case ImageMode.RGB:
                data = data.permute(2, 0, 1)

        return cls(data, mode)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def data(self) -> Tensor[Literal["_ H W"], Number]:
        """The underlying tensor."""
        return self._data

    @property
    def mode(self) -> ImageMode:
        """The image mode."""
        return self._mode

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

        return self.__class__(
            self._data.to(device, non_blocking=non_blocking),
            self._mode,
        )

    # ----------------------------------------------------------------------- #
    # Geometric Transformations
    # ----------------------------------------------------------------------- #

    def resize(
        self,
        size: tuple[int, int],
        interpolation_mode: InterpolationMode | str,
        antialias: bool = True,
    ) -> Self:
        """Resize the image.

        !!! note
            If the size is the same as the current size, `self` is returned.

        Args:
            size: The new size of the image as (height, width).
            interpolation_mode: The interpolation mode.
            antialias: Whether to use antialiasing.

        Returns:
            The resized image.

        Raises:
            ValueError: If `size` is not positive.
        """
        if size == self.size:
            return self
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(
                f"Expected `size` to be positive. Got {size[0]}x{size[1]}."
            )

        data = F.interpolate(
            self._data.unsqueeze(0),
            size=size,
            mode=str(interpolation_mode),
            antialias=antialias,
        )
        data.squeeze_(0)

        return self.__class__(data, self._mode)

    def horizontal_flip(self) -> Self:
        """Flip the image horizontally."""
        data = self._data.flip(-1)
        return self.__class__(data, self._mode)

    # ----------------------------------------------------------------------- #
    # Color Transformations
    # ----------------------------------------------------------------------- #

    def to_grayscale(self) -> Self:
        """Convert the image to grayscale.

        !!! note

            If the image is already grayscale or binary, `self` is returned.
        """
        match self._mode:
            case ImageMode.GRAYSCALE:
                return self
            case ImageMode.RGB:
                data = utils.rgb_to_grayscale(
                    self._data, num_output_channels=1, preserve_dtype=True
                )
                return self.__class__(data, ImageMode.GRAYSCALE)

    def to_rgb(self) -> Self:
        """Convert the image to RGB."""
        match self._mode:
            case ImageMode.GRAYSCALE:
                data = self._data.expand(3, -1, -1)
                return self.__class__(data, ImageMode.RGB)
            case ImageMode.RGB:
                return self

    def to_mode(self, mode: ImageMode | str) -> Self:
        """Convert the image to a different mode.

        Args:
            mode: The image mode.

        Returns:
            The converted image.
        """
        mode = ImageMode(mode)
        match mode:
            case ImageMode.GRAYSCALE:
                return self.to_grayscale()
            case ImageMode.RGB:
                return self.to_rgb()

    def adjust_brightness(self, brightness_factor: float) -> Self:
        """Adjust the brightness of the image.

        Args:
            brightness_factor: The brightness factor.

        Returns:
            The adjusted image.

        Raises:
            ValueError: If `brightness_factor` is negative.
        """
        if brightness_factor < 0:
            raise ValueError(
                f"Expected `brightness_factor` to be non-negative. Got "
                f"{brightness_factor}."
            )

        bound = utils.max_dtype_value(self.dtype)
        data = self._data.mul(brightness_factor).clamp_(0, bound)
        data = data.to(dtype=self.dtype)

        return self.__class__(data, self._mode)

    def adjust_contrast(self, contrast_factor: float) -> Self:
        """Adjust the contrast of the image.

        Args:
            contrast_factor: The contrast factor.

        Returns:
            The adjusted image.

        Raises:
            ValueError: If `contrast_factor` is negative.
        """
        if contrast_factor < 0:
            raise ValueError(
                f"Expected `contrast_factor` to be non-negative. Got "
                f"{contrast_factor}."
            )

        match self._mode:
            case ImageMode.GRAYSCALE:
                grayscale_data = (
                    self._data
                    if self._data.is_floating_point()
                    else self._data.to(torch.float32)
                )
            case ImageMode.RGB:
                grayscale_data = utils.rgb_to_grayscale(
                    self._data, num_output_channels=1, preserve_dtype=False
                )
                if not self.dtype.is_floating_point:
                    grayscale_data = grayscale_data.floor_()

        mean = torch.mean(grayscale_data, dim=(-3, -2, -1), keepdim=True)
        data = utils.blend(self._data, mean, contrast_factor)

        return self.__class__(data, self._mode)

    def adjust_saturation(self, saturation_factor: float) -> Self:
        """Adjust the saturation of the image.

        Args:
            saturation_factor: The saturation factor.

        Returns:
            The adjusted image.

        Raises:
            ValueError: If `saturation_factor` is negative.
        """
        if saturation_factor < 0:
            raise ValueError(
                f"Expected `saturation_factor` to be non-negative. Got "
                f"{saturation_factor}."
            )

        match self.mode:
            case ImageMode.GRAYSCALE:
                return self
            case ImageMode.RGB:
                grayscale_data = utils.rgb_to_grayscale(
                    self._data,
                    num_output_channels=1,
                    preserve_dtype=False,
                )
                if not grayscale_data.is_floating_point():
                    grayscale_data = grayscale_data.floor_()

                data = utils.blend(self._data, grayscale_data, saturation_factor)

                return self.__class__(data, self._mode)

    def adjust_hue(self, hue_factor: float) -> Self:
        """Adjust the hue of the image.

        Args:
            hue_factor: The hue factor.

        Returns:
            The adjusted image.

        Raises:
            ValueError: If `hue_factor` is not in the range [-0.5, 0.5].
        """
        if not -0.5 <= hue_factor <= 0.5:
            raise ValueError(
                f"Expected `hue_factor` to be in the range [-0.5, 0.5]. Got "
                f"{hue_factor}."
            )

        match self.mode:
            case ImageMode.GRAYSCALE:
                return self
            case ImageMode.RGB:
                if self._data.numel() == 0:
                    return self

                image = self.to_dtype(torch.float32, scale=True)
                hsv_data = utils.rgb_to_hsv(image.data)
                h, s, v = hsv_data.unbind(dim=0)
                h.add_(hue_factor).remainder_(1.0)
                hsv_data = torch.stack((h, s, v), dim=0)
                data = utils.hsv_to_rgb(hsv_data)
                data = utils.to_dtype(data, image.dtype, scale=True)

                return self.__class__(data, self._mode)

    # ----------------------------------------------------------------------- #
    # Miscellaneous Transformations
    # ----------------------------------------------------------------------- #

    def standardize(self, mean: Sequence[float], std: Sequence[float]) -> Self:
        """Standardize the image.

        Args:
            mean: The mean values for each channel.
            std: The standard deviation values for each channel.

        Returns:
            The standardized image.
        """
        if not self._data.is_floating_point():
            raise NotImplementedError(
                f"Currently, standardization is only supported for floating-point "
                f"images. Got {self.dtype}."
            )

        if len(mean) != self._data.shape[0]:
            raise ValueError(
                f"The number of values in `mean` must match the number of channels in "
                f"the image. Got {len(mean)} values for {self._data.shape[0]} channels."
            )

        if len(std) != self._data.shape[0]:
            raise ValueError(
                f"The number of values in `std` must match the number of channels in "
                f"the image. Got {len(std)} values for {self._data.shape[0]} channels."
            )

        mean_tensor = torch.as_tensor(mean, dtype=self.dtype, device=self.device)
        std_tensor = torch.as_tensor(std, dtype=self.dtype, device=self.device)
        mean_tensor = mean_tensor.view(-1, 1, 1)
        std_tensor = std_tensor.view(-1, 1, 1)

        data = self.data.clone()
        data.sub_(mean_tensor).div_(std_tensor)

        return self.__class__(data, self._mode)

    def to_dtype(self, dtype: torch.dtype, scale: bool) -> Self:
        """Convert the image to a different dtype."""
        data = utils.to_dtype(self._data, dtype, scale)
        return self.__class__(data, self._mode)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, mode={self.mode})"

    def __str__(self) -> str:
        return repr(self)

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_data",)
