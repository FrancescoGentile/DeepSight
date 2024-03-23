# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image as PILImage

from deepsight.typing import Detachable, EnumLike, Moveable, Number, PathLike, Tensor

from . import _utils as utils
from ._enums import (
    ColorSpace,
    ConstantPadding,
    InterpolationMode,
    PaddingMode,
    ReflectPadding,
    ReplicatePadding,
)


class Image(Detachable, Moveable):
    """A wrapper around a tensor representing an image."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        data: Tensor[Literal["_ H W"], Number],
        color_space: EnumLike[ColorSpace],
    ) -> None:
        if data.ndim != 3:
            msg = (
                f"Expected the image to be 3-dimensional. Got {data.ndim}-dimensional "
                f"image."
            )
            raise ValueError(msg)

        color_space = ColorSpace(color_space)
        if data.shape[0] != color_space.num_channels():
            msg = (
                f"Expected the image to have {color_space.num_channels()} channels. "
                f"Got {data.shape[0]} channels."
            )
            raise ValueError(msg)

        self._data = data
        self._color_space = color_space

    @classmethod
    def open(
        cls,
        path: PathLike,
        color_space: EnumLike[ColorSpace] | None = None,
    ) -> Self:
        """Opens an image from a file.

        Args:
            path: The path to the image file.
            color_space: The color space of the image. If `None`, no conversion is
                performed.

        Returns:
            The image.
        """
        pil_image = PILImage.open(str(path))
        if color_space is not None:
            color_space = ColorSpace(color_space)
            pil_image = pil_image.convert(mode=color_space.to_pil_mode())
        else:
            color_space = ColorSpace.from_pil_mode(pil_image.mode)

        data = torch.from_numpy(np.array(pil_image))
        match color_space:
            case ColorSpace.GRAYSCALE:
                if data.ndim == 2:
                    data = data.unsqueeze_(0)
            case ColorSpace.RGB:
                data = data.permute(2, 0, 1)

        return cls(data, color_space)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def data(self) -> Tensor[Literal["_ H W"], Number]:
        """The underlying tensor."""
        return self._data

    @property
    def color_space(self) -> ColorSpace:
        """The color space of the image."""
        return self._color_space

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
            self._color_space,
        )

    def detach(self) -> Self:
        return self.__class__(self._data.detach(), self._color_space)

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
            msg = f"Expected `size` to be positive. Got {size[0]}x{size[1]}."
            raise ValueError(msg)

        data = F.interpolate(
            self._data.unsqueeze(0),
            size=size,
            mode=str(interpolation_mode),
            antialias=antialias,
        )
        data.squeeze_(0)

        return self.__class__(data, self._color_space)

    def crop(self, top: int, left: int, bottom: int, right: int) -> Self:
        """Crop the image.

        Args:
            top: The top coordinate of the crop.
            left: The left coordinate of the crop.
            bottom: The bottom coordinate of the crop.
            right: The right coordinate of the crop.

        Returns:
            The cropped image.

        Raises:
            ValueError: If the crop region is not within the image bounds. If you want
                to crop a region of the image that is not (fully) within the image
                bounds, first call `self.pad` to pad the image such that the region is
                within the padded image bounds.
        """
        H, W = self.size  # noqa: N806
        if not (0 <= top < bottom <= H):
            msg = (
                f"Expected `top` and `bottom` to be in the range [0, {H}]. Got "
                f"{top} and {bottom}."
            )
            raise ValueError(msg)
        if not (0 <= left < right <= W):
            msg = (
                f"Expected `left` and `right` to be in the range [0, {W}]. Got "
                f"{left} and {right}."
            )
            raise ValueError(msg)

        data = self._data[:, top:bottom, left:right]
        return self.__class__(data.clone(), self._color_space)

    def horizontal_flip(self) -> Self:
        """Flip the image horizontally."""
        data = self._data.flip(-1)
        return self.__class__(data, self._color_space)

    # ----------------------------------------------------------------------- #
    # Color Transformations
    # ----------------------------------------------------------------------- #

    def to_grayscale(self) -> Self:
        """Convert the image to grayscale.

        !!! note

            If the image is already grayscale or binary, `self` is returned.
        """
        match self._color_space:
            case ColorSpace.GRAYSCALE:
                return self
            case ColorSpace.RGB:
                data = utils.rgb_to_grayscale(
                    self._data, num_output_channels=1, preserve_dtype=True
                )
                return self.__class__(data, ColorSpace.GRAYSCALE)

    def to_rgb(self) -> Self:
        """Convert the image to RGB."""
        match self._color_space:
            case ColorSpace.GRAYSCALE:
                data = self._data.expand(3, -1, -1)
                return self.__class__(data, ColorSpace.RGB)
            case ColorSpace.RGB:
                return self

    def to_color_space(self, color_space: EnumLike[ColorSpace]) -> Self:
        """Convert the image to a different color space.

        Args:
            color_space: The color space to convert to.

        Returns:
            The converted image.
        """
        color_space = ColorSpace(color_space)
        match color_space:
            case ColorSpace.GRAYSCALE:
                return self.to_grayscale()
            case ColorSpace.RGB:
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
            msg = (
                f"Expected `brightness_factor` to be non-negative. Got "
                f"{brightness_factor}."
            )
            raise ValueError(msg)

        bound = utils.max_dtype_value(self.dtype)
        data = self._data.mul(brightness_factor).clamp_(0, bound)
        data = data.to(dtype=self.dtype)

        return self.__class__(data, self._color_space)

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
            msg = (
                f"Expected `contrast_factor` to be non-negative. Got "
                f"{contrast_factor}."
            )
            raise ValueError(msg)

        match self._color_space:
            case ColorSpace.GRAYSCALE:
                grayscale_data = (
                    self._data
                    if self._data.is_floating_point()
                    else self._data.to(torch.float32)
                )
            case ColorSpace.RGB:
                grayscale_data = utils.rgb_to_grayscale(
                    self._data, num_output_channels=1, preserve_dtype=False
                )
                if not self.dtype.is_floating_point:
                    grayscale_data = grayscale_data.floor_()

        mean = torch.mean(grayscale_data, dim=(-3, -2, -1), keepdim=True)
        data = utils.blend(self._data, mean, contrast_factor)

        return self.__class__(data, self._color_space)

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
            msg = (
                f"Expected `saturation_factor` to be non-negative. Got "
                f"{saturation_factor}."
            )
            raise ValueError(msg)

        match self._color_space:
            case ColorSpace.GRAYSCALE:
                return self
            case ColorSpace.RGB:
                grayscale_data = utils.rgb_to_grayscale(
                    self._data,
                    num_output_channels=1,
                    preserve_dtype=False,
                )
                if not grayscale_data.is_floating_point():
                    grayscale_data = grayscale_data.floor_()

                data = utils.blend(self._data, grayscale_data, saturation_factor)

                return self.__class__(data, self._color_space)

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
            msg = (
                f"Expected `hue_factor` to be in the range [-0.5, 0.5]. Got "
                f"{hue_factor}."
            )
            raise ValueError(msg)

        match self._color_space:
            case ColorSpace.GRAYSCALE:
                return self
            case ColorSpace.RGB:
                if self._data.numel() == 0:
                    return self

                image = self.to_dtype(torch.float32, scale=True)
                hsv_data = utils.rgb_to_hsv(image.data)
                h, s, v = hsv_data.unbind(dim=0)
                h.add_(hue_factor).remainder_(1.0)
                hsv_data = torch.stack((h, s, v), dim=0)
                data = utils.hsv_to_rgb(hsv_data)
                data = utils.to_dtype(data, image.dtype, scale=True)

                return self.__class__(data, self._color_space)

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
            msg = (
                f"Currently, standardization is only supported for floating-point "
                f"images. Got {self.dtype}."
            )
            raise NotImplementedError(msg)

        if len(mean) != self._data.shape[0]:
            msg = (
                f"The number of values in `mean` must match the number of channels in "
                f"the image. Got {len(mean)} values for {self._data.shape[0]} channels."
            )
            raise ValueError(msg)

        if len(std) != self._data.shape[0]:
            msg = (
                f"The number of values in `std` must match the number of channels in "
                f"the image. Got {len(std)} values for {self._data.shape[0]} channels."
            )
            raise ValueError(msg)

        mean_tensor = torch.as_tensor(mean, dtype=self.dtype, device=self.device)
        std_tensor = torch.as_tensor(std, dtype=self.dtype, device=self.device)
        mean_tensor = mean_tensor.view(-1, 1, 1)
        std_tensor = std_tensor.view(-1, 1, 1)

        data = self.data.clone()
        data.sub_(mean_tensor).div_(std_tensor)

        return self.__class__(data, self._color_space)

    def to_dtype(self, dtype: torch.dtype, scale: bool) -> Self:
        """Convert the image to a different dtype."""
        data = utils.to_dtype(self._data, dtype, scale)
        return self.__class__(data, self._color_space)

    def pad(
        self,
        top: int,
        left: int,
        bottom: int,
        right: int,
        mode: PaddingMode,
    ) -> Self:
        """Pad the image.

        !!! note
            If the padding is (0, 0, 0, 0), `self` is returned.

        Args:
            top: The top padding.
            left: The left padding.
            bottom: The bottom padding.
            right: The right padding.
            mode: The padding mode.

        Returns:
            The padded image.
        """
        if (top, left, bottom, right) == (0, 0, 0, 0):
            return self

        match mode:
            case ConstantPadding(value):
                data = F.pad(
                    self._data,
                    pad=(left, right, top, bottom),
                    mode="constant",
                    value=value,
                )
            case ReplicatePadding():
                msg = "Replicate padding is not yet implemented."
                raise NotImplementedError(msg)
            case ReflectPadding():
                msg = "Reflection padding is not yet implemented."
                raise NotImplementedError(msg)

        return self.__class__(data, self._color_space)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"color_space={self._color_space}, "
            f"size={self.size})"
        )

    def __str__(self) -> str:
        return repr(self)

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_data",)
