# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (c) Soumith Chintala 2016. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_misc.py
# --------------------------------------------------------------------------- #

from collections.abc import Sequence
from typing import overload

import torch

from deepsight.structures import BoundingBoxes, Image, ImageMode
from deepsight.typing import Configs

from ._base import Transform


class ToDtype(Transform):
    """Convert an image to the given data type, optionally scaling the values.

    Scaling means that the values of the image are transformed to the expected range
    of values for the given data type. For example, if the data type is `torch.uint8`,
    then the values of the image are scaled to the range `[0, 255]`. If the data type
    is `torch.float32`, then the values of the image are scaled to the range `[0, 1]`.
    """

    def __init__(self, dtype: torch.dtype, scale: bool = False) -> None:
        """Initialize a to-dtype transform.

        Args:
            dtype: The desired data type.
            scale: Whether to scale the values of the image.
        """
        super().__init__()

        self.dtype = dtype
        self.scale = scale

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "dtype": self.dtype,
            "scale": self.scale,
        }

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    @overload
    def __call__(self, image: Image) -> Image: ...

    @overload
    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes,
    ) -> tuple[Image, BoundingBoxes]: ...

    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes | None = None,
    ) -> Image | tuple[Image, BoundingBoxes]:
        if image.dtype != torch.uint8 and not self.dtype.is_floating_point:
            raise NotImplementedError(
                f"Currently, only converting from `torch.uint8` to a floating-point "
                f"data type is supported. Got {image.dtype} -> {self.dtype}."
            )

        new_image = image.data.to(self.dtype)
        if self.scale:
            new_image = new_image / 255.0

        image = image.to_dtype(self.dtype, self.scale)
        match boxes:
            case None:
                return image
            case BoundingBoxes():
                return image, boxes


class ToMode(Transform):
    """Convert an image to the given mode."""

    def __init__(self, mode: ImageMode | str) -> None:
        """Initialize a to-mode transform.

        Args:
            mode: The desired mode.
        """
        super().__init__()

        self._mode = ImageMode(mode)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {"mode": str(self._mode)}

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    @overload
    def __call__(self, image: Image) -> Image: ...

    @overload
    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes,
    ) -> tuple[Image, BoundingBoxes]: ...

    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes | None = None,
    ) -> Image | tuple[Image, BoundingBoxes]:
        image = image.to_mode(self._mode)
        match boxes:
            case None:
                return image
            case BoundingBoxes():
                return image, boxes


class Standardize(Transform):
    """Standardize an image with the given mean and standard deviation.

    Standardization is performed by subtracting the mean and dividing by the standard
    deviation.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        """Initialize a standardize transform.

        Args:
            mean: The mean values for each channel.
            std: The standard deviation values for each channel.
        """
        super().__init__()

        if any(value <= 0 for value in std):
            raise ValueError("All values in `std` must be greater than 0.")

        self.mean = mean
        self.std = std

    def get_configs(self, recursive: bool) -> Configs:
        return {"mean": self.mean, "std": self.std}

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    @overload
    def __call__(self, image: Image) -> Image: ...

    @overload
    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes,
    ) -> tuple[Image, BoundingBoxes]: ...

    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes | None = None,
    ) -> Image | tuple[Image, BoundingBoxes]:
        image = image.standardize(self.mean, self.std)
        match boxes:
            case None:
                return image
            case BoundingBoxes():
                return image, boxes
