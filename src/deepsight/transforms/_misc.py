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
from typing import Any

import torch

from deepsight.structures import BoundingBoxes, ColorSpace, Image
from deepsight.typing import Configurable, EnumLike

from ._base import Transform

# --------------------------------------------------------------------------- #
# ToDtype
# --------------------------------------------------------------------------- #


class ToDtype(Transform[None], Configurable):
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

    def get_config(self, recursive: bool) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "scale": self.scale,
        }

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_parameters(self) -> None:
        return None

    def _apply_to_image(self, image: Image, parameters: None) -> Image:
        return image.to_dtype(self.dtype, self.scale)


# --------------------------------------------------------------------------- #
# ToColorSpace
# --------------------------------------------------------------------------- #


class ToColorSpace(Transform[None]):
    """Convert an image to the given color space."""

    def __init__(self, color_space: EnumLike[ColorSpace]) -> None:
        """Initializes the to-color-space transform.

        Args:
            color_space: The desired color space.
        """
        super().__init__()

        self._color_space = ColorSpace(color_space)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> dict[str, Any]:
        return {"color_space": str(self._color_space)}

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_parameters(self) -> None:
        return None

    def _apply_to_image(self, image: Image, parameters: None) -> Image:
        return image.to_color_space(self._color_space)


# --------------------------------------------------------------------------- #
# Standardize
# --------------------------------------------------------------------------- #


class Standardize(Transform[None]):
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
            msg = "All values in `std` must be greater than 0."
            raise ValueError(msg)

        self.mean = mean
        self.std = std

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_parameters(self) -> None:
        return None

    def _apply_to_image(self, image: Image, parameters: None) -> Image:
        return image.standardize(self.mean, self.std)


# --------------------------------------------------------------------------- #
# Clamp Bounding Boxes
# --------------------------------------------------------------------------- #


class ClampBoundingBoxes(Transform[None]):
    """Clamp bounding boxes to the image boundaries."""

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_parameters(self) -> None:
        return None

    def _apply_to_boxes(self, boxes: BoundingBoxes, parameters: None) -> BoundingBoxes:
        return boxes.clamp_to_image()
