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

import torch

from deepsight.structures import BoundingBoxes, Image, ImageMode
from deepsight.typing import Configs, Configurable

from ._base import Transform

# --------------------------------------------------------------------------- #
# ToDtype
# --------------------------------------------------------------------------- #


class ToDtype(Transform, Configurable):
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

    def transform_image(self, image: Image) -> Image:
        return image.to_dtype(self.dtype, scale=self.scale)


# --------------------------------------------------------------------------- #
# ToMode
# --------------------------------------------------------------------------- #


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

    def transform_image(self, image: Image) -> Image:
        return image.to_mode(self._mode)


# --------------------------------------------------------------------------- #
# Standardize
# --------------------------------------------------------------------------- #


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

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {"mean": self.mean, "std": self.std}

    def transform_image(self, image: Image) -> Image:
        return image.standardize(self.mean, self.std)


# --------------------------------------------------------------------------- #
# Clamp Bounding Boxes
# --------------------------------------------------------------------------- #


class ClampBoundingBoxes(Transform):
    """Clamp bounding boxes to the image boundaries."""

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        return boxes.clamp_to_image()
