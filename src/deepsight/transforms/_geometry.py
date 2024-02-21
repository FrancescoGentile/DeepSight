# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (c) Soumith Chintala 2016. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_geometry.py
# --------------------------------------------------------------------------- #

from deepsight import utils
from deepsight.structures import BoundingBoxes, Image, InterpolationMode
from deepsight.typing import Configs, Configurable

from ._base import Transform

# --------------------------------------------------------------------------- #
# Resize
# --------------------------------------------------------------------------- #


class Resize(Transform, Configurable):
    """Resize the input image to the given size."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation: InterpolationMode | str = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        """Initialize a resize transform.

        Args:
            size: The desired output size. If `size` is an integer, then both the height
                and width of the image will be resized to `size`. If `size` is a tuple,
                then the height and width of the image will be resized to `size[0]` and
                `size[1]` respectively.
            interpolation: The interpolation mode to use.
            antialias: Whether to use an anti-aliasing filter when downsampling the
                image.
        """
        super().__init__()

        size = utils.to_2tuple(size)
        if any(dim <= 0 for dim in size):
            raise ValueError("All values in `size` must be greater than 0.")

        self.size = size
        self.interpolation = InterpolationMode(interpolation)
        self.antialias = antialias

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "size": self.size,
            "interpolation": str(self.interpolation),
            "antialias": self.antialias,
        }

    def transform_image(self, image: Image) -> Image:
        return image.resize(
            self.size,
            interpolation_mode=self.interpolation,
            antialias=self.antialias,
        )

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        return boxes.resize(self.size)


# --------------------------------------------------------------------------- #
# Shortest Side Resize
# --------------------------------------------------------------------------- #


class ShortestSideResize(Transform, Configurable):
    """Resize the input image so that the shortest side is of the given size."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        size: int,
        max_size: int | None = None,
        interpolation: InterpolationMode | str = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        """Initialize a shortest side resize transform.

        Args:
            size: The desired size of the shorter edge of the image.
            max_size: The maximum allowed size of the longer edge of the image. If after
                resizing the shorter edge of the image to `size`, the longer edge is
                greater than `max_size`, then the image is resized again such that the
                longer edge is equal to `max_size` (meaning that the shorter edge will
                be smaller than `size`).
            interpolation: The interpolation mode to use.
            antialias: Whether to use an anti-aliasing filter when downsampling the
                image.
        """
        super().__init__()

        if size <= 0:
            raise ValueError("`size` must be greater than 0.")
        if max_size is not None and size > max_size:
            raise ValueError("`size` must be less than or equal to `max_size`.")

        self.size = size
        self.max_size = max_size
        self.interpolation = InterpolationMode(interpolation)
        self.antialias = antialias

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "interpolation": str(self.interpolation),
            "antialias": self.antialias,
        }

    def transform_image(self, image: Image) -> Image:
        return image.resize(
            size=self._compute_size(image.size),
            interpolation_mode=self.interpolation,
            antialias=self.antialias,
        )

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        return boxes.resize(self._compute_size(boxes.image_size))

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _compute_size(self, size: tuple[int, int]) -> tuple[int, int]:
        ratio = self.size / min(size)
        if self.max_size is not None:
            ratio = min(self.max_size / max(size), ratio)

        new_height = int(size[0] * ratio)
        new_width = int(size[1] * ratio)

        return new_height, new_width


# --------------------------------------------------------------------------- #
# Horizontal Flip
# --------------------------------------------------------------------------- #


class HorizontalFlip(Transform):
    """Flip the input image horizontally."""

    def __init__(self) -> None:
        super().__init__()

    def transform_image(self, image: Image) -> Image:
        return image.horizontal_flip()

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        return boxes.horizontal_flip()
