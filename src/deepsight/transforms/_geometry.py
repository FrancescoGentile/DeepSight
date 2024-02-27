# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (c) Soumith Chintala 2016. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_geometry.py
# --------------------------------------------------------------------------- #

import random
from dataclasses import dataclass
from types import TracebackType

from deepsight import utils
from deepsight.structures import (
    BoundingBoxes,
    ConstantPadding,
    Image,
    InterpolationMode,
    PaddingMode,
)
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


# --------------------------------------------------------------------------- #
# Random Position Crop
# --------------------------------------------------------------------------- #


class RandomCrop(Transform, Configurable):
    """Crop a region of the input image at a random position and size."""

    @dataclass
    class _Params:
        height: int
        width: int
        top: float
        left: float

    def __init__(
        self,
        height: int | tuple[int, int],
        width: int | tuple[int, int],
        padding_mode: PaddingMode | None = None,
    ) -> None:
        """Initialize a random crop transform.

        Args:
            height: The desired height of the cropped region. If `height` is an integer,
                then the height of the cropped region will be `height`. If `height` is a
                tuple, then the height of the cropped region will be a random value
                between `height[0]` and `height[1]`.
            width: The desired width of the cropped region. If `width` is an integer,
                then the width of the cropped region will be `width`. If `width` is a
                tuple, then the width of the cropped region will be a random value
                between `width[0]` and `width[1]`.
            padding_mode: The padding mode to use if the input image is smaller than the
                desired crop size. If `None`, then constant padding with a value of 0 is
                used.
        """
        super().__init__()

        if padding_mode is None:
            padding_mode = ConstantPadding(0)

        self._height = utils.to_2tuple(height)
        self._width = utils.to_2tuple(width)
        self._padding_mode = padding_mode

        self._params: RandomCrop._Params | None = None

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "height": self._height,
            "width": self._width,
            "padding_mode": str(self._padding_mode),
        }

    def transform_image(self, image: Image) -> Image:
        params = self._params or self._choose_params()

        pad_top, pad_left, pad_bottom, pad_right = 0, 0, 0, 0

        if image.height < params.height:
            diff = params.height - image.height
            pad_top = diff // 2
            pad_bottom = diff - pad_top

        if image.width < params.width:
            diff = params.width - image.width
            pad_left = diff // 2
            pad_right = diff - pad_left

        image = image.pad(
            top=pad_top,
            left=pad_left,
            bottom=pad_bottom,
            right=pad_right,
            mode=self._padding_mode,
        )

        top = int(params.top * (image.height - params.height + 1))
        left = int(params.left * (image.width - params.width + 1))

        return image.crop(
            top=top,
            left=left,
            bottom=top + params.height,
            right=left + params.width,
        )

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        params = self._params or self._choose_params()

        top = int(params.top * (boxes.image_size[0] - params.height + 1))
        left = int(params.left * (boxes.image_size[1] - params.width + 1))

        return boxes.crop(
            top=top,
            left=left,
            bottom=top + params.height,
            right=left + params.width,
        )

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __enter__(self) -> None:
        self._params = self._choose_params()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._params = None

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _choose_params(self) -> _Params:
        return RandomCrop._Params(
            height=random.randint(self._height[0], self._height[1]),
            width=random.randint(self._width[0], self._width[1]),
            top=random.random(),
            left=random.random(),
        )
