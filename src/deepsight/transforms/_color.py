# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (c) Soumith Chintala 2016. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_color.py
# --------------------------------------------------------------------------- #

import math
import random
from dataclasses import dataclass
from typing import Any

from deepsight.structures import Image
from deepsight.typing import Configurable

from ._base import Transform

# --------------------------------------------------------------------------- #
# ColorJitter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ColorJitterParameters:
    brightness: float | None
    contrast: float | None
    saturation: float | None
    hue: float | None


class ColorJitter(Transform[ColorJitterParameters], Configurable):
    """Randomly change the brightness, contrast, saturation and hue of an image."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        brightness: float | tuple[float, float] | None = None,
        contrast: float | tuple[float, float] | None = None,
        saturation: float | tuple[float, float] | None = None,
        hue: float | tuple[float, float] | None = None,
    ) -> None:
        """Initialize a color jitter transform.

        Args:
            brightness: The brightness factor. If `brightness` is a `float`, then the
                brightness of the image is changed by a factor drawn randomly from the
                range `[max(0, 1 - brightness), 1 + brightness]`. If `brightness` is a
                tuple `(min_factor, max_factor)`, then the brightness of the image is
                changed by a factor drawn randomly from the range `[min_factor,
                max_factor]`. Should be non negative numbers. If `brightness` is `None`,
                then the brightness is not changed.
            contrast: The contrast factor. If `contrast` is a `float`, then the contrast
                of the image is changed by a factor drawn randomly from the range
                `[max(0, 1 - contrast), 1 + contrast]`. If `contrast` is a tuple
                `(min_factor, max_factor)`, then the contrast of the image is changed by
                a factor drawn randomly from the range `[min_factor, max_factor]`.
                Should be non negative numbers. If `contrast` is `None`, then the
                contrast is not changed.
            saturation: The saturation factor. If `saturation` is a `float`, then the
                saturation of the image is changed by a factor drawn randomly from the
                range `[max(0, 1 - saturation), 1 + saturation]`. If `saturation` is a
                tuple `(min_factor, max_factor)`, then the saturation of the image is
                changed by a factor drawn randomly from the range `[min_factor,
                max_factor]`. Should be non negative numbers. If `saturation` is `None`,
                then the saturation is not changed.
            hue: The hue factor. If `hue` is a `float`, then the hue of the image is
                changed by a factor drawn randomly from the range `[-hue, hue]`. If
                `hue` is a tuple `(min_factor, max_factor)`, then the hue of the image
                is changed by a factor drawn randomly from the range `[min_factor,
                max_factor]`. Should be numbers between `-0.5` and `0.5`. If `hue` is
                `None`, then the hue is not changed.
        """
        super().__init__()

        self._brightness = _check_jitter_properties("brightness", brightness)
        self._contrast = _check_jitter_properties("contrast", contrast)
        self._saturation = _check_jitter_properties("saturation", saturation)
        self._hue = _check_jitter_properties(
            "hue", hue, center=0.0, bounds=(-0.5, 0.5), clip_first_on_zero=False
        )

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self, recursive: bool) -> dict[str, Any]:
        return {
            "brightness": self._brightness,
            "contrast": self._contrast,
            "saturation": self._saturation,
            "hue": self._hue,
        }

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_parameters(self) -> ColorJitterParameters:
        brightness = (
            random.uniform(self._brightness[0], self._brightness[1])
            if self._brightness
            else None
        )
        contrast = (
            random.uniform(self._contrast[0], self._contrast[1])
            if self._contrast
            else None
        )
        saturation = (
            random.uniform(self._saturation[0], self._saturation[1])
            if self._saturation
            else None
        )
        hue = random.uniform(self._hue[0], self._hue[1]) if self._hue else None

        return ColorJitterParameters(brightness, contrast, saturation, hue)

    def _apply_to_image(self, image: Image, parameters: ColorJitterParameters) -> Image:
        if parameters.brightness is not None:
            image = image.adjust_brightness(parameters.brightness)

        if parameters.contrast is not None:
            image = image.adjust_contrast(parameters.contrast)

        if parameters.saturation is not None:
            image = image.adjust_saturation(parameters.saturation)

        if parameters.hue is not None:
            image = image.adjust_hue(parameters.hue)

        return image


# --------------------------------------------------------------------------- #
# Private functions
# --------------------------------------------------------------------------- #


def _check_jitter_properties(
    property_name: str,
    value: float | tuple[float, float] | None,
    center: float = 1.0,
    bounds: tuple[float, float] = (0.0, math.inf),
    clip_first_on_zero: bool = True,
) -> tuple[float, float] | None:
    match value:
        case None:
            return None
        case float() | int():
            if value < 0:
                msg = f"If {property_name} is a single number, it must be non negative."
                raise ValueError(msg)
            value = float(value)
            value = (center - value, center + value)
            if clip_first_on_zero:
                value = (max(0, value[0]), value[1])
        case tuple():
            pass

    if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
        msg = (
            f"{property_name} values should be between {bounds[0]} and {bounds[1]}, "
            f"but got {value}."
        )
        raise ValueError(msg)

    return value
