# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (c) Soumith Chintala 2016. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_container.py
# --------------------------------------------------------------------------- #

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from deepsight import utils
from deepsight.structures import BoundingBoxes, Image
from deepsight.typing import Configurable

from ._base import Transform

# --------------------------------------------------------------------------- #
# Sequential Order
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SequentialOrderParameters:
    child_params: list[Any]


class SequentialOrder(Transform[SequentialOrderParameters], Configurable):
    """Sequentially apply a list of transforms."""

    def __init__(self, transforms: Iterable[Transform[Any]]) -> None:
        """Initialize a sequential order transform.

        Args:
            transforms: An iterable of transforms to apply sequentially.
        """
        super().__init__()

        self._transforms = transforms

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self, recursive: bool) -> dict[str, Any]:
        if not recursive:
            return {}

        return {
            "transforms": [
                utils.get_config(transform, recursive) for transform in self._transforms
            ]
        }

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_parameters(self) -> SequentialOrderParameters:
        return SequentialOrderParameters([
            transform._get_parameters() for transform in self._transforms
        ])

    def _apply_to_image(
        self,
        image: Image,
        parameters: SequentialOrderParameters,
    ) -> Image:
        for t, params in zip(self._transforms, parameters.child_params, strict=True):
            image = t._apply_to_image(image, params)

        return image

    def _apply_to_boxes(
        self,
        boxes: BoundingBoxes,
        parameters: SequentialOrderParameters,
    ) -> BoundingBoxes:
        for t, params in zip(self._transforms, parameters.child_params, strict=True):
            boxes = t._apply_to_boxes(boxes, params)

        return boxes


# --------------------------------------------------------------------------- #
# Random Order
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RandomOrderParameters:
    order: list[int]
    child_params: list[Any]


class RandomOrder(Transform[RandomOrderParameters], Configurable):
    """Apply a list of transforms in a random order."""

    def __init__(self, transforms: Sequence[Transform[Any]]) -> None:
        """Initialize a random order transform.

        Args:
            transforms: A sequence of transforms to apply in a random order.
        """
        super().__init__()

        self._transforms = transforms

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self, recursive: bool) -> dict[str, Any]:
        if not recursive:
            return {}

        return {
            "transforms": [
                utils.get_config(transform, recursive) for transform in self._transforms
            ]
        }

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_parameters(self) -> RandomOrderParameters:
        order = random.sample(range(len(self._transforms)), len(self._transforms))
        child_params = [transform._get_parameters() for transform in self._transforms]

        return RandomOrderParameters(order, child_params)

    def _apply_to_image(self, image: Image, parameters: RandomOrderParameters) -> Image:
        params = parameters.child_params
        for idx in parameters.order:
            image = self._transforms[idx]._apply_to_image(image, params[idx])

        return image

    def _apply_to_boxes(
        self,
        boxes: BoundingBoxes,
        parameters: RandomOrderParameters,
    ) -> BoundingBoxes:
        params = parameters.child_params
        for idx in parameters.order:
            boxes = self._transforms[idx]._apply_to_boxes(boxes, params[idx])

        return boxes


# --------------------------------------------------------------------------- #
# Random Apply
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RandomApplyParameters[T]:
    apply: bool
    child_params: T


class RandomApply[T](Transform[RandomApplyParameters[T]], Configurable):
    """Apply a transform with a probability."""

    def __init__(self, transform: Transform[T], p: float) -> None:
        """Initialize a random apply transform.

        Args:
            transform: The transform to apply.
            p: The probability to apply the transform.
        """
        super().__init__()

        self._transform = transform
        self._p = p

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self, recursive: bool) -> dict[str, Any]:
        config: dict[str, Any] = {"p": self._p}
        if recursive:
            config["transform"] = utils.get_config(self._transform, recursive)

        return config

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_parameters(self) -> RandomApplyParameters[T]:
        apply = random.random() < self._p
        child_params = self._transform._get_parameters()
        return RandomApplyParameters(apply, child_params)

    def _apply_to_image(
        self, image: Image, parameters: RandomApplyParameters[T]
    ) -> Image:
        if parameters.apply:
            image = self._transform._apply_to_image(image, parameters.child_params)

        return image


# --------------------------------------------------------------------------- #
# Random Choice
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RandomChoiceParameters:
    apply_index: int
    child_params: Any


class RandomChoice(Transform[RandomChoiceParameters], Configurable):
    """Choose a transform to apply at random."""

    def __init__(
        self,
        transforms: Sequence[Transform[Any]],
        p: Sequence[float] | None = None,
    ) -> None:
        """Initialize a random choice transform.

        Args:
            transforms: A sequence of transforms to choose from.
            p: The probabilities to choose each transform. If `None`, each transform
                is chosen with equal probability.
        """
        super().__init__()

        if p is None:
            p = [1 / len(transforms)] * len(transforms)
        else:
            if len(p) != len(transforms):
                raise ValueError(
                    "The number of probabilities must match the number of transforms."
                )
            if any(probability <= 0 for probability in p):
                raise ValueError("Probabilities must be positive.")

            total = sum(p)
            if total != 1:
                p = [probability / total for probability in p]

        self._transforms = transforms
        self._p = p

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self, recursive: bool) -> dict[str, Any]:
        config: dict[str, Any] = {"p": self._p}
        if recursive:
            config["transforms"] = [
                utils.get_config(transform, recursive) for transform in self._transforms
            ]

        return config

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_parameters(self) -> RandomChoiceParameters:
        apply_index = random.choices(range(len(self._transforms)), weights=self._p)[0]
        child_params = self._transforms[apply_index]._get_parameters()
        return RandomChoiceParameters(apply_index, child_params)

    def _apply_to_image(
        self,
        image: Image,
        parameters: RandomChoiceParameters,
    ) -> Image:
        transform = self._transforms[parameters.apply_index]
        return transform._apply_to_image(image, parameters.child_params)

    def _apply_to_boxes(
        self,
        boxes: BoundingBoxes,
        parameters: RandomChoiceParameters,
    ) -> BoundingBoxes:
        transform = self._transforms[parameters.apply_index]
        return transform._apply_to_boxes(boxes, parameters.child_params)
