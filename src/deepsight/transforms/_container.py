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
from types import TracebackType
from typing import Any

from deepsight import utils
from deepsight.structures import BoundingBoxes, Image
from deepsight.typing import Configurable

from ._base import Transform

# --------------------------------------------------------------------------- #
# Sequential Order
# --------------------------------------------------------------------------- #


class SequentialOrder(Transform, Configurable):
    """Sequentially apply a list of transforms."""

    def __init__(self, transforms: Iterable[Transform]) -> None:
        """Initialize a sequential order transform.

        Args:
            transforms: An iterable of transforms to apply sequentially.
        """
        super().__init__()

        self.transforms = transforms

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self, recursive: bool) -> dict[str, Any]:
        if not recursive:
            return {}

        return {
            "transforms": [
                utils.get_config(transform, recursive) for transform in self.transforms
            ]
        }

    def transform_image(self, image: Image) -> Image:
        for transform in self.transforms:
            image = transform.transform_image(image)

        return image

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        for transform in self.transforms:
            boxes = transform.transform_boxes(boxes)

        return boxes

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __enter__(self) -> None:
        for transform in self.transforms:
            transform.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for transform in self.transforms:
            transform.__exit__(exc_type, exc_value, traceback)


# --------------------------------------------------------------------------- #
# Random Order
# --------------------------------------------------------------------------- #


class RandomOrder(Transform, Configurable):
    """Apply a list of transforms in a random order."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """Initialize a random order transform.

        Args:
            transforms: A sequence of transforms to apply in a random order.
        """
        super().__init__()

        self.transforms = transforms

        self._order: Sequence[int] | None = None

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self, recursive: bool) -> dict[str, Any]:
        if not recursive:
            return {}

        return {
            "transforms": [
                utils.get_config(transform, recursive) for transform in self.transforms
            ]
        }

    def transform_image(self, image: Image) -> Image:
        order = self._order if self._order is not None else self._choose_order()

        for idx in order:
            image = self.transforms[idx].transform_image(image)

        return image

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        order = self._order if self._order is not None else self._choose_order()

        for idx in order:
            boxes = self.transforms[idx].transform_boxes(boxes)

        return boxes

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __enter__(self) -> None:
        self._order = self._choose_order()
        for idx in self._order:
            self.transforms[idx].__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._order is None:
            return

        for idx in self._order:
            self.transforms[idx].__exit__(exc_type, exc_value, traceback)
        self._order = None

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _choose_order(self) -> Sequence[int]:
        return random.sample(range(len(self.transforms)), len(self.transforms))


# --------------------------------------------------------------------------- #
# Random Apply
# --------------------------------------------------------------------------- #


class RandomApply(Transform, Configurable):
    """Apply a transform with a probability."""

    def __init__(self, transform: Transform, p: float) -> None:
        """Initialize a random apply transform.

        Args:
            transform: The transform to apply.
            p: The probability to apply the transform.
        """
        super().__init__()

        self.transform = transform
        self.p = p

        self._apply: bool | None = None

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self, recursive: bool) -> dict[str, Any]:
        config: dict[str, Any] = {"p": self.p}
        if recursive:
            config["transform"] = utils.get_config(self.transform, recursive)

        return config

    def transform_image(self, image: Image) -> Image:
        apply = self._apply if self._apply is not None else self._choose_apply()

        if apply:
            image = self.transform.transform_image(image)

        return image

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        apply = self._apply if self._apply is not None else self._choose_apply()

        if apply:
            boxes = self.transform.transform_boxes(boxes)

        return boxes

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __enter__(self) -> None:
        self._apply = self._choose_apply()
        if self._apply:
            self.transform.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._apply is None:
            return

        if self._apply:
            self.transform.__exit__(exc_type, exc_value, traceback)

        self._apply = None

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _choose_apply(self) -> bool:
        return random.random() < self.p


# --------------------------------------------------------------------------- #
# Random Choice
# --------------------------------------------------------------------------- #


class RandomChoice(Transform, Configurable):
    """Choose a transform to apply at random."""

    def __init__(
        self,
        transforms: Sequence[Transform],
        p: Sequence[float] | None = None,
    ) -> None:
        """Initialize a random choice transform.

        Args:
            transforms: A sequence of transforms to choose from.
            p: The probabilities to choose each transform. If `None`, all transforms
                will be chosen with equal probability.
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

        self.transforms = transforms
        self.p = p

        self._apply_index: int | None = None

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self, recursive: bool) -> dict[str, Any]:
        config: dict[str, Any] = {"p": self.p}
        if recursive:
            config["transforms"] = [
                utils.get_config(transform, recursive) for transform in self.transforms
            ]

        return config

    def transform_image(self, image: Image) -> Image:
        apply_index = (
            self._apply_index if self._apply_index is not None else self._choose_index()
        )

        return self.transforms[apply_index].transform_image(image)

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        apply_index = (
            self._apply_index if self._apply_index is not None else self._choose_index()
        )

        return self.transforms[apply_index].transform_boxes(boxes)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __enter__(self) -> None:
        self._apply_index = self._choose_index()
        self.transforms[self._apply_index].__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._apply_index is None:
            return

        self.transforms[self._apply_index].__exit__(exc_type, exc_value, traceback)
        self._apply_index = None

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _choose_index(self) -> int:
        return random.choices(range(len(self.transforms)), weights=self.p)[0]
