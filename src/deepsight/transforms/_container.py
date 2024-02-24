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
from typing import Literal

import torch

from deepsight import utils
from deepsight.structures import BoundingBoxes, Image
from deepsight.typing import Configs, Configurable, Tensor

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

    def get_configs(self, recursive: bool) -> Configs:
        if not recursive:
            return {}

        return {
            "transforms": [
                utils.get_configs(transform, recursive) for transform in self.transforms
            ]
        }

    def transform_image(self, image: Image) -> Image:
        for transform in self.transforms:
            image = transform.transform_image(image)

        return image

    def transform_boxes(
        self, boxes: BoundingBoxes
    ) -> tuple[BoundingBoxes, Tensor[Literal["N"], bool] | None]:
        keep = None
        for transform in self.transforms:
            boxes, last_keep = transform.transform_boxes(boxes)
            keep = _update_keep_mask(keep, last_keep)

        return boxes, keep


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

    def get_configs(self, recursive: bool) -> Configs:
        if not recursive:
            return {}

        return {
            "transforms": [
                utils.get_configs(transform, recursive) for transform in self.transforms
            ]
        }

    def transform_image(self, image: Image) -> Image:
        order = self._order if self._order is not None else self._choose_order()

        for idx in order:
            image = self.transforms[idx].transform_image(image)

        return image

    def transform_boxes(
        self, boxes: BoundingBoxes
    ) -> tuple[BoundingBoxes, Tensor[Literal["N"], bool] | None]:
        order = self._order if self._order is not None else self._choose_order()

        keep = None
        for idx in order:
            boxes, last_keep = self.transforms[idx].transform_boxes(boxes)
            keep = _update_keep_mask(keep, last_keep)

        return boxes, keep

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __enter__(self) -> None:
        self._order = self._choose_order()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
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

    def get_configs(self, recursive: bool) -> Configs:
        configs: Configs = {"p": self.p}
        if recursive:
            configs["transform"] = utils.get_configs(self.transform, recursive)

        return configs

    def transform_image(self, image: Image) -> Image:
        apply = self._apply if self._apply is not None else self._choose_apply()

        if apply:
            image = self.transform.transform_image(image)

        return image

    def transform_boxes(
        self, boxes: BoundingBoxes
    ) -> tuple[BoundingBoxes, Tensor[Literal["N"], bool] | None]:
        apply = self._apply if self._apply is not None else self._choose_apply()

        if apply:
            return self.transform.transform_boxes(boxes)
        else:
            return boxes, None

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __enter__(self) -> None:
        self._apply = self._choose_apply()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
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

    def get_configs(self, recursive: bool) -> Configs:
        configs: Configs = {"p": self.p}
        if recursive:
            configs["transforms"] = [
                utils.get_configs(transform, recursive) for transform in self.transforms
            ]

        return configs

    def transform_image(self, image: Image) -> Image:
        apply_index = (
            self._apply_index if self._apply_index is not None else self._choose_index()
        )

        return self.transforms[apply_index].transform_image(image)

    def transform_boxes(
        self, boxes: BoundingBoxes
    ) -> tuple[BoundingBoxes, Tensor[Literal["N"], bool] | None]:
        apply_index = (
            self._apply_index if self._apply_index is not None else self._choose_index()
        )

        return self.transforms[apply_index].transform_boxes(boxes)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __enter__(self) -> None:
        self._apply_index = self._choose_index()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._apply_index = None

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _choose_index(self) -> int:
        return random.choices(range(len(self.transforms)), weights=self.p)[0]


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _update_keep_mask(
    prev_keep: Tensor[Literal["N"], bool] | None,
    last_keep: Tensor[Literal["M"], bool] | None,
) -> Tensor[Literal["N"], bool] | None:
    """Update the current set of boxes to keep after a transform has been applied.

    Args:
        prev_keep: A tensor of shape (N,) indicating which of the original N
            boxes are still kept after all the previous transforms have been applied.
            The number of boxes kept so far is equal to the number of `True` values
            in the tensor and should be equal to `M`.
        last_keep: A tensor of shape (M,) indicating which of the last M boxes are
            still kept after the current transform has been applied.

    Returns:
        The updated set of boxes to keep. This is a tensor of shape (N,) indicating
        which of the original N boxes are still kept after all the previous transforms
        and the current transform have been applied. If all the boxes are still kept,
        `None` is returned.
    """
    match (prev_keep, last_keep):
        case (None, None):
            # No boxes have been removed so far and no boxes have been removed
            # by the current transform, so all the boxes are still kept and no
            # update is needed.
            new_keep = None
        case (None, torch.Tensor()):
            # No boxes have been removed so far, but the current transform has
            # removed some boxes, so we can simply set the current set of boxes
            # to keep to the new keep mask.
            new_keep = last_keep
        case (torch.Tensor(), None):
            # Some boxes have been removed so far, but the current transform
            # hasn't removed any boxes, so we don't need to update the current
            # set of boxes to keep.
            new_keep = prev_keep
        case (torch.Tensor(), torch.Tensor()):
            # Some boxes have been removed so far and the current transform has
            # removed some boxes, so we need to remove the boxes that have been
            # removed by the current transform from the current set of boxes to
            # keep.
            current_keep_idx = torch.nonzero(prev_keep).squeeze(1)
            current_keep_idx = current_keep_idx[last_keep]
            new_keep = torch.zeros_like(prev_keep)
            new_keep[current_keep_idx] = True
        case _:
            # This should never happen if the transforms are implemented
            # correctly.
            raise RuntimeError("Incompatible keep masks.")

    return new_keep
