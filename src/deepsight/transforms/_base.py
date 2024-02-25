# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from types import TracebackType
from typing import Protocol

from deepsight.structures import BoundingBoxes, Image


class Transform(Protocol):
    """Interface for all transforms."""

    def transform_image(self, image: Image) -> Image:
        """Apply the transform to the image."""
        return image

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        """Apply the transform to the bounding boxes."""
        return boxes

    def __enter__(self) -> None:
        """Initialize the parameters for the transform.

        All invocations of the transform within the context manager will use the same
        parameters. This is useful if the transform has any random components, such as
        random rotations or color jittering, and you want to ensure that the same random
        parameters are used for all invocations of the transform.
        """

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Clean up the parameters for the transform."""
