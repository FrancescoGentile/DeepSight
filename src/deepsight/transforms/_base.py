# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from types import TracebackType
from typing import Protocol

from deepsight.structures import BoundingBoxes, Image


class Transform(Protocol):
    """Interface for all transforms."""

    def transform_image(self, image: Image) -> Image:
        """Transform an image."""
        ...

    def transform_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        """Transform bounding boxes."""
        ...

    def __enter__(self) -> None:
        """Initialize the transform parameters."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Clean up the transform parameters."""
        ...
