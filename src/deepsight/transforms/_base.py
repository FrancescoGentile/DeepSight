##
##
##

import abc
from typing import overload

from deepsight.structures import BoundingBoxes, Image


class Transform(abc.ABC):
    @abc.abstractmethod
    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]: ...

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    @overload
    def __call__(self, image: Image) -> Image: ...

    @overload
    def __call__(
        self, image: Image, boxes: BoundingBoxes
    ) -> tuple[Image, BoundingBoxes]: ...

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> Image | tuple[Image, BoundingBoxes]:
        new_image, new_boxes = self._apply(image, boxes)
        if new_boxes is None:
            return new_image
        else:
            return new_image, new_boxes
