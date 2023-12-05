##
##
##

from typing import Protocol, overload

from deepsight.structures.vision import BoundingBoxes, Image
from deepsight.typing import Configurable


class Transform(Configurable, Protocol):
    @overload
    def __call__(self, image: Image) -> Image: ...

    @overload
    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes,
    ) -> tuple[Image, BoundingBoxes]: ...

    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes | None = None,
    ) -> Image | tuple[Image, BoundingBoxes]: ...
