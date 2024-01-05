##
##
##

from collections.abc import Callable
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


def check_image_boxes(
    func: Callable[..., Image | tuple[Image, BoundingBoxes]],
) -> Callable[..., Image | tuple[Image, BoundingBoxes]]:
    def wrapper(
        self: Transform,
        image: Image,
        boxes: BoundingBoxes | None = None,
    ) -> Image | tuple[Image, BoundingBoxes]:
        if boxes is not None and boxes.image_size != image.size:
            raise ValueError(
                f"Boxes' image size {boxes.image_size} does not match "
                f"the input image size {image.size}."
            )

        return func(self, image, boxes)

    return wrapper
