##
##
##

import random
from collections.abc import Sequence
from typing import overload

from deepsight import utils
from deepsight.structures.vision import BoundingBoxes, Image
from deepsight.typing import Configs
from deepsight.utils import InterpolationMode

from ._base import Transform


class Resize(Transform):
    """Resize the input image to the given size."""

    def __init__(
        self,
        size: int | tuple[int, int],
        max_size: int | None = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        """Initialize a resize transform.

        Args:
            size: The desired output size. If `size` is an `int`, then the smaller edge
                of the image is matched to this number. If `size` is a tuple `(height,
                width)`, then the image is resized to the specified size.
            max_size: The maximum allowed size of the longer edge of the image. If after
                resizing the smaller edge of the image to `size`, the longer edge is
                greater than `max_size`, then the image is resized again such that the
                longer edge is equal to `max_size` (meaning that the shorter edge will
                be smaller than `size`). This parameter is ignored if `size` is a tuple
                specifying the exact size of the output image.
            interpolation: The interpolation mode to use.
            antialias: Whether to use an anti-aliasing filter when downsampling the
                image.
        """
        super().__init__()

        if isinstance(size, int):
            if size <= 0:
                raise ValueError("`size` must be greater than 0.")
            if max_size is not None and size > max_size:
                raise ValueError("`size` must be less than or equal to `max_size`.")
        elif any(dim <= 0 for dim in size):
            raise ValueError("All values in `size` must be greater than 0.")

        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "interpolation": self.interpolation,
            "antialias": self.antialias,
        }

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

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
    ) -> Image | tuple[Image, BoundingBoxes]:
        if boxes is not None and boxes.image_size != image.size:
            raise ValueError(
                "The image size of the boxes does not match the size of the image, "
                f"got {boxes.image_size} and {image.size} respectively."
            )

        if isinstance(self.size, int):
            ratio = self.size / min(image.height, image.width)
            if self.max_size is not None:
                ratio = min(self.max_size / max(image.height, image.width), ratio)

            new_height = int(image.height * ratio)
            new_width = int(image.width * ratio)
        else:
            new_height, new_width = self.size

        image = image.resize(
            (new_height, new_width),
            interpolation_mode=self.interpolation,
            antialias=self.antialias,
        )

        match boxes:
            case None:
                return image
            case BoundingBoxes():
                boxes = boxes.resize(image.size)
                return image, boxes


class RandomShortestSize(Transform):
    """Resize the input image such that the shorter edge is equal to a random value."""

    def __init__(
        self,
        min_size: int | Sequence[int],
        max_size: int | None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        """Initialize a random shortest size transform.

        Args:
            min_size: Values to sample from to determine the minimum size of the
                shorter edge of the image.
            max_size: The maximum allowed size of the longer edge of the image. If after
                resizing the shorter edge of the image to the sampled value, the longer
                edge is greater than `max_size`, then the image is resized again such
                that the longer edge is equal to `max_size` (meaning that the shorter
                edge will be smaller than the sampled value).
            interpolation: The interpolation mode to use.
            antialias: Whether to use an anti-aliasing filter when downsampling the
                image.
        """
        super().__init__()

        min_size = utils.to_tuple(min_size)
        if any(size <= 0 for size in min_size):
            raise ValueError("All values in `min_size` must be greater than 0.")
        if max_size is not None and any(size > max_size for size in min_size):
            raise ValueError(
                "All values in `min_size` must be less than or equal to `max_size`."
            )

        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "min_size": self.min_size,
            "max_size": self.max_size,
            "interpolation": self.interpolation,
            "antialias": self.antialias,
        }

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

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
    ) -> Image | tuple[Image, BoundingBoxes]:
        if boxes is not None and boxes.image_size != image.size:
            raise ValueError(
                "The image size of the boxes does not match the size of the image, "
                f"got {boxes.image_size} and {image.size} respectively."
            )

        min_size = random.choice(self.min_size)
        ratio = min_size / min(image.height, image.width)
        if self.max_size is not None:
            ratio = min(self.max_size / max(image.height, image.width), ratio)

        new_height = int(image.height * ratio)
        new_width = int(image.width * ratio)

        image = image.resize(
            (new_height, new_width),
            interpolation_mode=self.interpolation,
            antialias=self.antialias,
        )

        match boxes:
            case None:
                return image
            case BoundingBoxes():
                boxes = boxes.resize(image.size)
                return image, boxes


class HorizonalFlip(Transform):
    def __init__(self) -> None:
        super().__init__()

    def get_configs(self, recursive: bool) -> Configs:
        return {}

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
    ) -> Image | tuple[Image, BoundingBoxes]:
        match boxes:
            case None:
                return image.horizontal_flip()
            case BoundingBoxes():
                return image.horizontal_flip(), boxes.horizontal_flip()
