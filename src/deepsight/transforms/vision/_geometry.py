##
##
##

from typing import overload

from deepsight import utils
from deepsight.structures import BoundingBoxes, Image
from deepsight.typing import Configs
from deepsight.utils import InterpolationMode

from ._base import Transform, check_image_boxes

# --------------------------------------------------------------------------- #
# Resize
# --------------------------------------------------------------------------- #


class Resize(Transform):
    """Resize the input image to the given size."""

    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        """Initialize a resize transform.

        Args:
            size: The desired output size. If `size` is an integer, then both the height
                and width of the image will be resized to `size`. If `size` is a tuple,
                then the height and width of the image will be resized to `size[0]` and
                `size[1]` respectively.
            interpolation: The interpolation mode to use.
            antialias: Whether to use an anti-aliasing filter when downsampling the
                image.
        """
        super().__init__()

        size = utils.to_2tuple(size)
        if any(dim <= 0 for dim in size):
            raise ValueError("All values in `size` must be greater than 0.")

        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "size": self.size,
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

    @check_image_boxes
    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes | None = None,
    ) -> Image | tuple[Image, BoundingBoxes]:
        image = image.resize(
            self.size,
            interpolation_mode=self.interpolation,
            antialias=self.antialias,
        )

        match boxes:
            case None:
                return image
            case BoundingBoxes():
                boxes = boxes.resize(image.size)
                return image, boxes


# --------------------------------------------------------------------------- #
# Resize Shortest Side
# --------------------------------------------------------------------------- #


class ShortestSideResize(Transform):
    """Resize the input image such that the shorter edge is equal to a given value."""

    def __init__(
        self,
        size: int,
        max_size: int | None = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        """Initialize a shortest side resize transform.

        Args:
            size: The desired size of the shorter edge of the image.
            max_size: The maximum allowed size of the longer edge of the image. If after
                resizing the shorter edge of the image to `size`, the longer edge is
                greater than `max_size`, then the image is resized again such that the
                longer edge is equal to `max_size` (meaning that the shorter edge will
                be smaller than `size`).
            interpolation: The interpolation mode to use.
            antialias: Whether to use an anti-aliasing filter when downsampling the
                image.
        """
        super().__init__()

        if size <= 0:
            raise ValueError("`size` must be greater than 0.")
        if max_size is not None and size > max_size:
            raise ValueError("`size` must be less than or equal to `max_size`.")

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

    @check_image_boxes
    def __call__(
        self,
        image: Image,
        boxes: BoundingBoxes | None = None,
    ) -> Image | tuple[Image, BoundingBoxes]:
        ratio = self.size / min(image.height, image.width)
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


# --------------------------------------------------------------------------- #
# Resize Longest Side
# --------------------------------------------------------------------------- #


class HorizonalFlip(Transform):
    """Flip the input image horizontally."""

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

    @check_image_boxes
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
