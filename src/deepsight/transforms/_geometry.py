##
##
##

import enum
import random
from collections.abc import Sequence

import torchvision.transforms.v2 as T  # noqa
import torchvision.transforms.v2.functional as F  # noqa

from deepsight.structures import BoundingBoxes, Image
from deepsight.typing import str_enum

from ._base import Transform


@str_enum
class InterpolationMode(enum.Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest_exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"

    def to_torchvision(self) -> T.InterpolationMode:
        match self:
            case InterpolationMode.NEAREST:
                return T.InterpolationMode.NEAREST
            case InterpolationMode.NEAREST_EXACT:
                return T.InterpolationMode.NEAREST_EXACT
            case InterpolationMode.BILINEAR:
                return T.InterpolationMode.BILINEAR
            case InterpolationMode.BICUBIC:
                return T.InterpolationMode.BICUBIC


class RandomShortestSize(Transform):
    def __init__(
        self,
        min_size: int | Sequence[int],
        max_size: int | None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        super().__init__()

        self.min_size = (min_size,) if isinstance(min_size, int) else min_size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]:
        if boxes is not None and boxes.image_size != image.size:
            raise ValueError(
                "The image size of the boxes does not match the size of the image, "
                f"got {boxes.image_size} and {image.size} respectively."
            )

        orig_height, orig_width = image.size
        min_size = random.choice(self.min_size)
        ratio = min_size / min(orig_height, orig_width)
        if self.max_size is not None:
            ratio = min(self.max_size / max(orig_height, orig_width), ratio)

        new_height = int(orig_height * ratio)
        new_width = int(orig_width * ratio)

        new_image = F.resize(
            image.data,
            [new_height, new_width],
            interpolation=self.interpolation.to_torchvision(),
            antialias=self.antialias,
        )
        new_image = Image(new_image)

        if boxes is not None:
            boxes = boxes.resize((new_height, new_width))

        return new_image, boxes
