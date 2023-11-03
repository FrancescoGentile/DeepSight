##
##
##

from collections.abc import Sequence

import torch
import torchvision.transforms.v2.functional as F  # noqa

from deepsight.structures import BoundingBoxes, Image

from ._base import Transform


class ToDtype(Transform):
    def __init__(self, dtype: torch.dtype, scale: bool = False) -> None:
        super().__init__()

        self.dtype = dtype
        self.scale = scale

    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]:
        data = F.to_dtype(image.data, self.dtype, self.scale)
        return Image(data), boxes


class Standardize(Transform):
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        inplace: bool = False,
    ) -> None:
        super().__init__()

        self.mean = list(mean)
        self.std = list(std)
        self.inplace = inplace

    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]:
        data = F.normalize(image.data, self.mean, self.std, self.inplace)
        return Image(data), boxes
