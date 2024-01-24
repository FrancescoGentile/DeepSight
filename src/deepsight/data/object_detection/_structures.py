##
##
##

from dataclasses import dataclass
from typing import Literal, Self

import torch

from deepsight.structures import BoundingBoxes, Image
from deepsight.typing import Moveable, Tensor


@dataclass(frozen=True)
class Sample(Moveable):
    """A sample for the object detection task.

    Attributes:
        image: The image containing the objects to detect.
    """

    image: Image

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        return self.__class__(self.image.to(device, non_blocking=non_blocking))


@dataclass(frozen=True)
class Annotation(Moveable):
    """The ground truth annotation for an object detection sample.

    Attributes:
        boxes: The bounding boxes of the objects in the image.
        classes: The classes of the objects. This is a tensor of length equal to the
            number of bounding boxes, where each element is an integer representing the
            class of the corresponding bounding box.
    """

    boxes: BoundingBoxes
    classes: Tensor[Literal["N"], int]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        return self.__class__(
            self.boxes.to(device, non_blocking=non_blocking),
            self.classes.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class Prediction(Moveable):
    """The prediction made by an object detection model.

    Attributes:
        boxes: The bounding boxes of the objects in the image.
        classes: The classes of the objects. This is a tensor of length equal to the
            number of bounding boxes, where each element is an integer representing the
            class of the corresponding bounding box.
        scores: The confidence scores of the predictions. This is a tensor of length
            equal to the number of bounding boxes, where each element is a float
            representing the confidence score of the corresponding bounding box.
    """

    boxes: BoundingBoxes
    classes: Tensor[Literal["N"], int]
    scores: Tensor[Literal["N"], float]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        return self.__class__(
            self.boxes.to(device, non_blocking=non_blocking),
            self.classes.to(device, non_blocking=non_blocking),
            self.scores.to(device, non_blocking=non_blocking),
        )
