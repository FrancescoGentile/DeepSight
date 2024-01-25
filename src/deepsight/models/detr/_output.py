##
##
##

from dataclasses import dataclass
from typing import Literal, Self

from deepsight.typing import Detachable, Tensor


@dataclass(frozen=True)
class Output(Detachable):
    """Output of DETR model.

    Attributes:
        class_logits: The predicted class logits.
        box_coords: The predicted bounding box coordinates. The coordinates are
            normalized to the range [0, 1] (i.e., they are relative to the image size)
            and in the CXCYWH format.
        image_sizes: The sizes of the input images.
    """

    class_logits: Tensor[Literal["L B Q C"], float]
    box_coords: Tensor[Literal["L B Q 4"], float]
    image_sizes: tuple[tuple[int, int], ...]

    def detach(self) -> Self:
        return self.__class__(
            self.class_logits.detach(),
            self.box_coords.detach(),
            self.image_sizes,
        )
