##
##
##

import random
from collections.abc import Iterable, Sequence

from deepsight.structures import BoundingBoxes, Image

from ._base import Transform


class SequentialOrder(Transform):
    def __init__(self, transforms: Iterable[Transform]) -> None:
        super().__init__()

        self.transforms = transforms

    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]:
        output = (image, boxes)
        for transform in self.transforms:
            output = transform._apply(*output)

        return output


class RandomOrder(Transform):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        super().__init__()

        self.transforms = transforms

    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]:
        permutation = random.sample(range(len(self.transforms)), len(self.transforms))
        output = (image, boxes)
        for index in permutation:
            output = self.transforms[index]._apply(*output)

        return output


class RandomApply(Transform):
    def __init__(
        self,
        transform: Transform,
        p: float,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.p = p

    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]:
        if random.random() < self.p:
            return self.transform._apply(image, boxes)

        return image, boxes


class RandomChoice(Transform):
    def __init__(
        self, transforms: Sequence[Transform], p: Sequence[float] | None = None
    ) -> None:
        super().__init__()

        if p is None:
            p = [1 / len(transforms)] * len(transforms)
        else:
            if len(p) != len(transforms):
                raise ValueError(
                    "The number of probabilities must match the number of transforms."
                )
            if any(probability <= 0 for probability in p):
                raise ValueError("Probabilities must be positive.")

            total = sum(p)
            if total != 1:
                p = [probability / total for probability in p]

        self.transforms = transforms
        self.p = p

    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]:
        transform = random.choices(self.transforms, weights=self.p)[0]
        return transform._apply(image, boxes)
