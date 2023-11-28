##
##
##

import random
from collections.abc import Iterable, Sequence

from deepsight.structures.vision import BoundingBoxes, Image

from ._base import Transform


class SequentialOrder(Transform):
    """Sequentially apply a list of transforms."""

    def __init__(self, transforms: Iterable[Transform]) -> None:
        """Initialize a sequential order transform.

        Args:
            transforms: An iterable of transforms to apply sequentially.
        """
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
    """Randomly apply a list of transforms."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """Initialize a random order transform.

        Args:
            transforms: A sequence of transforms to apply randomly.
        """
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
    """Apply a transform with a probability."""

    def __init__(
        self,
        transform: Transform,
        p: float,
    ) -> None:
        """Initialize a random apply transform.

        Args:
            transform: The transform to apply with a probability.
            p: The probability to apply the transform.
        """
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
    """Apply one of the transforms randomly."""

    def __init__(
        self, transforms: Sequence[Transform], p: Sequence[float] | None = None
    ) -> None:
        """Initialize a random choice transform.

        Args:
            transforms: A sequence of transforms to choose from.
            p: The probabilities to choose each transform. If `None`, all transforms
                will be chosen with equal probability.
        """
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
