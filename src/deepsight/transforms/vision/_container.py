##
##
##

import random
from collections.abc import Iterable, Sequence
from typing import overload

from deepsight import utils
from deepsight.structures.vision import BoundingBoxes, Image
from deepsight.typing import Configs

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

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        if not recursive:
            return {}

        return {
            "transforms": [
                utils.get_configs(transform, recursive) for transform in self.transforms
            ]
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
        match boxes:
            case None:
                for transform in self.transforms:
                    image = transform(image)
                return image
            case BoundingBoxes():
                for transform in self.transforms:
                    image, boxes = transform(image, boxes)
                return image, boxes


class RandomOrder(Transform):
    """Randomly apply a list of transforms."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """Initialize a random order transform.

        Args:
            transforms: A sequence of transforms to apply randomly.
        """
        super().__init__()

        self.transforms = transforms

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        if not recursive:
            return {}

        return {
            "transforms": [
                utils.get_configs(transform, recursive) for transform in self.transforms
            ]
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
        transform = random.choice(self.transforms)
        match boxes:
            case None:
                return transform(image)
            case BoundingBoxes():
                return transform(image, boxes)


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

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        configs: Configs = {"p": self.p}
        if recursive:
            configs["transform"] = utils.get_configs(self.transform, recursive)

        return configs

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
        match boxes:
            case None:
                if random.random() < self.p:
                    return self.transform(image)
                return image
            case BoundingBoxes():
                if random.random() < self.p:
                    return self.transform(image, boxes)
                return image, boxes


class RandomChoice(Transform):
    """Apply one of the transforms randomly."""

    def __init__(
        self,
        transforms: Sequence[Transform],
        p: Sequence[float] | None = None,
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

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        configs: Configs = {"p": self.p}
        if recursive:
            configs["transforms"] = [
                utils.get_configs(transform, recursive) for transform in self.transforms
            ]

        return configs

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
        transform = random.choices(self.transforms, weights=self.p)[0]
        match boxes:
            case None:
                return transform(image)
            case BoundingBoxes():
                return transform(image, boxes)
