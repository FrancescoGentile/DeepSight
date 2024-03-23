# Copyright 2024 francescogentile.
# SPDX-License-Identifier: Apache-2.0

import abc
from types import TracebackType

from deepsight.structures import BoundingBoxes, Image


class Transform[P](abc.ABC):
    """Base class for all transforms.

    Differently from [torchvision.transforms.v2.Transform](), this class does not
    provide a single `__call__` method that accepts different types of inputs (e.g.,
    image, boxes, segmentation) and applies the same transform to all of them.
    Instead, this class provides specific methods for each type of input (as of now,
    only images and bounding boxes are supported).

    If you need to ensure that the same transform is applied to multiple inputs, you
    should use the transform as a context manager. By doing so, all random parameters
    are generated only once, and the same parameters are used for all invocations of
    the transform. If the transform is used outside the context manager, the parameters
    are generated on the fly on each invocation. Note that it is safe to nest context
    managers. In this case, the parameters of the inner context manager will override
    the parameters of the outer context manager until the inner context manager is
    exited.

    Example:
        >>> transform = RandomTransform()
        >>> with transform:
        ...     # The same parameters are used for both the image and the boxes
        ...     image = transform.apply_to_image(image)
        ...     boxes = transform.apply_to_boxes(boxes)
        >>> # Different parameters are used for the image and the boxes
        >>> image = transform.apply_to_image(image)
        >>> boxes = transform.apply_to_boxes(boxes)
    """

    def __init__(self) -> None:
        self._context_parameters: list[P] = []

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def apply_to_image(self, image: Image) -> Image:
        """Applies the transform to the image."""
        if len(self._context_parameters) > 0:
            parameters = self._context_parameters[-1]
        else:
            parameters = self._get_parameters()

        return self._apply_to_image(image, parameters)

    def apply_to_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        """Applies the transform to the bounding boxes."""
        if len(self._context_parameters) > 0:
            parameters = self._context_parameters[-1]
        else:
            parameters = self._get_parameters()

        return self._apply_to_boxes(boxes, parameters)

    # ----------------------------------------------------------------------- #
    # Magic Method
    # ----------------------------------------------------------------------- #

    def __enter__(self) -> None:
        self._context_parameters.append(self._get_parameters())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if len(self._context_parameters) == 0:
            msg = "Cannot exit the context manager without entering it first."
            raise RuntimeError(msg)

        self._context_parameters.pop()

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    @abc.abstractmethod
    def _get_parameters(self) -> P:
        """Returns the non-fixed parameters of the transform.

        This method should be implemented by the subclasses to return those parameters
        that are not fixed and thus may change on each invocation of the transform.
        """
        ...

    def _apply_to_image(self, image: Image, parameters: P) -> Image:
        return self.apply_to_image(image)

    def _apply_to_boxes(self, boxes: BoundingBoxes, parameters: P) -> BoundingBoxes:
        return self.apply_to_boxes(boxes)
