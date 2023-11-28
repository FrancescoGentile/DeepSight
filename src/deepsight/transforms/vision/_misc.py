##
##
##

from collections.abc import Sequence

import torch

from deepsight.structures.vision import BoundingBoxes, Image

from ._base import Transform


class ToDtype(Transform):
    """Convert an image to the given data type, optionally scaling the values.

    Scaling means that the values of the image are transformed to the expected range
    of values for the given data type. For example, if the data type is `torch.uint8`,
    then the values of the image are scaled to the range `[0, 255]`. If the data type
    is `torch.float32`, then the values of the image are scaled to the range `[0, 1]`.
    """

    def __init__(self, dtype: torch.dtype, scale: bool = False) -> None:
        """Initialize a to-dtype transform.

        Args:
            dtype: The desired data type.
            scale: Whether to scale the values of the image.
        """
        super().__init__()

        self.dtype = dtype
        self.scale = scale

    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]:
        if image.dtype != torch.uint8 and not self.dtype.is_floating_point:
            raise NotImplementedError(
                f"Currently, only converting from `torch.uint8` to a floating-point "
                f"data type is supported. Got {image.dtype} -> {self.dtype}."
            )

        new_image = image.data.to(self.dtype)
        if self.scale:
            new_image = new_image / 255.0

        return Image(new_image), boxes


class Standardize(Transform):
    """Standardize an image with the given mean and standard deviation.

    Standardization is performed by subtracting the mean and dividing by the standard
    deviation.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        inplace: bool = False,
    ) -> None:
        """Initialize a standardize transform.

        Args:
            mean: The mean values for each channel.
            std: The standard deviation values for each channel.
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()

        if any(value <= 0 for value in std):
            raise ValueError("All values in `std` must be greater than 0.")

        self.mean = mean
        self.std = std
        self.inplace = inplace

    def _apply(
        self, image: Image, boxes: BoundingBoxes | None
    ) -> tuple[Image, BoundingBoxes | None]:
        if not image.dtype.is_floating_point:
            raise NotADirectoryError(
                f"Currently, standardization is only supported for floating-point "
                f"images. Got {image.dtype}."
            )

        if not self.inplace:
            data = image.data.clone()
        else:
            data = image.data

        mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
        std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)

        data.sub_(mean).div_(std)

        return Image(data), boxes
