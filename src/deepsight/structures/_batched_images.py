##
##
##

import math
from collections.abc import Sequence
from typing import Literal, Self

import torch

from deepsight import utils
from deepsight.typing import Moveable, Number, Tensor

from ._batched_sequences import BatchedSequences


class BatchedImages(Moveable):
    """Structure to hold a batch of images as a single tensor."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        data: Tensor[Literal["B C H W"], Number],
        image_sizes: tuple[tuple[int, int], ...] | None = None,
        mask: Tensor[Literal["B H W"], bool] | None = None,
        *,
        check_validity: bool = True,
    ) -> None:
        """Initialize the batched images.

        !!! note

            If neither `image_sizes` nor `mask` are provided, it is assumed
            that the images are not padded (i.e., the images in the batch
            have the same height and width).

        Args:
            data: The tensor containing the batched images.
            image_sizes: The sizes of the images in the batch. If not
                provided, the sizes are computed from the mask.
            mask: The mask indicating which pixels are padded and which
                not. The mask is `True` for padded pixels and `False` for
                valid pixels. If not provided, the mask is computed from
                the image sizes.
            check_validity: Whether to check the validity of the inputs.
                Checking the validity may be expensive, so it can be
                disabled if the inputs are known to be valid.

        Raises:
            ValueError: if `image_sizes` and `mask` are provided and are incompatible.
            ValueError: f the `data` and `mask`/`image_sizes` are incompatible.
        """
        if image_sizes is None and mask is None:
            image_sizes = ((data.shape[2], data.shape[3]),) * data.shape[0]

        if check_validity:
            _check_mask_sizes(data, mask, image_sizes)

        self._data = data
        self._image_sizes = image_sizes
        self._mask = mask

    @classmethod
    def batch(
        cls,
        images: Sequence[Tensor[Literal["C H W"], Number]],
        padding_value: float = 0,
        size_divisible_by: int | tuple[int, int] | None = None,
    ) -> Self:
        """Batch a list of images into a single tensor.

        The images are padded to the largest height and width in the batch (plus
        the optional `size_divisible_by` padding) and stacked into a single
        tensor.

        Args:
            images: The images to batch.
            padding_value: The value to pad the images with.
            size_divisible_by: The height and width of the images will be
                further padded to be divisible by this value. If a single
                value is provided, it is used for both the height and width.
                If `None`, the images are padded to the largest width and
                height in the batch.

        Returns:
            The batched images.
        """
        _check_images(images)

        image_sizes = tuple((img.size(1), img.size(2)) for img in images)
        max_height = max(s[0] for s in image_sizes)
        max_width = max(s[1] for s in image_sizes)

        if size_divisible_by is not None:
            stride_h, stride_w = utils.to_2tuple(size_divisible_by)
            max_height = math.ceil(max_height / stride_h) * stride_h
            max_width = math.ceil(max_width / stride_w) * stride_w

        data = torch.full(
            (len(images), images[0].shape[0], max_height, max_width),
            padding_value,
            dtype=images[0].dtype,
            device=images[0].device,
        )

        for i, image in enumerate(images):
            data[i, :, : image.shape[1], : image.shape[2]].copy_(image)

        return cls(data, image_sizes=image_sizes, check_validity=False)

    @classmethod
    def from_sequences(
        cls,
        sequences: BatchedSequences,
        image_sizes: tuple[tuple[int, int], ...],
    ) -> Self:
        """Convert a batch of sequences to a batch of images."""
        if len(sequences) != len(image_sizes):
            raise ValueError("The number of sequences and image_sizes must be equal.")

        if not sequences.is_padded():
            # if the sequence are not padded, it means that all images have the same
            # height and width, so we can simply reshape the sequences to images
            h, w = image_sizes[0]
            images = sequences.data.view(len(sequences), h, w, -1)
            return cls(images, image_sizes=image_sizes)
        else:
            # if the sequences are padded, we need to unbatch them and pad each image
            # individually
            images = [
                seq.view(image_size[0], image_size[1], -1).permute(2, 0, 1)
                for seq, image_size in zip(
                    sequences.unbatch(), image_sizes, strict=True
                )
            ]

            return cls.batch(images)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def data(self) -> Tensor[Literal["B C H W"], Number]:
        """The tensor containing the batched images.

        The tensor has shape `(B, C, H, W)`, where `B` is the batch size, `C` is
        the number of channels, `H` is the maximum height of the images in the
        batch, and `W` is the maximum width of the images in the batch.
        """
        return self._data

    @property
    def image_sizes(self) -> tuple[tuple[int, int], ...]:
        """The sizes of the images in the batch before padding."""
        if self._image_sizes is None:
            assert self._mask is not None
            self._image_sizes = _compute_sizes_from_mask(self._mask)

        return self._image_sizes

    @property
    def padding_mask(self) -> Tensor[Literal["B H W"], bool]:
        """The mask indicating which pixels are padded and which not.

        The mask is a boolean tensor of shape `(B, H, W)`, where `B` is the
        batch size, `H` is the maximum height of the images in the batch, and
        `W` is the maximum width of the images in the batch. The entries of the
        mask are `True` for valid pixels and `False` for padded pixels.
        """
        if self._mask is None:
            assert self._image_sizes is not None
            self._mask = _compute_mask_from_sizes(
                self._image_sizes,
                max_height=self._data.shape[2],
                max_width=self._data.shape[3],
                device=self._data.device,
            )

        return self._mask

    @property
    def shape(self) -> torch.Size:
        """The shape of the batched images."""
        return self._data.shape

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the batched images."""
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        """The device of the batched images."""
        return self._data.device

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def is_padded(self) -> bool:
        """Whether the images in the batch are padded."""
        return any(
            (h < self._data.shape[2] or w < self._data.shape[3])
            for h, w in self.image_sizes
        )

    def unbatch(self) -> tuple[Tensor[Literal["C H W"], Number], ...]:
        """Unbatch the images into a list of tensors."""
        return tuple(self[i] for i in range(len(self)))

    def new_with(self, data: Tensor[Literal["B C H W"], Number]) -> Self:
        """Return new batched images with the given data tensor.

        Raises:
            ValueError: If the new data tensor does have a different batch size
                or spatial dimensions than the current data tensor.
        """
        if self.data.shape[0] != data.shape[0]:
            raise ValueError("The batch size cannot be changed.")
        if self.data.shape[2:] != data.shape[2:]:
            raise ValueError("The spatial dimensions cannot be changed.")

        return self.__class__(
            data,
            image_sizes=self._image_sizes,
            mask=self._mask,
            check_validity=False,
        )

    def to_sequences(self) -> BatchedSequences:
        """Convert the batched images to a batch of sequences."""
        # We can't use the following code because the padding tokens in the batched
        # sequences would no longer be at the end of the sequences, but rather
        # scattered throughout the sequences.
        # data = self._data.flatten(2).permute(0, 2, 1)
        # mask = self._mask.flatten(1)
        # sizes = tuple(s[0] * s[1] for s in self._image_sizes)
        # return BatchedSequences(data, sizes, mask, check_validity=False)
        if not self.is_padded():
            return BatchedSequences(self._data.flatten(2).permute(0, 2, 1))
        else:
            flattened_images = [image.flatten(1).T for image in self.unbatch()]
            return BatchedSequences.batch(flattened_images)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        if self.device == torch.device(device):
            return self

        return self.__class__(
            self._data.to(device, non_blocking=non_blocking),
            image_sizes=self._image_sizes,
            mask=self._mask.to(device, non_blocking=non_blocking)
            if self._mask is not None
            else None,
            check_validity=False,
        )

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Get the number of images in the batch."""
        return self._data.shape[0]

    def __getitem__(self, index: int) -> Tensor[Literal["C H W"], Number]:
        """Get the image in the batch at the given index."""
        h, w = self.image_sizes[index]
        return self._data[index, :, :h, :w]

    def __str__(self) -> str:
        """Get the string representation of the batched images."""
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, dtype={self.dtype}, device={self.device})"
        )

    def __repr__(self) -> str:
        """Get the string representation of the batched images."""
        return str(self)

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_data", "_image_sizes", "_mask")


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _compute_sizes_from_mask(
    mask: Tensor[Literal["B H W"], bool],
) -> tuple[tuple[int, int], ...]:
    """Get the sizes of the images from the mask.

    Args:
        mask: The mask indicating which pixels are padded and which not.
            The mask is `True` for valid pixels and `False` for padded pixels.

    Returns:
        The sizes of the images.
    """
    sizes: list[tuple[int, int]] = []
    for m in mask:
        h = int(m.sum(0).max().item())
        w = int(m.sum(1).max().item())

        sizes.append((h, w))

    return tuple(sizes)


def _compute_mask_from_sizes(
    sizes: tuple[tuple[int, int], ...],
    max_height: int,
    max_width: int,
    device: torch.device,
) -> Tensor[Literal["B H W"], bool]:
    """Get the mask from the image sizes.

    Args:
        sizes: The sizes of the images.
        max_height: The maximum height of the images.
        max_width: The maximum width of the images.
        device: The device to put the mask on.

    Returns:
        The mask indicating which pixels are padded and which not.
        The mask is `True` for valid pixels and `False` for padded pixels.
    """
    mask = torch.zeros(
        (len(sizes), max_height, max_width), dtype=torch.bool, device=device
    )

    for i, size in enumerate(sizes):
        mask[i, : size[0], : size[1]] = True

    return mask


def _check_images(images: Sequence[torch.Tensor]) -> None:
    """Check that the images are valid.

    Args:
        images: The images to check.

    Raises:
        ValueError: If no tensors are provided.
        ValueError: If any tensor is not three-dimensional.
        ValueError: If any tensor does not have the same number of channels.
        ValueError: If any tensor does not have the same dtype.
        ValueError: If any tensor is not on the same device.
    """
    if len(images) == 0:
        raise ValueError("At least one image must be provided.")

    if any(image.ndim != 3 for image in images):
        raise ValueError("All images must have 3 dimensions.")

    if any(image.shape[0] != images[0].shape[0] for image in images):
        raise ValueError("All images must have the same number of channels.")

    if any(image.dtype != images[0].dtype for image in images):
        raise ValueError("All images must have the same dtype.")

    if any(image.device != images[0].device for image in images):
        raise ValueError("All images must be on the same device.")


def _check_mask_sizes(
    data: Tensor[Literal["B C H W"], Number],
    mask: Tensor[Literal["B H W"], bool] | None,
    image_sizes: tuple[tuple[int, int], ...] | None,
) -> None:
    """Check that the mask and image sizes are valid."""
    if mask is not None:
        if mask.device != data.device:
            raise ValueError("The data and mask must be on the same device.")
        if mask.dtype != torch.bool:
            raise ValueError("The mask must be of dtype bool.")
        if data.shape[0] != mask.shape[0] or data.shape[2:] != mask.shape[1:]:
            raise ValueError("The data and mask are incompatible.")

    if image_sizes is not None:
        if len(image_sizes) != len(data):
            raise ValueError("The data and image_sizes are incompatible.")
        if any((h > data.shape[2] or w > data.shape[3]) for h, w in image_sizes):
            raise ValueError("The data and image_sizes are incompatible.")

    if mask is not None and image_sizes is not None:
        mask_image_sizes = _compute_sizes_from_mask(mask)
        if image_sizes != mask_image_sizes:
            raise ValueError("The image_sizes and mask are incompatible.")
