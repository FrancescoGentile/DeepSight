##
##
##

from collections.abc import Sequence
from typing import Literal, Self

import torch

from deepsight.typing import Moveable, Number, Tensor


class BatchedSequences(Moveable):
    """Structure to hold a batch of sequences as a single tensor.

    The tensor is obtained by padding the sequences to the largest length in
    the batch. Since the sequences are padded, a mask is also stored to
    indicate which elements are padded and which not. The mask is True for
    padded elements and False for valid elements.
    """

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        data: Tensor[Literal["B L D"], Number],
        sequence_lengths: tuple[int, ...] | None = None,
        mask: Tensor[Literal["B L"], bool] | None = None,
        *,
        check_validity: bool = True,
    ) -> None:
        """Initialize the batched sequences.

        !!! note

            If neither `sequence_lengths` nor `mask` is provided, it is assumed that
            the sequences are not padded (i.e., all sequences have the same length).

        Args:
            data: The tensor containing the batched sequences.
            sequence_lengths: The lengths of the sequences in the batch.
            mask: The mask indicating which elements are padded and which not.
                The mask is `True` for padded elements and `False` for valid elements.
                If not provided, the mask is computed from the sequence lengths.
            check_validity: Whether to check the validity of the inputs.

        Raises:
            ValueError: If `sequence_lengths` and `mask` are both provided and are
                incompatible.
            ValueError: If the `data` and `mask` (thus `sequence_lengths`) are
                incompatible.
        """
        match sequence_lengths, mask:
            case None, None:
                sequence_lengths = (data.shape[1],) * data.shape[0]
                mask = torch.zeros(data.shape[:2], dtype=torch.bool, device=data.device)
                check_validity = False
            case None, _:
                sequence_lengths = _compute_lengths_from_mask(mask)  # type: ignore
            case _, None:
                mask = _compute_mask_from_lengths(
                    sequence_lengths, data.shape[1], data.device
                )
            case _, _:
                if check_validity:
                    mask_lengths = _compute_lengths_from_mask(mask)
                    if mask_lengths != sequence_lengths:
                        raise ValueError(
                            "The sequence_lengths and mask are incompatible."
                        )

        if check_validity:
            if mask.device != data.device:  # type: ignore
                raise ValueError("The data and mask must be on the same device.")
            if mask.dtype != torch.bool:  # type: ignore
                raise ValueError("The mask must be of dtype bool.")
            if data.shape[:-1] != mask.shape:  # type: ignore
                raise ValueError("The data and mask are incompatible.")

        self._data = data
        self._sequence_lengths = sequence_lengths
        self._mask: Tensor = mask  # type: ignore

    @classmethod
    def batch(
        cls,
        sequences: Sequence[Tensor[Literal["L D"], Number]],
        padding_value: int | float = 0.0,
    ) -> Self:
        """Batch a list of sequences.

        Args:
            sequences: The sequences to batch.
            padding_value: The value to use for padding.

        Returns:
            The batched sequences.
        """
        _check_sequences(sequences)

        sequence_lengths = tuple(sequences.shape[0] for sequences in sequences)
        max_length = max(sequence_lengths)

        data = torch.full(
            (len(sequences), max_length, sequences[0].shape[1]),
            padding_value,
            dtype=sequences[0].dtype,
            device=sequences[0].device,
        )

        for i, sequence in enumerate(sequences):
            data[i, : sequence.shape[0]].copy_(sequence)

        return cls(data, sequence_lengths=sequence_lengths, check_validity=False)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def data(self) -> Tensor[Literal["B L D"], Number]:
        """The tensor containing the batched sequences.

        The tensor has shape `(B, L, D)`, where `B` is the batch size, `L` is the
        maximum length of the sequences in the batch, and `D` is the dimension
        of the sequences.
        """
        return self._data

    @property
    def sequence_lengths(self) -> tuple[int, ...]:
        """The lengths of the sequences in the batch.

        These are the lengths of the sequences before padding.
        """
        return self._sequence_lengths

    @property
    def mask(self) -> Tensor[Literal["B L"], bool]:
        """The mask indicating which elements are padded and which not.

        The mask is `True` for padded elements and `False` for valid elements.
        """
        return self._mask

    @property
    def shape(self) -> torch.Size:
        """The shape of the batched sequences."""
        return self._data.shape

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the batched sequences."""
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        """The device of the batched sequences."""
        return self._data.device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def unbatch(self) -> tuple[Tensor[Literal["L D"], Number], ...]:
        """Unbatch the sequences."""
        return tuple(
            self._data[i, :length] for i, length in enumerate(self._sequence_lengths)
        )

    def replace(self, data: Tensor[Literal["B L D"], Number]) -> Self:
        """Replace the data tensor.

        Raises:
            ValueError: If the data and mask (thus sequence_lengths) are incompatible.
        """
        if data.shape[:-1] != self._data.shape[:-1]:
            raise ValueError("The data and mask are incompatible.")

        return self.__class__(
            data,
            sequence_lengths=self._sequence_lengths,
            mask=self._mask,
            check_validity=False,
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        if self.device == torch.device(device):
            return self

        return self.__class__(
            self._data.to(device, non_blocking=non_blocking),
            sequence_lengths=self._sequence_lengths,
            mask=self._mask.to(device, non_blocking=non_blocking),
            check_validity=False,
        )

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Get the number of sequences in the batch."""
        return len(self._sequence_lengths)

    def __getitem__(self, index: int) -> Tensor[Literal["L D"], Number]:
        """Get the sequence in the batch at the given index."""
        return self._data[index, : self._sequence_lengths[index]]

    def __str__(self) -> str:
        """Get the string representation of the batched sequences."""
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, dtype={self.dtype}, device={self.device})"
        )

    def __repr__(self) -> str:
        """Get the string representation of the batched sequences."""
        return str(self)

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_data", "_sequence_lengths", "_mask")


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _compute_lengths_from_mask(mask: Tensor[Literal["B L"], bool]) -> tuple[int, ...]:
    """Compute the lengths of the sequences from the mask.

    Args:
        mask: The mask indicating which elements are padded and which not.
            The mask is `True` for padded elements and `False` for valid elements.

    Returns:
        The lengths of the sequences in the batch.
    """
    lengths = mask.shape[1] - mask.sum(dim=1)
    return tuple(lengths.tolist())  # type: ignore


def _compute_mask_from_lengths(
    sequence_lengths: tuple[int, ...], max_length: int, device: torch.device
) -> Tensor[Literal["B L"], bool]:
    """Compute the mask from the sequence lengths.

    Args:
        sequence_lengths: The lengths of the sequences in the batch.
        max_length: The maximum length of the sequences in the batch.
        device: The device to put the mask on.

    Returns:
        The mask indicating which elements are padded and which not.
        The mask is `True` for padded elements and `False` for valid elements.
    """
    indices = torch.arange(max_length, device=device)
    indices = indices.expand(len(sequence_lengths), max_length)

    mask = indices >= torch.tensor(sequence_lengths, device=device).unsqueeze(1)
    return mask


def _check_sequences(sequences: Sequence[torch.Tensor]) -> None:
    """Check that the sequences are valid.

    Args:
        sequences: The sequences to check.

    Raises:
        ValueError: If no sequences are provided.
        ValueError: If any sequence does not have 2 dimensions.
        ValueError: If any sequence does not have the same number of features.
        ValueError: If any sequence does not have the same dtype.
        ValueError: If any sequence is not on the same device.
    """
    if len(sequences) == 0:
        raise ValueError("At least one sequence must be provided.")

    if any(sequence.ndim != 2 for sequence in sequences):
        raise ValueError("All sequences must have 2 dimensions.")

    if any(sequence.shape[1] != sequences[0].shape[1] for sequence in sequences):
        raise ValueError("All sequences must have the same number of features.")

    if any(sequence.dtype != sequences[0].dtype for sequence in sequences):
        raise ValueError("All sequences must have the same dtype.")

    if any(sequence.device != sequences[0].device for sequence in sequences):
        raise ValueError("All sequences must be on the same device.")
