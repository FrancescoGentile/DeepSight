# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Literal, Self

import torch

from deepsight.typing import Detachable, Moveable, Number, Tensor


class BatchedSequences(Detachable, Moveable):
    """Structure to hold a batch of sequences as a single tensor."""

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
                The mask is `True` for valid elements and `False` for padded elements.
                If not provided, the mask is computed from the sequence lengths.
            check_validity: Whether to check the validity of the inputs.
                Checking the validity may be expensive, so it can be disabled
                if the inputs are known to be valid.

        Raises:
            ValueError: If `sequence_lengths` and `mask` are both provided and are
                incompatible.
            ValueError: If the `data` and `mask`/`sequence_lengths` are incompatible.
        """
        if sequence_lengths is None and mask is None:
            sequence_lengths = (data.shape[1],) * data.shape[0]

        if check_validity:
            _check_mask_lengths(data, mask, sequence_lengths)

        self._data = data
        self._sequence_lengths = sequence_lengths
        self._mask = mask

    @classmethod
    def batch(
        cls,
        sequences: Sequence[Tensor[Literal["L D"], Number]],
        padding_value: float = 0.0,
    ) -> Self:
        """Batch a list of sequences.

        The sequences are padded to the length of the longest sequence in the batch
        and then stacked into a single tensor.

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
        if self._sequence_lengths is None:
            assert self._mask is not None
            self._sequence_lengths = _compute_lengths_from_mask(self._mask)

        return self._sequence_lengths

    @property
    def padding_mask(self) -> Tensor[Literal["B L"], bool]:
        """The mask indicating which elements are padded and which not.

        The mask is `True` for valid elements and `False` for padded elements.
        """
        if self._mask is None:
            assert self._sequence_lengths is not None
            self._mask = _compute_mask_from_lengths(
                self._sequence_lengths, self._data.shape[1], self._data.device
            )

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

    def is_padded(self) -> bool:
        """Whether the sequences are padded."""
        return any(length < self._data.shape[1] for length in self.sequence_lengths)

    def unbatch(self) -> tuple[Tensor[Literal["L D"], Number], ...]:
        """Unbatch the sequences."""
        return tuple(self[i] for i in range(len(self)))

    def new_with(self, data: Tensor[Literal["B L D"], Number]) -> Self:
        """Return new batched sequences with the given data tensor.

        Raises:
            ValueError: If the new data tensor does not have the same shape as the
                current data tensor (except for the feature dimension, i.e., the
                last dimension).
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
            mask=self._mask.to(device, non_blocking=non_blocking)
            if self._mask is not None
            else None,
            check_validity=False,
        )

    def detach(self) -> Self:
        return self.__class__(
            self._data.detach(),
            sequence_lengths=self._sequence_lengths,
            mask=self._mask.detach() if self._mask is not None else None,
            check_validity=False,
        )

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Get the number of sequences in the batch."""
        return self._data.shape[0]

    def __getitem__(self, index: int) -> Tensor[Literal["L D"], Number]:
        """Get the sequence in the batch at the given index."""
        return self._data[index, : self.sequence_lengths[index]]

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
            The mask is `True` for valid elements and `False` for padded elements.

    Returns:
        The lengths of the sequences in the batch.
    """
    lengths = mask.sum(dim=1)
    return tuple(lengths.tolist())


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
        The mask is `True` for valid elements and `False` for padded elements.
    """
    indices = torch.arange(max_length, device=device)
    indices = indices.expand(len(sequence_lengths), max_length)

    mask = indices < torch.tensor(sequence_lengths, device=device).unsqueeze(1)
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


def _check_mask_lengths(
    data: Tensor[Literal["B L D"], Number],
    mask: Tensor[Literal["B L"], bool] | None,
    sequence_lengths: tuple[int, ...] | None,
) -> None:
    """Check that the mask and sequence lengths are valid."""
    if mask is not None:
        if mask.device != data.device:
            raise ValueError("The data and mask must be on the same device.")
        if mask.dtype != torch.bool:
            raise ValueError("The mask must be of dtype bool.")
        if data.shape[:-1] != mask.shape:
            raise ValueError("The data and mask are incompatible.")

    if sequence_lengths is not None:
        if len(sequence_lengths) != data.shape[0]:
            raise ValueError("The data and sequence_lengths are incompatible.")
        if any(length > data.shape[1] for length in sequence_lengths):
            raise ValueError("The data and sequence_lengths are incompatible.")

    if mask is not None and sequence_lengths is not None:
        mask_lengths = _compute_lengths_from_mask(mask)
        if mask_lengths != sequence_lengths:
            raise ValueError("The sequence_lengths and mask are incompatible.")
