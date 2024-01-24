##
##
##

from typing import Literal, Self

import torch

from deepsight.typing import Moveable, Tensor

# make the class MultiHeadAttentionLayer more modular


class Mask(Moveable):
    r"""Mask used when computing attention scores.

    The shape of the mask is (B, H, Q, K) where B is the batch size, H is the number of
    attention heads, Q is the number of query elements, and K is the number of key
    elements. Each dimension can be broadcasted to match the dimensions of the attention
    score matrix (i.e. it is possible to have B = 1, H = 1, Q = 1, or K = 1).

    If the mask is a float tensor, the attention mask is applied by adding the mask
    to the attention logit matrix.

    If the mask is a boolean tensor, a `False` value indicates that the corresponding
    element should be ignored in the attention computation. This is equivalent to
    setting the corresponding attention logit to $-\infty$ in the attention logit
    matrix.
    """

    # ----------------------------------------------------------------------- #
    # Constructor and Factory methods
    # ----------------------------------------------------------------------- #

    def __init__(self, mask: Tensor[Literal["B H Q K"], bool | float]) -> None:
        """Initialize the attention mask.

        Args:
            mask: The attention mask tensor.
        """
        if mask.ndim != 4:
            raise ValueError(
                f"The attention mask tensor must be 4-dimensional, but the given "
                f"tensor is {mask.ndim}-dimensional."
            )
        if not (torch.is_floating_point(mask) or mask.dtype == torch.bool):
            raise ValueError(
                f"The attention mask tensor must be a boolean or float tensor, but "
                f"the given tensor has dtype {mask.dtype}."
            )

        self._mask = mask

    @classmethod
    def from_key_padding_mask(
        cls,
        padding_mask: Tensor[Literal["B K"], bool],
        padding_value: bool = False,
    ) -> Self:
        """Create an attention mask from a key padding mask.

        Args:
            padding_mask: The key padding mask tensor. This mask should be a boolean
                tensor with shape (B, K) where B is the batch size and K is the number
                of key elements indicating whether the corresponding key element is
                a valid key element (that should be used in the attention computation)
                or a padding element (that should be masked out in the attention
                computation).
            padding_value: The value used to indicate a padding element in the key
                padding mask.

        Returns:
            The attention mask.
        """
        mask = padding_mask[:, None, None]  # (B, 1, 1, K)
        if padding_value:
            mask = ~mask

        return cls(mask)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def tensor(self) -> Tensor[Literal["B H Q K"], bool | float]:
        """The attention mask tensor."""
        return self._mask

    @property
    def device(self) -> torch.device:
        """The device of the attention mask tensor."""
        return self._mask.device

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def is_float(self) -> bool:
        """Check whether the attention mask is a float tensor."""
        return torch.is_floating_point(self._mask)

    def is_bool(self) -> bool:
        """Check whether the attention mask is a boolean tensor."""
        return self._mask.dtype == torch.bool

    def combine_with(self, other: Self) -> Self:
        r"""Combine this attention mask with another attention mask.

        The two attention masks are combined in the following way:
        - If both attention masks are float tensors, the two tensors are added.
        - If one attention mask is a float tensor and the other is a boolean tensor,
            the boolean tensor is used to mask out the float tensor. Each element in
            the new attention mask is equal to the corresponding element in the float
            tensor if the corresponding element in the boolean tensor is `True`,
            otherwise the element is set to $-\infty$.
        - If both attention masks are boolean tensors, the two tensors are combined
            using the logical AND operator.

        Args:
            other: The other attention mask.

        Returns:
            The combined attention mask.
        """
        match (self.is_float(), other.is_float()):
            case (True, True):
                mask = self._mask + other._mask
            case (True, False):
                mask = self._mask.masked_fill(other._mask.logical_not(), -torch.inf)
            case (False, True):
                mask = other._mask.masked_fill(self._mask.logical_not(), -torch.inf)
            case (False, False):
                mask = self._mask.logical_and(other._mask)
            case _:
                # to make type checker happy
                raise RuntimeError("Unreachable code.")

        return self.__class__(mask)

    def apply_to(
        self,
        attn_logit: Tensor[Literal["B H Q K"], float],
        inplace: bool = False,
    ) -> Tensor[Literal["B H Q K"], float]:
        """Apply the attention mask to the attention logit matrix.

        Args:
            attn_logit: The attention logit matrix.
            inplace: Whether to apply the attention mask inplace, i.e. modify the
                attention logit matrix in-place.

        Returns:
            The masked attention logit matrix.
        """
        match (self.is_float(), inplace):
            case (True, True):
                attn_logit += self._mask
            case (True, False):
                attn_logit = attn_logit + self._mask
            case (False, True):
                attn_logit.masked_fill_(self._mask.logical_not(), -torch.inf)
            case (False, False):
                attn_logit = attn_logit.masked_fill(
                    self._mask.logical_not(), -torch.inf
                )
            case _:
                # to make type checker happy
                raise RuntimeError("Unreachable code.")

        return attn_logit

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        return self.__class__(self._mask.to(device, non_blocking=non_blocking))
