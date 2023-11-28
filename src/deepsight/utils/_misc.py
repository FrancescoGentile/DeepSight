##
##
##

from collections.abc import Iterable

import torch
from torch import Tensor


def is_float_tensor(tensor: Tensor) -> bool:
    """Check if a tensor is a float tensor."""
    return torch.is_floating_point(tensor)


def is_integer_tensor(tensor: Tensor) -> bool:
    """Check if a tensor is an integer tensor."""
    return tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]


def to_2tuple[T](value: T | tuple[T, T]) -> tuple[T, T]:
    """Convert a value to a 2-tuple."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple, got {value}.")
        return value

    return value, value


def to_tuple[T](value: T | Iterable[T]) -> tuple[T, ...]:
    """Convert a value to a tuple."""
    if isinstance(value, tuple):
        return value
    elif isinstance(value, Iterable):
        return tuple(value)
    else:
        return (value,)
