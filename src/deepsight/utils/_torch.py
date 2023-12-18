##
##
##

import torch


def is_float_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a float tensor."""
    return torch.is_floating_point(tensor)


def is_integer_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is an integer tensor."""
    return tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
