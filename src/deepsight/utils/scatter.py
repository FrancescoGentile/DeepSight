##
##
##

"""Scatter operations for PyTorch tensors."""

from torch import Tensor

from ._misc import is_float_tensor


def scatter_sum(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_output_size: int | None = None,
) -> Tensor:
    """Scatter sum operation."""
    index = _broadcast(index, src, dim)
    out = _create_output_tensor(src, index, dim, dim_output_size)
    out.scatter_reduce_(dim, index, src, reduce="sum", include_self=False)

    return out


def scatter_mul(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_output_size: int | None = None,
) -> Tensor:
    """Scatter multiply operation."""
    index = _broadcast(index, src, dim)
    out = _create_output_tensor(src, index, dim, dim_output_size)
    out.scatter_reduce_(dim, index, src, reduce="prod", include_self=False)

    return out


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_output_size: int | None = None,
) -> Tensor:
    """Scatter mean operation."""
    index = _broadcast(index, src, dim)
    out = _create_output_tensor(src, index, dim, dim_output_size)
    out.scatter_reduce_(dim, index, src, reduce="mean", include_self=False)

    return out


def scatter_min(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_output_size: int | None = None,
) -> Tensor:
    """Scatter min operation."""
    index = _broadcast(index, src, dim)
    out = _create_output_tensor(src, index, dim, dim_output_size)
    out.scatter_reduce_(dim, index, src, reduce="amin", include_self=False)

    return out


def scatter_max(  # noqa
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_output_size: int | None = None,
) -> Tensor:
    """Scatter max operation."""
    index = _broadcast(index, src, dim)
    out = _create_output_tensor(src, index, dim, dim_output_size)
    out.scatter_reduce_(dim, index, src, reduce="amax", include_self=False)

    return out


def scatter_softmax(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_output_size: int | None = None,
) -> Tensor:
    """Scatter softmax operation."""
    if not is_float_tensor(src):
        raise TypeError("Expected a float tensor.")

    index = _broadcast(index, src, dim)
    max_value_per_index = scatter_max(src, index, dim, dim_output_size)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_exp_scores = recentered_scores.exp_()

    sum_per_index = scatter_sum(recentered_exp_scores, index, dim, dim_output_size)
    normalized_scores = recentered_exp_scores / sum_per_index.gather(dim, index)

    return normalized_scores


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _broadcast(src: Tensor, other: Tensor, dim: int) -> Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())

    return src


def _create_output_tensor(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_output_size: int | None = None,
) -> Tensor:
    out_size = list(src.size())
    if dim_output_size is not None:
        out_size[dim] = dim_output_size
    elif index.numel() == 0:
        out_size[dim] = 0
    else:
        out_size[dim] = int(index.max()) + 1

    return src.new_zeros(out_size)
