##
##
##

from numbers import Number
from typing import Annotated, overload

import torch
from torch import Tensor

from ._reduce import ReduceOp
from ._scatter import scatter


@overload
def coalesce(
    indices: Annotated[Tensor, "2 N", int],
    values: Annotated[Tensor, "N *", Number],
    size: int | tuple[int, int] | None = None,
    reduce: ReduceOp | str = ReduceOp.SUM,
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> tuple[Annotated[Tensor, "2 N", int], Annotated[Tensor, "N *", Number]]:
    ...


@overload
def coalesce(
    indices: Annotated[Tensor, "2 N", int],
    values: None = None,
    size: int | tuple[int, int] | None = None,
    reduce: ReduceOp | str = ReduceOp.SUM,
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> tuple[Annotated[Tensor, "2 N", int], None]:
    ...


def coalesce(
    indices: Annotated[Tensor, "2 N", int],
    values: Annotated[Tensor, "N *", Number] | None = None,
    size: int | tuple[int, int] | None = None,
    reduce: ReduceOp | str = ReduceOp.SUM,
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> tuple[Annotated[Tensor, "2 N", int], Annotated[Tensor, "N *", Number] | None]:
    num_edges = indices.size(1)
    match size:
        case None:
            num_nodes = int(indices.max().item()) + 1
        case int(size):
            num_nodes = size
        case (int(size0), int(size1)):
            num_nodes = max(size0, size1)

    idx = indices.new_empty((num_edges + 1,))
    idx[0] = -1
    idx[1:] = indices[0] if sort_by_row else indices[1]
    idx[1:].mul_(num_nodes).add_(indices[1] if sort_by_row else indices[0])

    if not is_sorted:
        idx[1:], perm = idx[1:].sort()
        indices = indices[:, perm]

    mask = idx[1:] > idx[:-1]
    if mask.all():
        return indices, values

    indices = indices[:, mask]

    if values is None:
        return indices, None

    dim_size = indices.size(1)
    idx = torch.arange(0, num_edges, device=indices.device)
    idx.sub_(mask.logical_not_().cumsum(0))
    values = scatter(values, idx, 0, dim_size, reduce)

    return indices, values
