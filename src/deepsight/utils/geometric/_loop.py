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
def add_remaining_self_loops(
    indices: Annotated[Tensor, "2 N", int],
    values: Annotated[Tensor, "N *", Number],
    size: int | tuple[int, int] | None = None,
    fill_value: Number | Tensor | ReduceOp | str = 1.0,  # type: ignore
) -> tuple[Annotated[Tensor, "2 N", int], Annotated[Tensor, "N *", Number]]: ...


@overload
def add_remaining_self_loops(
    indices: Annotated[Tensor, "2 N", int],
    values: None = None,
    size: int | tuple[int, int] | None = None,
    fill_value: None = None,
) -> tuple[Annotated[Tensor, "2 N", int], None]: ...


def add_remaining_self_loops(
    indices: Annotated[Tensor, "2 N", int],
    values: Annotated[Tensor, "N *", Number] | None = None,
    size: int | tuple[int, int] | None = None,
    fill_value: Number | Tensor | ReduceOp | str | None = None,
) -> tuple[Annotated[Tensor, "2 N", int], Annotated[Tensor, "N *", Number] | None]:
    """Adds remaining self-loop edges to the adjacency matrix.

    Args:
        indices: The indices of the non-zero elements in the adjacency matrix.
        values: The values associated to the non-zero elements in the adjacency
            matrix. These can correspond to edge weights if the adjacency matrix
            is weighted or to arbitrary edge features.
        size: The shape of the corresponding dense adjacency matrix. If set to
            `None`, the maximum node index is used to infer the size. If set to
            a tuple, the corresponding dense adjacency matrix must be square.
        fill_value: The way to generate the values of the self-loops (if `values` is not
            `None`). If a `Number` or `Tensor`, all self-loop values are set to this
            value. If a `ReduceOp` or `str`, the self-loop values are computed by
            aggregating all features of the edges that point to the specific node,
            according to the reduce operation.

    Returns:
        A tuple holding the new indices and (optionally) the new values.
    """
    match size:
        case None:
            num_nodes = int(indices.max().item()) + 1
        case int(size):
            num_nodes = size
        case (int(size0), int(size1)):
            if size0 != size1:
                raise ValueError(
                    "Cannot add self-loops for non-square adjacency matrices."
                )
            num_nodes = size0

    mask = indices[0] != indices[1]

    loop_indices = torch.arange(num_nodes, dtype=indices.dtype, device=indices.device)
    loop_indices = loop_indices.unsqueeze(0).expand(2, -1)

    if values is not None:
        if fill_value is None:
            raise ValueError("Expected to fill self-loop values, found None instead.")

        if isinstance(fill_value, Number):
            loop_values = values.new_full((num_nodes,) + values.shape[1:], fill_value)  # type: ignore
        elif isinstance(fill_value, Tensor):
            loop_values = fill_value.to(values.device, values.dtype)
            if loop_values.shape == (num_nodes,) + values.shape[1:]:
                pass
            elif loop_values.shape == values.shape[1:]:
                sizes = [num_nodes] + [-1] * loop_values.ndim
                loop_values = loop_values.unsqueeze(0).expand(*sizes)
            else:
                raise ValueError(
                    f"Expected fill_value to have shape {num_nodes, *values.shape[1:]} "
                    f"or {values.shape[1:]}, found {loop_values.shape}."
                )
        else:
            loop_values = scatter(
                values, indices[1], dim=0, dim_output_size=num_nodes, reduce=fill_value
            )

        not_mask = ~mask
        loop_values[indices[0, not_mask]] = values[not_mask]
        values = torch.cat([values[mask], loop_values], dim=0)

    indices = torch.cat([indices[:, mask], loop_indices], dim=1)

    return indices, values
