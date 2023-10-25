##
##
##

from numbers import Number
from typing import Annotated

import torch
from torch import Tensor
from typing_extensions import Self

from deepsight.typing import Moveable


class Graph(Moveable):
    """A structure representing a graph."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        adjacency_matrix: Annotated[Tensor, "N N", Number],
        node_features: Annotated[Tensor, "N D", Number],
        edge_features: Annotated[Tensor, "E C", Number] | None = None,
    ) -> None:
        """Initialize a graph.

        Args:
            adjacency_matrix: The adjacency matrix of the graph. The adjacency matrix
                can be passed as a dense or sparse tensor.
            node_features: The node features of the graph.
            edge_features: The edge features of the graph. If provided, the number of
                rows must match the number of edges in the adjacency matrix.
        """
        adjacency_matrix = adjacency_matrix.to_sparse_coo()
        _check_tensors(adjacency_matrix, node_features, edge_features)

        self._adj = adjacency_matrix
        self._node_features = node_features
        self._edge_features = edge_features

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def adjacency_matrix(self) -> Annotated[Tensor, "N N", Number, torch.sparse_coo]:
        """The adjacency matrix of the graph as a sparse COO tensor."""
        return self._adj

    @property
    def node_features(self) -> Annotated[Tensor, "N D", Number]:
        """The node features of the graph."""
        return self._node_features

    @property
    def edge_features(self) -> Annotated[Tensor, "E C", Number] | None:
        """The edge features of the graph."""
        return self._edge_features

    @property
    def num_nodes(self) -> int:
        """The number of nodes in the graph."""
        return self._adj.shape[0]

    @property
    def num_edges(self) -> int:
        """The number of edges in the graph."""
        return self._adj.indices().shape[1]

    @property
    def device(self) -> torch.device:
        return self._node_features.device

    # ----------------------------------------------------------------------- #
    # Methods
    # ----------------------------------------------------------------------- #

    def replace(
        self,
        node_features: Annotated[Tensor, "N D", Number] | None = None,
        edge_features: Annotated[Tensor, "E C", Number] | None = None,
    ) -> Self:
        """Return a new graph with the given node and edge features.

        Args:
            node_features: The node features of the new graph. If `None`, the node
                features of the current graph are used.
            edge_features: The edge features of the new graph. If `None`, the edge
                features of the current graph are used.

        Returns:
            A new graph with the given node and edge features.
        """
        if node_features is None:
            node_features = self.node_features

        if edge_features is None:
            edge_features = self.edge_features

        return self.__class__(
            adjacency_matrix=self._adj,
            node_features=node_features,
            edge_features=edge_features,
        )

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
        if self.device == device:
            return self

        return self.__class__(
            adjacency_matrix=self._adj.to(device, non_blocking=non_blocking),
            node_features=self.node_features.to(device, non_blocking=non_blocking),
            edge_features=self.edge_features.to(device, non_blocking=non_blocking)
            if self.edge_features is not None
            else None,
        )

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges})"
        )

    def __str__(self) -> str:
        return repr(self)

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_adj", "_node_features", "_edge_features")


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _check_tensors(adj: Tensor, nodes: Tensor, edges: Tensor | None) -> None:
    """Check that the tensors have the correct shapes and are on the same device."""
    if nodes.ndim != 2:
        raise ValueError(f"The nodes tensor must have 2 dimensions, got {nodes.ndim}.")

    if adj.shape != (nodes.shape[0], nodes.shape[0]):
        raise ValueError(
            f"The adjacency matrix must have shape (N, N), got {adj.shape}."
        )

    if nodes.device != adj.device:
        raise ValueError(
            "The adjacency matrix and the node features must be on the same device."
        )

    if edges is not None:
        if edges.ndim != 2:
            raise ValueError(
                f"The edges tensor must have 2 dimensions, got {edges.ndim}."
            )

        if edges.shape[0] != adj.indices().shape[1]:
            raise ValueError(
                f"The number of rows in the edges tensor must match the number of "
                f"edges in the adjacency matrix, got {edges.shape[0]} and "
                f"{adj.indices().shape[1]} respectively."
            )

        if edges.device != adj.device:
            raise ValueError(
                "The adjacency matrix and the edge features must be on the same device."
            )
