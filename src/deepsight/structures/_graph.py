##
##
##

from collections.abc import Iterable, Sequence
from typing import Annotated, Literal, overload

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from typing_extensions import Self

from deepsight.typing import Moveable, number

from ._common import BatchMode


class Graph(Moveable):
    """A structure representing a graph."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        adjacency_matrix: Annotated[Tensor, "N N", number],
        node_features: Annotated[Tensor, "N D", number],
        edge_features: Annotated[Tensor, "E C", number] | None = None,
        num_nodes: Sequence[int] | None = None,
        num_edges: Sequence[int] | None = None,
    ) -> None:
        """Initialize a graph.

        Args:
            adjacency_matrix: The adjacency matrix of the graph. The adjacency matrix
                can be passed as a dense or sparse tensor.
            node_features: The node features of the graph.
            edge_features: The edge features of the graph. If provided, the number of
                rows must match the number of edges in the adjacency matrix.
            num_nodes: If the graph is obtained by batching multiple graphs, this is
                the number of nodes in each graph. If the graph is not batched, this
                argument should be `None`.
            num_edges: If the graph is obtained by batching multiple graphs, this is
                the number of edges in each graph. If the graph is not batched, this
                argument should be `None`.
        """
        if (num_nodes is not None) != (num_edges is not None):
            raise ValueError(
                "The arguments `num_nodes` and `num_edges` must be both `None` or both "
                "not `None`."
            )

        if num_nodes is None and num_edges is None:
            num_nodes = (node_features.shape[0],)
            num_edges = (adjacency_matrix.indices().shape[1],)
        elif num_nodes is None or num_edges is None:
            raise ValueError(
                "The arguments `num_nodes` and `num_edges` must be both `None` or both "
                "not `None`."
            )

        adjacency_matrix = adjacency_matrix.to_sparse_coo()

        _check_tensors(
            adjacency_matrix, node_features, edge_features, num_nodes, num_edges
        )

        self._adj = adjacency_matrix
        self._node_features = node_features
        self._edge_features = edge_features
        self._num_nodes = num_nodes
        self._num_edges = num_edges

    @classmethod
    def batch(cls, graphs: Sequence[Self]) -> Self:
        """Batch multiple graphs into a single graph.

        !!! note

            If the given list contains only one graph, this method returns the graph
            itself.
        """
        if len(graphs) == 0:
            raise ValueError("Expected at least one graph, got none.")
        if len(graphs) == 1:
            return graphs[0]

        _check_graphs(graphs)  # type: ignore

        num_nodes = [graph.num_nodes() for graph in graphs]
        num_edges = [graph.num_edges() for graph in graphs]

        node_features = torch.cat([graph.node_features() for graph in graphs])
        if graphs[0].edge_features() is not None:
            edge_features = torch.cat([graph.edge_features() for graph in graphs])  # type: ignore
        else:
            edge_features = None

        adj_matrices = [graph.adjacency_matrix().coalesce() for graph in graphs]

        indices = []
        node_offset = 0
        for adj_matrix, n_nodes in zip(adj_matrices, num_nodes, strict=True):
            if node_offset > 0:
                indices.append(adj_matrix.indices() + node_offset)
            else:
                indices.append(adj_matrix.indices())
            node_offset += n_nodes

        total_num_nodes = sum(num_nodes)
        adj = torch.sparse_coo_tensor(
            indices=torch.cat(indices, dim=1),
            values=torch.cat([adj_matrix.values() for adj_matrix in adj_matrices]),
            size=(total_num_nodes, total_num_nodes),
            is_coalesced=True,
        )

        return cls(
            adjacency_matrix=adj,
            node_features=node_features,
            edge_features=edge_features,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        return self._node_features.device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    @overload
    def num_nodes(
        self, batch_mode: Literal[BatchMode.CONCAT] = BatchMode.CONCAT
    ) -> int: ...

    @overload
    def num_nodes(self, batch_mode: Literal[BatchMode.SEQUENCE]) -> tuple[int, ...]: ...

    def num_nodes(
        self,
        batch_mode: Literal[BatchMode.CONCAT, BatchMode.SEQUENCE] = BatchMode.CONCAT,
    ) -> int | tuple[int, ...]:
        """Get the number of nodes in the graph.

        Args:
            batch_mode: The batch mode to use. If `BatchMode.CONCAT`, the total number
                of nodes of the batched graphs is returned, otherwise the number of
                nodes of each graph is returned.
        """
        match batch_mode:
            case BatchMode.CONCAT:
                return sum(self._num_nodes)
            case BatchMode.SEQUENCE:
                return tuple(self._num_nodes)

    @overload
    def num_edges(
        self, batch_mode: Literal[BatchMode.CONCAT] = BatchMode.CONCAT
    ) -> int: ...

    @overload
    def num_edges(self, batch_mode: Literal[BatchMode.SEQUENCE]) -> tuple[int, ...]: ...

    def num_edges(
        self,
        batch_mode: Literal[BatchMode.CONCAT, BatchMode.SEQUENCE] = BatchMode.CONCAT,
    ) -> int | tuple[int, ...]:
        """Get the number of edges in the graph.

        Args:
            batch_mode: The batch mode to use. If `BatchMode.CONCAT`, the total number
                of edges of the batched graphs is returned, otherwise the number of
                edges of each graph is returned.
        """
        match batch_mode:
            case BatchMode.CONCAT:
                return sum(self._num_edges)
            case BatchMode.SEQUENCE:
                return tuple(self._num_edges)

    @overload
    def adjacency_matrix(
        self, batch_mode: Literal[BatchMode.CONCAT] = BatchMode.CONCAT
    ) -> Annotated[Tensor, "N N", number, torch.sparse_coo]: ...

    @overload
    def adjacency_matrix(
        self, batch_mode: Literal[BatchMode.STACK]
    ) -> Annotated[Tensor, "B N N", number, torch.sparse_coo]: ...

    @overload
    def adjacency_matrix(
        self, batch_mode: Literal[BatchMode.SEQUENCE]
    ) -> tuple[Annotated[Tensor, "N N", number, torch.sparse_coo], ...]: ...

    def adjacency_matrix(
        self, batch_mode: BatchMode = BatchMode.CONCAT
    ) -> (
        Annotated[Tensor, "N N", number, torch.sparse_coo]
        | Annotated[Tensor, "B N N", number, torch.sparse_coo]
        | tuple[Annotated[Tensor, "N N", number, torch.sparse_coo], ...]
    ):
        """Get the adjacency matrix of the graph."""
        match batch_mode:
            case BatchMode.CONCAT:
                return self._adj
            case BatchMode.STACK:
                raise NotImplementedError
            case BatchMode.SEQUENCE:
                values = self._adj.values().split_with_sizes(self._num_edges)
                indices = self._adj.indices().split_with_sizes(self._num_edges, dim=1)
                node_offset = 0
                for idx, n_nodes in enumerate(self._num_nodes):
                    if node_offset > 0:
                        indices[idx] = indices[idx] - node_offset
                    node_offset += n_nodes

                return tuple(
                    torch.sparse_coo_tensor(
                        indices=indices[idx],
                        values=values[idx],
                        size=(n_nodes, n_nodes),
                    )
                    for idx, n_nodes in enumerate(self._num_nodes)
                )

    @overload
    def node_features(
        self, batch_mode: Literal[BatchMode.CONCAT] = BatchMode.CONCAT
    ) -> Annotated[Tensor, "N D", number]: ...

    @overload
    def node_features(
        self, batch_mode: Literal[BatchMode.STACK]
    ) -> Annotated[Tensor, "B N D", number]: ...

    @overload
    def node_features(
        self, batch_mode: Literal[BatchMode.SEQUENCE]
    ) -> tuple[Annotated[Tensor, "N D", number], ...]: ...

    def node_features(
        self, batch_mode: BatchMode = BatchMode.CONCAT
    ) -> (
        Annotated[Tensor, "N D", number]
        | Annotated[Tensor, "B N D", number]
        | tuple[Annotated[Tensor, "N D", number], ...]
    ):
        """Get the features of the nodes in the graph."""
        match batch_mode:
            case BatchMode.CONCAT:
                return self._node_features
            case BatchMode.STACK:
                nodes = self._node_features.split_with_sizes(self._num_nodes)
                return pad_sequence(nodes, batch_first=True)
            case BatchMode.SEQUENCE:
                return tuple(self._node_features.split_with_sizes(self._num_nodes))

    @overload
    def edge_features(
        self, batch_mode: Literal[BatchMode.CONCAT] = BatchMode.CONCAT
    ) -> Annotated[Tensor, "E C", number] | None: ...

    @overload
    def edge_features(
        self, batch_mode: Literal[BatchMode.STACK]
    ) -> Annotated[Tensor, "B E C", number] | None: ...

    @overload
    def edge_features(
        self, batch_mode: Literal[BatchMode.SEQUENCE]
    ) -> tuple[Annotated[Tensor, "E C", number], ...] | None: ...

    def edge_features(
        self, batch_mode: BatchMode = BatchMode.CONCAT
    ) -> (
        Annotated[Tensor, "E C", number]
        | Annotated[Tensor, "B E C", number]
        | tuple[Annotated[Tensor, "E C", number], ...]
        | None
    ):
        """Get the features of the edges in the graph."""
        if self._edge_features is None:
            return None

        match batch_mode:
            case BatchMode.CONCAT:
                return self._edge_features
            case BatchMode.STACK:
                edges = self._edge_features.split_with_sizes(self._num_edges)
                return pad_sequence(edges, batch_first=True)
            case BatchMode.SEQUENCE:
                return tuple(self._edge_features.split_with_sizes(self._num_edges))

    def replace(  # noqa
        self,
        node_features: Annotated[Tensor, "N D", number]
        | Annotated[Tensor, "B N D", number]
        | Iterable[Annotated[Tensor, "N D", number]]
        | None = None,
        edge_features: Annotated[Tensor, "E C", number]
        | Annotated[Tensor, "B E C", number]
        | Iterable[Annotated[Tensor, "E C", number]]
        | None = None,
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
        if node_features is None and edge_features is None:
            return self

        if node_features is None:
            node_features = self._node_features
        elif isinstance(node_features, Tensor):
            match node_features.ndim:
                case 2:
                    if node_features.shape[0] != self._node_features.shape[0]:
                        raise ValueError(
                            f"The number of nodes in the node features must match the "
                            f"number of nodes in the current graph, got "
                            f"{node_features.shape[0]} and "
                            f"{self._node_features.shape[0]} respectively."
                        )
                case 3:
                    node_features_list = []
                    for idx, n_nodes in enumerate(self._num_nodes):
                        node_features_list.append(node_features[idx, :n_nodes])
                    node_features = torch.cat(node_features_list)
                case _:
                    raise TypeError("The node features must have 2 or 3 dimensions.")
        else:
            for nf, nnodes in zip(node_features, self._num_nodes, strict=True):
                if nf.shape[0] != nnodes:
                    raise ValueError(
                        "The number of nodes in each batched graph must remain the "
                        f"same, got {nf.shape[0]} and {nnodes} respectively."
                    )

            node_features = torch.cat(list(node_features))

        if edge_features is None:
            edge_features = self._edge_features
        elif isinstance(edge_features, Tensor):
            match edge_features.ndim:
                case 2:
                    if edge_features.shape[0] != sum(self._num_edges):
                        raise ValueError(
                            f"The number of edges in the edge features must match the "
                            f"number of edges in the current graph, got "
                            f"{edge_features.shape[0]} and "
                            f"{sum(self._num_edges)} respectively."
                        )
                case 3:
                    edge_features_list = []
                    for idx, n_edges in enumerate(self._num_edges):
                        edge_features_list.append(edge_features[idx, :n_edges])
                    edge_features = torch.cat(edge_features_list)
                case _:
                    raise TypeError("The edge features must have 2 or 3 dimensions.")
        else:
            for ef, nedges in zip(edge_features, self._num_edges, strict=True):
                if ef.shape[0] != nedges:
                    raise ValueError(
                        "The number of edges in each batched graph must remain the "
                        f"same, got {ef.shape[0]} and {nedges} respectively."
                    )

            edge_features = torch.cat(list(edge_features))

        return self.__class__(
            adjacency_matrix=self._adj,
            node_features=node_features,
            edge_features=edge_features,
            num_nodes=self._num_nodes,
            num_edges=self._num_edges,
        )

    def unbatch(self) -> Iterable[Self]:
        """Unbatch the graph into a sequence of graphs.

        !!! note

            If the graph is not batched, this method returns a sequence containing
            only the graph itself.

        Returns:
            A sequence of graphs.
        """
        if len(self._num_nodes) < 2:
            return (self,)

        node_features = list(self.node_features(batch_mode=BatchMode.SEQUENCE))
        edge_features = self.edge_features(batch_mode=BatchMode.SEQUENCE)
        if edge_features is not None:
            edge_features = list(edge_features)
        else:
            edge_features = [None] * len(self._num_nodes)
        adjacency_matrix = list(self.adjacency_matrix(batch_mode=BatchMode.SEQUENCE))

        return [
            self.__class__(
                adjacency_matrix=adjacency_matrix[idx],
                node_features=node_features[idx],
                edge_features=edge_features[idx],
            )
            for idx in range(len(self._num_nodes))
        ]

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
        if self.device == device:
            return self

        return self.__class__(
            adjacency_matrix=self._adj.to(device, non_blocking=non_blocking),
            node_features=self._node_features.to(device, non_blocking=non_blocking),
            edge_features=self._edge_features.to(device, non_blocking=non_blocking)
            if self._edge_features is not None
            else None,
        )

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_nodes={self.num_nodes()}, "
            f"num_edges={self.num_edges()})"
        )

    def __str__(self) -> str:
        return repr(self)

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_adj", "_node_features", "_edge_features", "_num_nodes", "_num_edges")


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _check_tensors(  # noqa: C901
    adj: Tensor,
    nodes: Tensor,
    edges: Tensor | None,
    num_nodes: Sequence[int],
    num_edges: Sequence[int],
) -> None:
    """Check that the tensors have the correct shapes and are on the same device."""
    if nodes.ndim != 2:
        raise ValueError(f"The nodes tensor must have 2 dimensions, got {nodes.ndim}.")

    if nodes.shape[0] != sum(num_nodes):
        raise ValueError(
            f"The number of nodes in the node features must match the sum of the "
            f"number of nodes of each batched graph, got {nodes.shape[0]} and "
            f"{sum(num_nodes)} respectively."
        )

    if adj.shape != (nodes.shape[0], nodes.shape[0]):
        raise ValueError(
            f"The adjacency matrix must have shape (N, N), got {adj.shape}."
        )

    if adj.indices().shape[1] != sum(num_edges):
        raise ValueError(
            f"The number of edges in the adjacency matrix must match the sum of the "
            f"number of edges of each batched graph, got {adj.indices().shape[1]} and "
            f"{sum(num_edges)} respectively."
        )

    if len(num_nodes) != len(num_edges):
        raise ValueError("Inconsistent number of batches.")

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


def _check_graphs(graphs: Sequence[Graph]) -> None:
    """Check that the graphs are compatible to be batched."""
    if any(
        graphs[0].node_features().shape[1] != graph.node_features().shape[1]
        for graph in graphs
    ):
        raise ValueError("All graphs must have the same number of node features.")

    if any(graphs[0].device != graph.device for graph in graphs):
        raise ValueError("All graphs must be on the same device.")

    if any(
        (graphs[0].edge_features() is not None) != (graph.edge_features() is not None)
        for graph in graphs
    ):
        raise ValueError("Cannot batch graphs some of which have edges and some not.")

    if graphs[0].edge_features() is not None:
        if any(
            graphs[0].edge_features().shape[1] != graph.edge_features().shape[1]  # type: ignore
            for graph in graphs
        ):
            raise ValueError("All graphs must have the same number of edge features.")
