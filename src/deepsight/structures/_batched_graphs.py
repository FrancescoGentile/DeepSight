##
##
##

from numbers import Number
from typing import Annotated

import torch
from torch import Tensor
from typing_extensions import Self

from deepsight.structures import Graph
from deepsight.typing import Moveable


class BatchedGraphs(Moveable):
    """Structure to hold a batch of graphs."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(self, graph: Graph, graph_sizes: tuple[tuple[int, int], ...]) -> None:
        """Initialize a batched graph.

        Args:
            graph: The batch of graphs stored as a single graph.
            graph_sizes: The sizes (number of nodes and edges) of the original graphs.
        """
        self._graph = graph
        self._graph_sizes = graph_sizes

    @classmethod
    def batch(cls, graphs: list[Graph]) -> Self:
        """Batch a list of graphs into a single batched graph."""
        _check_graphs(graphs)

        sizes = tuple((graph.num_nodes, graph.num_edges) for graph in graphs)

        node_features = torch.cat([graph.node_features for graph in graphs])
        if graphs[0].edge_features is not None:
            edge_features = torch.cat([graph.edge_features for graph in graphs])  # type: ignore
        else:
            edge_features = None

        adj_indices = []
        adj_values = []

        node_offset = 0
        for graph in graphs:
            adj = graph.adjacency_matrix.indices()
            adj_indices.append(graph.adjacency_matrix.indices() + node_offset)
            adj_values.append(adj.values())

            node_offset += graph.num_nodes

        adj_indices = torch.cat(adj_indices, dim=1)
        adj_values = torch.cat(adj_values)

        adjacency = torch.sparse_coo_tensor(
            adj_indices,
            adj_values,
            size=(node_offset, node_offset),
        )

        graph = Graph(adjacency, node_features, edge_features)
        return cls(graph, sizes)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def adjacency_matrix(self) -> Annotated[Tensor, "N N", Number, torch.sparse_coo]:
        """The adjacency matrix of the batched graph."""
        return self._graph.adjacency_matrix

    @property
    def node_features(self) -> Annotated[Tensor, "N D", Number]:
        """The node features of the batched graph."""
        return self._graph.node_features

    @property
    def edge_features(self) -> Annotated[Tensor, "E C", Number] | None:
        """The edge features of the batched graph."""
        return self._graph.edge_features

    @property
    def num_nodes(self) -> int:
        """The number of nodes in the batched graph."""
        return self._graph.num_nodes

    @property
    def device(self) -> torch.device:
        """The device on which the batched graph is stored."""
        return self._graph.device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def unbatch(self) -> list[Graph]:
        """Unbatch the batched graph into a list of graphs."""
        graphs: list[Graph] = []

        node_offset = 0
        edge_offset = 0

        for num_nodes, num_edges in self._graph_sizes:
            node_limit = node_offset + num_nodes
            edge_limit = edge_offset + num_edges

            node_features = self.node_features[node_offset:node_limit]
            if self.edge_features is not None:
                edge_features = self.edge_features[edge_offset:edge_limit]
            else:
                edge_features = None

            adj_indices = self.adjacency_matrix.indices()[edge_offset:edge_limit]
            adj_indices = adj_indices - node_offset
            adj_values = self.adjacency_matrix.values()[edge_offset:edge_limit]
            adj = torch.sparse_coo_tensor(
                adj_indices,
                adj_values,
                size=(num_nodes, num_nodes),
            )

            graph = Graph(adj, node_features, edge_features)
            graphs.append(graph)

            node_offset = node_limit
            edge_offset = edge_limit

        return graphs

    def replace(
        self,
        node_features: Annotated[Tensor, "N D", Number] | None = None,
        edge_features: Annotated[Tensor, "E C", Number] | None = None,
    ) -> Self:
        """Return a new instance of the batched graph with the given features."""
        graph = self._graph.replace(node_features, edge_features)
        return self.__class__(graph, self._graph_sizes)

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
        if self.device == device:
            return self

        graph = self._graph.move(device, non_blocking)
        return self.__class__(graph, self._graph_sizes)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Return the number of graphs in the batched graph."""
        return len(self._graph_sizes)

    def __getitem__(self, index: int) -> Graph:
        """Return the graph at the given index in the batched graph."""
        node_offset = sum(num_nodes for num_nodes, _ in self._graph_sizes[:index])
        edge_offset = sum(num_edges for _, num_edges in self._graph_sizes[:index])
        node_limit = node_offset + self._graph_sizes[index][0]
        edge_limit = edge_offset + self._graph_sizes[index][1]

        node_features = self.node_features[node_offset:node_limit]
        if self.edge_features is not None:
            edge_features = self.edge_features[edge_offset:edge_limit]
        else:
            edge_features = None

        adj_indices = self.adjacency_matrix.indices()[edge_offset:edge_limit]
        adj_indices = adj_indices - node_offset
        adj_values = self.adjacency_matrix.values()[edge_offset:edge_limit]
        adj = torch.sparse_coo_tensor(
            adj_indices,
            adj_values,
            size=(self._graph_sizes[index][0], self._graph_sizes[index][0]),
        )

        return Graph(adj, node_features, edge_features)

    def __repr__(self) -> str:
        """Return a string representation of the batched graph."""
        return (
            f"{self.__class__.__name__}(num_graphs={len(self)}, "
            f"num_nodes={self._graph.num_nodes}, num_edges={self._graph.num_edges})"
        )

    def __str__(self) -> str:
        """Return a string representation of the batched graph."""
        return repr(self)

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_graph", "_graph_sizes")


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _check_graphs(graphs: list[Graph]) -> None:
    """Check that the graphs are compatible to be batched."""
    if len(graphs) == 0:
        raise ValueError("Cannot batch an empty list of graphs.")

    if any(
        graphs[0].node_features.shape[1] != graph.node_features.shape[1]
        for graph in graphs
    ):
        raise ValueError("All graphs must have the same number of node features.")

    if any(graphs[0].device != graph.device for graph in graphs):
        raise ValueError("All graphs must be on the same device.")

    if any(
        graphs[0].edge_features is not None == graph.edge_features is not None
        for graph in graphs
    ):
        raise ValueError("Cannot batch graphs some of which have edges and some not.")
