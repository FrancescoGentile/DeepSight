# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Literal, Self, overload

import torch
from torch.nn.utils.rnn import pad_sequence

from deepsight.typing import Number, SparseTensor, Tensor

from ._common import BatchMode


class CombinatorialComplex:
    """A structure representing a combinatorial complex."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        cell_features: Sequence[Tensor[Literal["N D"], Number]],
        boundary_matrices: Sequence[Tensor[Literal["N M"], Number]],
        num_cells: Sequence[Sequence[int]] | None = None,
        boundary_matrices_sizes: Sequence[Sequence[int]] | None = None,
    ) -> None:
        r"""Initialize a combinatorial complex.

        Args:
            cell_features: An iterable of tensors representing the features of the
                cells of the complex. The cells are assumed to be ordered in ascending
                order of their rank, i.e. the first tensor in the iterable represents
                the 0-cells (vertices), the second tensor represents the 1-cells
                (hyper-edges), the third tensor represents the 2-cells (faces), and
                so on.
            boundary_matrices: An iterable of tensors representing the boundary
                matrices of the cells of the complex. The matrices are assumed to
                be ordered in ascending order of their rank, i.e. the first tensor
                in the iterable is the $B_1$ matrix, the second tensor is the $B_2$
                matrix, and so on. For more information on the boundary matrices,
                see the documentation of the `boundary_matrix` method.
            num_cells: If the combinatorial complex is obtained by batching multiple
                combinatorial complexes, then this argument should be an iterable
                of tuples, each containing the number of cells of the corresponding
                rank in each of the complexes. If the combinatorial complex is not
                batched, then this argument should be `None`.
            boundary_matrices_sizes: If the combinatorial complex is obtained by
                batching multiple combinatorial complexes, then this argument should
                be an iterable of tuples, each containing the number of non-zero
                entries in the corresponding boundary matrix in each of the complexes.
                If the combinatorial complex is not batched, then this argument
                should be `None`.
        """
        cell_features = tuple(cell_features)
        boundary_matrices = tuple(b.to_sparse_coo() for b in boundary_matrices)
        if num_cells is None:
            num_cells = tuple((c.shape[0],) for c in cell_features)

        if boundary_matrices_sizes is None:
            boundary_matrices_sizes = tuple((bm._nnz(),) for bm in boundary_matrices)

        _check_tensors(
            cell_features, boundary_matrices, num_cells, boundary_matrices_sizes
        )

        self._cell_features = cell_features
        self._boundary_matrices = tuple(boundary_matrices)
        self._num_cells = num_cells
        self._boundary_matrices_sizes = boundary_matrices_sizes

    @classmethod
    def batch(cls, complexes: Sequence[Self]) -> Self:
        """Batch multiple combinatorial complexes into a single complex.

        !!! note

            If the given list contains only one element, then that element is
            returned.
        """
        if len(complexes) == 0:
            raise ValueError("Expected at least one complex, got none.")
        if len(complexes) == 1:
            return complexes[0]

        if any(complexes[0].rank != c.rank for c in complexes):
            raise ValueError(
                "Expected all complexes to have the same rank, got "
                f"{[c.rank for c in complexes]}."
            )

        cell_features: list[torch.Tensor] = []
        boundary_matrices: list[torch.Tensor] = []
        num_cells: list[tuple[int, ...]] = []
        boundary_matrices_sizes: list[tuple[int, ...]] = []

        for r in range(complexes[0].rank + 1):
            cell_features_r = torch.cat([c.cell_features(r) for c in complexes], dim=0)
            cell_features.append(cell_features_r)

            num_cells.append(tuple(c.cell_features(r).shape[0] for c in complexes))

            if r == 0:
                # the 0-cells have no boundary matrix
                continue

            indices = []
            values = []

            r_minus_1_cell_offset = 0
            r_cell_offset = 0
            bm_sizes = []

            for c in complexes:
                bm_indices = c.boundary_matrix(r).indices()
                bm_values = c.boundary_matrix(r).values()
                bm_sizes.append(bm_values.shape[0])

                bm_indices_r_minus_1 = bm_indices[0] + r_minus_1_cell_offset
                bm_indices_r = bm_indices[1] + r_cell_offset

                indices.append(torch.stack([bm_indices_r_minus_1, bm_indices_r], dim=0))
                values.append(bm_values)

                r_minus_1_cell_offset += c.cell_features(r - 1).shape[0]
                r_cell_offset += c.cell_features(r).shape[0]

            indices = torch.cat(indices, dim=1)
            values = torch.cat(values, dim=0)

            boundary_matrices.append(
                torch.sparse_coo_tensor(
                    indices,
                    values,
                    size=(r_minus_1_cell_offset, r_cell_offset),
                    is_coalesced=True,
                )
            )
            boundary_matrices_sizes.append(tuple(bm_sizes))

        return cls(cell_features, boundary_matrices, num_cells, boundary_matrices_sizes)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def rank(self) -> int:
        """The rank of the complex.

        The rank of a combinatiorial complex is the maximum rank of its cells.
        """
        return len(self._cell_features) - 1

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    @overload
    def num_cells(
        self, rank: int, batch_mode: Literal[BatchMode.CONCAT] = BatchMode.CONCAT
    ) -> int: ...

    @overload
    def num_cells(
        self, rank: int, batch_mode: Literal[BatchMode.SEQUENCE]
    ) -> tuple[int, ...]: ...

    def num_cells(
        self,
        rank: int,
        batch_mode: Literal[BatchMode.CONCAT, BatchMode.SEQUENCE] = BatchMode.CONCAT,
    ) -> int | tuple[int, ...]:
        """Get the number of cells of the given rank.

        Args:
            rank: The rank of the cells whose number to get. Must be between 0
                and the rank of the complex.
            batch_mode: The mode in which to return the number of cells. This is
                only relevant if the complex is batched.

        Returns:
            The number of cells of the given rank.

        Raises:
            ValueError: If `rank` is not between 0 and the rank of the complex.
        """
        if not 0 <= rank <= self.rank:
            raise ValueError(f"Expected a rank between 0 and {self.rank}, got {rank}.")

        match batch_mode:
            case BatchMode.CONCAT:
                return sum(self._num_cells[rank])
            case BatchMode.SEQUENCE:
                return tuple(self._num_cells[rank])

    @overload
    def cell_features(
        self, rank: int, batch_mode: Literal[BatchMode.CONCAT] = BatchMode.CONCAT
    ) -> Tensor[Literal["N D"], Number]: ...

    @overload
    def cell_features(
        self, rank: int, batch_mode: Literal[BatchMode.STACK]
    ) -> Tensor[Literal["B N D"], Number]: ...

    @overload
    def cell_features(
        self, rank: int, batch_mode: Literal[BatchMode.SEQUENCE]
    ) -> tuple[Tensor[Literal["N D"], Number], ...]: ...

    def cell_features(
        self, rank: int, batch_mode: BatchMode = BatchMode.CONCAT
    ) -> (
        Tensor[Literal["N D"], Number]
        | Tensor[Literal["B N D"], Number]
        | tuple[Tensor[Literal["N D"], Number], ...]
    ):
        """Get the features of the cells of the given rank.

        Args:
            rank: The rank of the cells whose features to get. Must be between 0
                and the rank of the complex.
            batch_mode: The mode in which to return the cell features. This is only
                relevant if the complex is batched. If `batch_mode` is set to `CONCAT`,
                then the cell features of all batched complexes are concatenated.
                If `batch_mode` is set to `STACK`, then the cell features are stacked
                along the batch dimension (padding with zeros is used if the number of
                cells is not the same across the batched complexes). If `batch_mode` is
                set to `SEQUENCE`, then the cell features for each batched complex are
                returned.

        Returns:
            The features of the cells of the given rank.

        Raises:
            ValueError: If `rank` is not between 0 and the rank of the complex.
        """
        if not 0 <= rank <= self.rank:
            raise ValueError(f"Expected a rank between 0 and {self.rank}, got {rank}.")

        match batch_mode:
            case BatchMode.CONCAT:
                return self._cell_features[rank]
            case BatchMode.STACK:
                return pad_sequence(
                    self._cell_features[rank].split_with_sizes(self._num_cells[rank]),
                    batch_first=True,
                )
            case BatchMode.SEQUENCE:
                return tuple(
                    self._cell_features[rank].split_with_sizes(self._num_cells[rank])
                )

    @overload
    def boundary_matrix(
        self, rank: int, batch_mode: Literal[BatchMode.CONCAT] = BatchMode.CONCAT
    ) -> SparseTensor[Literal["N M"], Number]: ...

    @overload
    def boundary_matrix(
        self, rank: int, batch_mode: Literal[BatchMode.STACK]
    ) -> SparseTensor[Literal["B N M"], Number]: ...

    @overload
    def boundary_matrix(
        self, rank: int, batch_mode: Literal[BatchMode.SEQUENCE]
    ) -> tuple[SparseTensor[Literal["N M"], Number], ...]: ...

    def boundary_matrix(
        self, rank: int, batch_mode: BatchMode = BatchMode.CONCAT
    ) -> (
        SparseTensor[Literal["N M"], Number]
        | SparseTensor[Literal["B N M"], Number]
        | tuple[SparseTensor[Literal["N M"], Number], ...]
    ):
        r"""Get the boundary matrix for the given rank.

        The boundary matrix $B_r$ is a matrix that records to which cells of rank
        $r$ each cell of rank $r - 1$ is bound. More formally, it is a matrix of
        size $n_{r-1} \times n_{r}$, with $n_r$ denoting the number of cells of rank
        $r \geq 1$, defined as follows:

        $$
        B_{r, i, j} = \begin{cases}
            1 & \text{if x_{i}^{r-1} \prec x_{j}^{r}} \\
            0 & \text{otherwise},
        \end{cases}
        $$

        where $x_{i}^{r-1}$ denotes the $i$-th cell of rank $r - 1$ and $x_{j}^{r}$
        denotes the $j$-th cell of rank $r$. In case of a weighted complex, the
        value 1 is replaced by the weight of the connection between the two cells.

        Args:
            rank: The rank of the boundary matrix to get. Must be between 1 and the
                rank of the complex.
            batch_mode: The mode in which to return the boundary matrix. This is only
                relevant if the complex is batched.

        Returns:
            The boundary matrix of the given rank as a sparse COO tensor.

        Raises:
            ValueError: If `rank` is not between 1 and the rank of the complex.
        """
        if not 1 <= rank <= self.rank:
            raise ValueError(f"Expected a rank between 1 and {self.rank}, got {rank}.")

        if batch_mode == BatchMode.CONCAT:
            return self._boundary_matrices[rank - 1]

        indices = list(
            self._boundary_matrices[rank - 1]
            .indices()
            .split_with_sizes(self._boundary_matrices_sizes[rank - 1], dim=1)
        )
        values = list(
            self._boundary_matrices[rank - 1]
            .values()
            .split_with_sizes(self._boundary_matrices_sizes[rank - 1])
        )

        r_minus_1_cell_offset = 0
        r_cell_offset = 0
        for idx in range(len(self._num_cells[rank])):
            r_minus_1_ncells = self._num_cells[rank - 1][idx]
            r_ncells = self._num_cells[rank][idx]

            bm_indices_r_minus_1 = indices[idx][0] - r_minus_1_cell_offset
            bm_indices_r = indices[idx][1] - r_cell_offset

            if batch_mode == BatchMode.STACK:
                batch_idx = bm_indices_r.new_full((bm_indices_r.shape[0],), idx)
                indices[idx] = torch.stack(
                    [batch_idx, bm_indices_r_minus_1, bm_indices_r], dim=0
                )
            else:
                indices[idx] = torch.stack([bm_indices_r_minus_1, bm_indices_r], dim=0)

            r_minus_1_cell_offset += r_minus_1_ncells
            r_cell_offset += r_ncells

        if batch_mode == BatchMode.STACK:
            indices = torch.cat(indices, dim=1)
            values = torch.cat(values, dim=0)
            max_r_minus_1_ncells = max(self._num_cells[rank - 1])
            max_r_ncells = max(self._num_cells[rank])

            return torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=(len(self._num_cells[rank]), max_r_minus_1_ncells, max_r_ncells),
            )

        return tuple(
            torch.sparse_coo_tensor(
                indices=indices[idx],
                values=values[idx],
                size=(self._num_cells[rank - 1][idx], self._num_cells[rank][idx]),
            )
            for idx in range(len(indices))
        )

    @overload
    def coboundary_matrix(
        self, rank: int, batch_mode: Literal[BatchMode.CONCAT] = BatchMode.CONCAT
    ) -> SparseTensor[Literal["M N"], Number]: ...

    @overload
    def coboundary_matrix(
        self, rank: int, batch_mode: Literal[BatchMode.STACK]
    ) -> SparseTensor[Literal["B M N"], Number]: ...

    @overload
    def coboundary_matrix(
        self, rank: int, batch_mode: Literal[BatchMode.SEQUENCE]
    ) -> tuple[SparseTensor[Literal["M N"], Number], ...]: ...

    def coboundary_matrix(
        self, rank: int, batch_mode: BatchMode = BatchMode.CONCAT
    ) -> (
        SparseTensor[Literal["M N"], Number]
        | SparseTensor[Literal["B M N"], Number]
        | tuple[SparseTensor[Literal["M N"], Number], ...]
    ):
        r"""Get the coboundary matrix for the given rank.

        The coboundary matrix $B_r^T$ is the transpose of the boundary matrix $B_r$.
        It records to which cells of rank $r - 1$ each cell of rank $r$ is bound.
        More formally, it is a matrix of size $n_{r} \times n_{r-1}$, with $n_r$
        denoting the number of cells of rank $r \geq 1$, defined as follows:

        $$
        B_{r, i, j} = \begin{cases}
            1 & \text{if x_{j}^{r-1} \prec x_{i}^{r}} \\
            0 & \text{otherwise},
        \end{cases}
        $$

        where $x_{i}^{r-1}$ denotes the $i$-th cell of rank $r - 1$ and $x_{j}^{r}$
        denotes the $j$-th cell of rank $r$. In case of a weighted complex, the
        value 1 is replaced by the weight of the connection between the two cells.

        Args:
            rank: The rank of the coboundary matrix to get. Must be between 1 and
                the rank of the complex.
            batch_mode: The mode in which to return the coboundary matrix. This is
                only relevant if the complex is batched.

        Returns:
            The coboundary matrix of the given rank as a sparse COO tensor.
        """
        match batch_mode:
            case BatchMode.CONCAT:
                return self.boundary_matrix(rank).transpose(0, 1)
            case BatchMode.STACK:
                return self.boundary_matrix(rank, batch_mode).transpose(1, 2)
            case BatchMode.SEQUENCE:
                return tuple(bm.T for bm in self.boundary_matrix(rank, batch_mode))

    def lower_laplacian_matrix(self, rank: int) -> SparseTensor[Literal["N N"], Number]:
        r"""Get the lower Laplacian matrix for the given rank.

        The lower Laplacian matrix of rank $r$ records to which cells of rank $r$
        each cell of rank $r$ is lower adjacent. Two cells $x_i^r$ and $x_j^r$ are
        lower adjacent if there is a cell $x_k^{r-1}$ such that $x_k^{r-1} \prec x_i^r$
        and $x_k^{r-1} \prec x_j^r$. In other words, two cells are lower adjacent if
        there is a cell of rank $r - 1$ that is bound to both of them.

        More formally, the lower Laplacian matrix $L_{\downarrow, r}$ is defined as
        $L_{\downarrow, r} = B_r^T B_r$.

        Args:
            rank: The rank of the lower Laplacian matrix to get. Must be between 1
                and the rank of the complex.

        Returns:
            The lower Laplacian matrix of the given rank as a sparse COO tensor.
        """
        return torch.mm(self.coboundary_matrix(rank), self.boundary_matrix(rank))

    def upper_laplacian_matrix(self, rank: int) -> SparseTensor[Literal["N N"], Number]:
        r"""Get the upper Laplacian matrix for the given rank.

        The upper Laplacian matrix of rank $r$ records to which cells of rank $r$
        each cell of rank $r$ is upper adjacent. Two cells $x_i^r$ and $x_j^r$ are
        upper adjacent if there is a cell $x_k^{r+1}$ such that $x_i^r \prec x_k^{r+1}$
        and $x_j^r \prec x_k^{r+1}$. In other words, two cells are upper adjacent if
        they are both bound to the same cell of rank $r + 1$.

        More formally, the upper Laplacian matrix $L_{\uparrow, r}$ is defined as
        $L_{\uparrow, r} = B_{r+1} B_{r+1}^T$.

        Args:
            rank: The rank of the upper Laplacian matrix to get. Must be between 0
                and the rank of the complex - 1.

        Returns:
            The upper Laplacian matrix of the given rank as a sparse COO tensor.
        """
        return torch.mm(
            self.boundary_matrix(rank + 1), self.coboundary_matrix(rank + 1)
        )

    def lower_degree_matrix(self, rank: int) -> SparseTensor[Literal["N N"], Number]:
        """Get the lower degree matrix for the given rank.

        The lower degree matrix of rank $r$ is a diagonal matrix that records the
        number of cells of rank $r - 1$ that are bound to each cell of rank $r$.
        For example, in a hypergraph, the lower degree matrix of rank 1 (the
        hyper-edges) records the number of vertices that are part of each hyper-edge.

        Args:
            rank: The rank of the lower degree matrix to get. Must be between 1
                and the rank of the complex.

        Returns:
            The lower degree matrix of the given rank as a sparse COO tensor.
        """
        coboundary_matrix = self.coboundary_matrix(rank)
        degree = torch.sparse.sum(coboundary_matrix, dim=-1).to_dense()
        return torch.diag_embed(degree).to_sparse_coo()

    def upper_degree_matrix(self, rank: int) -> SparseTensor[Literal["N N"], Number]:
        """Get the upper degree matrix for the given rank.

        The upper degree matrix of rank $r$ is a diagonal matrix that records the
        number of cells of rank $r + 1$ to which each cell of rank $r$ is bound.
        For example, in a graph, the upper degree matrix of rank 0 (the vertices)
        records the number of edges that are incident to each vertex.

        Args:
            rank: The rank of the upper degree matrix to get. Must be between 0
                and the rank of the complex - 1.

        Returns:
            The upper degree matrix of the given rank as a sparse COO tensor.
        """
        boundary_matrix = self.boundary_matrix(rank + 1)
        degree = torch.sparse.sum(boundary_matrix, dim=-1).to_dense()
        return torch.diag_embed(degree).to_sparse_coo()

    def lower_adjacency_matrix(self, rank: int) -> SparseTensor[Literal["N N"], Number]:
        r"""Get the lower adjacency matrix for the given rank.

        The lower adjacency matrix is defined as
        $D_{\downarrow, r} - L_{\downarrow, r}$, where $D_{\downarrow, r}$ is the
        lower degree matrix of rank $r$ and $L_{\downarrow, r}$ is the lower Laplacian
        matrix of rank $r$.

        Args:
            rank: The rank of the lower adjacency matrix to get. Must be between 1
                and the rank of the complex.

        Returns:
            The lower adjacency matrix of the given rank as a sparse COO tensor.
        """
        return self.lower_degree_matrix(rank) - self.lower_laplacian_matrix(rank)

    def upper_adjacency_matrix(self, rank: int) -> SparseTensor[Literal["N N"], Number]:
        r"""Get the upper adjacency matrix for the given rank.

        The upper adjacency matrix is defined as
        $D_{\uparrow, r} - L_{\uparrow, r}$, where $D_{\uparrow, r}$ is the upper
        degree matrix of rank $r$ and $L_{\uparrow, r}$ is the upper Laplacian
        matrix of rank $r$.

        Args:
            rank: The rank of the upper adjacency matrix to get. Must be between 0
                and the rank of the complex - 1.

        Returns:
            The upper adjacency matrix of the given rank as a sparse COO tensor.
        """
        return self.upper_degree_matrix(rank) - self.upper_laplacian_matrix(rank)

    def replace(  # noqa
        self,
        *args: tuple[
            Tensor[Literal["N D"], Number]
            | Tensor[Literal["B N D"], Number]
            | Iterable[Tensor[Literal["N D"], Number]],
            int,
        ],
    ) -> Self:
        """Replace the features of the cells of the given ranks.

        !!! note

            If no arguments are given, then `self` is returned.

        Args:
            *args: A sequence of tuples, each containing the new features of the
                cells of the corresponding rank and the rank of the cells. The features
                can be given as a matrix, a batched matrix, or a sequence of matrices
                similar to the return value of the `cell_features` method. The rank
                must be between 0 and the rank of the complex.
        """
        if len(args) == 0:
            return self

        cell_features = list(self._cell_features)
        for cell_feature, rank in args:
            if not 0 <= rank <= self.rank:
                raise ValueError(
                    f"Expected a rank between 0 and {self.rank}, got {rank}."
                )

            if isinstance(cell_feature, torch.Tensor):
                match cell_feature.ndim:
                    case 2:
                        if cell_feature.shape[0] != cell_features[rank].shape[0]:
                            raise ValueError(
                                f"Expected the number of cells of rank {rank} to be "
                                f"{cell_features[rank].shape[0]}, "
                                f"got {cell_feature.shape[0]}."
                            )
                    case 3:
                        rank_cell_features = []
                        for idx, num_cells in enumerate(self._num_cells[rank]):
                            rank_cell_features.append(cell_feature[idx, :num_cells])
                        cell_feature = torch.cat(rank_cell_features, dim=0)
                    case _:
                        raise TypeError("Expected a matrix or a batched matrix.")
            else:
                # check that the number of cells for each batched complex is correct
                for cf, ncells in zip(cell_feature, self._num_cells[rank], strict=True):
                    if cf.shape[0] != ncells:
                        raise ValueError(
                            "The number of cells for each batched complex must remain "
                            "the same when replacing the cell features."
                        )

                cell_feature = torch.cat(list(cell_feature), dim=0)

            cell_features[rank] = cell_feature

        return self.__class__(
            cell_features,
            self._boundary_matrices,
            self._num_cells,
            self._boundary_matrices_sizes,
        )

    def unbatch(self) -> tuple[Self, ...]:
        """Unbatch the combinatorial complex into a sequence of complexes.

        !!! note

            If the combinatorial complex was not batched, then the sequence will
            contain only one element, which is `self`.
        """
        B = len(self._num_cells[0])  # noqa
        if B < 2:
            return (self,)

        cell_features = [[] for _ in range(B)]
        boundary_matrices = [[] for _ in range(B)]

        for r in range(self.rank + 1):
            cf = tuple(self.cell_features(r, batch_mode=BatchMode.SEQUENCE))
            for idx in range(B):
                cell_features[idx].append(cf[idx])

            if r == 0:
                continue

            bm = tuple(self.boundary_matrix(r, batch_mode=BatchMode.SEQUENCE))
            for idx in range(B):
                boundary_matrices[idx].append(bm[idx])

        return tuple(
            self.__class__(cf, bm)
            for cf, bm in zip(cell_features, boundary_matrices, strict=True)
        )

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = (
        "_cell_features",
        "_boundary_matrices",
        "_num_cells",
        "_boundary_matrices_sizes",
    )


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _check_tensors(  # noqa
    cell_features: Sequence[Tensor[Literal["N D"], Number]],
    boundary_matrices: Sequence[Tensor[Literal["N M"], Number]],
    num_cells: Sequence[Sequence[int]],
    boundary_matrices_sizes: Sequence[Sequence[int]],
) -> None:
    if len(cell_features) == 0:
        raise ValueError("Expected at least one cell feature, got none.")

    if len(cell_features) != len(boundary_matrices) + 1:
        raise ValueError(
            f"Expected the number of cell features to be one more than the "
            f"number of boundary matrices, got {len(cell_features)} cell "
            f"features and {len(boundary_matrices)} boundary matrices."
        )

    if len(cell_features) != len(num_cells):
        raise ValueError(
            f"Expected the number of cell features to be equal to the number "
            f"of cell sizes, got {len(cell_features)} cell features and "
            f"{len(num_cells)} cell sizes."
        )

    if len(boundary_matrices) != len(boundary_matrices_sizes):
        raise ValueError(
            f"Expected the number of boundary matrices to be equal to the "
            f"number of boundary matrix sizes, got {len(boundary_matrices)} "
            f"boundary matrices and {len(boundary_matrices_sizes)} boundary "
            f"matrix sizes."
        )

    for r, (cell_feature, cell_splits) in enumerate(
        zip(cell_features, num_cells, strict=True)
    ):
        if cell_feature.ndim != 2:
            raise ValueError(
                f"Expected the cell features of rank {r} to be a matrix, "
                f"got a tensor of shape {cell_feature.shape}."
            )

        if cell_feature.shape[0] != sum(cell_splits):
            raise ValueError(
                f"Expected the number of cells of rank {r} to be "
                f"{sum(cell_splits)}, got {cell_feature.shape[0]}."
            )

    for r_minus_1, (boundary_matrix, bm_sizes) in enumerate(
        zip(boundary_matrices, boundary_matrices_sizes, strict=True)
    ):
        # check that the incidence matrices have the correct shape
        num_cells_r_minus_1 = cell_features[r_minus_1].shape[0]
        num_cells_r = cell_features[r_minus_1 + 1].shape[0]

        if boundary_matrix.shape != (num_cells_r_minus_1, num_cells_r):
            raise ValueError(
                f"Expected the boundary matrix B_{r_minus_1 + 1} to have shape "
                f"({num_cells_r_minus_1}, {num_cells_r}), got "
                f"{boundary_matrix.shape}."
            )

        if boundary_matrix._nnz() != sum(bm_sizes):
            raise ValueError(
                f"Expected the number of non-zero entries in the boundary "
                f"matrix B_{r_minus_1 + 1} to be {sum(bm_sizes)}, got "
                f"{boundary_matrix._nnz()}."
            )

    if any(len(num_cells[0]) != len(nc) for nc in num_cells):
        raise ValueError("Inconsistent number of batches.")

    if any(
        len(boundary_matrices_sizes[0]) != len(bm) for bm in boundary_matrices_sizes
    ):
        raise ValueError("Inconsistent number of batches.")

    if len(num_cells[0]) != len(boundary_matrices_sizes[0]):
        raise ValueError("Inconsistent number of batches.")

    # for nc, bm in zip(num_cells, boundary_matrices_sizes, strict=True):
    #     if len(nc) != len(bm):
    #         raise ValueError("Inconsistent number of batches.")
