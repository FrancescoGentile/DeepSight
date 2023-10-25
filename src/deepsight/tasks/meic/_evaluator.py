##
##
##

from collections.abc import Iterable
from typing import Annotated, Any

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from typing_extensions import Self

from deepsight.structures import Batch
from deepsight.tasks import Evaluator as _Evaluator
from deepsight.tasks import MetricType
from deepsight.typing import Moveable, Stateful

from ._structures import Predictions


class Evaluator(_Evaluator[Predictions], Moveable, Stateful):
    """Evaluator for the Multi-Entity Interaction Classification task."""

    def __init__(self, num_interaction_classes: int) -> None:
        self._cluster_metrics = MetricCollection(
            {"clustering_jaccard": ClusteringJaccard()}
        )

        self._meic_metrics = MetricCollection(
            {
                "meic_accuracy": MultilabelAccuracy(
                    num_interaction_classes, average="weighted"
                ),
                "meic_precision": MultilabelPrecision(
                    num_interaction_classes, average="weighted"
                ),
                "meic_recall": MultilabelRecall(
                    num_interaction_classes, average="weighted"
                ),
                "meic_f1score": MultilabelF1Score(
                    num_interaction_classes, average="weighted"
                ),
            }
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def metrics(self) -> Iterable[tuple[str, MetricType]]:
        return [
            ("clustering_jaccard", MetricType.NUMERIC),
            ("meic_accuracy", MetricType.NUMERIC),
            ("meic_precision", MetricType.NUMERIC),
            ("meic_recall", MetricType.NUMERIC),
            ("meic_f1score", MetricType.NUMERIC),
        ]

    @property
    def device(self) -> torch.device:
        return next(iter(self._cluster_metrics.values())).device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def update(
        self,
        predictions: Batch[Predictions],
        ground_truth: Batch[Predictions],
    ) -> None:
        for prediction, target in zip(predictions, ground_truth, strict=True):
            pbm = prediction.interactions.int().to_dense()  # (N, H)
            tbm = target.interactions.int().to_dense()  # (N, H')
            shared_nodes = pbm.t().mm(tbm)  # (H, H')
            nodes_per_cluster = tbm.sum(dim=0, keepdim=True)  # (H', 1)
            equal = shared_nodes.eq(nodes_per_cluster)  # (H, H')

            self._cluster_metrics.update(equal)

            matched_indices = equal.nonzero(as_tuple=False)  # (K, 2)
            pred_labels = prediction.interaction_labels
            target_labels = torch.zeros_like(pred_labels)
            target_labels[matched_indices[:, 0]] = target.interaction_labels[
                matched_indices[:, 1]
            ]
            self._meic_metrics.update(pred_labels, target_labels)

    def compute_numeric_metrics(self) -> dict[str, float]:
        return {
            **self._cluster_metrics.compute(),
            **self._meic_metrics.compute(),
        }

    def reset(self) -> None:
        self._cluster_metrics.reset()
        self._meic_metrics.reset()

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
        self._cluster_metrics = self._cluster_metrics.to(
            device, non_blocking=non_blocking
        )
        self._meic_metrics = self._meic_metrics.to(device, non_blocking=non_blocking)

        return self

    def get_state(self) -> dict[str, Any]:
        return {
            "cluster_metrics": self._cluster_metrics.state_dict(),
            "meic_metrics": self._meic_metrics.state_dict(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._cluster_metrics.load_state_dict(state["cluster_metrics"])
        self._meic_metrics.load_state_dict(state["meic_metrics"])


class ClusteringJaccard(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.add_state("num_equal", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_equal: Tensor
        self.num_total: Tensor

    def update(self, equal: Annotated[Tensor, "H H'", bool]) -> None:
        num_equal = equal.sum()
        self.num_equal += num_equal
        self.num_total += equal.shape[0] + equal.shape[1] - num_equal

    def compute(self) -> float:
        return (self.num_equal / self.num_total).item()
