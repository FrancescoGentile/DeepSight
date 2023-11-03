##
##
##

from collections.abc import Iterable
from typing import Annotated, Any

import sklearn.metrics
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
        # Multi-Entity Interaction Classification Metrics
        self._jaccard_index = JaccardIndex()
        self._adjusted_rand_index = RandIndex(adjusted=True)
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

        # Binary Human-Object Interaction Classification Metrics
        self._hoic_metrics = MetricCollection(
            {
                "hoic_accuracy": MultilabelAccuracy(
                    num_interaction_classes, average="weighted"
                ),
                "hoic_precision": MultilabelPrecision(
                    num_interaction_classes, average="weighted"
                ),
                "hoic_recall": MultilabelRecall(
                    num_interaction_classes, average="weighted"
                ),
                "hoic_f1score": MultilabelF1Score(
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
            ("jaccard_index", MetricType.NUMERIC),
            ("adjusted_rand_index", MetricType.NUMERIC),
            ("meic_accuracy", MetricType.NUMERIC),
            ("meic_precision", MetricType.NUMERIC),
            ("meic_recall", MetricType.NUMERIC),
            ("meic_f1score", MetricType.NUMERIC),
            ("hoic_accuracy", MetricType.NUMERIC),
            ("hoic_precision", MetricType.NUMERIC),
            ("hoic_recall", MetricType.NUMERIC),
            ("hoic_f1score", MetricType.NUMERIC),
        ]

    @property
    def device(self) -> torch.device:
        return self._jaccard_index.device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def update(
        self, predictions: Batch[Predictions], ground_truth: Batch[Predictions]
    ) -> None:
        for prediction, target in zip(predictions, ground_truth, strict=True):
            self._compute_meic_metrics(prediction, target)
            self._compute_hoic_metrics(prediction, target)

    def compute_numeric_metrics(self) -> dict[str, float]:
        return {
            "jaccard_index": self._jaccard_index.compute(),
            "adjusted_rand_index": self._adjusted_rand_index.compute(),
            **self._meic_metrics.compute(),
            **self._hoic_metrics.compute(),
        }

    def reset(self) -> None:
        self._jaccard_index.reset()
        self._adjusted_rand_index.reset()
        self._meic_metrics.reset()
        self._hoic_metrics.reset()

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
        self._jaccard_index.to(device, non_blocking=non_blocking)
        self._adjusted_rand_index.to(device, non_blocking=non_blocking)
        self._meic_metrics.to(device, non_blocking=non_blocking)
        self._hoic_metrics.to(device, non_blocking=non_blocking)

        return self

    def get_state(self) -> dict[str, Any]:
        return {
            "jaccard_index": self._jaccard_index.state_dict(),
            "adjusted_rand_index": self._adjusted_rand_index.state_dict(),
            "meic_metrics": self._meic_metrics.state_dict(),
            "hoic_metrics": self._hoic_metrics.state_dict(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._jaccard_index.load_state_dict(state["jaccard_index"])
        self._adjusted_rand_index.load_state_dict(state["adjusted_rand_index"])
        self._meic_metrics.load_state_dict(state["meic_metrics"])
        self._hoic_metrics.load_state_dict(state["hoic_metrics"])

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _compute_meic_metrics(self, pred: Predictions, target: Predictions) -> None:
        pred_bm = pred.interactions.int().to_dense()  # (N, H)
        gt_bm = target.interactions.int().to_dense()  # (N, H')
        self._adjusted_rand_index.update(pred_bm, gt_bm)

        shared_nodes = pred_bm.t().mm(gt_bm)  # (H, H')
        nodes_per_cluster = gt_bm.sum(dim=0, keepdim=True)  # (H', 1)
        matched = shared_nodes.eq(nodes_per_cluster)  # (H, H')
        self._jaccard_index.update(matched)

        pred_labels, gt_labels = _match_labels(
            pred.interaction_labels, target.interaction_labels, matched
        )
        self._meic_metrics.update(pred_labels, gt_labels)

    def _compute_hoic_metrics(self, pred: Predictions, target: Predictions) -> None:
        pred_indices = pred.binary_interactions  # (E, 2)
        gt_indices = target.binary_interactions  # (E', 2)

        matched = pred_indices.unsqueeze(1) == gt_indices  #  (E, E', 2)
        matched = matched.all(dim=2)  # (E, E')

        pred_labels, gt_labels = _match_labels(
            pred.interaction_labels, target.interaction_labels, matched
        )

        self._hoic_metrics.update(pred_labels, gt_labels)


# --------------------------------------------------------------------------- #
# Clustering Metrics
# --------------------------------------------------------------------------- #


class JaccardIndex(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.add_state("num_equal", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_equal: Tensor
        self.num_total: Tensor

    def update(self, matched: Annotated[Tensor, "H H'", bool]) -> None:
        num_equal = matched.sum()
        self.num_equal += num_equal
        self.num_total += matched.shape[0] + matched.shape[1] - num_equal

    def compute(self) -> float:
        return (self.num_equal / self.num_total).item()


class RandIndex(Metric):
    def __init__(self, adjusted: bool = True) -> None:
        super().__init__()

        self.adjusted = adjusted
        self.add_state("score", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.score: Tensor
        self.total: Tensor

    def update(
        self,
        pred_bm: Annotated[Tensor, "N H", bool],
        gt_bm: Annotated[Tensor, "N H'", bool],
    ) -> None:
        pred_labels = _compute_cluster_labels(pred_bm).cpu().numpy()
        gt_labels = _compute_cluster_labels(gt_bm).cpu().numpy()

        if self.adjusted:
            score = sklearn.metrics.adjusted_rand_score(pred_labels, gt_labels)
        else:
            score = sklearn.metrics.rand_score(pred_labels, gt_labels)

        self.score += score
        self.total += 1

    def compute(self) -> float:
        return (self.score / self.total).item()


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _compute_cluster_labels(
    bm: Annotated[Tensor, "N H", bool]
) -> Annotated[Tensor, "NH", int]:
    cluster_ids = torch.arange(bm.shape[1], device=bm.device)
    cluster_ids = cluster_ids.unsqueeze_(0).expand_as(bm)  # (N, H)

    node_ids = torch.arange(bm.shape[0], device=bm.device)
    node_ids += bm.shape[1]
    node_ids = node_ids.unsqueeze_(1).expand_as(bm)  # (N, H)

    labels = torch.where(bm, cluster_ids, node_ids)  # (N, H)
    labels = labels.flatten()  # (NH,)
    return labels


def _match_labels(
    pred_labels: Annotated[Tensor, "N C", float],
    gt_labels: Annotated[Tensor, "M C", float],
    matched: Annotated[Tensor, "N M", bool],
) -> tuple[Annotated[Tensor, "T C", float], Annotated[Tensor, "T C", float]]:
    matched_pred, matched_target = torch.nonzero(matched, as_tuple=True)

    not_matched = matched.logical_not_()
    not_matched_pred = not_matched.all(dim=1).nonzero(as_tuple=True)[0]
    not_matched_target = not_matched.all(dim=0).nonzero(as_tuple=True)[0]

    total = len(matched_pred) + len(not_matched_pred) + len(not_matched_target)
    predictions = pred_labels.new_zeros((total, pred_labels.shape[1]))
    targets = pred_labels.new_zeros((total, pred_labels.shape[1]))

    predictions[: len(matched_pred)] = pred_labels[matched_pred]
    targets[: len(matched_target)] = gt_labels[matched_target]

    predictions[
        len(matched_pred) : len(matched_pred) + len(not_matched_pred)
    ] = pred_labels[not_matched_pred]

    targets[len(matched_target) + len(not_matched_pred) :] = gt_labels[
        not_matched_target
    ]

    return predictions, targets
