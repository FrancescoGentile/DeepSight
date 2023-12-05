##
##
##

import enum
from typing import Any, Literal, Self

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelAveragePrecision

from deepsight.core import Batch, MetricInfo
from deepsight.core import Evaluator as _Evaluator
from deepsight.typing import Configs, Configurable, Moveable, Stateful, Tensor, str_enum

from ._structures import Predictions


@str_enum
class ErrorStrategy(enum.Enum):
    """The error strategy to use when a prediction is invalid.

    A prediction is invalid if it contains an interaction where the subject is not a
    human or the target is not an object (if `allow_human_human` is `False`).

    Attributes:
        NONE: No validation is performed.
        IGNORE: Invalid predictions are ignored.
        WARN: A warning is raised when an invalid prediction is encountered and the
            prediction is ignored.
        RAISE: An error is raised when an invalid prediction is encountered.
    """

    NONE = "none"
    IGNORE = "ignore"
    WARN = "warn"
    RAISE = "raise"


class Evaluator(_Evaluator[Predictions], Moveable, Stateful, Configurable):
    """Evaluator for the Human-Object Interaction (HOI) task.

    The evaluator computes the accuracy and mean average precision (mAP) for the
    predictions.
    """

    def __init__(
        self,
        num_interaction_classes: int,
        average: Literal["micro", "macro", "weighted"] = "weighted",
        thresholds: int | list[float] | None = None,
        error_strategy: ErrorStrategy = ErrorStrategy.NONE,
        human_class_id: int | None = None,
        allow_human_human: bool | None = None,
    ) -> None:
        """Initializes a new evaluator.

        Args:
            num_interaction_classes: The number of interaction classes.
            average: The averaging strategy to use for the metrics.
            thresholds: The thresholds to use for the computation of the mean average
                precision. See `torchmetrics.classification.MultilabelAveragePrecision`
                for more information.
            error_strategy: The error strategy to use when a prediction is invalid.
            human_class_id: The class ID of the human class. This must be set if
                `error_strategy` is not set to `ErrorStrategy.NONE`.
            allow_human_human: Whether to allow human-human interactions. This must
                be set if `error_strategy` is not set to `ErrorStrategy.NONE`.
        """
        self._error_strategy = error_strategy
        self._human_class_id = human_class_id
        self._allow_human_human = allow_human_human

        if error_strategy != ErrorStrategy.NONE:
            if human_class_id is None:
                raise ValueError(
                    "The class ID of the human class must be set if the error strategy "
                    f"is set to {error_strategy}."
                )

            if allow_human_human is None:
                raise ValueError(
                    "Whether to allow human-human interactions must be set if the "
                    f"error strategy is set to {error_strategy}."
                )

        self._metrics = MetricCollection(
            {
                "accuracy": MultilabelAccuracy(
                    num_labels=num_interaction_classes, average=average
                ),
                "mAP": MultilabelAveragePrecision(
                    num_labels=num_interaction_classes,
                    average=average,
                    thresholds=thresholds,
                ),
            },
            compute_groups=False,
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        return next(iter(self._metrics.values())).device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_metrics_info(self) -> tuple[MetricInfo, ...]:
        return tuple(
            MetricInfo(name=str(name), type=MetricInfo.Type.NUMERIC)
            for name in self._metrics.keys()
        )

    def update(
        self, predictions: Batch[Predictions], ground_truth: Batch[Predictions]
    ) -> None:
        predicted_labels = []
        target_labels = []

        for prediction, target in zip(predictions, ground_truth, strict=True):
            pred, tgt = _match_prediction_ground_truth(prediction, target)
            predicted_labels.append(pred)
            target_labels.append(tgt)

        predicted_labels = torch.cat(predicted_labels, dim=0)
        target_labels = torch.cat(target_labels, dim=0)

        match self._error_strategy:
            case ErrorStrategy.NONE:
                pass
            case ErrorStrategy.IGNORE:
                # TODO: Implement
                raise NotImplementedError
            case ErrorStrategy.WARN:
                # TODO: Implement
                raise NotImplementedError
            case ErrorStrategy.RAISE:
                # TODO: Implement
                raise NotImplementedError

        self._metrics.update(predicted_labels, target_labels.round_().int())

    def compute_numeric_metrics(self) -> dict[str, float]:
        return self._metrics.compute()

    def reset(self) -> None:
        self._metrics.reset()

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        self._metrics = self._metrics.to(device, non_blocking=non_blocking)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {"metrics": self._metrics.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        self._metrics.load_state_dict(state_dict["metrics"])

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "average": self._metrics["mAP"].average,
            "thresholds": list(self._metrics["mAP"].thresholds),
            "allow_human_human": self._allow_human_human,
        }


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _match_prediction_ground_truth(
    pred: Predictions, ground_truth: Predictions
) -> tuple[Tensor[Literal["M V"], float], Tensor[Literal["M V"], float]]:
    num_classes = pred.interaction_labels.shape[1]

    matched = (
        pred.interaction_indices.unsqueeze(1) == ground_truth.interaction_indices
    )  # (I, I', 2)
    matched = matched.all(dim=2)  # (I, I')
    matched_pred, matched_target = torch.nonzero(matched, as_tuple=True)

    not_matched = matched.logical_not_()
    not_matched_pred = not_matched.all(dim=1).nonzero(as_tuple=True)[0]
    not_matched_target = not_matched.all(dim=0).nonzero(as_tuple=True)[0]

    total = len(matched_pred) + len(not_matched_pred) + len(not_matched_target)
    predictions = pred.interaction_labels.new_zeros((total, num_classes))
    targets = pred.interaction_labels.new_zeros((total, num_classes))

    # MATCHED PRED - MATCHED TARGET
    # NOT MATCHED PRED - fake target
    # fake pred - NOT MATCHED TARGET

    predictions[: len(matched_pred)] = pred.interaction_labels[matched_pred]
    targets[: len(matched_target)] = ground_truth.interaction_labels[matched_target]

    predictions[
        len(matched_pred) : len(matched_pred) + len(not_matched_pred)
    ] = pred.interaction_labels[not_matched_pred]

    targets[
        len(matched_target) + len(not_matched_pred) :
    ] = ground_truth.interaction_labels[not_matched_target]

    return predictions, targets
