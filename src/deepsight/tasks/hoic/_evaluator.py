##
##
##

import enum
from collections.abc import Iterable
from typing import Annotated, Any

import torch
from torch import Tensor
from torchmetrics import MetricCollection
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
from deepsight.typing import Moveable, Stateful, str_enum

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


class Evaluator(_Evaluator[Predictions], Moveable, Stateful):
    """Evaluator for the Human-Object Interaction (HOI) task.

    The evaluator computes the accuracy, precision, recall, and F1 score of the
    predictions. At the moment, only the weighted-averaged metrics are supported.
    """

    def __init__(
        self,
        num_interaction_classes: int,
        error_strategy: ErrorStrategy = ErrorStrategy.NONE,
        human_class_id: int | None = None,
        allow_human_human: bool | None = None,
    ) -> None:
        """Initializes a new evaluator.

        Args:
            num_interaction_classes: The number of interaction classes.
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
                    num_labels=num_interaction_classes, average="weighted"
                ),
                "precision": MultilabelPrecision(
                    num_labels=num_interaction_classes, average="weighted"
                ),
                "recall": MultilabelRecall(
                    num_labels=num_interaction_classes, average="weighted"
                ),
                "f1_score": MultilabelF1Score(
                    num_labels=num_interaction_classes, average="weighted"
                ),
            }
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def metrics(self) -> Iterable[tuple[str, MetricType]]:
        return [
            ("accuracy", MetricType.NUMERIC),
            ("precision", MetricType.NUMERIC),
            ("recall", MetricType.NUMERIC),
            ("f1_score", MetricType.NUMERIC),
        ]

    @property
    def device(self) -> torch.device:
        return next(iter(self._metrics.values())).device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def update(
        self,
        predictions: Batch[Predictions],
        ground_truth: Batch[Predictions],
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

        self._metrics.update(predicted_labels, target_labels)

    def compute_numeric_metrics(self) -> dict[str, float]:
        return self._metrics.compute()

    def reset(self) -> None:
        self._metrics.reset()

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
        self._metrics = self._metrics.to(device, non_blocking=non_blocking)
        return self

    def get_state(self) -> dict[str, Any]:
        return {
            "metrics": self._metrics.state_dict(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._metrics.load_state_dict(state["metrics"])


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _match_prediction_ground_truth(
    pred: Predictions, ground_truth: Predictions
) -> tuple[Annotated[Tensor, "M V", float], Annotated[Tensor, "M V", bool]]:
    num_classes = pred.interaction_labels.shape[1]
    same_subject = pred.interactions[0].unsqueeze(1) == ground_truth.interactions[0]
    same_object = pred.interactions[1].unsqueeze(1) == ground_truth.interactions[1]

    matched = same_subject & same_object  # (E_pred, E_target)
    matched_pred, matched_target = torch.nonzero(matched, as_tuple=True)

    not_matched = matched.logical_not_()
    not_matched_pred = not_matched.all(dim=1).nonzero(as_tuple=True)[0]
    not_matched_target = not_matched.all(dim=0).nonzero(as_tuple=True)[0]

    total = len(matched_pred) + len(not_matched_pred) + len(not_matched_target)
    predictions = pred.interaction_labels.new_zeros((total, num_classes))
    targets = pred.interaction_labels.new_zeros((total, num_classes), dtype=torch.bool)

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
