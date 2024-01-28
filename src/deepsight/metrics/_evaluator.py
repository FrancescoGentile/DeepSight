# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass
from typing import Protocol

from deepsight.data import Batch
from deepsight.typing import Number, str_enum


@str_enum
class MetricType(enum.Enum):
    """Type of a metric computed by an evaluator."""

    NUMERIC = "numeric"
    TABULAR = "tabular"


@dataclass(frozen=True)
class MetricInfo:
    name: str
    type: MetricType
    higher_is_better: bool = True


class Evaluator[P](Protocol):
    """Interface for evaluators.

    Evaluators are used to compute metrics on the predictions of a model trained
    on a particular task. Since evaluators are task-specific, they can be shared
    between models and datasets of the same task.
    """

    def get_metrics_info(self) -> tuple[MetricInfo, ...]:
        """Get the info of the metrics computed by the evaluator.

        Returns:
            The metrics info.
        """
        ...

    def update(self, predictions: Batch[P], ground_truth: Batch[P]) -> None:
        """Update the metrics with the given predictions and ground-truth."""
        ...

    def compute_numeric_metrics(self) -> dict[str, Number]:
        """Compute the numeric metrics.

        Numeric metrics are metrics that can be represented as a number. For
        example, the (micro, macro, and weighted) F1 score of a classification
        model can be represented as a single number.

        Returns:
            The numeric metrics computed by the evaluator.
        """
        ...

    # def compute_tabular_metrics(self) -> tuple[tuple[str, dict[str, Number]]]:
    #     """Compute the tabular metrics.

    #     Tabular metrics are metrics that can be represented as a table. For
    #     example, the class-wise metrics of a classification model can be
    #     represented as a table where each row corresponds to a class and each
    #     column corresponds to a metric (e.g. precision, recall, etc.).
    #     """
    #     ...

    def reset(self) -> None:
        """Reset the metrics."""
        ...
