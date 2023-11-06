##
##
##

import enum
from collections.abc import Iterable
from typing import Generic, Protocol, TypeVar

from deepsight.structures import Batch
from deepsight.typing import number, str_enum

P = TypeVar("P")


@str_enum
class MetricType(enum.Enum):
    """Metric types."""

    NUMERIC = "numeric"
    TABULAR = "tabular"


class Evaluator(Generic[P], Protocol):
    """Interface for evaluators.

    Evaluators are used to compute metrics on the predictions of a model trained
    on a particular task. Since evaluators are task-specific, they can be shared
    between models and datasets of the same task.
    """

    @property
    def metrics(self) -> Iterable[tuple[str, MetricType]]:
        """The metrics computed by the evaluator.

        Returns:
            The metrics computed by the evaluator. Each metric is represented as a
            tuple containing the name of the metric and its type.
        """
        ...

    def update(self, predictions: Batch[P], ground_truth: Batch[P]) -> None:
        """Update the metrics with the given predictions and ground-truth."""
        ...

    def compute_numeric_metrics(self) -> dict[str, number]:
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
