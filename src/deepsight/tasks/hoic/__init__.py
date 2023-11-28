##
##
##

"""The Human-Object Interaction Classification task."""

from ._evaluator import ErrorStrategy, Evaluator
from ._structures import Annotations, Predictions, Sample

__all__ = [
    # _evaluator
    "ErrorStrategy",
    "Evaluator",
    # _structures
    "Annotations",
    "Predictions",
    "Sample",
]
