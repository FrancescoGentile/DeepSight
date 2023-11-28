##
##
##

"""Multi-Entity Interaction Classification task."""

from ._evaluator import Evaluator
from ._structures import Annotations, Predictions, Sample

__all__ = [
    # _evaluator
    "Evaluator",
    # _structures
    "Annotations",
    "Predictions",
    "Sample",
]
