##
##
##

"""The Human-Object Interaction Classification task."""

from ._evaluator import ErrorStrategy, Evaluator
from ._h2o_dataset import H2ODataset
from ._structures import Annotations, Predictions, Sample

__all__ = [
    "Annotations",
    "ErrorStrategy",
    "Evaluator",
    "H2ODataset",
    "Predictions",
    "Sample",
]
