##
##
##

import abc

from torch import nn

from deepsight.typing import Moveable, Stateful
from deepsight.utils import Batch


class Model[S, O, A, P](nn.Module, Moveable, Stateful, abc.ABC):
    """An interface for all models in DeepSight."""

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    @abc.abstractmethod
    def forward(self, samples: Batch[S], annotations: Batch[A] | None) -> O: ...

    @abc.abstractmethod
    def postprocess(self, output: O) -> Batch[P]: ...

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __call__(self, samples: Batch[S], annotations: Batch[A] | None = None) -> O:
        return super().__call__(samples, annotations)
