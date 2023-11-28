##
##
##

import abc

import torch
from torch import nn

from deepsight.typing import Moveable, Stateful

from ._batch import Batch


class Model[S, O, A, P](nn.Module, Moveable, Stateful, abc.ABC):
    """An interface for all models in DeepSight."""

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

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
