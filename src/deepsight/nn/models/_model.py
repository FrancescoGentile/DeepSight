##
##
##

import abc
from typing import Any, Generic, TypeVar

import torch
from torch import nn
from typing_extensions import Self

from deepsight.structures import Batch
from deepsight.typing import Moveable, Stateful

S = TypeVar("S")
O = TypeVar("O")  # noqa
A = TypeVar("A")
P = TypeVar("P")


class DeepSightModel(nn.Module, Moveable, Stateful, Generic[S, O, A, P], abc.ABC):
    """An interface for all models in DeepSight."""

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    @abc.abstractmethod
    def forward(self, samples: Batch[S], annotations: Batch[A] | None) -> O: ...

    @abc.abstractmethod
    def postprocess(self, output: O) -> Batch[P]: ...

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
        return self.to(device, non_blocking=non_blocking)

    def get_state(self) -> dict[str, Any]:
        return self.state_dict()

    def set_state(self, state: dict[str, Any]) -> None:
        self.load_state_dict(state)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __call__(self, samples: Batch[S], annotations: Batch[A] | None = None) -> O:
        return super().__call__(samples, annotations)
