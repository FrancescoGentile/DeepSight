# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import abc

from deepsight.data import Batch
from deepsight.modules import Module
from deepsight.typing import Moveable, Stateful


class Model[S, O, A, P](Module, Moveable, Stateful, abc.ABC):
    """An interface for all models in DeepSight."""

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    @abc.abstractmethod
    def postprocess(self, output: O) -> Batch[P]: ...

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    @abc.abstractmethod
    def __call__(self, samples: Batch[S], annotations: Batch[A] | None = None) -> O:
        return super().__call__(samples, annotations)
